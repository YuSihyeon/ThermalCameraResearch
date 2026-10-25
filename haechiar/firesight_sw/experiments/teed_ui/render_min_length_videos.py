from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from edge_filter import EdgeFilterConfig  # noqa: E402
from hud import HUDConfig, HUDRenderer  # noqa: E402
from replay import ReplaySource  # noqa: E402
from teed_core import TEEDCore  # noqa: E402

OUT = Path(__file__).with_name("min_length_videos")
SIZE = (1440, 540)


def video_frames(path, start):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for index in range(start, total):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Decode failed: {index}/{total}")
            yield index, index / fps, frame
    finally:
        cap.release()


def smoke_frames(rows, root):
    for index, row in enumerate(rows):
        frame = cv2.imread(str(root / row["image"]))
        if frame is None:
            raise RuntimeError(row["image"])
        yield index, float(row["sensor_timestamp_seconds"]), frame


def bands(title):
    large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    image = Image.new("RGB", SIZE, "#101820")
    draw = ImageDraw.Draw(image)
    draw.text((22, 8), title + " | 최소 길이 필터 비교", font=small, fill="#d7e2ee")
    draw.text((22, 43), "필터 OFF · 길이 제한 없음", font=large, fill="white")
    draw.text((742, 43), "필터 ON · 12 px 미만 제거", font=large, fill="#8bf0a9")
    draw.text((22, 497), "동일 TEED·히스테리시스·밝기 조건 / 최소 길이만 0 → 12 변경", font=small, fill="#d7e2ee")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(name, title, frames, fps, expected, regions, core, before, after):
    base = bands(title)
    writers = []
    records = []
    try:
        for extension, codec in [("webm", "VP80"), ("mp4", "mp4v")]:
            writer = cv2.VideoWriter(str(OUT / f"{name}.{extension}"), cv2.VideoWriter_fourcc(*codec), fps, SIZE)
            if not writer.isOpened():
                raise RuntimeError("Cannot encode " + extension)
            writers.append(writer)
        for ordinal, (index, seconds, frame) in enumerate(frames):
            edge = core.infer(frame, ignore_regions=regions)
            a = before.render(frame, edge, pixel_origin="bgr8", ignore_regions=regions)
            b = after.render(frame, edge, pixel_origin="bgr8", ignore_regions=regions)
            count_a = int(a.edge_metrics["output_edge_pixels"])
            count_b = int(b.edge_metrics["output_edge_pixels"])
            assert count_a >= count_b
            panel = base.copy()
            panel[84:489, :720] = a.image
            panel[84:489, 720:] = b.image
            cv2.line(panel, (720, 84), (720, 489), (90, 90, 90), 1)
            cv2.putText(panel, f"frame {index} | t={seconds:.2f}s | edge px {count_a} -> {count_b}", (935, 28), cv2.FONT_HERSHEY_SIMPLEX, .42, (225, 225, 225), 1, cv2.LINE_AA)
            for writer in writers:
                writer.write(panel)
            records.append({"source_frame": index, "source_seconds": seconds, "before": count_a, "after": count_b, "metrics": b.edge_metrics})
            if ordinal in {0, expected // 2, expected - 1}:
                cv2.imwrite(str(OUT / f"{name}_qa_{ordinal:04d}.png"), panel)
            if ordinal % 100 == 0:
                print(f"{name}: {ordinal + 1}/{expected}", flush=True)
    finally:
        for writer in writers:
            writer.release()
    assert len(records) == expected
    result = {"name": name, "frames": len(records), "fps": fps, "seconds": len(records) / fps, "records": records}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"COMPLETE {name}: {len(records)} frames", flush=True)
    return {k: v for k, v in result.items() if k != "records"}


def main():
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=True)
    off = EdgeFilterConfig(min_length=0.0, min_branch_length=0.0, min_extent=0.0)
    on = EdgeFilterConfig(min_length=12.0, min_branch_length=0.0, min_extent=0.0)
    assert dataclasses.replace(off, min_length=12.0) == on
    before = HUDRenderer(HUDConfig(output_size=(720, 405), edge_filter=off))
    after = HUDRenderer(HUDConfig(output_size=(720, 405), edge_filter=on))
    core = TEEDCore(ROOT / "edge/teed/models/5_model.pth", width=320, height=256, threshold=.75, edge_width=3, contrast_clip_limit=2.0, device="cpu", half=False)
    with (ROOT.parent / "data/external/smokebasement/indexes/run5_left.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    first = float(rows[0]["sensor_timestamp_seconds"])
    selected = [r for r in rows if first + 6 <= float(r["sensor_timestamp_seconds"]) < first + 12]
    fps = (len(selected) - 1) / (float(selected[-1]["sensor_timestamp_seconds"]) - float(selected[0]["sensor_timestamp_seconds"]))
    smoke_root = ROOT.parent / "data/external/smokebasement/raw/20250121"
    summaries = [render("smokebasement_min_length", "SmokeBasement", smoke_frames(selected, smoke_root), fps, len(selected), (), core, before, after)]
    video = ROOT.parent / "data/fire360/raw/IFSI Video 4 (2).mp4"
    replay = ReplaySource(str(video), profile_path=str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"), video_key=video.name, max_frames=1)
    regions = replay.info.osd_regions
    replay.close()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 30
    cap.release()
    summaries.append(render("fire360_thermal_min_length", "Fire360 / IFSI Video 4", video_frames(video, 30), fps, expected, regions, core, before, after))
    report = {"comparison": "Only minimum skeleton length changes: 0 versus 12 model pixels", "other_settings": dataclasses.asdict(on), "playback": "Original scene timing, not live inference speed", "files": summaries, "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["edge/orin/edge_filter.py", "edge/orin/hud.py", "experiments/teed_ui/render_min_length_videos.py"]}}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
