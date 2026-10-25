"""Apply the validated contour HUD to every frame of local thermal recordings."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from hud import HUDConfig, HUDRenderer  # noqa: E402
from replay import load_fire360_osd_regions  # noqa: E402
from teed_core import TEEDCore, sha256_file  # noqa: E402

OUT = Path(__file__).with_name("contour_stability") / "thermal_full"
CLIPS = [("fire360_ifsi4", "IFSI Video 4 (2).mp4"),
         ("fire360_ifsi8", "IFSI Video 8 (2).mp4")]
CODE_FILES = ["edge/orin/teed_core.py", "edge/orin/teed_model.py",
              "edge/orin/edge_filter.py", "edge/orin/temporal_edges.py",
              "edge/orin/contour_tracker.py", "edge/orin/contour_render.py",
              "edge/orin/hud.py", "edge/orin/thermal_overlay.py",
              "edge/orin/replay.py", "edge/teed/profiles/fire360_video_profiles.json",
              "experiments/teed_ui/render_full_thermal_videos.py"]


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def panel_template(name: str) -> np.ndarray:
    canvas = Image.new("RGB", (1440, 540), "#101820")
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    draw.text((22, 8), name + " | 열화상 원본 전체", font=small, fill="#d7e2ee")
    draw.text((22, 43), "기존 · 확률 안정화", font=large, fill="white")
    draw.text((742, 43), "적용 · 윤곽 추적 + 부드러운 선", font=large, fill="#8bf0a9")
    draw.text((22, 500), "첫 프레임부터 끝까지 / 같은 시점·원본 재생 속도 / 무음", font=small, fill="#d7e2ee")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def render_clip(core: TEEDCore, name: str, filename: str, hashes: dict) -> dict:
    source = ROOT.parent / "data/fire360/raw" / filename
    digest = sha256_file(source)
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open thermal source: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(prop)) for prop in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT])
    if fps <= 0 or expected <= 0:
        cap.release()
        raise ValueError(f"Invalid source timeline: {filename}")
    regions = load_fire360_osd_regions(str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"), filename)
    before = HUDRenderer(HUDConfig(output_size=(720, 405), temporal_edges=True))
    after = HUDRenderer(HUDConfig(output_size=(720, 405), temporal_edges=True, contour_edges=True))
    outputs = [
        {"path": str((OUT / f"{name}_full_applied.webm").resolve()), "role": "applied", "width": 720, "height": 406, "frames": expected, "fps": fps},
        {"path": str((OUT / f"{name}_full_comparison.webm").resolve()), "role": "comparison", "width": 1440, "height": 540, "frames": expected, "fps": fps},
    ]
    writers = []
    records = []
    panel_base = panel_template(filename)
    try:
        for output in outputs:
            writer = cv2.VideoWriter(output["path"], cv2.VideoWriter.fourcc(*"VP80"), fps, (output["width"], output["height"]))
            writers.append(writer)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create {output['path']}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = len(records)
            edge = core.infer(frame, regions)
            background = cv2.resize(frame, (720, 405), interpolation=cv2.INTER_AREA)
            timestamp = index / fps
            old = before.render(background, edge, ignore_regions=regions, frame_index=index, timestamp_s=timestamp)
            new = after.render(background, edge, ignore_regions=regions, frame_index=index, timestamp_s=timestamp)
            writers[0].write(cv2.copyMakeBorder(new.image, 0, 1, 0, 0, cv2.BORDER_CONSTANT))
            panel = panel_base.copy()
            panel[84:489, :720] = old.image
            panel[84:489, 720:] = new.image
            cv2.line(panel, (720, 84), (720, 489), (90, 90, 90), 1)
            cv2.putText(panel, f"{index + 1}/{expected} | {timestamp:.2f}/{expected / fps:.2f}s", (1050, 28), cv2.FONT_HERSHEY_SIMPLEX, .42, (225, 225, 225), 1, cv2.LINE_AA)
            writers[1].write(panel)
            assert old.edge_mask is not None and new.edge_mask is not None
            records.append({"frame_index": index, "source_seconds": timestamp,
                            "teed_ms": edge.timings_ms["total"],
                            "hud_ms_before": old.render_ms, "hud_ms_after": new.render_ms,
                            "edge_pixels_before": int(np.count_nonzero(old.edge_mask)),
                            "edge_pixels_after": int(np.count_nonzero(new.edge_mask)),
                            "contour": new.edge_metrics.get("contour", {})})
            if index in {0, expected // 4, expected // 2, 3 * expected // 4, expected - 1}:
                cv2.imwrite(str(OUT / f"{name}_qa_{index:04d}.png"), panel)
            if index % 60 == 0 or index == expected - 1:
                save_json(OUT / "progress.json", {"status": "RUNNING", "clip": name, "processed": index + 1, "expected": expected})
                print(f"{name}: {index + 1}/{expected}", flush=True)
    finally:
        cap.release()
        for writer in writers:
            writer.release()
    if len(records) != expected:
        raise RuntimeError(f"Incomplete source decode {filename}: {len(records)}/{expected}")
    if sha256_file(source) != digest:
        raise RuntimeError(f"Source changed during rendering: {filename}")
    for file, value in hashes.items():
        if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != value:
            raise RuntimeError(f"Code changed during rendering: {file}")
    result = {"name": name, "source": {"path": str(source.resolve()), "sha256": digest,
              "expected_frames": expected, "fps": fps, "width": width, "height": height},
              "frames": len(records), "start_frame": 0, "end_frame": len(records) - 1,
              "duration_s": len(records) / fps, "code_sha256": hashes, "outputs": outputs,
              "config": dataclasses.asdict(after.config), "osd_regions": regions,
              "model": core.describe(), "audio": "silent exports; source files unchanged",
              "records": records}
    save_json(OUT / f"{name}.json", result)
    print(f"COMPLETE {name}: {len(records)} frames, {len(records) / fps:.3f}s", flush=True)
    return {key: result[key] for key in ["name", "source", "frames", "start_frame", "end_frame", "duration_s", "outputs"]}


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in CODE_FILES}
    core = TEEDCore(ROOT / "edge/teed/models/5_model.pth", width=320, height=256,
                    threshold=.75, edge_width=3, contrast_clip_limit=2, device="cpu")
    try:
        clips = [render_clip(core, name, filename, hashes) for name, filename in CLIPS]
        save_json(OUT / "manifest.json", {"status": "RENDER_COMPLETE", "scope": "complete local thermal sources only", "clips": clips, "code_sha256": hashes})
        save_json(OUT / "progress.json", {"status": "RENDER_COMPLETE", "frames": sum(clip["frames"] for clip in clips)})
    except Exception as error:
        save_json(OUT / "progress.json", {"status": "FAILED", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
