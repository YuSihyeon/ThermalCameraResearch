import csv
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import render_min_length_videos as comparison
import torch
from edge_filter import EdgeFilterConfig
from hud import HUDConfig, HUDRenderer
from PIL import Image, ImageDraw, ImageFont
from replay import ReplaySource
from teed_core import TEEDCore


def labels(title):
    canvas = Image.new("RGB", comparison.SIZE, "#101820")
    draw = ImageDraw.Draw(canvas)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    draw.text((22, 8), title + " | 잔여 엣지 개선 비교", font=small, fill="#d7e2ee")
    draw.text((22, 43), "기존 · 최소 길이 12 px", font=large, fill="white")
    draw.text((742, 43), "개선 · 짧은 가지 + 작은 덩어리 제거", font=large, fill="#8bf0a9")
    draw.text((22, 497), "동일 TEED / 가지 길이 6 px · 덩어리 범위 8 px / 시간 필터 없음", font=small, fill="#d7e2ee")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = comparison.ROOT
    comparison.OUT = Path(__file__).with_name("residual_edge_results")
    comparison.OUT.mkdir(exist_ok=True)
    comparison.bands = labels
    before_config = EdgeFilterConfig(min_branch_length=0, min_extent=0)
    after_config = EdgeFilterConfig()
    before = HUDRenderer(HUDConfig(output_size=(720, 405), edge_filter=before_config))
    after = HUDRenderer(HUDConfig(output_size=(720, 405), edge_filter=after_config))
    core = TEEDCore(root / "edge/teed/models/5_model.pth", width=320, height=256,
                    threshold=.75, edge_width=3, contrast_clip_limit=2.0, device="cpu", half=False)
    with (root.parent / "data/external/smokebasement/indexes/run5_left.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    first = float(rows[0]["sensor_timestamp_seconds"])
    selected = [r for r in rows if first + 6 <= float(r["sensor_timestamp_seconds"]) < first + 12]
    fps = (len(selected) - 1) / (float(selected[-1]["sensor_timestamp_seconds"]) - float(selected[0]["sensor_timestamp_seconds"]))
    summaries = [comparison.render("smokebasement_cleanup", "SmokeBasement",
        comparison.smoke_frames(selected, root.parent / "data/external/smokebasement/raw/20250121"),
        fps, len(selected), (), core, before, after)]
    video = root.parent / "data/fire360/raw/IFSI Video 4 (2).mp4"
    replay = ReplaySource(str(video), profile_path=str(root / "edge/teed/profiles/fire360_video_profiles.json"),
                          video_key=video.name, max_frames=1)
    regions = replay.info.osd_regions
    replay.close()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    summaries.append(comparison.render("fire360_cleanup", "Fire360 / IFSI Video 4",
        itertools.islice(comparison.video_frames(video, 360), 180),
        fps, 180, regions, core, before, after))
    report = {"before": dataclasses.asdict(before_config), "after": dataclasses.asdict(after_config),
              "model_resolution": [320, 256], "files": summaries,
              "limits": ["Spatial filtering only; no temporal stabilization",
                         "Pixel reduction is not obstacle recall or detection accuracy",
                         "No hardware performance validation"],
              "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                for p in ["edge/orin/edge_filter.py", "edge/orin/hud.py",
                          "experiments/teed_ui/render_residual_edges.py",
                          "experiments/teed_ui/render_min_length_videos.py"]}}
    (comparison.OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
