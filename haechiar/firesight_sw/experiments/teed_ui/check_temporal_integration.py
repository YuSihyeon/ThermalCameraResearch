from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from edge_filter import EdgeFilterConfig  # noqa: E402
from hud import HUDConfig, HUDRenderer  # noqa: E402
from replay import ReplaySource  # noqa: E402
from run_pipeline import build_parser  # noqa: E402
from teed_core import EdgeResult  # noqa: E402

OUT = Path(__file__).with_name("temporal_stability") / "integration_checks"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    parser = build_parser()
    assert parser.parse_args([]).temporal_edges
    assert not parser.parse_args(["--no-temporal-edges"]).temporal_edges
    checks.append("CLI default enabled and explicit disable")
    for high, value in [(0.75, 0.80), (0.60, 0.65)]:
        renderer = HUDRenderer(
            HUDConfig(
                temporal_edges=True,
                edge_filter=EdgeFilterConfig(low_threshold=0.30, high_threshold=high),
            )
        )
        source = np.zeros((96, 128, 3), np.uint8)
        source[:, 32:80] = 200
        probability = np.zeros((96, 128), np.float32)
        first = EdgeResult(
            probability, np.zeros((96, 128), np.uint8), source, (128, 96), {}
        )
        renderer.render(source, first, frame_index=0, timestamp_s=0)
        probability[20:80, 31:34] = value
        current = EdgeResult(probability, first.mask, source, (128, 96), {})
        shown = renderer.render(source, current, frame_index=1, timestamp_s=1 / 30)
        assert shown.edge_metrics["output_edge_pixels"] > 0
        assert shown.edge_metrics["temporal"]["immediate_pixels"] == 180
        checks.append(f"new above-threshold edge immediately displayed at high={high}")
    with tempfile.TemporaryDirectory(prefix="firesight_temporal_") as temporary:
        path = Path(temporary) / "loop.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (64, 64)
        )
        assert writer.isOpened()
        for x in [10, 30]:
            frame = np.zeros((64, 64, 3), np.uint8)
            frame[10:50, x : x + 5] = 255
            writer.write(frame)
        writer.release()
        with ReplaySource(str(path), loop=True) as source:
            frames = list(itertools.islice(source.frames(), 5))
        assert [frame.discontinuity for frame in frames] == [
            False,
            False,
            True,
            False,
            True,
        ]
        checks.append("actual decoder loop signals history discontinuity")
    base = [
        sys.executable,
        str(ROOT / "edge/orin/run_pipeline.py"),
        "--source",
        str(ROOT.parent / "data/fire360/raw/IFSI Video 4 (2).mp4"),
        "--profiles",
        str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"),
        "--device",
        "cpu",
        "--headless",
        "--warmup",
        "0",
        "--max-frames",
        "4",
    ]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name, args in [
        ("enabled", ["--start-frame", "300"]),
        ("disabled", ["--start-frame", "300", "--no-temporal-edges"]),
        ("loop", ["--start-frame", "623", "--loop"]),
    ]:
        with (OUT / (name + ".log")).open("w") as log:
            result = subprocess.run(
                base + args + ["--metrics", str(OUT / (name + ".json"))],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        assert result.returncode == 0, (name, result.returncode)
        metrics = json.loads((OUT / (name + ".json")).read_text())
        assert metrics["processed_frames"] == 4
        frames = metrics["hud"]["edge_frames"]
        if name == "enabled":
            assert metrics["hud"]["temporal_edges_enabled"]
            assert frames[0]["temporal"]["temporal_reset"] == 1
            assert frames[1]["temporal"]["temporal_reset"] == 0
        elif name == "disabled":
            assert not metrics["hud"]["temporal_edges_enabled"]
            assert all(frame["temporal"]["status"] == "disabled" for frame in frames)
        else:
            assert frames[2]["temporal"]["temporal_reset"] == 1
        checks.append("production replay CLI " + name)
    (OUT / "checks.json").write_text(
        json.dumps({"status": "PASS", "checks": checks}, indent=2), encoding="utf-8"
    )
    print("PASS:", len(checks), "integration contracts")


if __name__ == "__main__":
    main()
