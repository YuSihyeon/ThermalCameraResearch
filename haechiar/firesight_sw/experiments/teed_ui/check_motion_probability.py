"""Protect static probability stabilization and immediate dynamic evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from check_motion_separation import (  # noqa: E402
    scene_with_foreground,
    textured_scene,
    vertical_line,
)
from contour_tracker import ContourConfig  # noqa: E402
from temporal_edges import TemporalEdgeConfig, TemporalEdgeStabilizer  # noqa: E402


def main() -> None:
    cv2.setNumThreads(1)
    checks = []
    motion = ContourConfig(motion_separation=True)

    def tracker():
        return TemporalEdgeStabilizer(TemporalEdgeConfig(), motion_config=motion)

    test = tracker()
    frame = textured_scene()
    p = vertical_line(50)[1]
    test.update(p, frame, frame_index=0)
    weak = p * (0.65 / 0.9)
    out, metrics = test.update(weak, frame, frame_index=1)
    assert metrics["motion_probability_status"] == "ok"
    assert np.mean(out[25:95, 50]) > 0.8
    checks.append("supported static boundary dip is buffered upstream")
    blank = np.zeros_like(p)
    out, _ = test.update(blank, frame, frame_index=2)
    assert not out.any()
    checks.append("vanished static boundary is never resurrected")
    fresh = tracker()
    fresh.update(blank, frame, frame_index=0)
    out, _ = fresh.update(p, frame, frame_index=1)
    assert np.allclose(out, p)
    checks.append("strong new boundary remains immediate")
    for shift in [1.5, 4.0, 10.0]:
        test = tracker()
        for index in range(5):
            x = int(round(45 + index * shift))
            current = vertical_line(x)[1]
            out, _ = test.update(
                current, scene_with_foreground(index, x), frame_index=index
            )
            assert float(out[25:95, x].mean()) > 0.75
            assert not np.any(out[current == 0])
        checks.append(
            f"moving foreground {shift} px/frame stays current without trails"
        )
    test = tracker()
    test.update(p, frame, frame_index=0)
    out, metrics = test.update(weak, frame, frame_index=3)
    assert np.allclose(out, weak) and metrics["reset_reason"] == "frame_discontinuity"
    checks.append("discontinuity clears static probability history")
    regions = ((0.2, 0, 0.4, 1),)
    test = tracker()
    for index in range(3):
        out, _ = test.update(p, frame, ignore_regions=regions, frame_index=index)
        assert not out[:, 32:64].any()
    checks.append("OSD remains excluded from history and output")
    dest = Path(__file__).with_name("motion_split") / "probability_checks.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"status": "PASS", "checks": checks}, indent=2))
    print(f"PASS: {len(checks)} motion probability contracts")


if __name__ == "__main__":
    main()
