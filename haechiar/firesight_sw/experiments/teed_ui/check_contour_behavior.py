"""Black-box requirements: persistent geometry, onset, and disappearance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from contour_tracker import (  # noqa: E402
    ContourConfig,
    ContourPath,
    ContourTracker,
    _deduplicate,
    _smooth_points,
)

OUT = Path(__file__).with_name("contour_stability")


def inputs(xs, value=0.8, length=50):
    mask = np.zeros((96, 128), np.uint8)
    for x in xs:
        mask[20 : 20 + length, x] = 255
    return mask, (mask > 0).astype(np.float32) * value


def main():
    cv2.setNumThreads(1)
    frame = np.random.default_rng(42).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    checks = []
    tracker = ContourTracker()
    centers, ids = [], []
    for index in range(20):
        x = 40 + index % 2
        mask, p = inputs([x, 85])
        r = tracker.update(mask, p, frame, frame_index=index, timestamp_s=index / 30)
        assert len(r.paths) == 2, len(r.paths)
        paths = sorted(r.paths, key=lambda path: path.points[:, 0].mean())
        assert len({path.track_id for path in paths}) == 2, (
            "two different lines need distinct IDs"
        )
        ids.append([path.track_id for path in paths])
        centers.append(float(paths[0].points[:, 0].mean()))
    assert all(value == ids[0] for value in ids[1:]), "same lines must retain IDs"
    checks.append("distinct persistent IDs across 20 jittering frames")
    assert np.std(centers[5:]) < 0.4, centers
    assert any(abs(x - round(x)) > 0.05 for x in centers[1:]), centers
    checks.append(
        "geometry smoothing produces subpixel coordinates and reduces static jitter"
    )
    weak = ContourTracker()
    visible = []
    for index in range(3):
        mask, p = inputs([30], 0.65, 16)
        result = weak.update(mask, p, frame, frame_index=index, timestamp_s=index / 30)
        visible.append(bool(result.paths))
    assert visible == [False, False, True], visible
    checks.append("weak short curve requires observations from first frame onward")
    blank = np.zeros((96, 128), np.uint8)
    result = weak.update(
        blank, np.zeros_like(blank, np.float32), frame, frame_index=3, timestamp_s=0.1
    )
    assert not result.paths and not result.mask.any()
    checks.append("zero current evidence removes an established curve immediately")
    pending = ContourTracker()
    weak_mask, weak_p = inputs([30], 0.65, 16)
    assert not pending.update(weak_mask, weak_p, frame, frame_index=0).paths
    support_p = (weak_mask > 0).astype(np.float32) * 0.3
    assert not pending.update(blank, support_p, frame, frame_index=1).paths
    assert not pending.update(weak_mask, weak_p, frame, frame_index=2).paths
    assert pending.update(weak_mask, weak_p, frame, frame_index=3).paths
    checks.append(
        "pending weak curve remains hidden in support-only gap and confirms 3 of 4"
    )
    held_tracker = ContourTracker(ContourConfig(max_missing_frames=2))
    strong_mask, strong_p = inputs([30], 0.8, 16)
    held_tracker.update(strong_mask, strong_p, frame, frame_index=0)
    held_visibility = []
    for index in range(1, 4):
        held_visibility.append(
            bool(held_tracker.update(blank, support_p, frame, frame_index=index).paths)
        )
    assert held_visibility == [True, True, False], held_visibility
    checks.append("supported disappearance holds at most two frames")
    default_tracker = ContourTracker()
    default_tracker.update(strong_mask, strong_p, frame, frame_index=0)
    default_result = default_tracker.update(blank, support_p, frame, frame_index=1)
    assert not default_result.paths and default_result.metrics["held_paths"] == 0
    checks.append("default display does not add a second hold after probability stabilization")
    split_tracker = ContourTracker()
    whole, whole_p = inputs([30], 0.8, 50)
    split_tracker.update(whole, whole_p, frame, frame_index=0)
    split_mask = whole.copy()
    split_mask[36:54] = 0
    split_p = (whole > 0).astype(np.float32) * 0.3
    split_p[split_mask > 0] = 0.8
    split_result = split_tracker.update(split_mask, split_p, frame, frame_index=1)
    assert split_result.metrics["held_paths"] == 0
    assert not split_result.mask[42:48, 30].any()
    checks.append("old contour is not held over changed current path topology")
    junction_tracker = ContourTracker()
    for index in range(12):
        junction = np.zeros((96, 128), np.uint8)
        junction[20:70, 45 + index % 2] = 255
        junction[40, 20:76] = 255
        junction_result = junction_tracker.update(junction, (junction > 0).astype(np.float32) * 0.8, frame, frame_index=index)
        assert cv2.connectedComponents(junction_result.mask)[0] == 2
    checks.append("shared junctions remain connected through coordinate smoothing")
    output_filtered = ContourTracker(ContourConfig(min_output_length=12))
    short_mask, short_probability = inputs([30], 0.9, 8)
    long_mask, long_probability = inputs([70], 0.9, 16)
    output_result = output_filtered.update(short_mask | long_mask, np.maximum(short_probability, long_probability), frame)
    assert len(output_result.paths) == 1
    assert output_result.paths[0].points[:, 0].mean() == 70
    assert not output_result.mask[:, 30].any()
    assert output_result.metrics["dropped_output_components"] == 1
    checks.append("final length cleanup removes short raster and vector fragments together")
    compact_mask = long_mask.copy()
    cv2.rectangle(compact_mask, (25, 25), (30, 30), 255, 1)
    compact_tracker = ContourTracker(ContourConfig(min_output_length=12, min_output_extent=8))
    compact_result = compact_tracker.update(compact_mask, (compact_mask > 0).astype(np.float32) * 0.9, frame)
    assert len(compact_result.paths) == 1 and not compact_result.mask[24:32, 24:32].any()
    checks.append("final extent cleanup removes tiny loop while preserving strong line")
    bridge = np.zeros((96, 128), np.uint8)
    bridge[20:60, 30] = 255
    bridge[20:60, 44] = 255
    bridge[40, 30:45] = 255
    bridge_p = (bridge > 0).astype(np.float32) * 0.8
    bridge_p[40, 30:45] = 0.6
    bridge_result = ContourTracker().update(bridge, bridge_p, frame)
    assert cv2.connectedComponents(bridge_result.mask)[0] == 2
    assert bridge_result.mask[40, 35:40].all()
    checks.append("weak bridge in an observed structural graph is not delayed as isolated noise")
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    circle = np.column_stack((50 + 10 * np.cos(theta), 45 + 10 * np.sin(theta))).astype(
        np.float32
    )
    stable_circle = _smooth_points(
        np.roll(circle[::-1], 7, axis=0), circle, True, ContourConfig()
    )
    assert np.max(np.linalg.norm(stable_circle - circle, axis=1)) < 0.1
    checks.append("closed contour cyclic/reversed order does not collapse geometry")
    base = np.array([[30, 20], [30, 60]], np.float32)
    kept, dropped = _deduplicate(
        [
            ContourPath(base, 1.0, 1),
            ContourPath(base + [0.5, 0], 0.7, 2),
            ContourPath(base + [2.0, 0], 1.0, 3),
        ],
        ContourConfig(),
    )
    assert dropped == 1 and {path.track_id for path in kept} == {1, 3}
    checks.append(
        "weak coincident duplicate removed while close strong boundaries remain"
    )
    strong = ContourTracker()
    mask, p = inputs([30], 0.8, 16)
    result = strong.update(mask, p, frame, frame_index=0, timestamp_s=0)
    assert result.paths and result.mask.any()
    checks.append("new strong short curve is visible on its first frame")
    fast = ContourTracker()
    for index in range(5):
        x = 20 + index * 10
        mask, p = inputs([x])
        moving = np.roll(frame, index * 10, axis=1)
        result = fast.update(mask, p, moving, frame_index=index, timestamp_s=index / 30)
        assert result.paths
        coords = np.concatenate([path.points for path in result.paths])
        assert np.max(np.abs(coords[:, 0] - x)) <= 1.1, (index, coords[:, 0])
    checks.append(
        "fast 10px/frame movement stays within 1.1px without old-position trails"
    )
    close = ContourTracker()
    mask, p = inputs([40, 42], 0.9)
    result = close.update(mask, p, frame, frame_index=0, timestamp_s=0)
    assert (
        len(result.paths) == 2
        and np.any(result.mask[:, 40])
        and np.any(result.mask[:, 42])
    )
    checks.append("nearby distinct strong parallel boundaries are both retained")
    evidence = {
        "status": "PASS",
        "checks": checks,
        "static_raw_jitter_std_px": 0.5,
        "static_output_jitter_std_px": float(np.std(centers[5:])),
        "centers": centers,
        "weak_onset_frames": visible,
    }
    (OUT / "behavior_checks.json").write_text(json.dumps(evidence, indent=2))
    print("PASS:", len(checks), "independent contour behavior contracts")


if __name__ == "__main__":
    main()
