from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from contour_tracker import ContourConfig, ContourTracker  # noqa: E402

OUT = Path(__file__).with_name("contour_stability") / "core_checks"


def textured_frame(size: tuple[int, int] = (96, 128), shift_x: float = 0) -> np.ndarray:
    height, width = size
    y, x = np.indices((height, width), dtype=np.float32)
    gray = (
        100
        + 50 * np.sin((x + shift_x) / 8)
        + 40 * np.cos(y / 7)
        + 20 * np.sin((x + y + shift_x) / 13)
    )
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def line(x0: int, x1: int, y: int = 45, value: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    prob = np.zeros((96, 128), np.float32)
    mask = np.zeros((96, 128), np.uint8)
    prob[y, x0:x1] = value
    mask[y, x0:x1] = 255
    return mask, prob


def curved(offset: int = 0, value: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    prob = np.zeros((96, 128), np.float32)
    mask = np.zeros((96, 128), np.uint8)
    points = np.array(
        [[30 + offset, 30], [38 + offset, 38], [44 + offset, 51], [53 + offset, 65]],
        np.int32,
    )
    cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8)
    prob[mask > 0] = value
    return mask, prob


def centroid(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    assert xs.size
    return float(xs.mean())


def check(name: str, condition: bool, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def main() -> None:
    cv2.setNumThreads(1)
    checks: list[str] = []

    tracker = ContourTracker()
    mask, prob = line(30, 60, value=0.8)
    result = tracker.update(mask, prob, textured_frame(), frame_index=0, timestamp_s=0)
    check("strong line appears on first frame", bool(np.count_nonzero(result.mask) >= 28), checks)
    check("first strong line has persistent id", result.paths[0].track_id == 1, checks)

    weak = ContourTracker()
    for index in range(2):
        mask, prob = line(70, 78, value=0.6)
        result = weak.update(mask, prob, textured_frame(shift_x=index * 0.1), frame_index=index)
        check(f"weak short fragment hidden before confirmation {index}", not bool(np.any(result.mask)), checks)
    result = weak.update(mask, prob, textured_frame(shift_x=0.2), frame_index=2)
    check("weak short fragment appears after 3 of 4 confirmations", bool(np.any(result.mask)), checks)

    unsupported_pending = ContourTracker(ContourConfig(min_registration_ratio=0.0))
    mask, prob = line(80, 88, value=0.6)
    unsupported_pending.update(mask, prob, textured_frame(), frame_index=0)
    supported_prob = np.zeros((96, 128), np.float32)
    supported_prob[45, 80:88] = 0.3
    held = unsupported_pending.update(
        np.zeros((96, 128), np.uint8),
        supported_prob,
        textured_frame(shift_x=0.1),
        frame_index=1,
    )
    check("unconfirmed weak pending fragment is not held", not bool(np.any(held.mask)), checks)

    jitter = ContourTracker(ContourConfig(min_registration_ratio=0.0))
    m0, p0 = line(30, 70, value=0.8)
    r0 = jitter.update(m0, p0, textured_frame(), frame_index=0)
    m1, p1 = line(31, 71, value=0.8)
    r1 = jitter.update(m1, p1, textured_frame(shift_x=0.1), frame_index=1)
    check("slow one-pixel jitter is damped", centroid(r1.mask) - centroid(r0.mask) < 0.75, checks)
    check("same contour keeps track id", r1.paths[0].track_id == r0.paths[0].track_id, checks)

    moving = ContourTracker(ContourConfig(min_registration_ratio=0.0))
    m0, p0 = curved(0)
    moving.update(m0, p0, textured_frame(), frame_index=0)
    m1, p1 = curved(6)
    r1 = moving.update(m1, p1, textured_frame(shift_x=0.2), frame_index=1)
    check("moving curved contour remains visible", bool(np.count_nonzero(r1.mask) > 20), checks)
    check("moving curved contour is not frozen at old location", centroid(r1.mask) > 32, checks)

    vanish = ContourTracker(ContourConfig(min_registration_ratio=0.0))
    vanish.update(*line(30, 60, value=0.9), textured_frame(), frame_index=0)
    gone = vanish.update(np.zeros((96, 128), np.uint8), np.zeros((96, 128), np.float32), textured_frame(), frame_index=1)
    check("zero current evidence clears immediately", not np.any(gone.mask), checks)

    osd = ContourTracker()
    mask, prob = line(30, 60, value=0.9)
    blocked = osd.update(mask, prob, textured_frame(), ignore_regions=((0.2, 0.3, 0.55, 0.6),))
    check("ignored OSD region suppresses contour", not np.any(blocked.mask), checks)

    distinct_mask = np.zeros((96, 128), np.uint8)
    distinct_prob = np.zeros((96, 128), np.float32)
    distinct_mask[40, 30:70] = 255
    distinct_mask[44, 30:70] = 255
    distinct_prob[distinct_mask > 0] = 0.9
    distinct = ContourTracker().update(distinct_mask, distinct_prob, textured_frame())
    check("nearby distinct strong boundaries are preserved", len(distinct.paths) == 2, checks)

    reset = ContourTracker()
    reset.update(*line(10, 40), textured_frame(), frame_index=0, timestamp_s=0)
    gap = reset.update(*line(10, 40), textured_frame(), frame_index=1, timestamp_s=1)
    check("large time gap resets contour tracker", gap.metrics["reset_reason"] == "time_gap", checks)

    perf = ContourTracker()
    frame = textured_frame((256, 320))
    prob = np.zeros((256, 320), np.float32)
    mask = np.zeros((256, 320), np.uint8)
    cv2.rectangle(mask, (40, 80), (270, 180), 255, 1)
    prob[mask > 0] = 0.8
    perf.update(mask, prob, frame, frame_index=0)
    start = time.perf_counter()
    perf.update(mask, prob, textured_frame((256, 320), 0.2), frame_index=1)
    elapsed_ms = (time.perf_counter() - start) * 1000
    check("default 320x256 contour update stays under 220 ms synthetic budget", elapsed_ms < 220, checks)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(
        json.dumps({"status": "PASS", "checks": checks, "elapsed_ms": elapsed_ms}, indent=2),
        encoding="utf-8",
    )
    print(f"PASS: {len(checks)} contour tracker contracts ({elapsed_ms:.1f} ms 320x256)")


if __name__ == "__main__":
    main()
