from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from contour_tracker import ContourConfig, ContourTracker  # noqa: E402

OUT = Path(__file__).with_name("motion_split")


def textured_scene(
    size: tuple[int, int] = (120, 160), shift_x: float = 0.0
) -> np.ndarray:
    height, width = size
    y, x = np.indices((height, width), dtype=np.float32)
    gray = (
        120
        + 45 * np.sin((x - shift_x) / 7.0)
        + 36 * np.cos((y + 2.0 * shift_x) / 9.0)
        + 28 * np.sin((x + y - shift_x) / 11.0)
    )
    noise = ((x * 17 + y * 29) % 31).astype(np.float32)
    gray = np.clip(gray + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def scene_with_foreground(shift_x: float, object_x: int) -> np.ndarray:
    frame = textured_scene(shift_x=shift_x)
    y, x = np.indices(frame.shape[:2], dtype=np.float32)
    patch = 135 + 70 * np.sin((x - object_x) / 2.5) + 55 * np.cos((y + object_x) / 4.0)
    patch_mask = (np.abs(x - float(object_x)) <= 5.0) & (y >= 18.0) & (y <= 102.0)
    frame[patch_mask] = cv2.cvtColor(
        np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )[patch_mask]
    return frame


def vertical_line(
    x: int, y0: int = 25, y1: int = 95, value: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((120, 160), np.uint8)
    prob = np.zeros((120, 160), np.float32)
    mask[y0:y1, x] = 255
    prob[mask > 0] = value
    return mask, prob


def t_junction(x: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((120, 160), np.uint8)
    mask[30:92, x] = 255
    mask[58, x - 30 : x + 1] = 255
    prob = (mask > 0).astype(np.float32) * 0.9
    return mask, prob


def center_x(paths) -> float:
    assert paths
    points = np.concatenate([path.points for path in paths])
    return float(points[:, 0].mean())


def single_center_x(result) -> float:
    assert len(result.paths) == 1
    return float(result.paths[0].points[:, 0].mean())


def check(name: str, condition: bool, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def main() -> None:
    cv2.setNumThreads(1)
    checks: list[str] = []

    cfg = ContourConfig(min_registration_ratio=0.0, motion_separation=True)
    static_tracker = ContourTracker(cfg)
    legacy_tracker = ContourTracker(ContourConfig(min_registration_ratio=0.0))
    static_centers: list[float] = []
    legacy_centers: list[float] = []
    raw_centers: list[float] = []
    for index in range(8):
        camera_shift = float(index)
        jitter = 1 if index % 2 else 0
        x = 45 + index + jitter
        raw_centers.append(float(x))
        mask, prob = vertical_line(x)
        frame = textured_scene(shift_x=camera_shift)
        static_result = static_tracker.update(mask, prob, frame, frame_index=index)
        legacy_result = legacy_tracker.update(mask, prob, frame, frame_index=index)
        static_centers.append(single_center_x(static_result))
        legacy_centers.append(single_center_x(legacy_result))
    static_jitter = float(np.std(np.diff(static_centers[3:])))
    legacy_jitter = float(np.std(np.diff(legacy_centers[3:])))
    raw_jitter = float(np.std(np.diff(raw_centers[3:])))
    check(
        "static camera-pan contour has less frame-to-frame jitter than legacy",
        static_jitter < legacy_jitter and static_jitter < raw_jitter,
        checks,
    )
    clean_static = ContourTracker(cfg)
    clean_static_seen = False
    for index in range(4):
        clean_result = clean_static.update(
            *vertical_line(45 + index),
            textured_scene(shift_x=float(index)),
            frame_index=index,
        )
        clean_static_seen = (
            clean_static_seen or int(clean_result.metrics["static_paths"]) >= 1
        )
    check(
        "static paths are classified after consecutive low residual evidence",
        clean_static_seen,
        checks,
    )

    slow = ContourTracker(cfg)
    slow_lag: list[float] = []
    slow_states: list[dict[str, int]] = []
    for index in range(6):
        camera_shift = float(index)
        x = int(round(80 + index * 1.5))
        result = slow.update(
            *vertical_line(x),
            scene_with_foreground(camera_shift, x),
            frame_index=index,
        )
        slow_lag.append(abs(single_center_x(result) - x))
        slow_states.append(
            {
                "static": int(result.metrics["static_paths"]),
                "dynamic": int(result.metrics["dynamic_paths"]),
                "unknown": int(result.metrics["unknown_paths"]),
            }
        )
    check(
        "slow independent foreground motion is not persistently promoted to static",
        sum(state["static"] for state in slow_states[2:]) <= 1
        and slow_states[-1]["static"] == 0,
        checks,
    )
    check(
        "slow independent foreground lag stays within 1.1 model pixels",
        max(slow_lag[2:]) <= 1.1,
        checks,
    )

    fast = ContourTracker(cfg)
    for index in range(4):
        x = 30 + index * 10
        result = fast.update(
            *vertical_line(x),
            scene_with_foreground(float(index), x),
            frame_index=index,
        )
        check(
            f"fast dynamic contour stays current at frame {index}",
            abs(single_center_x(result) - x) <= 1.1,
            checks,
        )
    medium = ContourTracker(
        ContourConfig(
            min_registration_ratio=0.0,
            motion_separation=True,
            match_distance_px=5.0,
        )
    )
    for index in range(3):
        medium_result = medium.update(
            *vertical_line(36 + index * 4),
            scene_with_foreground(float(index), 36 + index * 4),
            frame_index=index,
        )
    check(
        "matched independent motion is classified dynamic once residual is large",
        int(medium_result.metrics["dynamic_paths"]) >= 1,
        checks,
    )

    junction = ContourTracker(cfg)
    for index in range(6):
        mask, prob = t_junction(80 + index + (index % 2))
        result = junction.update(
            mask, prob, textured_scene(shift_x=index), frame_index=index
        )
    check(
        "shared static junction endpoints remain connected",
        cv2.connectedComponents(result.mask)[0] == 2,
        checks,
    )

    slow_support = np.zeros((120, 160), np.float32)
    slow_support[25:95, 88:92] = 0.4
    slow_gone = slow.update(
        np.zeros((120, 160), np.uint8),
        slow_support,
        textured_scene(shift_x=6),
        frame_index=6,
    )
    check(
        "slow moving foreground disappearance is not statically held",
        not bool(slow_gone.paths),
        checks,
    )

    hold = ContourTracker(cfg)
    for index in range(4):
        hold.update(
            *vertical_line(55 + index), textured_scene(shift_x=index), frame_index=index
        )
    support_prob = np.zeros((120, 160), np.float32)
    support_prob[25:95, 57:61] = 0.4
    blank = np.zeros((120, 160), np.uint8)
    held = hold.update(blank, support_prob, textured_scene(shift_x=4), frame_index=4)
    gone = hold.update(
        blank, np.zeros_like(support_prob), textured_scene(shift_x=5), frame_index=5
    )
    check(
        "supported static dropout is held for one frame",
        bool(held.paths) and int(held.metrics["static_held_paths"]) == 1,
        checks,
    )
    check(
        "unsupported disappearance clears without a stale line",
        not bool(gone.paths),
        checks,
    )

    fallback = ContourTracker(cfg)
    flat = np.full((120, 160, 3), 80, np.uint8)
    fallback.update(*vertical_line(30), flat, frame_index=0)
    low = fallback.update(*vertical_line(31), flat, frame_index=1)
    check(
        "low-texture global fit falls back to unknown dense prediction",
        low.metrics["global_motion_status"] != "ok"
        and int(low.metrics["unknown_paths"]) >= 1,
        checks,
    )

    osd = ContourTracker(cfg)
    osd.update(
        *vertical_line(50),
        textured_scene(),
        ignore_regions=((0.2, 0, 0.5, 1),),
        frame_index=0,
    )
    moved_osd = osd.update(
        *vertical_line(50),
        textured_scene(shift_x=1),
        ignore_regions=((0.5, 0, 0.8, 1),),
        frame_index=1,
    )
    check(
        "ignore-region changes reset motion-separated contour history",
        moved_osd.metrics["reset_reason"] == "ignore_regions_changed",
        checks,
    )

    default = ContourTracker()
    default_result = default.update(*vertical_line(70), textured_scene(), frame_index=0)
    check(
        "default tracker does not expose motion-separation metrics",
        "global_motion_status" not in default_result.metrics,
        checks,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motion_separation_checks.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": checks,
                "static_jitter_px": static_jitter,
                "legacy_jitter_px": legacy_jitter,
                "raw_jitter_px": raw_jitter,
                "slow_lag_max_px": max(slow_lag[2:]),
                "slow_states": slow_states,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PASS: {len(checks)} motion separation contracts")


if __name__ == "__main__":
    main()
