from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .intrinsic import board_points, calibrate_observations, save_intrinsic


def run_self_check(output: str | Path, seed: int = 7) -> dict:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pattern = (9, 5)
    image_size = (320, 256)
    true_matrix = np.asarray([[285.0, 0.0, 160.0], [0.0, 284.0, 128.0], [0.0, 0.0, 1.0]])
    true_distortion = np.asarray([-0.08, 0.025, 0.0005, -0.0003, 0.0])
    obj = board_points(pattern, 0.05)
    observations: list[np.ndarray] = []
    attempts = 0
    while len(observations) < 24 and attempts < 500:
        attempts += 1
        rvec = rng.uniform([-0.35, -0.35, -0.25], [0.35, 0.35, 0.25]).astype(np.float64)
        tvec = rng.uniform([-0.20, -0.12, 0.75], [0.05, 0.03, 1.35]).astype(np.float64)
        pixels, _ = cv2.projectPoints(obj, rvec, tvec, true_matrix, true_distortion)
        pixels = pixels.reshape(-1, 2)
        if (pixels[:, 0].min() < 5 or pixels[:, 0].max() >= image_size[0] - 5 or pixels[:, 1].min() < 5 or pixels[:, 1].max() >= image_size[1] - 5):
            continue
        observations.append((pixels + rng.normal(0.0, 0.12, pixels.shape)).astype(np.float32))
    if len(observations) < 15:
        raise RuntimeError("synthetic generator could not create enough valid views")
    result = calibrate_observations(observations, image_size, pattern, 0.05)
    save_intrinsic(target / "boson_intrinsic.yaml", result, pattern, 0.05)
    fx_error_pct = abs(result.camera_matrix[0, 0] - true_matrix[0, 0]) / true_matrix[0, 0] * 100.0
    summary = {
        "passed": bool(result.rms_px < 0.5 and fx_error_pct < 8.0),
        "views": len(observations),
        "rms_px": result.rms_px,
        "fx_error_pct": float(fx_error_pct),
        "output": str(target / "boson_intrinsic.yaml"),
    }
    with (target / "self_check.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    if not summary["passed"]:
        raise RuntimeError(f"self-check quality gate failed: {summary}")
    return summary

