"""Rasterized thermal circle board for a hardware-free end-to-end check."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .target import load_target
from .workflow import write_image


def generate_dataset(directory, views=24, seed=19) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    target = load_target()
    matrix = np.array([[343.0, 0, 160.0], [0, 350.0, 128.0], [0, 0, 1.0]])
    distortion = np.array([-0.12, 0.03, 0.0005, -0.0003, 0.0])
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    boundary = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(64)]) * 0.03
    board = np.array([[0, 0, 0], [0.6, 0, 0], [0.6, 0.4, 0], [0, 0.4, 0]], float)
    count = 0
    for _ in range(2000):
        rv = rng.uniform([-0.55, -0.55, -0.7], [0.55, 0.55, 0.7])
        rot = cv2.Rodrigues(rv)[0]
        tv = np.array(
            [rng.uniform(-0.35, 0.35), rng.uniform(-0.25, 0.25), rng.uniform(1.0, 1.9)]
        ) - rot @ np.array([0.3, 0.2, 0])
        corners = cv2.projectPoints(board, rv, tv, matrix, distortion)[0].reshape(-1, 2)
        if np.any(corners.min(axis=0) < 4) or np.any(corners.max(axis=0) > [316, 252]):
            continue
        image = np.full((256 * 4, 320 * 4), 44000, np.uint16)
        for center in target.object_points():
            pixels = cv2.projectPoints(boundary + center, rv, tv, matrix, distortion)[0].reshape(-1, 2)
            cv2.fillConvexPoly(image, np.round((pixels + 0.5) * 4 - 0.5).astype(np.int32), 16000)
        image = cv2.resize(image, (320, 256), interpolation=cv2.INTER_AREA)
        image = cv2.GaussianBlur(image, (3, 3), 0.45)
        if count % 2:
            image = 60000 - image
        image = np.clip(image.astype(float) + rng.normal(0, 65, image.shape), 0, 65535).astype(np.uint16)
        write_image(directory / f"view_{count:03d}.png", image)
        count += 1
        if count == views:
            return {"matrix": matrix, "distortion": distortion, "views": count, "synthetic": True}
    raise RuntimeError("could not generate enough synthetic views")


def run_image_check(directory) -> dict:
    from .io import load_yaml
    from .workflow import calibrate_dataset

    directory = Path(directory)
    expected = generate_dataset(directory / "images")
    result = calibrate_dataset(
        sorted((directory / "images").glob("*.png")), directory / "boson_intrinsic.yaml"
    )
    relative = np.abs(np.diag(result.camera_matrix)[:2] / np.diag(expected["matrix"])[:2] - 1)
    quality = load_yaml(directory / "boson_intrinsic.yaml")["quality"]
    passed = bool(result.rms_px < 0.8 and np.max(relative) < 0.08 and quality["passed"])
    summary = {
        "passed": passed,
        "accepted_views": len(result.accepted_images),
        "rms_px": result.rms_px,
        "focal_error_pct": (relative * 100).tolist(),
        "output": str(directory / "boson_intrinsic.yaml"),
        "synthetic": True,
        "hardware_verified": False,
        "numerical_quality_passed": quality["passed"],
    }
    if not passed:
        raise RuntimeError(f"thermal image self-check failed: {summary}")
    return summary
