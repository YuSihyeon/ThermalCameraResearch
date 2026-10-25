from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import save_yaml
from .thermal import find_thermal_corners, normalize_thermal


@dataclass
class IntrinsicResult:
    rms_px: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    per_view_error_px: list[float]
    accepted_images: list[str]
    rejected_images: list[str]


def board_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    x_count, y_count = pattern
    points = np.zeros((x_count * y_count, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:x_count, 0:y_count].T.reshape(-1, 2) * float(square_m)
    return points


def calibrate_observations(
    image_points: list[np.ndarray], image_size: tuple[int, int], pattern: tuple[int, int], square_m: float
) -> IntrinsicResult:
    if len(image_points) < 3:
        raise ValueError("at least 3 accepted views are required")
    object_template = board_points(pattern, square_m)
    object_points = [object_template.copy() for _ in image_points]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, [p.reshape(-1, 1, 2) for p in image_points], image_size, None, None
    )
    errors: list[float] = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        delta = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return IntrinsicResult(float(rms), matrix, distortion.reshape(-1), image_size, errors, [], [])


def calibrate_images(
    paths: list[str], output: str | Path, pattern: tuple[int, int], square_m: float, min_views: int = 15
) -> IntrinsicResult:
    accepted: list[str] = []
    rejected: list[str] = []
    points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    debug_dir = Path(output).with_suffix("").with_name(Path(output).stem + "_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in paths:
        image = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            rejected.append(raw_path)
            continue
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if size is None:
            size = current_size
        if current_size != size:
            rejected.append(raw_path)
            continue
        corners, method = find_thermal_corners(image, pattern)
        if corners is None:
            rejected.append(raw_path)
            continue
        accepted.append(raw_path)
        points.append(corners)
        preview = cv2.cvtColor(normalize_thermal(image), cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(preview, pattern, corners.reshape(-1, 1, 2), True)
        cv2.putText(preview, method or "detected", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(str(debug_dir / f"{len(accepted):03d}_{Path(raw_path).stem}.png"), preview)

    if size is None:
        raise ValueError("no readable input images")
    if len(points) < min_views:
        raise ValueError(f"only {len(points)} valid views; at least {min_views} required")
    result = calibrate_observations(points, size, pattern, square_m)
    result.accepted_images = accepted
    result.rejected_images = rejected
    save_intrinsic(output, result, pattern, square_m)
    return result


def save_intrinsic(path: str | Path, result: IntrinsicResult, pattern: tuple[int, int], square_m: float) -> Path:
    return save_yaml(
        path,
        {
            "schema": "firesight.camera_calibration/v1",
            "camera": "FLIR Boson 320",
            "image_size": list(result.image_size),
            "camera_matrix": result.camera_matrix,
            "distortion_model": "plumb_bob",
            "distortion_coefficients": result.distortion,
            "rms_px": result.rms_px,
            "per_view_error_px": result.per_view_error_px,
            "pattern": {"internal_corners": list(pattern), "square_length_m": square_m},
            "accepted_images": result.accepted_images,
            "rejected_images": result.rejected_images,
        },
    )

