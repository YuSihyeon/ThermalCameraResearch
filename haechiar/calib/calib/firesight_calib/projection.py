from __future__ import annotations

import cv2
import numpy as np


def project_points(
    points_source: np.ndarray,
    transform_boson_from_source: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_source, dtype=np.float64).reshape(-1, 3)
    rotation = transform_boson_from_source[:3, :3]
    translation = transform_boson_from_source[:3, 3]
    points_boson = (rotation @ points.T).T + translation
    positive = points_boson[:, 2] > 0
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(positive):
        pixels, _ = cv2.projectPoints(
            points_boson[positive], np.zeros(3), np.zeros(3), camera_matrix, distortion
        )
        projected[positive] = pixels.reshape(-1, 2)
    width, height = image_size
    valid = positive & np.isfinite(projected).all(axis=1)
    valid &= (projected[:, 0] >= 0) & (projected[:, 0] < width)
    valid &= (projected[:, 1] >= 0) & (projected[:, 1] < height)
    return projected, valid

