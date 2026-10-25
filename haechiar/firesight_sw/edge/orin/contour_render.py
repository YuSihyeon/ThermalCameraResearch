"""Draw tracked model-space curves at fractional output-pixel coordinates."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from contour_tracker import ContourPath
from edge_filter import _active_region


def render_contour_alpha(
    paths: Sequence[ContourPath],
    model_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    placement: tuple[int, int, int, int],
    line_width: float = 1.5,
    ignore_regions: Sequence[tuple[float, float, float, float]] = (),
) -> np.ndarray:
    if not np.isfinite(line_width) or not 0.5 <= line_width <= 8.0:
        raise ValueError("contour line width must be within [0.5, 8] output pixels")
    model_h, model_w = model_shape
    canvas_h, canvas_w = canvas_shape
    x, y, width, height = placement
    if min(model_h, model_w, canvas_h, canvas_w, width, height) <= 0:
        raise ValueError("contour render dimensions must be positive")
    if x < 0 or y < 0 or x + width > canvas_w or y + height > canvas_h:
        raise ValueError("contour placement must fit the output canvas")
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    if not paths:
        return canvas
    # Two samples per output pixel retain a consistent half-pixel line width.
    coverage = np.zeros((height * 2, width * 2), dtype=np.uint8)
    scale = np.array([width / model_w, height / model_h], dtype=np.float32)
    thickness = max(1, int(round(line_width * 2)))
    for path in sorted(paths, key=lambda item: item.alpha):
        points = np.asarray(path.points, dtype=np.float32)
        if len(points) < 2:
            continue
        output_points = (points + 0.5) * scale * 2 - 0.5
        fixed = np.rint(output_points * 256).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            coverage,
            [fixed],
            path.closed,
            int(round(255 * np.clip(path.alpha, 0, 1))),
            thickness,
            cv2.LINE_AA,
            shift=8,
        )
    alpha = (
        cv2.resize(coverage, (width, height), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        / 255
    )
    active = _active_region(model_shape, ignore_regions).astype(np.uint8)
    alpha *= cv2.resize(active, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas[y : y + height, x : x + width] = alpha
    return canvas
