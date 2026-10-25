from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class MotionRegistrationSettings(Protocol):
    @property
    def motion_global_min_samples(self) -> int: ...
    @property
    def motion_global_min_coverage(self) -> float: ...
    @property
    def motion_global_min_inlier_ratio(self) -> float: ...
    @property
    def motion_global_max_error_px(self) -> float: ...
    @property
    def motion_static_residual_px(self) -> float: ...


@dataclass(frozen=True)
class _GlobalMotion:
    status: str
    affine: np.ndarray | None = None
    inliers: int = 0
    coverage: float = 0.0
    error_px: float = 0.0


def _estimate_global_motion(
    current_to_prev: np.ndarray,
    confident_current: np.ndarray,
    active: np.ndarray,
    prev_gray: np.ndarray,
    config: MotionRegistrationSettings,
) -> _GlobalMotion:
    height, width = confident_current.shape
    if not np.any(confident_current):
        return _GlobalMotion("insufficient_samples")
    safe_active = cv2.erode(active.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    prev_texture = _texture_mask(prev_gray)
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    prev_x = xs + current_to_prev[..., 0]
    prev_y = ys + current_to_prev[..., 1]
    in_prev = (
        (prev_x >= 0.0)
        & (prev_x <= float(width - 1))
        & (prev_y >= 0.0)
        & (prev_y <= float(height - 1))
    )
    texture_current = cv2.remap(
        prev_texture.astype(np.uint8),
        prev_x.astype(np.float32),
        prev_y.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    usable = confident_current & safe_active & in_prev & texture_current
    ys_i, xs_i = np.nonzero(usable)
    if len(xs_i) < config.motion_global_min_samples:
        return _GlobalMotion("insufficient_samples")
    step = max(1, len(xs_i) // 800)
    xs_i = xs_i[::step]
    ys_i = ys_i[::step]
    src = np.column_stack((prev_x[ys_i, xs_i], prev_y[ys_i, xs_i])).astype(np.float32)
    dst = np.column_stack((xs_i, ys_i)).astype(np.float32)
    coverage = _spatial_coverage(xs_i, ys_i, width, height)
    if coverage < config.motion_global_min_coverage:
        return _GlobalMotion("low_coverage", coverage=coverage)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=max(0.5, config.motion_global_max_error_px),
        maxIters=2000,
        confidence=0.98,
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        return _GlobalMotion("fit_failed", coverage=coverage)
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < config.motion_global_min_samples:
        return _GlobalMotion(
            "insufficient_inliers", coverage=coverage, inliers=inlier_count
        )
    inlier_ratio = float(inlier_count / max(1, len(inliers)))
    if inlier_ratio < config.motion_global_min_inlier_ratio:
        return _GlobalMotion(
            "low_inlier_ratio", coverage=coverage, inliers=inlier_count
        )
    inlier_coverage = _spatial_coverage(
        dst[inliers, 0].astype(np.int32),
        dst[inliers, 1].astype(np.int32),
        width,
        height,
    )
    if inlier_coverage < config.motion_global_min_coverage:
        return _GlobalMotion(
            "low_inlier_coverage",
            affine=affine,
            inliers=inlier_count,
            coverage=inlier_coverage,
        )
    projected = _transform_points(src[inliers], affine)
    error = float(np.median(np.linalg.norm(projected - dst[inliers], axis=1)))
    if error > config.motion_global_max_error_px:
        return _GlobalMotion(
            "high_error",
            affine=affine,
            inliers=inlier_count,
            coverage=inlier_coverage,
            error_px=error,
        )
    return _GlobalMotion(
        "ok",
        affine=affine,
        inliers=inlier_count,
        coverage=inlier_coverage,
        error_px=error,
    )


def _texture_mask(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = max(8.0, float(np.percentile(magnitude, 55.0)))
    return magnitude >= threshold


def _spatial_coverage(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> float:
    if len(xs) == 0:
        return 0.0
    cols = np.clip((xs.astype(np.float32) / max(1, width) * 4).astype(int), 0, 3)
    rows = np.clip((ys.astype(np.float32) / max(1, height) * 3).astype(int), 0, 2)
    bins = rows * 4 + cols
    return float(len(set(int(value) for value in bins)) / 12.0)


def _transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return (points @ affine[:, :2].T + affine[:, 2].reshape(1, 2)).astype(np.float32)
