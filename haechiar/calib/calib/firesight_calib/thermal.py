from __future__ import annotations

import cv2
import numpy as np

from .target import Target


def normalize_thermal(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("thermal image contains no finite pixels")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return scaled.astype(np.uint8)


def detection_candidates(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    base = normalize_thermal(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base)
    return [
        ("normalized", base),
        ("normalized_inverted", 255 - base),
        ("clahe", clahe),
        ("clahe_inverted", 255 - clahe),
    ]


def find_thermal_corners(image: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    for name, candidate in detection_candidates(image):
        found, corners = cv2.findChessboardCornersSB(candidate, pattern, flags=flags)
        if found:
            return corners.reshape(-1, 2).astype(np.float32), name
    return None, None


def find_target(image: np.ndarray, target: Target) -> tuple[np.ndarray | None, str | None]:
    if target.pattern_type == "checkerboard":
        return find_thermal_corners(image, target.pattern)
    scale = 2.0 if min(image.shape[:2]) <= 320 else 1.0
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold, params.maxThreshold, params.thresholdStep = 5, 250, 10
    params.filterByColor, params.blobColor = True, 0
    params.filterByArea = True
    params.minArea = 8 * scale * scale
    params.maxArea = image.shape[0] * image.shape[1] * 0.04 * scale * scale
    params.filterByCircularity, params.minCircularity = True, 0.45
    params.filterByConvexity, params.minConvexity = True, 0.75
    params.filterByInertia, params.minInertiaRatio = True, 0.2
    params.minDistBetweenBlobs = 3 * scale
    detector = cv2.SimpleBlobDetector_create(params)
    for name, candidate in detection_candidates(image):
        if not np.any(candidate != candidate.flat[0]):
            continue
        enlarged = cv2.resize(candidate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for clustering in (False, True):
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID | (cv2.CALIB_CB_CLUSTERING if clustering else 0)
            found, centers = cv2.findCirclesGrid(enlarged, target.pattern, flags=flags, blobDetector=detector)
            if found and centers is not None:
                # Invert resize's pixel-center transform; calibration remains native resolution.
                points = ((centers.reshape(-1, 2) + 0.5) / scale - 0.5).astype(np.float32)
                if np.isfinite(points).all() and len(points) == np.prod(target.pattern):
                    return points, f"circles_{name}" + ("_cluster" if clustering else "")
    return None, None
