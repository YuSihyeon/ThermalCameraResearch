"""Conservative acquisition/fit diagnostics, not a hardware certification."""

from __future__ import annotations
import cv2
import numpy as np
from .target import Target


def duplicate_pose(points: np.ndarray, previous: list[np.ndarray], tolerance_px: float = 2.5) -> bool:
    # Nearest-set distance also catches reversed ordering on the symmetric board.
    return any(
        float(np.sqrt(np.mean(np.min(np.sum((points[:, None, :] - p[None, :, :]) ** 2, axis=2), axis=1))))
        < tolerance_px
        for p in previous
    )


def coverage(points: list[np.ndarray], size: tuple[int, int]) -> tuple[float, list[list[int]]]:
    bins = np.zeros((3, 3), int)
    if not points:
        return 0.0, bins.tolist()
    all_points = np.concatenate(points).astype(np.float32)
    area = float(cv2.contourArea(cv2.convexHull(all_points)) / (size[0] * size[1]))
    for x, y in all_points:
        bins[min(2, max(0, int(y / size[1] * 3))), min(2, max(0, int(x / size[0] * 3)))] += 1
    return area, bins.tolist()


def homography_condition(observations: list[np.ndarray], target: Target, size=(320, 256)) -> float:
    """Rank-five Zhang constraint strength, independent of the fitted K.

    Isotropic coordinate normalization and unit constraint rows make the 0.01
    heuristic reproducible. A near-zero fifth singular value means focal length
    is poorly constrained (e.g. frontoparallel translation/scale only).
    """
    obj = target.object_points()[:, :2].astype(np.float64)
    obj = (obj - obj.mean(axis=0)) / np.ptp(obj, axis=0).max()
    constraints = []

    def vij(h, i, j):
        a, b = h[:, i], h[:, j]
        return np.array(
            [
                a[0] * b[0],
                a[0] * b[1] + a[1] * b[0],
                a[1] * b[1],
                a[2] * b[0] + a[0] * b[2],
                a[2] * b[1] + a[1] * b[2],
                a[2] * b[2],
            ]
        )

    for points in observations:
        if points.shape != (len(obj), 2) or not np.isfinite(points).all():
            return 0.0
        normalized = (points - np.array(size) / 2) / max(size)
        h, _ = cv2.findHomography(obj, normalized, 0)
        if h is None:
            return 0.0
        for row in (vij(h, 0, 1), vij(h, 0, 0) - vij(h, 1, 1)):
            norm = np.linalg.norm(row)
            if norm < 1e-12:
                return 0.0
            constraints.append(row / norm)
    if len(constraints) < 6:
        return 0.0
    singular = np.linalg.svd(np.asarray(constraints), compute_uv=False)
    return float(singular[4] / singular[0])


def assess_quality(result, observations: list[np.ndarray], target: Target, max_rms: float = 0.8) -> dict:
    failures = []
    condition = homography_condition(observations, target, result.image_size)
    if condition < 0.01:
        failures.append("pose_conditioning")
    matrix = result.camera_matrix
    dist = result.distortion
    w, h = result.image_size
    valid = bool(
        np.isfinite(matrix).all()
        and np.isfinite(dist).all()
        and matrix[0, 0] > 0
        and matrix[1, 1] > 0
        and 0 < matrix[0, 2] < w
        and 0 < matrix[1, 2] < h
    )
    if not valid:
        failures.append("invalid_camera_parameters")
    if not np.isfinite(result.rms_px) or result.rms_px > max_rms:
        failures.append("rms")
    area, bins = coverage(observations, result.image_size)
    if area < 0.35 or np.count_nonzero(bins) < 6:
        failures.append("coverage")
    normals = []
    if valid:
        obj = target.object_points()
        for p in observations:
            if len(p) != len(obj):
                continue
            ok, rvec, _ = cv2.solvePnP(obj, p, matrix, dist)
            if ok:
                normal = cv2.Rodrigues(rvec)[0][:, 2]
                normals.append(normal if normal[2] >= 0 else -normal)
    spread = 0.0
    if len(normals) > 1:
        n = np.asarray(normals)
        spread = float(np.degrees(np.arccos(np.clip((n @ n.T).min(), -1, 1))))
    if spread < 12:
        failures.append("tilt_diversity")
    radial_ok = False
    if valid:
        radius = np.hypot(
            max(matrix[0, 2], w - matrix[0, 2]) / matrix[0, 0],
            max(matrix[1, 2], h - matrix[1, 2]) / matrix[1, 1],
        )
        r = np.linspace(0, radius, 100)
        k1, k2, _, _, k3 = dist[:5]
        radial_ok = bool(np.all(1 + 3 * k1 * r * r + 5 * k2 * r**4 + 7 * k3 * r**6 > 0))
    if not radial_ok:
        failures.append("nonmonotonic_radial_distortion")
    return {
        "passed": not failures,
        "deployment_ready": not failures and target.status == "verified",
        "failures": failures,
        "max_rms_px": max_rms,
        "coverage_fraction": area,
        "coverage_3x3": bins,
        "normal_spread_deg": spread,
        "homography_condition_ratio": condition,
        "min_homography_condition_ratio": 0.01,
        "target_status": target.status,
        "notes": [
            "Heuristic acquisition checks; inspect overlays and repeat on real hardware.",
            "RMS is training reprojection error, not independent accuracy.",
        ],
    }
