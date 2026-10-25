from __future__ import annotations

import cv2
import numpy as np

from ..projection import project_points
from .calibration import Camera, Registration


def deproject_depth(depth_m, camera: Camera):
    depth = np.asarray(depth_m)
    if depth.shape != camera.size[::-1]:
        raise ValueError("depth shape must match its intrinsic calibration; implicit resizing is forbidden")
    if depth.dtype.kind != "f":
        raise ValueError("depth must be floating metres")
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(rows):
        return np.empty((0, 3), dtype=float)
    uv = np.column_stack([cols, rows]).astype(np.float64)
    rays = cv2.undistortPoints(uv.reshape(-1, 1, 2), camera.matrix, camera.distortion).reshape(-1, 2)
    z = depth[rows, cols]
    return np.column_stack([rays * z[:, None], z])


def registered_depth(points, registration: Registration):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3 metres")
    camera = registration.camera
    width, height = camera.size
    output = np.full((height, width), np.nan, dtype=np.float32)
    if not len(points):
        return output
    uv, valid = project_points(points, registration.transform, camera.matrix, camera.distortion, camera.size)
    uv = np.floor(uv[valid] + 0.5).astype(np.int64)
    z = (points @ registration.transform[:3, :3].T + registration.transform[:3, 3])[valid, 2]
    inside = (uv[:, 0] < width) & (uv[:, 1] < height)
    flat = uv[inside, 1] * width + uv[inside, 0]
    buffer = np.full(width * height, np.inf)
    np.minimum.at(buffer, flat, z[inside])
    finite = np.isfinite(buffer)
    output.reshape(-1)[finite] = buffer[finite]
    return output


def display_thermal(native):
    """Per-frame display normalization only. Never a temperature conversion."""
    lo, hi = np.percentile(native, [1, 99])
    if hi <= lo:
        return np.zeros(native.shape, dtype=np.uint8)
    return np.clip((native.astype(float) - lo) * 255. / (hi - lo), 0, 255).astype(np.uint8)


def fuse(bundle, mode, registration=None, edge_model=None):
    thermal = bundle["thermal"]
    if thermal.kind != "thermal":
        raise ValueError("thermal stream must contain native thermal samples")
    view = display_thermal(thermal.data)
    if edge_model is None:
        edges = cv2.Canny(view, 50, 150)
        method = "canny_baseline_not_ai"
    else:
        # A plugin receives a copy of the native array and returns probabilities.
        probabilities = np.asarray(edge_model.predict(thermal.data.copy()))
        if (probabilities.shape != view.shape or probabilities.dtype.kind != "f"
                or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError("edge model must return finite float HxW probabilities in [0,1] at native size")
        edges = np.where(probabilities >= 0.5, 255, 0).astype(np.uint8)
        method = str(edge_model.name)
    result = {"edges": edges, "edge_method": method, "view": view}
    if mode == "thermal_ai":
        return result
    if mode not in ("rgbd_thermal", "radar_thermal"):
        raise ValueError("unsupported experiment mode")
    if registration is None:
        raise ValueError("fusion requires measured registration; run collection first")
    source = bundle["depth" if mode == "rgbd_thermal" else "radar"]
    if source.kind != ("depth" if mode == "rgbd_thermal" else "radar"):
        raise ValueError("source stream kind does not match experiment")
    if thermal.frame != registration.target_frame or source.frame != registration.source_frame:
        raise ValueError("sample frame does not match calibration direction")
    if any(s.provenance != registration.provenance for s in bundle.values()):
        raise ValueError("sample/calibration provenance mismatch")
    if thermal.data.shape != registration.camera.size[::-1]:
        raise ValueError("thermal shape must match intrinsic calibration")
    if mode == "rgbd_thermal":
        if registration.source_camera is None:
            raise ValueError("RGB-D fusion requires source_intrinsic for the depth image frame")
        if "intrinsic" in source.metadata:
            recorded = source.metadata["intrinsic"]
            measured_camera = Camera(recorded["camera_matrix"], recorded["distortion_coefficients"],
                                     tuple(recorded["image_size"]))
            expected = registration.source_camera
            # Different coefficient vector lengths are equivalent only for zero distortion.
            distortions_match = ((not np.any(measured_camera.distortion) and not np.any(expected.distortion))
                                  or (measured_camera.distortion.shape == expected.distortion.shape
                                      and np.allclose(measured_camera.distortion, expected.distortion)))
            if (measured_camera.size != expected.size or not np.allclose(measured_camera.matrix, expected.matrix)
                    or not distortions_match):
                raise ValueError("source_intrinsic differs from recorded device intrinsic")
        points = deproject_depth(source.data, registration.source_camera)
    else:
        points = source.data
    result["depth_m"] = registered_depth(points, registration)
    return result
