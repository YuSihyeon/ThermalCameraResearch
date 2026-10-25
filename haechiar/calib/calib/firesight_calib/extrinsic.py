from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import average_transforms, invert_transform, rotation_error_deg, transform_from_rvec_tvec
from .io import camera_from_yaml, save_yaml


def _views(array: np.ndarray) -> list[np.ndarray]:
    if array.dtype == object:
        return [np.asarray(v, dtype=np.float64) for v in array.tolist()]
    return [np.asarray(v, dtype=np.float64) for v in array]


def calibrate_extrinsic(
    correspondences: str | Path,
    boson_intrinsic: str | Path,
    sensor_intrinsic: str | Path,
    sensor_frame: str,
    output: str | Path,
) -> dict:
    data = np.load(correspondences, allow_pickle=True)
    objects = _views(data["object_points"])
    boson_points = _views(data["boson_points"])
    sensor_points = _views(data["sensor_points"])
    if not (len(objects) == len(boson_points) == len(sensor_points)) or len(objects) < 3:
        raise ValueError("correspondences require at least 3 equally sized views")

    kb, db, _ = camera_from_yaml(boson_intrinsic)
    ks, ds, _ = camera_from_yaml(sensor_intrinsic)
    transforms: list[np.ndarray] = []
    for index, (obj, bp, sp) in enumerate(zip(objects, boson_points, sensor_points)):
        if not (len(obj) == len(bp) == len(sp)) or len(obj) < 4:
            raise ValueError(f"view {index} has inconsistent or insufficient points")
        ok_b, rb, tb = cv2.solvePnP(obj, bp, kb, db, flags=cv2.SOLVEPNP_ITERATIVE)
        ok_s, rs, ts = cv2.solvePnP(obj, sp, ks, ds, flags=cv2.SOLVEPNP_ITERATIVE)
        if not (ok_b and ok_s):
            raise ValueError(f"PnP failed for view {index}")
        boson_from_board = transform_from_rvec_tvec(rb, tb)
        sensor_from_board = transform_from_rvec_tvec(rs, ts)
        transforms.append(boson_from_board @ invert_transform(sensor_from_board))

    mean = average_transforms(transforms)
    translation_errors = [float(np.linalg.norm(t[:3, 3] - mean[:3, 3]) * 1000.0) for t in transforms]
    rotation_errors = [rotation_error_deg(t, mean) for t in transforms]
    key = "T_boson_from_" + sensor_frame.removesuffix("_optical_frame")
    result = {
        "schema": "firesight.rig_transform/v1",
        "source_frame": sensor_frame,
        "target_frame": "boson_optical_frame",
        key: mean,
        "views": len(transforms),
        "consistency": {
            "translation_mean_mm": float(np.mean(translation_errors)),
            "translation_max_mm": float(np.max(translation_errors)),
            "rotation_mean_deg": float(np.mean(rotation_errors)),
            "rotation_max_deg": float(np.max(rotation_errors)),
        },
    }
    save_yaml(output, result)
    return result

