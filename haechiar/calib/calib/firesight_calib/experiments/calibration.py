from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import camera_from_yaml, load_yaml, transform_from_yaml


@dataclass
class Camera:
    matrix: np.ndarray
    distortion: np.ndarray
    size: tuple[int, int]

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if (self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all()
                or self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0
                or not np.allclose(self.matrix[2], [0, 0, 1])
                or abs(self.matrix[0, 1]) > 1e-12 or abs(self.matrix[1, 0]) > 1e-12):
            raise ValueError("camera_matrix must be a finite canonical pinhole matrix")
        if self.distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError("invalid OpenCV distortion coefficients")
        if len(self.size) != 2 or any(type(v) is not int or v <= 0 for v in self.size):
            raise ValueError("image_size must contain two positive integers")

    @classmethod
    def load(cls, path):
        # Preserve strict dimension types before the legacy helper's int conversion.
        doc = load_yaml(path)
        matrix, distortion, _ = camera_from_yaml(path)
        return cls(matrix, distortion, tuple(doc["image_size"]))

    def document(self):
        return {"camera_matrix": self.matrix.tolist(), "distortion_coefficients": self.distortion.tolist(),
                "image_size": list(self.size)}


@dataclass
class Registration:
    calibration_id: str
    source_frame: str
    target_frame: str
    camera: Camera
    transform: np.ndarray
    provenance: str
    source_camera: Camera | None = None

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.calibration_id, self.source_frame, self.target_frame)):
            raise ValueError("calibration ID and coordinate frames must be nonempty")
        if self.provenance not in ("measured", "synthetic"):
            raise ValueError("registration provenance must be measured or synthetic")
        self.transform = np.asarray(self.transform, dtype=np.float64)
        if self.transform.shape != (4, 4) or not np.isfinite(self.transform).all():
            raise ValueError("transform must be finite 4x4")
        r = self.transform[:3, :3]
        if (not np.allclose(self.transform[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(r @ r.T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
            raise ValueError("transform must be a proper rigid source-to-target transform")

    @classmethod
    def load(cls, path):
        path = Path(path)
        doc = load_yaml(path)
        if doc.get("schema") != "firesight.experiment_registration/v1":
            raise ValueError("unsupported registration schema")
        extrinsic = path.parent / doc["extrinsic"]
        ext = load_yaml(extrinsic)
        for key in ("source_frame", "target_frame"):
            if ext.get(key) != doc[key]:
                raise ValueError(f"registration {key} does not match extrinsic file")
        source_camera = Camera.load(path.parent / doc["source_intrinsic"]) if doc.get("source_intrinsic") else None
        return cls(doc["calibration_id"], doc["source_frame"], doc["target_frame"],
                   Camera.load(path.parent / doc["intrinsic"]), transform_from_yaml(extrinsic),
                   doc["provenance"], source_camera)

    def document(self):
        return {"calibration_id": self.calibration_id, "source_frame": self.source_frame,
                "target_frame": self.target_frame, "provenance": self.provenance,
                "transform": self.transform.tolist(), "camera": self.camera.document(),
                "source_camera": self.source_camera.document() if self.source_camera else None}


def fit_rigid(source, target):
    """Fit target = R @ source + t to explicitly associated 3D points in metres.

    Correspondences must refer to the same physical radar-visible target centres,
    measured in both frames. Image checker corners alone are insufficient.
    Residuals are fitting errors, not held-out accuracy or an approval decision.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape or len(source) < 4
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("at least four finite corresponding Nx3 points in metres are required")
    a, b = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(a, tol=1e-8) < 2 or np.linalg.matrix_rank(b, tol=1e-8) < 2:
        raise ValueError("collinear/coincident correspondences cannot determine a rigid transform")
    u, singular, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target.mean(0) - rotation @ source.mean(0)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return {"transform": transform, "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "max_error_m": float(errors.max()), "errors_m": errors.tolist(),
            "singular_values": singular.tolist(), "correspondences": len(source),
            "deployment_ready": False}
