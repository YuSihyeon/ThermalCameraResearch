"""Native thermal images -> observations -> intrinsics and review artifacts."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .intrinsic import IntrinsicResult
from .io import save_yaml
from .quality import assess_quality, duplicate_pose, homography_condition
from .review import write_review
from .target import Target, load_target
from .thermal import find_target, normalize_thermal


def read_image(path: str | Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error):
        return None


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"cannot encode image: {path}")
    encoded.tofile(path)


def solve_target(observations: list[np.ndarray], target: Target, size=(320, 256)) -> IntrinsicResult:
    if len(observations) < 3:
        raise ValueError("at least 3 valid views required")
    unique = []
    obj = target.object_points()
    for p in observations:
        if p.shape != (len(obj), 2) or not np.isfinite(p).all():
            raise ValueError("invalid observation coordinates")
        if not duplicate_pose(p, unique):
            unique.append(p)
    if len(unique) < 3:
        raise ValueError("insufficient pose diversity")
    if homography_condition(observations, target, size) < 0.01:
        raise ValueError(
            "insufficient pose conditioning: tilt the board in both directions; frontoparallel views cannot constrain intrinsics"
        )
    objects = [obj.copy() for _ in observations]
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objects, [p.astype(np.float32).reshape(-1, 1, 2) for p in observations], size, None, None
    )
    if not np.isfinite(matrix).all() or not np.isfinite(dist).all() or not np.isfinite(rms):
        raise ValueError("calibration is degenerate: non-finite parameters")
    errors = []
    for p, rv, tv in zip(observations, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rv, tv, matrix, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - p) ** 2, axis=1)))))
    return IntrinsicResult(float(rms), matrix, dist.ravel(), size, errors, [], [])


def calibrate_dataset(
    paths,
    output,
    target: Target | None = None,
    min_views: int = 15,
    max_rms: float = 0.8,
    expected_size=(320, 256),
) -> IntrinsicResult:
    target = target or load_target()
    if min_views < 3 or not np.isfinite(max_rms) or max_rms <= 0:
        raise ValueError("min_views must be >=3 and max_rms must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists; choose a new output path: {output}")
    debug = output.parent / (output.stem + "_debug")
    debug.mkdir(parents=True, exist_ok=True)
    observations = []
    accepted = []
    rejected = []
    records = []
    sample = None
    dtype = None
    for path in paths:
        record = {"path": str(path), "reason": None}
        image = read_image(path)
        points = None
        if image is None:
            record["reason"] = "unreadable"
        elif image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
            record["reason"] = "expected_mono8_or_mono16"
        elif (image.shape[1], image.shape[0]) != tuple(expected_size):
            record["reason"] = "image_size"
        elif dtype is not None and image.dtype != dtype:
            record["reason"] = "mixed_encoding"
        else:
            points, method = find_target(image, target)
            record["method"] = method
            if points is None:
                record["reason"] = "pattern_not_found"
            elif duplicate_pose(points, observations):
                record["reason"] = "duplicate_pose"
            else:
                dtype = image.dtype
                observations.append(points)
                accepted.append(str(path))
                record["points_px"] = points.tolist()
                if sample is None:
                    sample = image.copy()
                preview = cv2.cvtColor(normalize_thermal(image), cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(preview, target.pattern, points.reshape(-1, 1, 2), True)
                write_image(debug / f"{len(accepted):03d}_detected.png", preview)
        if record["reason"]:
            rejected.append(str(path))
        records.append(record)
    report = {
        "accepted_views": len(accepted),
        "rejected_views": len(rejected),
        "frames": records,
        "target": target.metadata(),
        "status": "insufficient_views",
    }
    report_path = debug / "report.json"

    def write_report():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report()
    if len(accepted) < min_views:
        raise ValueError(
            f"only {len(accepted)} valid views; at least {min_views} required. See {report_path}"
        )
    try:
        result = solve_target(observations, target, tuple(expected_size))
    except (ValueError, cv2.error) as exc:
        report.update(status="calibration_failed", error=str(exc))
        write_report()
        raise
    result.accepted_images, result.rejected_images = accepted, rejected
    quality = assess_quality(result, observations, target, max_rms)
    report.update(
        status="passed" if quality["passed"] else "review_required",
        quality=quality,
        rms_px=result.rms_px,
        per_view_error_px=result.per_view_error_px,
    )
    write_report()
    save_yaml(
        output,
        {
            "schema": "firesight.camera_calibration/v2",
            "camera": "FLIR Boson 320",
            "image_size": list(result.image_size),
            "camera_matrix": result.camera_matrix,
            "distortion_model": "plumb_bob",
            "distortion_coefficients": result.distortion,
            "rms_px": result.rms_px,
            "per_view_error_px": result.per_view_error_px,
            "accepted_images": accepted,
            "rejected_images": rejected,
            "target": target.metadata(),
            "quality": quality,
            "report": str(report_path),
            "encoding": str(dtype),
        },
    )
    preview = normalize_thermal(sample)
    write_image(debug / "sample_original.png", preview)
    write_image(
        debug / "sample_undistorted.png", cv2.undistort(preview, result.camera_matrix, result.distortion)
    )
    # Keep the native 16-bit image separate from the display-only images.
    write_image(
        debug / "sample_undistorted_native.png",
        cv2.undistort(sample, result.camera_matrix, result.distortion),
    )
    heat = np.zeros((expected_size[1], expected_size[0], 3), np.uint8)
    for points in observations:
        for x, y in points:
            cv2.circle(heat, (int(round(x)), int(round(y))), 2, (0, 200, 255), -1)
    for i in (1, 2):
        cv2.line(
            heat,
            (expected_size[0] * i // 3, 0),
            (expected_size[0] * i // 3, expected_size[1] - 1),
            (100, 100, 100),
            1,
        )
        cv2.line(
            heat,
            (0, expected_size[1] * i // 3),
            (expected_size[0] - 1, expected_size[1] * i // 3),
            (100, 100, 100),
            1,
        )
    write_image(debug / "coverage.png", heat)
    review_path = write_review(result, observations, target, quality, debug, records)
    (debug / "review_manifest.json").write_text(
        json.dumps(
            {
                "schema": "firesight.review/v1",
                "calibration_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
