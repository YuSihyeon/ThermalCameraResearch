from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("contour_stability/thermal_full")
EXPECTED_CLIPS = {
    "fire360_ifsi4": ("IFSI Video 4 (2).mp4", 625),
    "fire360_ifsi8": ("IFSI Video 8 (2).mp4", 661),
}
EXPECTED_OUTPUTS = {
    "applied": (720, 406),
    "comparison": (1440, 540),
}
CONTACT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


class Inspection(TypedDict):
    path: str
    role: str
    opened: bool
    frames: int
    fps: float
    width: int
    height: int
    expected_frames: int
    expected_fps: float
    expected_width: int
    expected_height: int
    passed: bool
    failures: list[str]
    decoded_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(
    path: Path,
    role: str,
    expected_frames: int,
    expected_fps: float,
    expected_size: tuple[int, int],
) -> Inspection:
    capture = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    opened = capture.isOpened()
    frames = 0
    width = height = 0
    digest = hashlib.sha256()
    size_failures: list[str] = []
    expected_width, expected_height = expected_size
    while opened:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        height, width = frame.shape[:2]
        if (width, height) != (expected_width, expected_height) and not size_failures:
            size_failures.append(
                f"frame {frames - 1} size {width}x{height} != "
                f"{expected_width}x{expected_height}"
            )
        digest.update(frame.tobytes())
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    capture.release()

    failures: list[str] = []
    if not opened:
        failures.append("not_opened")
    if frames != expected_frames:
        failures.append(f"frames {frames} != {expected_frames}")
    if abs(fps - expected_fps) > 0.01:
        failures.append(f"fps {fps:.6f} != {expected_fps:.6f}")
    if (width, height) != (expected_width, expected_height):
        failures.append(
            f"size {width}x{height} != {expected_width}x{expected_height}"
        )
    failures.extend(size_failures)

    return {
        "path": str(path),
        "role": role,
        "opened": bool(opened),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "expected_frames": expected_frames,
        "expected_fps": expected_fps,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "passed": not failures,
        "failures": failures,
        "decoded_sha256": digest.hexdigest(),
    }


def require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def expected_index(frame_count: int, fraction: float) -> int:
    if frame_count <= 1:
        return 0
    return round((frame_count - 1) * fraction)


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    if not capture.isOpened():
        raise ValueError(f"unable to open {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"unable to read frame {index} from {path}")
    return frame


def fit_thumbnail(frame: np.ndarray, width: int = 360) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_contact_sheet(
    name: str,
    outputs: list[dict[str, Any]],
    frame_count: int,
    output_path: Path,
) -> str:
    by_role = {str(item["role"]): item for item in outputs}
    sample_indices = [expected_index(frame_count, fraction) for fraction in CONTACT_FRACTIONS]
    rows: list[np.ndarray] = []
    for role in ("applied", "comparison"):
        path = Path(str(by_role[role]["path"]))
        thumbs = [fit_thumbnail(read_frame(path, index)) for index in sample_indices]
        row_height = max(thumb.shape[0] for thumb in thumbs)
        padded: list[np.ndarray] = []
        for thumb in thumbs:
            if thumb.shape[0] < row_height:
                pad = np.zeros(
                    (row_height - thumb.shape[0], thumb.shape[1], 3), dtype=np.uint8
                )
                thumb = np.vstack((thumb, pad))
            padded.append(thumb)
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), sheet):
        raise ValueError(f"unable to write contact sheet {output_path}")
    return str(output_path)


def is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def verify_records(data: dict[str, Any], frame_count: int, fps: float) -> list[str]:
    failures: list[str] = []
    records = require_list(data, "records")
    if not is_finite_positive(fps):
        failures.append(f"records cannot validate source_seconds with fps {fps}")
        return failures
    if len(records) != frame_count:
        failures.append(f"records {len(records)} != {frame_count}")
        return failures
    for expected, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"record {expected} is not an object")
            continue
        if record.get("frame_index") != expected:
            failures.append(f"record frame_index {record.get('frame_index')} != {expected}")
            break
        source_seconds = record.get("source_seconds")
        if not isinstance(source_seconds, int | float):
            failures.append(f"record {expected} source_seconds is not numeric")
            break
        expected_seconds = expected / fps
        if abs(float(source_seconds) - expected_seconds) > 1e-9:
            failures.append(
                f"record {expected} source_seconds {source_seconds} != {expected_seconds}"
            )
            break
    return failures


def verify_code_hashes(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    code_sha = require_dict(data, "code_sha256")
    if not code_sha:
        failures.append("code_sha256 is empty")
        return failures
    for relative, expected in sorted(code_sha.items()):
        if not isinstance(relative, str) or not isinstance(expected, str) or not expected:
            failures.append("code_sha256 must map repo-relative paths to strings")
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"code file missing: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"code sha {relative} {actual} != {expected}")
    return failures


def verify_source(name: str, data: dict[str, Any]) -> tuple[Inspection, list[str]]:
    failures: list[str] = []
    source = require_dict(data, "source")
    path = Path(str(source.get("path", "")))
    expected_file, expected_frames = EXPECTED_CLIPS[name]
    if path.name != expected_file:
        failures.append(f"source file {path.name} != {expected_file}")
    if not path.is_file():
        raise ValueError(f"source file missing: {path}")
    recorded_sha = str(source.get("sha256", ""))
    actual_sha = sha256_file(path)
    if recorded_sha != actual_sha:
        failures.append(f"source sha {actual_sha} != {recorded_sha}")
    frame_count = int(source.get("expected_frames", -1))
    if frame_count != expected_frames:
        failures.append(f"source expected_frames {frame_count} != {expected_frames}")
    fps = float(source.get("fps", 0.0))
    if not is_finite_positive(fps):
        failures.append(f"source fps is not finite positive: {fps}")
    inspection = inspect_video(
        path,
        "source",
        frame_count,
        fps,
        (int(source.get("width", 0)), int(source.get("height", 0))),
    )
    failures.extend(inspection["failures"])
    if inspection["frames"] != frame_count:
        failures.append(
            f"source metadata expected_frames {frame_count} != decoded {inspection['frames']}"
        )
    return inspection, failures


def verify_outputs(
    data: dict[str, Any], frame_count: int, fps: float
) -> tuple[list[Inspection], list[str]]:
    failures: list[str] = []
    output_items = require_list(data, "outputs")
    if len(output_items) != 2:
        failures.append(f"outputs {len(output_items)} != 2")
    roles = sorted(str(item.get("role", "")) for item in output_items if isinstance(item, dict))
    if roles != sorted(EXPECTED_OUTPUTS):
        failures.append(f"output roles {roles} != {sorted(EXPECTED_OUTPUTS)}")

    inspections: list[Inspection] = []
    for item in output_items:
        if not isinstance(item, dict):
            failures.append("output item is not an object")
            continue
        role = str(item.get("role", ""))
        path = Path(str(item.get("path", "")))
        if path.suffix.lower() != ".webm":
            failures.append(f"{role} is not webm: {path}")
        if role not in EXPECTED_OUTPUTS:
            continue
        if not path.is_file():
            failures.append(f"output missing: {path}")
            continue
        expected_size = EXPECTED_OUTPUTS[role]
        expected_width, expected_height = expected_size
        if int(item.get("width", 0)) != expected_width:
            failures.append(f"{role} metadata width {item.get('width')} != {expected_width}")
        if int(item.get("height", 0)) != expected_height:
            failures.append(f"{role} metadata height {item.get('height')} != {expected_height}")
        if int(item.get("frames", -1)) != frame_count:
            failures.append(f"{role} metadata frames {item.get('frames')} != {frame_count}")
        if abs(float(item.get("fps", 0.0)) - fps) > 0.01:
            failures.append(f"{role} metadata fps {item.get('fps')} != {fps:.6f}")
        inspections.append(inspect_video(path, role, frame_count, fps, expected_size))
    return inspections, failures


def verify_metadata(path: Path, root: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name", path.stem))
    if name not in EXPECTED_CLIPS:
        raise ValueError(f"unexpected clip name: {name}")
    frame_count = int(data.get("frames", -1))
    source = require_dict(data, "source")
    fps = float(source.get("fps", 0.0))
    failures: list[str] = []
    if frame_count != int(source.get("expected_frames", -1)):
        failures.append(
            f"frames {frame_count} != source.expected_frames {source.get('expected_frames')}"
        )
    if not is_finite_positive(fps):
        failures.append(f"source fps is not finite positive: {fps}")
    if int(data.get("start_frame", -1)) != 0:
        failures.append(f"start_frame {data.get('start_frame')} != 0")
    if int(data.get("end_frame", -1)) != frame_count - 1:
        failures.append(f"end_frame {data.get('end_frame')} != {frame_count - 1}")
    expected_duration = frame_count / fps if is_finite_positive(fps) else math.inf
    duration = float(data.get("duration_s", 0.0))
    if not math.isfinite(duration) or abs(duration - expected_duration) > 1e-9:
        failures.append(f"duration_s {duration} != {expected_duration}")

    source_inspection, source_failures = verify_source(name, data)
    failures.extend(source_failures)
    failures.extend(verify_records(data, frame_count, fps))
    failures.extend(verify_code_hashes(data))
    output_inspections, output_failures = verify_outputs(data, frame_count, fps)
    failures.extend(output_failures)
    for inspection in output_inspections:
        failures.extend(inspection["failures"])

    contact_path = root / f"{name}_contact.png"
    contact = ""
    if not failures:
        contact = make_contact_sheet(name, require_list(data, "outputs"), frame_count, contact_path)

    return {
        "name": name,
        "metadata": str(path),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "source": source_inspection,
        "outputs": output_inspections,
        "contact_sheet": contact,
    }


def resolve_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "contour_stability":
        return Path(__file__).parent / path
    return Path(__file__).with_name("contour_stability") / path


def main() -> None:
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = resolve_root(args.root)
    root.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(root.glob("fire360_ifsi*.json"))
    names = [path.stem for path in metadata_paths]
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    if names != sorted(EXPECTED_CLIPS):
        failures.append(f"metadata set {names} != {sorted(EXPECTED_CLIPS)}")
    for path in metadata_paths:
        try:
            results.append(verify_metadata(path, root))
        except Exception as exc:
            results.append(
                {
                    "name": path.stem,
                    "metadata": str(path),
                    "status": "FAIL",
                    "failures": [str(exc)],
                    "source": None,
                    "outputs": [],
                    "contact_sheet": "",
                }
            )
    failures.extend(
        f"{item['name']}: {failure}"
        for item in results
        for failure in item["failures"]
    )
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "root": str(root),
        "expected_clips": sorted(EXPECTED_CLIPS),
        "failures": failures,
        "clips": results,
    }
    qa_path = root / "qa.json"
    qa_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
