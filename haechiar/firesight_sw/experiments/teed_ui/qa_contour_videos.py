from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import cv2


class VideoInspection(TypedDict):
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
    sample_sha256: str


def inspect_video(
    path: Path,
    role: str,
    expected_frames: int,
    expected_fps: float,
    expected_size: tuple[int, int],
) -> VideoInspection:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frames = 0
    digest = hashlib.sha256()
    width = height = 0
    while opened:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        height, width = frame.shape[:2]
        digest.update(frame.tobytes())
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    capture.release()
    failures: list[str] = []
    expected_width, expected_height = expected_size
    if not opened:
        failures.append("not_opened")
    if frames != expected_frames:
        failures.append(f"frames {frames} != {expected_frames}")
    if abs(fps - expected_fps) > 0.01:
        failures.append(f"fps {fps:.6f} != {expected_fps:.6f}")
    if (width, height) != (expected_width, expected_height):
        failures.append(f"size {width}x{height} != {expected_width}x{expected_height}")
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
        "sample_sha256": digest.hexdigest(),
    }


def expected_videos(root: Path) -> list[tuple[Path, str, int, float, tuple[int, int]]]:
    expected: list[tuple[Path, str, int, float, tuple[int, int]]] = []
    json_paths = sorted(
        path
        for path in root.glob("fire360_*.json")
        if path.name != "video_decode_qa.json"
    )
    for json_path in json_paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        name = str(data["name"])
        frames = int(data["frames"])
        fps = float(data["fps"])
        expected.extend(
            [
                (root / f"{name}.webm", "pair_webm", frames, fps, (1440, 540)),
                (root / f"{name}.mp4", "pair_mp4", frames, fps, (1440, 540)),
                (root / f"{name}_contour.webm", "single_contour", frames, fps, (720, 406)),
            ]
        )
        if name == "fire360_ifsi8":
            expected.append(
                (root / f"{name}_detail.webm", "detail", frames, fps, (960, 720))
            )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="video_iteration1")
    args = parser.parse_args()
    root = Path(__file__).with_name("contour_stability") / args.root
    expected = expected_videos(root)
    results = [
        inspect_video(path, role, frames, fps, size)
        if path.is_file()
        else {
            "path": str(path),
            "role": role,
            "opened": False,
            "frames": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "expected_frames": frames,
            "expected_fps": fps,
            "expected_width": size[0],
            "expected_height": size[1],
            "passed": False,
            "failures": ["missing"],
            "sample_sha256": "",
        }
        for path, role, frames, fps, size in expected
    ]
    payload = {
        "status": "PASS"
        if len(expected) == 10 and all(item["passed"] for item in results)
        else "FAIL",
        "expected_video_count": 10,
        "video_count": len(results),
        "videos": results,
    }
    (root / "video_decode_qa.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
