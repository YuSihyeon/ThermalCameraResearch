from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from hud import HUDConfig, HUDRenderer  # noqa: E402
from replay import load_fire360_osd_regions  # noqa: E402
from teed_core import TEEDCore, sha256_file  # noqa: E402

OUT = Path(__file__).with_name("contour_stability") / "thermal_additional"
RAW = ROOT.parent / "data/fire360/raw"
PROFILE = ROOT / "edge/teed/profiles/fire360_video_profiles.json"
OSD_PROFILE_KEY = "IFSI Video 4 (2).mp4"
CLIPS = {
    5: ("fire360_ifsi5", "IFSI Video 5 (2).mp4"),
    6: ("fire360_ifsi6", "IFSI Video 6 (2).mp4"),
    7: ("fire360_ifsi7", "IFSI Video 7 (2).mp4"),
    9: ("fire360_ifsi9", "IFSI Video 9 (2).mp4"),
    10: ("fire360_ifsi10", "IFSI Video 10 (2).mp4"),
}
CODE_FILES = [
    "edge/orin/teed_core.py",
    "edge/orin/teed_model.py",
    "edge/orin/edge_filter.py",
    "edge/orin/temporal_edges.py",
    "edge/orin/contour_tracker.py",
    "edge/orin/contour_render.py",
    "edge/orin/hud.py",
    "edge/orin/thermal_overlay.py",
    "edge/orin/thermal_palette.py",
    "edge/orin/replay.py",
    "edge/teed/profiles/fire360_video_profiles.json",
    "experiments/teed_ui/render_additional_thermal_videos.py",
]
CONTACT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
OUTPUT_SIZE = (720, 405)
PADDED_SIZE = (720, 406)
PROGRESS_EVERY = 120


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    keys = ("teed_ms", "hud_ms", "frame_total_ms", "write_ms")
    return {key: stats([float(row[key]) for row in records]) for key in keys}


def open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return capture


def expected_index(frame_count: int, fraction: float) -> int:
    if frame_count <= 1:
        return 0
    return round((frame_count - 1) * fraction)


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = open_capture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return frame


def make_contact_sheet(path: Path, frame_count: int, output_path: Path) -> str:
    sample_indices = [expected_index(frame_count, value) for value in CONTACT_FRACTIONS]
    thumbs: list[np.ndarray] = []
    for index in sample_indices:
        frame = read_frame(path, index)
        thumb_width = 360
        thumb_height = max(1, round(frame.shape[0] * thumb_width / frame.shape[1]))
        thumbs.append(
            cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        )
    height = max(thumb.shape[0] for thumb in thumbs)
    padded: list[np.ndarray] = []
    for thumb in thumbs:
        if thumb.shape[0] < height:
            pad = np.zeros((height - thumb.shape[0], thumb.shape[1], 3), dtype=np.uint8)
            thumb = np.vstack((thumb, pad))
        padded.append(thumb)
    sheet = np.hstack(padded)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Cannot write contact sheet: {output_path}")
    return str(output_path.resolve())


def inspect_output(
    path: Path,
    expected_frames: int,
    expected_fps: float,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    failures: list[str] = []
    decoded = hashlib.sha256()
    capture = open_capture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    width = height = 0
    expected_width, expected_height = expected_size
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if (width, height) != expected_size:
                failures.append(
                    f"frame {frames} size {width}x{height} != "
                    f"{expected_width}x{expected_height}"
                )
                break
            decoded.update(np.ascontiguousarray(frame).tobytes())
            frames += 1
    finally:
        capture.release()

    if frames != expected_frames:
        failures.append(f"frames {frames} != {expected_frames}")
    if not math.isfinite(fps) or abs(fps - expected_fps) > 0.01:
        failures.append(f"fps {fps:.6f} != {expected_fps:.6f}")
    if (width, height) != expected_size:
        failures.append(f"last size {width}x{height} != {expected_width}x{expected_height}")

    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "decoded_sha256": decoded.hexdigest(),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "expected_frames": expected_frames,
        "expected_fps": expected_fps,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def setup_logger(name: str, output_root: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(output_root / f"{name}.log", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def check_code_hashes(hashes: dict[str, str]) -> dict[str, Any]:
    mismatches: list[str] = []
    for relative, expected in hashes.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"{relative}: {actual} != {expected}")
    return {"checked": True, "passed": not mismatches, "mismatches": mismatches}


def render_clip(
    number: int, output_root: Path = OUT, thermal_palette_overlay: bool = False
) -> dict[str, Any]:
    name, filename = CLIPS[number]
    source = RAW / filename
    if not source.is_file():
        raise FileNotFoundError(f"Missing Fire360 source: {source}")

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(name, output_root)
    progress_path = output_root / f"{name}_progress.json"
    metadata_path = output_root / f"{name}.json"
    qa_path = output_root / f"{name}_qa.json"
    suffix = "hot_outline" if thermal_palette_overlay else "applied"
    output_path = output_root / f"{name}_full_{suffix}.webm"
    contact_path = output_root / f"{name}_contact.png"

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    source_start_sha256 = sha256_file(source)
    code_hashes = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in CODE_FILES
    }
    osd_regions = load_fire360_osd_regions(str(PROFILE), OSD_PROFILE_KEY)
    core = TEEDCore(
        ROOT / "edge/teed/models/5_model.pth",
        width=320,
        height=256,
        threshold=0.75,
        edge_width=3,
        contrast_clip_limit=2,
        device="cpu",
    )
    renderer = HUDRenderer(
        HUDConfig(output_size=OUTPUT_SIZE, temporal_edges=True, contour_edges=True,
                  thermal_palette_overlay=thermal_palette_overlay)
    )

    capture = open_capture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or expected_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source timeline: {source}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"VP80"),
        fps,
        PADDED_SIZE,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video: {output_path}")

    records: list[dict[str, Any]] = []
    first_source_frame_sha256 = ""
    last_source_frame_sha256 = ""
    first_rendered_frame_sha256 = ""
    last_rendered_frame_sha256 = ""
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_started = time.perf_counter()
            index = len(records)
            source_hash = frame_sha256(frame)
            if index == 0:
                first_source_frame_sha256 = source_hash
            last_source_frame_sha256 = source_hash

            timestamp = index / fps
            edge = core.infer(frame, osd_regions)
            rendered = renderer.render(
                frame,
                edge,
                ignore_regions=osd_regions,
                frame_index=index,
                timestamp_s=timestamp,
            )
            padded = cv2.copyMakeBorder(
                rendered.image, 0, 1, 0, 0, cv2.BORDER_CONSTANT
            )
            rendered_hash = frame_sha256(padded)
            if index == 0:
                first_rendered_frame_sha256 = rendered_hash
            last_rendered_frame_sha256 = rendered_hash

            before_write = time.perf_counter()
            writer.write(padded)
            after_write = time.perf_counter()
            edge_pixels = (
                0
                if rendered.edge_mask is None
                else int(np.count_nonzero(rendered.edge_mask))
            )
            records.append(
                {
                    "frame_index": index,
                    "source_seconds": timestamp,
                    "teed_ms": edge.timings_ms["total"],
                    "hud_ms": rendered.render_ms,
                    "frame_total_ms": (after_write - frame_started) * 1000.0,
                    "write_ms": (after_write - before_write) * 1000.0,
                    "edge_pixels_after": edge_pixels,
                    "contour": rendered.edge_metrics.get("contour", {}),
                    "palette": rendered.palette_metrics,
                }
            )
            if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == expected_frames:
                elapsed = time.perf_counter() - started
                payload = {
                    "status": "RUNNING",
                    "clip": name,
                    "processed": index + 1,
                    "expected": expected_frames,
                    "elapsed_s": elapsed,
                    "average_processing_fps": (index + 1) / elapsed if elapsed > 0 else 0,
                    "source": str(source.resolve()),
                    "output": str(output_path.resolve()),
                }
                save_json(progress_path, payload)
                logger.info("%s: %s/%s", name, index + 1, expected_frames)
    finally:
        capture.release()
        writer.release()

    elapsed_s = time.perf_counter() - started
    if len(records) != expected_frames:
        save_json(
            progress_path,
            {
                "status": "FAILED",
                "clip": name,
                "processed": len(records),
                "expected": expected_frames,
                "error": "source_decode_incomplete",
            },
        )
        raise RuntimeError(f"Incomplete source decode: {len(records)}/{expected_frames}")

    source_end_sha256 = sha256_file(source)
    if source_end_sha256 != source_start_sha256:
        raise RuntimeError(f"Source changed during rendering: {source}")
    code_check = check_code_hashes(code_hashes)
    if not code_check["passed"]:
        raise RuntimeError("Code changed during rendering")

    output_inspection = inspect_output(output_path, len(records), fps, PADDED_SIZE)
    contact_sheet = ""
    if output_inspection["status"] == "PASS":
        contact_sheet = make_contact_sheet(output_path, len(records), contact_path)

    metadata = {
        "name": name,
        "source": {
            "path": str(source.resolve()),
            "sha256": source_start_sha256,
            "sha256_end": source_end_sha256,
            "expected_frames": expected_frames,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "frames": len(records),
        "start_frame": 0,
        "end_frame": len(records) - 1,
        "duration_s": len(records) / fps,
        "elapsed_s": elapsed_s,
        "average_processing_fps": len(records) / elapsed_s if elapsed_s > 0 else 0.0,
        "render_timings_ms": summarize_records(records),
        "first_last": {
            "first_source_frame_sha256": first_source_frame_sha256,
            "last_source_frame_sha256": last_source_frame_sha256,
            "first_rendered_frame_sha256": first_rendered_frame_sha256,
            "last_rendered_frame_sha256": last_rendered_frame_sha256,
        },
        "code_sha256": code_hashes,
        "code_sha256_end_check": code_check,
        "outputs": [
            {
                "path": str(output_path.resolve()),
                "role": "applied",
                "width": PADDED_SIZE[0],
                "height": PADDED_SIZE[1],
                "frames": len(records),
                "fps": fps,
            }
        ],
        "config": dataclasses.asdict(renderer.config),
        "osd_regions": osd_regions,
        "osd_regions_source_profile_key": OSD_PROFILE_KEY,
        "model": core.describe(),
        "model_sha256": core.checkpoint_sha256,
        "audio": "silent export; source file unchanged",
        "method": "Full source decode and fresh TEED inference with configured HUD",
        "records": records,
    }
    save_json(metadata_path, metadata)

    qa = {
        "status": "PASS" if output_inspection["status"] == "PASS" else "FAIL",
        "clip": name,
        "metadata": str(metadata_path.resolve()),
        "output": output_inspection,
        "contact_sheet": contact_sheet,
        "source_sha256_verified": source_start_sha256 == source_end_sha256,
        "code_sha256_end_check": code_check,
    }
    save_json(qa_path, qa)
    save_json(
        progress_path,
        {
            "status": "COMPLETE" if qa["status"] == "PASS" else "FAILED",
            "clip": name,
            "processed": len(records),
            "expected": expected_frames,
            "elapsed_s": elapsed_s,
            "qa": str(qa_path.resolve()),
        },
    )
    logger.info("%s complete: %s frames in %.3fs", name, len(records), elapsed_s)
    if qa["status"] != "PASS":
        raise RuntimeError(f"Output QA failed: {qa_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number",
        type=int,
        required=True,
        choices=sorted(CLIPS),
        help="Fire360 IFSI clip number to render: 5, 6, 7, 9, or 10.",
    )
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--thermal-palette-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_clip(int(args.number), args.output_root, args.thermal_palette_overlay)


if __name__ == "__main__":
    main()
