#!/usr/bin/env python3
"""Shared FireSight thermal input helpers for depth baseline wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RAW_EXTS = {".raw", ".y16"}


@dataclass
class Frame:
    index: int
    name: str
    rgb: np.ndarray
    source_min: float
    source_max: float


def _robust_u8(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> tuple[np.ndarray, float, float]:
    arr = gray.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, np.uint8), 0.0, 0.0
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8), lo, hi
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8), lo, hi


def thermal_to_rgb_u8(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Convert 8/16-bit thermal-like image to 3-channel uint8 RGB.

    16-bit images are percentile-scaled per frame. This is an inference input
    conversion, not radiometric calibration or temperature conversion.
    """
    if image is None:
        raise ValueError("empty image")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype == np.uint8:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return rgb, float(gray.min()), float(gray.max())
        gray = image[..., 0]
    else:
        gray = image

    gray_u8, lo, hi = _robust_u8(gray)
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    return rgb, lo, hi


def read_y16_raw(path: Path, width: int, height: int, big_endian: bool = False) -> np.ndarray:
    if not width or not height:
        raise ValueError("raw Y16 input requires --raw-width and --raw-height")
    dtype = np.dtype(">u2" if big_endian else "<u2")
    arr = np.fromfile(path, dtype=dtype)
    expected = width * height
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} uint16 values, got {arr.size}")
    return arr.reshape(height, width)


def iter_frames(
    input_path: str,
    raw_width: Optional[int] = None,
    raw_height: Optional[int] = None,
    raw_big_endian: bool = False,
    max_frames: Optional[int] = None,
) -> Iterator[Frame]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS | RAW_EXTS)
        for idx, file in enumerate(files):
            if max_frames is not None and idx >= max_frames:
                break
            if file.suffix.lower() in RAW_EXTS:
                img = read_y16_raw(file, int(raw_width or 0), int(raw_height or 0), raw_big_endian)
            else:
                img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            rgb, lo, hi = thermal_to_rgb_u8(img)
            yield Frame(idx, file.stem, rgb, lo, hi)
        return

    suffix = path.suffix.lower()
    if suffix in RAW_EXTS:
        img = read_y16_raw(path, int(raw_width or 0), int(raw_height or 0), raw_big_endian)
        rgb, lo, hi = thermal_to_rgb_u8(img)
        yield Frame(0, path.stem, rgb, lo, hi)
        return

    if suffix in IMAGE_EXTS:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        rgb, lo, hi = thermal_to_rgb_u8(img)
        yield Frame(0, path.stem, rgb, lo, hi)
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input as image/raw/video: {path}")
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            yield Frame(idx, f"{idx:06d}", rgb, float(gray.min()), float(gray.max()))
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break
    finally:
        cap.release()


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def colorize_depth(depth: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    arr = depth.astype(np.float32)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(valid, lo_pct))
    hi = float(np.percentile(valid, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = np.round(norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_preview_video(frame_paths: list[Path], output: Path, fps: float = 15.0) -> None:
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for path in frame_paths:
            img = cv2.imread(str(path))
            if img is not None:
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                writer.write(img)
    finally:
        writer.release()
