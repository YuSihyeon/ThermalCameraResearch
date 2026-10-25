#!/usr/bin/env python3
"""RIDERS stage-1 style global metric scale alignment for FireSight experiments.

This is intentionally labelled stage-1, not full RIDERS. It aligns a dense
monocular inverse-depth/depth prediction with sparse projected radar depth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))
from thermal_io import colorize_depth  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global radar scale alignment (RIDERS stage-1 proxy)")
    p.add_argument("--mono-depth", required=True, help="dense depth .npy produced by a monocular baseline")
    p.add_argument("--radar-depth", required=True, help="projected sparse radar depth PNG/NPY in meters")
    p.add_argument("--output", required=True, help="output aligned .npy")
    p.add_argument("--mono-type", choices=["depth", "inverse"], default="depth")
    p.add_argument("--min-radar", type=float, default=0.1)
    p.add_argument("--max-radar", type=float, default=50.0)
    p.add_argument("--radar-scale", type=float, default=1.0, help="multiply loaded radar values by this factor")
    return p.parse_args()


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"cannot read {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    mono = load_depth(Path(args.mono_depth))
    radar = load_depth(Path(args.radar_depth)) * args.radar_scale
    if radar.shape != mono.shape:
        radar = cv2.resize(radar, (mono.shape[1], mono.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid = np.isfinite(radar) & (radar >= args.min_radar) & (radar <= args.max_radar) & np.isfinite(mono) & (mono > 0)
    if valid.sum() < 3:
        raise RuntimeError(f"not enough valid radar correspondences: {int(valid.sum())}")

    if args.mono_type == "inverse":
        estimate = 1.0 / np.maximum(mono, 1e-6)
    else:
        estimate = mono

    ratios = radar[valid] / np.maximum(estimate[valid], 1e-6)
    scale = float(np.median(ratios[np.isfinite(ratios)]))
    aligned = estimate * scale

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, aligned.astype(np.float32))
    vis = colorize_depth(aligned)
    cv2.imwrite(str(out.with_suffix(".png")), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"valid radar pixels: {int(valid.sum())}")
    print(f"global scale: {scale:.6f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
