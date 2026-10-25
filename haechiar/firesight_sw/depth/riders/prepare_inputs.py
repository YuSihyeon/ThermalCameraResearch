#!/usr/bin/env python3
"""Prepare FireSight thermal frames and projected radar depth for RIDERS experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))
from thermal_io import iter_frames, save_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare RIDERS-style FireSight scene inputs")
    p.add_argument("--thermal", required=True, help="thermal image/video/directory/Y16")
    p.add_argument("--radar-dir", help="projected radar depth PNG directory, one file per frame")
    p.add_argument("--output", default="./outputs/riders_scene")
    p.add_argument("--scene", default="firesight")
    p.add_argument("--raw-width", type=int)
    p.add_argument("--raw-height", type=int)
    p.add_argument("--raw-big-endian", action="store_true")
    p.add_argument("--max-frames", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene = Path(args.output).resolve() / args.scene
    thermal_dir = scene / "thermal_undistort"
    radar_out = scene / "radar_png"
    thermal_dir.mkdir(parents=True, exist_ok=True)
    radar_out.mkdir(parents=True, exist_ok=True)

    frames = list(iter_frames(
        args.thermal,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        raw_big_endian=args.raw_big_endian,
        max_frames=args.max_frames,
    ))
    for frame in frames:
        save_rgb(thermal_dir / f"{frame.index:06d}.png", frame.rgb)

    radar_files = []
    if args.radar_dir:
        src = Path(args.radar_dir)
        radar_files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".png")
        if len(radar_files) != len(frames):
            raise ValueError(f"thermal frames={len(frames)}, radar PNGs={len(radar_files)}; counts must match")
        for idx, p in enumerate(radar_files):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"cannot read radar depth: {p}")
            cv2.imwrite(str(radar_out / f"{idx:06d}.png"), img)

    metadata = {
        "scene": args.scene,
        "thermal_frames": len(frames),
        "radar_frames": len(radar_files),
        "radar_requirement": "radar_png must already be geometrically projected into the thermal camera image plane",
        "note": "This prepares RIDERS input structure; it does not perform radar-to-camera calibration/projection.",
    }
    (scene / "firesight_input.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"prepared: {scene}")


if __name__ == "__main__":
    main()
