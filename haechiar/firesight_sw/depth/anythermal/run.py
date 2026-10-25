#!/usr/bin/env python3
"""FireSight input adapter for AnyThermal + MiDaS official depth inference."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))
from thermal_io import iter_frames, save_rgb, write_preview_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AnyThermal MiDaS depth on FireSight thermal input")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="./outputs/anythermal")
    p.add_argument("--upstream", default=str(DEPTH_ROOT / "third_party" / "AnyThermal"))
    p.add_argument("--ckpt", help="Override released Midas_anythermal checkpoint path")
    p.add_argument("--raw-width", type=int)
    p.add_argument("--raw-height", type=int)
    p.add_argument("--raw-big-endian", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--preview-fps", type=float, default=15.0)
    p.add_argument("--keep-prepared", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    upstream = Path(args.upstream).resolve()
    bridge = upstream / "baselines" / "depth" / "BridgeMultiSpectralDepth"
    if not bridge.exists():
        raise FileNotFoundError(f"AnyThermal depth submodule not found: {bridge}. Run setup.sh first.")

    ckpt = Path(args.ckpt).resolve() if args.ckpt else (
        upstream / "pretrained_checkpoints" / "depth" / "Midas_anythermal" / "ckpt_epoch=28_step=145000.ckpt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    temp_root_obj = tempfile.TemporaryDirectory(prefix="firesight_anythermal_")
    prepared = Path(temp_root_obj.name)
    left = prepared / "left"
    left.mkdir(parents=True)

    count = 0
    for frame in iter_frames(
        args.input,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        raw_big_endian=args.raw_big_endian,
        max_frames=args.max_frames,
    ):
        save_rgb(left / f"{frame.index:06d}_thr.png", frame.rgb)
        count += 1
    if count == 0:
        raise RuntimeError("no frames found")

    cmd = [
        sys.executable,
        "inference_depth.py",
        "--config", "configs/AnyThermal/Midas_anythermal.yaml",
        "--input_dir", str(prepared),
        "--save_dir", str(output),
        "--modality", "thr",
        "--ckpt_path", str(ckpt),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=bridge, check=True)

    previews = sorted((output / "all").glob("*.png"))
    if len(previews) > 1:
        write_preview_video(previews, output / "preview.mp4", fps=args.preview_fps)

    if args.keep_prepared:
        kept = output / "prepared_input"
        if kept.exists():
            shutil.rmtree(kept)
        shutil.copytree(prepared, kept)
    temp_root_obj.cleanup()
    print(f"done: {output}")


if __name__ == "__main__":
    main()
