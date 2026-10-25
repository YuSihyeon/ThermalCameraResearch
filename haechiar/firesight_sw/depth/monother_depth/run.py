#!/usr/bin/env python3
"""FireSight wrapper for MonoTher-Depth thermal-branch inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))
from thermal_io import colorize_depth, iter_frames, write_preview_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MonoTher-Depth on FireSight thermal image/video/Y16 input")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="./outputs/monother_depth")
    p.add_argument("--upstream", default=str(DEPTH_ROOT / "third_party" / "monother_depth"))
    p.add_argument("--weights", required=True, help="MonoTher MS2 checkpoint (.pt)")
    p.add_argument("--depth-anything-weights", required=True, help="depth_anything_vitl14.pth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw-width", type=int)
    p.add_argument("--raw-height", type=int)
    p.add_argument("--raw-big-endian", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--preview-fps", type=float, default=15.0)
    return p.parse_args()


def load_model(upstream: Path, weights: Path, base_weights: Path, device: torch.device):
    if not upstream.exists():
        raise FileNotFoundError(f"upstream repo not found: {upstream}. Run setup.sh first.")
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not base_weights.exists():
        raise FileNotFoundError(base_weights)

    (upstream / "checkpoints").mkdir(exist_ok=True)
    target_base = upstream / "checkpoints" / "depth_anything_vitl14.pth"
    if target_base.resolve() != base_weights.resolve():
        if target_base.exists() or target_base.is_symlink():
            target_base.unlink()
        try:
            target_base.symlink_to(base_weights.resolve())
        except OSError:
            import shutil
            shutil.copy2(base_weights, target_base)

    sys.path.insert(0, str(upstream))
    old_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config

        config = get_config(
            "zoedepth_rgbt",
            "eval",
            "ms2thermal",
            config_version="ms2_teach_nofs",
            dataloader_type=2,
            modality="rgbtd",
            pretrained_resource=f"local::{weights.resolve()}",
        )
        model = build_model(config).to(device)
        model.eval()
        return model
    finally:
        os.chdir(old_cwd)


def main() -> None:
    args = parse_args()
    upstream = Path(args.upstream).resolve()
    weights = Path(args.weights).resolve()
    base_weights = Path(args.depth_anything_weights).resolve()
    output = Path(args.output).resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(upstream, weights, base_weights, device)
    preview_paths: list[Path] = []

    old_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        for frame in iter_frames(
            args.input,
            raw_width=args.raw_width,
            raw_height=args.raw_height,
            raw_big_endian=args.raw_big_endian,
            max_frames=args.max_frames,
        ):
            h0, w0 = frame.rgb.shape[:2]
            rgb = cv2.resize(frame.rgb, (640, 256), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

            with torch.inference_mode():
                pred_thermal, _ = model(tensor, tensor, dataset="ms2thermal")
                depth = pred_thermal["metric_depth"].squeeze().float().cpu().numpy()

            depth = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            stem = f"{frame.index:06d}_{frame.name}"
            np.save(frames_dir / f"{stem}_depth.npy", depth)

            depth_rgb = colorize_depth(depth)
            input_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
            preview = np.hstack((input_bgr, depth_bgr))
            preview_path = frames_dir / f"{stem}_preview.png"
            cv2.imwrite(str(preview_path), preview)
            preview_paths.append(preview_path)
            print(f"[{frame.index:06d}] saved {preview_path.name}")
    finally:
        os.chdir(old_cwd)

    if len(preview_paths) > 1:
        write_preview_video(preview_paths, output / "preview.mp4", fps=args.preview_fps)
    print(f"done: {output}")


if __name__ == "__main__":
    main()
