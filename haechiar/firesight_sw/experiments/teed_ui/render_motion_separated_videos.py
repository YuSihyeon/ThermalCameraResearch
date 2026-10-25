"""Whole thermal videos with camera-compensated static/dynamic contour policy."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from contour_tracker import ContourConfig  # noqa: E402
from hud import HUDConfig, HUDRenderer  # noqa: E402
from render_additional_thermal_videos import (  # noqa: E402
    inspect_output,
    make_contact_sheet,
    open_capture,
)
from replay import load_fire360_osd_regions  # noqa: E402
from teed_core import TEEDCore, sha256_file  # noqa: E402

OUT = Path(__file__).with_name("motion_split")


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def run(number: int, destination: Path, threads: int, comparison: bool) -> None:
    name = f"fire360_ifsi{number}"
    source = ROOT.parent / f"data/fire360/raw/IFSI Video {number} (2).mp4"
    source_hash = sha256_file(source)
    code_paths = list((ROOT / "edge/orin").glob("*.py")) + [Path(__file__).resolve()]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths}
    destination.mkdir(parents=True, exist_ok=True)
    cap = open_capture(source)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    regions = load_fire360_osd_regions(
        str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"), source.name
    )
    assert len(regions) == 2
    config = HUDConfig(
        output_size=(720, 405),
        temporal_edges=True,
        contour_edges=True,
        thermal_palette_overlay=True,
        contour_config=ContourConfig(motion_separation=True),
    )
    renderer = HUDRenderer(config)
    model = TEEDCore(ROOT / "edge/teed/models/5_model.pth", device="cpu")
    output = destination / f"{name}_full_motion_separated.webm"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter.fourcc(*"VP80"), fps, (720, 406)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output}")
    baseline_path = (
        OUT.with_name("thermal_color_full") / f"{name}_full_thermal_colors.webm"
    )
    baseline = open_capture(baseline_path) if comparison else None
    side_path = destination / f"{name}_full_comparison.webm"
    side_writer = (
        cv2.VideoWriter(
            str(side_path), cv2.VideoWriter.fourcc(*"VP80"), fps, (1440, 446)
        )
        if comparison
        else None
    )
    if side_writer is not None and not side_writer.isOpened():
        raise RuntimeError(f"Cannot write {side_path}")
    records = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = len(records)
            t0 = time.perf_counter()
            edge = model.infer(frame, regions)
            t1 = time.perf_counter()
            result = renderer.render(
                frame,
                edge,
                ignore_regions=regions,
                frame_index=index,
                timestamp_s=index / fps,
            )
            t2 = time.perf_counter()
            writer.write(
                cv2.copyMakeBorder(result.image, 0, 1, 0, 0, cv2.BORDER_CONSTANT)
            )
            if baseline is not None and side_writer is not None:
                old_ok, old = baseline.read()
                if not old_ok:
                    raise RuntimeError(f"Short baseline at {index}")
                panel = np.zeros((446, 1440, 3), np.uint8)
                panel[40:445, :720] = old[:405]
                panel[40:445, 720:] = result.image
                cv2.putText(
                    panel,
                    "Before: shared motion policy",
                    (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    "After: static / dynamic separation",
                    (736, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 255, 160),
                    1,
                    cv2.LINE_AA,
                )
                side_writer.write(panel)
            assert result.edge_mask is not None
            records.append(
                {
                    "frame_index": index,
                    "source_seconds": index / fps,
                    "teed_ms": (t1 - t0) * 1000,
                    "hud_ms": (t2 - t1) * 1000,
                    "contour": result.edge_metrics["contour"],
                    "palette": result.palette_metrics,
                    "pixels": int(np.count_nonzero(result.edge_mask)),
                }
            )
            if len(records) % 120 == 0:
                save(
                    destination / f"{name}_progress.json",
                    {
                        "status": "RUNNING",
                        "processed": len(records),
                        "expected": expected,
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                print(name, len(records), "/", expected, flush=True)
        assert len(records) == expected, (len(records), expected)
        if baseline is not None:
            assert not baseline.read()[0], "Baseline has unmatched extra frames"
    finally:
        cap.release()
        writer.release()
        if baseline is not None:
            baseline.release()
        if side_writer is not None:
            side_writer.release()
    qa = inspect_output(output, expected, fps, (720, 406))
    assert qa["status"] == "PASS", qa
    side_qa = (
        inspect_output(side_path, expected, fps, (1440, 446)) if comparison else {}
    )
    assert not comparison or side_qa["status"] == "PASS", side_qa
    assert sha256_file(source) == source_hash
    assert all(sha256_file(ROOT / p) == sha for p, sha in code_hashes.items()), (
        "Code changed during render"
    )
    contact = make_contact_sheet(output, expected, destination / f"{name}_contact.png")
    statuses = {}
    for row in records:
        status = str(row["contour"].get("global_motion_status", "unreported"))
        statuses[status] = statuses.get(status, 0) + 1
    record = {
        "status": "PASS",
        "name": name,
        "source": str(source),
        "source_sha256": source_hash,
        "frames": expected,
        "start_frame": 0,
        "end_frame": expected - 1,
        "fps": fps,
        "duration_s": expected / fps,
        "code_sha256": code_hashes,
        "model": model.describe(),
        "config": dataclasses.asdict(config),
        "osd_regions": regions,
        "output": qa,
        "comparison": side_qa,
        "comparison_baseline": str(baseline_path) if comparison else None,
        "comparison_baseline_sha256": sha256_file(baseline_path)
        if comparison
        else None,
        "contact_sheet": contact,
        "elapsed_s": time.perf_counter() - started,
        "library_threads": threads,
        "global_motion_status_counts": statuses,
        "records": records,
        "limits": "Whole sequential TEED inference with original thermal palette. Encoding and parallel render load make export time unsuitable as realtime FPS. Silent lossy VP8; last row black padding.",
    }
    save(destination / f"{name}.json", record)
    save(
        destination / f"{name}_progress.json",
        {
            "status": "COMPLETE",
            "processed": expected,
            "expected": expected,
            "qa": "PASS",
        },
    )
    print(name, "COMPLETE QA PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numbers", nargs="+", type=int, choices=range(4, 11), required=True
    )
    parser.add_argument("--output", default="full_v1")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for number in args.numbers:
        try:
            run(number, OUT / args.output, args.threads, args.comparison)
        except Exception as error:
            save(
                OUT / args.output / f"fire360_ifsi{number}_progress.json",
                {"status": "FAILED", "error": str(error)},
            )
            raise


if __name__ == "__main__":
    main()
