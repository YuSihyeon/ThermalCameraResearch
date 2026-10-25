#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIN_DIR = REPO_ROOT / "edge" / "orin"
if str(ORIN_DIR) not in sys.path:
    sys.path.insert(0, str(ORIN_DIR))

from hud import HUDConfig, HUDRenderer  # noqa: E402
from replay import ReplaySource  # noqa: E402
from teed_core import EdgeResult, TEEDCore, clear_regions, sha256_file  # noqa: E402

DEFAULT_FIRE360 = (
    "IFSI Video 4 (2).mp4",
    "IFSI Video 8 (2).mp4",
    "02815 (2).MTS",
)
DEFAULT_GIT = Path("git")
BASELINE_REVISION = "481569b7412522f9f6f90cde24b320538dce1e13"
DISPLAY_SIZE = (960, 540)
CONTACT_CELL_SIZE = (480, 270)
DESCRIPTION = (
    "Replay TEED HUD filtering against the old HUD on real local clips. "
    "Overlap scores are display proxies only: no obstacle labels, no metric "
    "depth, and no calibrated Celsius maps are used."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--output-dir", default="experiments/teed_ui/results")
    parser.add_argument("--checkpoint", default="edge/teed/models/5_model.pth")
    parser.add_argument("--fire360-root", default="../data/fire360/raw")
    parser.add_argument(
        "--fire360-profile",
        default="edge/teed/profiles/fire360_video_profiles.json",
    )
    parser.add_argument(
        "--fire360-video",
        action="append",
        dest="fire360_videos",
        help="Fire360 video basename; repeatable",
    )
    parser.add_argument(
        "--smoke-root",
        default="../data/external/smokebasement/raw/20250121",
    )
    parser.add_argument(
        "--smoke-index",
        default="../data/external/smokebasement/indexes/run5_left.csv",
    )
    parser.add_argument("--fire360-frames", type=int, default=40)
    parser.add_argument("--smoke-frames", type=int, default=60)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--edge-width", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--git", default=str(DEFAULT_GIT))
    parser.add_argument("--max-video-frames", type=int, default=12)
    return parser


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def set_single_thread_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def run_git(git: Path, args: list[str]) -> str:
    command = [str(git), *args]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def load_baseline_hud(git: Path, output_dir: Path):
    baseline_path = output_dir / "_baseline_hud_HEAD.py"
    baseline_path.write_text(
        run_git(git, ["show", f"{BASELINE_REVISION}:edge/orin/hud.py"]), encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("baseline_hud_head", baseline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import baseline HUD")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_video_frames(
    path: Path, start: int, stride: int, count: int
) -> list[tuple[int, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    samples: list[tuple[int, float, np.ndarray]] = []
    for offset in range(count):
        frame_index = start + offset * stride
        if total and frame_index >= total:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        seconds = frame_index / fps if fps > 0.0 else 0.0
        samples.append((frame_index, seconds, frame))
    capture.release()
    return samples


def choose_evenly(
    items: list[dict[str, str]], count: int
) -> list[tuple[int, dict[str, str]]]:
    if not items:
        return []
    if len(items) <= count:
        return list(enumerate(items))
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [(int(index), items[int(index)]) for index in indices]


def sample_smoke_rows(
    index_path: Path, root: Path, count: int
) -> list[tuple[int, float, np.ndarray]]:
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty SmokeBasement index: {index_path}")
    first_seconds = float(rows[0]["sensor_timestamp_seconds"])
    window = [
        row
        for row in rows
        if first_seconds + 6.0
        <= float(row["sensor_timestamp_seconds"])
        <= first_seconds + 12.0
    ]
    selected_rows = choose_evenly(window or rows, count)
    samples: list[tuple[int, float, np.ndarray]] = []
    for row_index, row in selected_rows:
        image_path = root / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unable to read SmokeBasement image: {image_path}")
        samples.append((row_index, float(row["sensor_timestamp_seconds"]), image))
    return samples


def dilated(mask: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance <= 0:
        return mask > 0
    size = tolerance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1) > 0


def overlap_proxy(
    reference: np.ndarray, candidate: np.ndarray, tolerance: int = 2
) -> dict[str, float | int]:
    ref = reference > 0
    cand = candidate > 0
    ref_count = int(np.count_nonzero(ref))
    cand_count = int(np.count_nonzero(cand))
    ref_near = dilated(reference, tolerance)
    cand_near = dilated(candidate, tolerance)
    recovered = int(np.count_nonzero(ref & cand_near))
    supported = int(np.count_nonzero(cand & ref_near))
    recall = recovered / ref_count if ref_count else 0.0
    precision = supported / cand_count if cand_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_pixels": ref_count,
        "candidate_pixels": cand_count,
        "proxy_recall": recall,
        "proxy_precision": precision,
        "proxy_f1": f1,
        "tolerance_px": tolerance,
    }


def mask_coverage(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def render_source(frame: np.ndarray) -> np.ndarray:
    output = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
    return output


def label_cell(image: np.ndarray, label: str) -> np.ndarray:
    cell = cv2.resize(image, CONTACT_CELL_SIZE, interpolation=cv2.INTER_AREA)
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (cell.shape[1], 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, cell, 0.45, 0, cell)
    cv2.putText(
        cell,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return cell


def comparison_panel(
    source: np.ndarray, baseline: np.ndarray, filtered: np.ndarray, title: str
) -> np.ndarray:
    row = cv2.hconcat(
        [
            label_cell(source, "Source"),
            label_cell(baseline, "Old HUD: raw TEED + red colour heuristic"),
            label_cell(filtered, "Current HUD: filtered TEED + TEMP N/A"),
        ]
    )
    header = np.full((36, row.shape[1], 3), 22, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([header, row])


def contact_sheet(panels: Iterable[np.ndarray], cols: int = 1) -> np.ndarray:
    panel_list = list(panels)
    if not panel_list:
        raise RuntimeError("no panels available")
    width = panel_list[0].shape[1]
    height = panel_list[0].shape[0]
    blank = np.zeros_like(panel_list[0])
    rows = []
    for start in range(0, len(panel_list), cols):
        cells = panel_list[start : start + cols]
        while len(cells) < cols:
            cells.append(blank)
        rows.append(cv2.hconcat(cells))
    sheet = cv2.vconcat(rows)
    if sheet.shape[1] != width * cols or sheet.shape[0] < height:
        raise RuntimeError("invalid contact sheet shape")
    return sheet


def edge_result_with_mask(edge: EdgeResult, mask: np.ndarray) -> EdgeResult:
    return EdgeResult(
        probability=edge.probability,
        mask=mask.astype(np.uint8, copy=False),
        source_view=edge.source_view,
        size=edge.size,
        timings_ms=edge.timings_ms,
    )


def make_binary_from_probability(
    probability: np.ndarray,
    threshold: float,
    width: int,
    ignore_regions: tuple,
) -> np.ndarray:
    mask = (probability >= threshold).astype(np.uint8) * 255
    if width > 1:
        kernel_size = width + (1 - width % 2)
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8))
    if ignore_regions:
        mask = clear_regions(mask, ignore_regions)
    return mask


def aggregate(records: list[dict]) -> dict[str, object]:
    numeric_keys = (
        "legacy_edge_coverage",
        "filtered_edge_coverage",
        "relative_edge_coverage",
        "proxy_recall",
        "proxy_precision",
        "proxy_f1",
        "baseline_fire_candidate_pixels",
        "filtered_hot_pixels",
        "filtered_contour_pixels",
        "teed_total_ms",
        "filtered_hud_ms",
        "baseline_hud_ms",
    )
    output: dict[str, object] = {"frames": len(records)}
    for key in numeric_keys:
        values = [float(record[key]) for record in records if key in record]
        if values:
            output[key] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("thermal_status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    output["thermal_status_counts"] = statuses
    return output


def process_dataset(
    name: str,
    samples: list[tuple[int, float, np.ndarray]],
    core: TEEDCore,
    baseline_renderer,
    filtered_renderer: HUDRenderer,
    out_dir: Path,
    max_video_frames: int,
    ignore_regions: tuple = (),
    pixel_origin: str = "bgr8",
) -> tuple[list[dict], list[np.ndarray], np.ndarray]:
    records: list[dict] = []
    panels: list[np.ndarray] = []
    overview_panel = None
    overview_ordinal = len(samples) // 2
    video_path = out_dir / f"{name}_comparison.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (CONTACT_CELL_SIZE[0] * 3, CONTACT_CELL_SIZE[1] + 36),
    )
    if not writer.isOpened():
        raise RuntimeError(f"unable to create video: {video_path}")

    first_saved = False
    for ordinal, (frame_index, seconds, frame) in enumerate(samples):
        started = time.perf_counter()
        edge = core.infer(frame, ignore_regions=ignore_regions)
        legacy_mask = make_binary_from_probability(
            edge.probability,
            core.threshold,
            core.edge_width,
            ignore_regions,
        )
        legacy_edge = edge_result_with_mask(edge, legacy_mask)
        stats = {
            "source": name,
            "frame": frame_index,
            **edge.timings_ms,
        }
        baseline = baseline_renderer.render(
            frame, legacy_edge, stats, pixel_origin=pixel_origin
        )
        filtered = filtered_renderer.render(
            frame,
            edge,
            stats,
            pixel_origin=pixel_origin,
            temperature_c=None,
            depth_m=None,
            ignore_regions=ignore_regions,
        )
        selected_pixels = int(filtered.edge_metrics.get("output_edge_pixels", 0))
        legacy_pixels = int(np.count_nonzero(legacy_mask))
        proxy = overlap_proxy(
            legacy_mask, make_filtered_mask(edge, ignore_regions, filtered.edge_metrics)
        )
        source_view = render_source(frame)
        panel = comparison_panel(
            source_view,
            baseline.image,
            filtered.image,
            f"{name} frame={frame_index} t={seconds:.2f}s | temperature unavailable",
        )
        if ordinal < max_video_frames:
            writer.write(panel)
        if ordinal in representative_ordinals(len(samples)):
            panel_path = out_dir / f"{name}_frame_{frame_index:06d}.png"
            cv2.imwrite(str(panel_path), panel)
            panels.append(panel)
        if ordinal == overview_ordinal:
            overview_panel = panel
        if not first_saved:
            cv2.imwrite(str(out_dir / f"{name}_first_source.png"), source_view)
            prob_u8 = np.clip(edge.probability * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{name}_first_teed_probability.png"), prob_u8)
            first_saved = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record = {
            "dataset": name,
            "frame_index": frame_index,
            "source_seconds": seconds,
            "legacy_edge_pixels": legacy_pixels,
            "filtered_edge_pixels": selected_pixels,
            "legacy_edge_coverage": mask_coverage(legacy_mask),
            "filtered_edge_coverage": selected_pixels / float(edge.probability.size),
            "relative_edge_coverage": (selected_pixels / legacy_pixels)
            if legacy_pixels
            else 0.0,
            "proxy_reference": (
                "legacy TEED mask dilated by tolerance; proxy only, "
                "not obstacle recall"
            ),
            "proxy_recall": float(proxy["proxy_recall"]),
            "proxy_precision": float(proxy["proxy_precision"]),
            "proxy_f1": float(proxy["proxy_f1"]),
            "baseline_fire_candidate_pixels": int(
                getattr(baseline, "fire_candidate_pixels", 0)
            ),
            "thermal_status": filtered.thermal_metrics.get("status"),
            "filtered_hot_pixels": int(filtered.thermal_metrics.get("hot_pixels", 0)),
            "filtered_contour_pixels": int(
                filtered.thermal_metrics.get("contour_pixels", 0)
            ),
            "teed_total_ms": float(edge.timings_ms["total"]),
            "baseline_hud_ms": float(baseline.render_ms),
            "filtered_hud_ms": float(filtered.render_ms),
            "frame_total_ms": elapsed_ms,
            "edge_metrics": filtered.edge_metrics,
            "thermal_metrics": filtered.thermal_metrics,
        }
        records.append(record)
    writer.release()
    if overview_panel is None:
        overview_panel = panels[0]
    return records, panels, overview_panel


def make_filtered_mask(
    edge: EdgeResult, ignore_regions: tuple, metrics: dict
) -> np.ndarray:
    from edge_filter import EdgeFilterConfig, select_edges

    selection = select_edges(
        edge.probability,
        edge.source_view,
        EdgeFilterConfig(),
        ignore_regions,
        None,
    )
    expected = int(metrics.get("output_edge_pixels", -1))
    actual = int(np.count_nonzero(selection.mask))
    if expected >= 0 and expected != actual:
        raise RuntimeError(f"edge metric mismatch: expected {expected}, got {actual}")
    return selection.mask


def representative_ordinals(length: int) -> set[int]:
    if length <= 1:
        return {0}
    picks = np.linspace(0, length - 1, min(3, length), dtype=int)
    return {int(value) for value in picks}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    set_single_thread_runtime()
    args = build_parser().parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    git_executable = shutil.which(args.git)
    if git_executable is None:
        raise RuntimeError(f"git executable not found: {args.git}")
    git = Path(git_executable)

    baseline_module = load_baseline_hud(git, output_dir)
    baseline_renderer = baseline_module.HUDRenderer(
        baseline_module.HUDConfig(
            output_size=DISPLAY_SIZE,
            show_panel=False,
            full_res_background=True,
        )
    )
    filtered_renderer = HUDRenderer(
        HUDConfig(output_size=DISPLAY_SIZE, show_panel=False, full_res_background=True)
    )
    core = TEEDCore(
        checkpoint=REPO_ROOT / args.checkpoint,
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        edge_width=args.edge_width,
        contrast_clip_limit=2.0,
        device=args.device,
        half=False,
    )

    git_head = run_git(git, ["rev-parse", "HEAD"]).strip()
    git_status = run_git(git, ["status", "--short"])
    profile_path = REPO_ROOT / args.fire360_profile
    fire360_root = (REPO_ROOT / args.fire360_root).resolve()
    fire_videos = tuple(args.fire360_videos or DEFAULT_FIRE360)

    all_records: list[dict] = []
    all_panels: list[np.ndarray] = []
    overview_panels: list[np.ndarray] = []
    dataset_summaries: dict[str, object] = {}
    inputs: list[dict[str, object]] = []

    for video_name in fire_videos:
        video_path = fire360_root / video_name
        source = ReplaySource(
            str(video_path),
            profile_path=str(profile_path),
            video_key=video_name,
            max_frames=1,
        )
        try:
            info = source.info.as_dict()
            ignore_regions = source.info.osd_regions
        finally:
            source.close()
        samples = sample_video_frames(
            video_path, args.start, args.stride, args.fire360_frames
        )
        safe_name = "fire360_" + Path(video_name).stem.replace(" ", "_").replace(
            "(", ""
        ).replace(")", "")
        records, panels, overview_panel = process_dataset(
            safe_name,
            samples,
            core,
            baseline_renderer,
            filtered_renderer,
            output_dir,
            args.max_video_frames,
            ignore_regions,
            pixel_origin="bgr8",
        )
        all_records.extend(records)
        all_panels.extend(panels)
        overview_panels.append(overview_panel)
        dataset_summaries[safe_name] = aggregate(records)
        inputs.append(
            {
                "dataset": safe_name,
                "path": rel(video_path),
                "sha256": sha256_file(video_path),
                "source_info": info,
                "sample_policy": {
                    "start": args.start,
                    "stride": args.stride,
                    "requested_frames": args.fire360_frames,
                    "actual_frames": len(samples),
                },
            }
        )

    smoke_root = (REPO_ROOT / args.smoke_root).resolve()
    smoke_index = (REPO_ROOT / args.smoke_index).resolve()
    smoke_samples = sample_smoke_rows(smoke_index, smoke_root, args.smoke_frames)
    smoke_records, smoke_panels, smoke_overview_panel = process_dataset(
        "smokebasement_run5_left",
        smoke_samples,
        core,
        baseline_renderer,
        filtered_renderer,
        output_dir,
        args.max_video_frames,
        (),
        pixel_origin="bgr8",
    )
    all_records.extend(smoke_records)
    all_panels.extend(smoke_panels)
    overview_panels.append(smoke_overview_panel)
    dataset_summaries["smokebasement_run5_left"] = aggregate(smoke_records)
    inputs.append(
        {
            "dataset": "smokebasement_run5_left",
            "root": rel(smoke_root),
            "index": rel(smoke_index),
            "index_sha256": sha256_file(smoke_index),
            "sample_policy": {
                "window": "first timestamp + 6s through +12s",
                "requested_frames": args.smoke_frames,
                "actual_frames": len(smoke_samples),
            },
        }
    )

    contact_path = output_dir / "teed_ui_multiscene_comparison.png"
    cv2.imwrite(str(contact_path), contact_sheet(all_panels[:12], cols=1))
    overview_path = output_dir / "teed_ui_compact_overview.png"
    cv2.imwrite(str(overview_path), contact_sheet(overview_panels, cols=1))
    metrics_path = output_dir / "teed_ui_metrics.json"
    production_files = {
        name: sha256_file(REPO_ROOT / path)
        for name, path in {
            "hud": "edge/orin/hud.py",
            "edge_filter": "edge/orin/edge_filter.py",
            "thermal_overlay": "edge/orin/thermal_overlay.py",
        }.items()
    }
    payload = {
        "protocol": "firesight_teed_ui_replay_eval_v1",
        "generated_at_unix": time.time(),
        "repo": {
            "root": str(REPO_ROOT),
            "git_head": git_head,
            "git_status_short": git_status.splitlines(),
            "baseline_hud_source": f"git show {BASELINE_REVISION}:edge/orin/hud.py",
            "baseline_hud_sha256": sha256_text(
                (output_dir / "_baseline_hud_HEAD.py").read_text(encoding="utf-8")
            ),
        },
        "runtime": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "cpu_threads": 1,
        },
        "teed": core.describe(),
        "production_sources": production_files,
        "hud_config": {
            "output_size": list(DISPLAY_SIZE),
            "background_scale": filtered_renderer.config.background_scale,
            "edge_filter": dataclasses.asdict(filtered_renderer.config.edge_filter),
            "thermal": dataclasses.asdict(filtered_renderer.config.thermal),
            "filter_edges": filtered_renderer.config.filter_edges,
            "show_panel": filtered_renderer.config.show_panel,
            "full_res_background": filtered_renderer.config.full_res_background,
        },
        "inputs": inputs,
        "limitations": [
            (
                "No calibrated Celsius map was supplied to HUDRenderer; "
                "thermal status is expected to be temperature_unavailable."
            ),
            (
                "No metric depth map was supplied; near/far alpha behavior "
                "cannot be validated on these replays."
            ),
            (
                "SmokeBasement comparison uses legacy TEED edge overlap "
                "as a proxy, not obstacle avoidance recall."
            ),
            (
                "Fire360 clips are decoded display videos; camera model and "
                "radiometric output format are not established by these files."
            ),
        ],
        "aggregate": aggregate(all_records),
        "by_dataset": dataset_summaries,
        "frames": all_records,
        "artifacts": {
            "contact_png": rel(contact_path),
            "compact_overview_png": rel(overview_path),
            "result_dir": rel(output_dir),
        },
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"metrics": str(metrics_path), "contact": str(contact_path)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
