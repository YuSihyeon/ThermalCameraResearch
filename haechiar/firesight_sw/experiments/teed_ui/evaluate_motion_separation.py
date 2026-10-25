"""Paired thermal replay from identical cached TEED probabilities."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from contour_tracker import ContourConfig  # noqa: E402
from evaluate_contour_stability import (  # noqa: E402
    compact_components,
    compare_masks,
    summarize,
)
from hud import HUDConfig, HUDRenderer  # noqa: E402
from render_additional_thermal_videos import inspect_output  # noqa: E402
from teed_core import EdgeResult  # noqa: E402

OUT = Path(__file__).with_name("motion_split")


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def green_alpha(image: np.ndarray) -> np.ndarray:
    data = image.astype(np.float32)
    return np.clip((data[..., 1] - np.maximum(data[..., 0], data[..., 2])) / 255, 0, 1)


def alpha_churn(
    current: np.ndarray, previous: np.ndarray, flow: np.ndarray
) -> tuple[float, float]:
    h, w = current.shape
    scaled = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    scaled[..., 0] *= w / flow.shape[1]
    scaled[..., 1] *= h / flow.shape[0]
    yy, xx = np.indices((h, w), dtype=np.float32)
    old = cv2.remap(
        previous,
        xx + scaled[..., 0],
        yy + scaled[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return float(np.abs(current - old).sum()), float(current.sum() + old.sum())


def run(name: str, destination: Path, limit: int | None, render: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    paths = list((ROOT / "edge/orin").glob("*.py")) + [Path(__file__).resolve()]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    with np.load(
        Path(__file__).with_name("temporal_stability") / f"{name}_inputs.npz",
        allow_pickle=False,
    ) as data:
        probability, source, background = (
            data["probability"],
            data["source"],
            data["background"],
        )
        indices, seconds = data["indices"], data["seconds"]
        metadata = json.loads(str(data["metadata"]))
    count = min(limit or len(indices), len(indices))
    configs = [
        HUDConfig(
            output_size=(720, 405),
            temporal_edges=True,
            contour_edges=True,
            thermal_palette_overlay=True,
            contour_config=ContourConfig(motion_separation=value),
        )
        for value in (False, True)
    ]
    renderers = [HUDRenderer(config) for config in configs]
    writer = None
    video = destination / f"{name}_comparison.webm"
    if render:
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter.fourcc(*"VP80"), metadata["fps"], (1440, 446)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open {video}")
    previous_gray = None
    previous_masks: list[np.ndarray] = []
    previous_alpha: list[np.ndarray] = []
    records = []
    try:
        for i in range(count):
            p = probability[i]
            edge = EdgeResult(
                p,
                (p >= 0.75).astype(np.uint8) * 255,
                source[i],
                (p.shape[1], p.shape[0]),
                {},
            )
            frames = [
                renderer.render(
                    background[i],
                    edge,
                    ignore_regions=metadata["regions"],
                    frame_index=int(indices[i]),
                    timestamp_s=float(seconds[i]),
                )
                for renderer in renderers
            ]
            masks = [
                frame.edge_mask > 0 for frame in frames if frame.edge_mask is not None
            ]
            assert len(masks) == 2
            alphas = [green_alpha(frame.image) for frame in frames]
            for alpha in alphas:
                for left, top, right, bottom in metadata["regions"]:
                    alpha[
                        round(top * 405) : round(bottom * 405) + 1,
                        round(left * 720) : round(right * 720) + 1,
                    ] = 0
                alpha[-22:] = 0
            near = cv2.dilate(masks[1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            row = {
                "frame": int(indices[i]),
                "seconds": float(seconds[i]),
                "before_pixels": int(masks[0].sum()),
                "after_pixels": int(masks[1].sum()),
                "baseline_supported_pixels": int(np.count_nonzero(masks[0] & near)),
                "compact_before": compact_components(masks[0]),
                "compact_after": compact_components(masks[1]),
                "render_ms_before": frames[0].render_ms,
                "render_ms_after": frames[1].render_ms,
                "contour": frames[1].edge_metrics["contour"],
            }
            gray = cv2.cvtColor(source[i], cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    gray,
                    previous_gray,
                    np.empty((*gray.shape, 2), dtype=np.float32),
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                for index, prefix in enumerate(["before", "after"]):
                    row[prefix + "_unmatched"], row[prefix + "_total"] = compare_masks(
                        masks[index], previous_masks[index], flow
                    )
                    row[prefix + "_alpha_diff"], row[prefix + "_alpha_total"] = (
                        alpha_churn(alphas[index], previous_alpha[index], flow)
                    )
            records.append(row)
            previous_gray, previous_masks, previous_alpha = gray, masks, alphas
            if writer is not None:
                panel = np.zeros((446, 1440, 3), np.uint8)
                panel[40:445, :720] = frames[0].image
                panel[40:445, 720:] = frames[1].image
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
                writer.write(panel)
                if i in {0, count // 4, count // 2, 3 * count // 4, count - 1}:
                    cv2.imwrite(str(destination / f"{name}_qa_{i:04}.png"), panel)
            if (i + 1) % 30 == 0 or i + 1 == count:
                save(
                    destination / f"{name}_progress.json",
                    {"status": "RUNNING", "processed": i + 1, "expected": count},
                )
                print(name, i + 1, "/", count, flush=True)
    finally:
        if writer is not None:
            writer.release()
    summary = {}
    for prefix in ["before", "after"]:
        summary[prefix + "_churn"] = sum(
            row.get(prefix + "_unmatched", 0) for row in records
        ) / max(1, sum(row.get(prefix + "_total", 0) for row in records))
        summary[prefix + "_alpha_churn"] = sum(
            row.get(prefix + "_alpha_diff", 0) for row in records
        ) / max(1, sum(row.get(prefix + "_alpha_total", 0) for row in records))
        summary[prefix + "_compact_count"] = float(
            np.mean([row["compact_" + prefix] for row in records])
        )
        summary[prefix + "_compact_delta"] = (
            float(
                np.mean(np.abs(np.diff([row["compact_" + prefix] for row in records])))
            )
            if count > 1
            else 0
        )
        summary[prefix + "_hud_ms"] = summarize(
            [row["render_ms_" + prefix] for row in records]
        )
    summary["churn_reduction_percent"] = 100 * (
        1 - summary["after_churn"] / max(summary["before_churn"], 1e-9)
    )
    summary["alpha_churn_reduction_percent"] = 100 * (
        1 - summary["after_alpha_churn"] / max(summary["before_alpha_churn"], 1e-9)
    )
    summary["baseline_edge_support_2px"] = sum(
        row["baseline_supported_pixels"] for row in records
    ) / max(1, sum(row["before_pixels"] for row in records))
    summary["edge_pixel_ratio"] = sum(row["after_pixels"] for row in records) / max(
        1, sum(row["before_pixels"] for row in records)
    )
    for key in ["static_paths", "dynamic_paths", "unknown_paths", "static_held_paths"]:
        summary[key + "_mean"] = float(
            np.mean([row["contour"].get(key, 0) for row in records])
        )
    statuses = {}
    for row in records:
        status = str(row["contour"].get("global_motion_status", "unreported"))
        statuses[status] = statuses.get(status, 0) + 1
    summary["global_motion_status_counts"] = statuses
    qa = inspect_output(video, count, metadata["fps"], (1440, 446)) if render else {}
    assert not render or qa["status"] == "PASS", qa
    assert all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
        for p, sha in hashes.items()
    ), "Code changed while evaluating"
    result = {
        "status": "PASS",
        "name": name,
        "frames": count,
        "source": metadata,
        "code_sha256": hashes,
        "configs": [dataclasses.asdict(config) for config in configs],
        "summary": summary,
        "output": qa,
        "records": records,
        "limits": "Paired cached inference, not realtime. Motion-aligned churn/green-alpha difference are display stability proxies, not obstacle recall or labeled motion segmentation accuracy.",
    }
    save(destination / f"{name}.json", result)
    save(
        destination / f"{name}_progress.json",
        {"status": "COMPLETE", "processed": count, "expected": count},
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip", choices=["fire360_ifsi4", "fire360_ifsi8"], required=True
    )
    parser.add_argument("--output", default="probe_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run(args.clip, OUT / args.output, args.limit, args.render)


if __name__ == "__main__":
    main()
