#!/usr/bin/env python3
"""Paired original / dense-smoke edge evaluation.

The experiment already contains its own ground truth: the edges TEED finds in
the clean frame are the real structure. Running the same model on the smoked
frame and comparing against that reference turns "how much green appeared" into
"how much real structure survived, and how much of what we drew was invented".

    recall    = reference edges recovered in the smoked frame
    precision = drawn edges that correspond to a real edge
    density   = fraction of pixels drawn (the number the screenshots report)

Edge maps never line up pixel-perfectly, so both matches are computed against a
dilated counterpart. --tolerance sets that slack in pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

from replay import IMAGE_SUFFIXES
from teed_core import TEEDCore


def collect_pairs(original: Path, smoke: Path) -> list[tuple[Path, Path]]:
    """Match clean and smoked frames. Directories pair up by file stem."""
    if original.is_file() and smoke.is_file():
        return [(original, smoke)]
    if not (original.is_dir() and smoke.is_dir()):
        raise ValueError("--original and --smoke must both be files or both directories")
    smoked = {
        path.stem: path
        for path in smoke.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    pairs = []
    missing = []
    for path in sorted(original.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = smoked.get(path.stem)
        if match is None:
            missing.append(path.name)
        else:
            pairs.append((path, match))
    if missing:
        print(
            "warning: {} clean frames had no smoked counterpart (first: {})".format(
                len(missing), missing[0]
            ),
            file=sys.stderr,
        )
    if not pairs:
        raise ValueError("no matching original/smoke pairs were found")
    return pairs


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    return image


def dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance < 1:
        return mask
    size = tolerance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def compare(reference: np.ndarray, detected: np.ndarray, tolerance: int) -> dict:
    reference_on = reference > 0
    detected_on = detected > 0
    total = reference.size
    reference_count = int(reference_on.sum())
    detected_count = int(detected_on.sum())

    recall_hits = int((detected_on & (dilate(reference, tolerance) > 0)).sum())
    precision_hits = int((reference_on & (dilate(detected, tolerance) > 0)).sum())

    recall = precision_hits / reference_count if reference_count else 0.0
    precision = recall_hits / detected_count if detected_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "reference_density": reference_count / total,
        "detected_density": detected_count / total,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def panel(
    original: np.ndarray,
    smoked: np.ndarray,
    reference_mask: np.ndarray,
    detected_mask: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    def edge_view(mask: np.ndarray) -> np.ndarray:
        view = np.zeros((height, width, 3), dtype=np.uint8)
        view[mask > 0] = (0, 255, 0)
        return view
    cells = [
        cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA),
        cv2.resize(smoked, (width, height), interpolation=cv2.INTER_AREA),
        edge_view(reference_mask),
        edge_view(detected_mask),
    ]
    labels = ("Original", "Dense smoke", "Reference edges", "Detected edges")
    for cell, label in zip(cells, labels):
        cv2.putText(
            cell, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return cv2.hconcat(cells)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="clean frame or directory")
    parser.add_argument("--smoke", required=True, help="smoked frame or directory")
    here = Path(__file__).resolve().parent
    shared = here.parent / "teed" / "models" / "5_model.pth"
    default_ckpt = shared if shared.is_file() else here / "models" / "teed" / "5_model.pth"
    parser.add_argument("--checkpoint", default=str(default_ckpt))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--reference-threshold", type=float, default=0.75,
        help="threshold used on the clean frame to define real structure",
    )
    parser.add_argument(
        "--threshold", type=float, action="append",
        help="threshold to evaluate on the smoked frame; repeatable",
    )
    parser.add_argument(
        "--edge-width", type=int, action="append",
        help="edge width to evaluate on the smoked frame; repeatable",
    )
    parser.add_argument("--tolerance", type=int, default=2, help="match slack in pixels")
    parser.add_argument("--panel-dir", help="write a 4-up comparison PNG per pair")
    parser.add_argument("--metrics", default="smoke_eval.json")
    return parser


def run(args: argparse.Namespace) -> dict:
    thresholds = args.threshold or [0.75, 0.55, 0.40]
    edge_widths = args.edge_width or [1, 3]
    pairs = collect_pairs(Path(args.original), Path(args.smoke))
    print(f"pairs: {len(pairs)}")

    reference_core = TEEDCore(
        checkpoint=Path(args.checkpoint), width=args.width, height=args.height,
        threshold=args.reference_threshold, edge_width=1,
        device=args.device, half=args.half,
    )
    # One core per setting keeps the checkpoint load out of the inner loop.
    settings = {
        (threshold, edge_width): TEEDCore(
            checkpoint=Path(args.checkpoint), width=args.width, height=args.height,
            threshold=threshold, edge_width=edge_width,
            device=args.device, half=args.half,
        )
        for threshold in thresholds
        for edge_width in edge_widths
    }

    panel_dir = Path(args.panel_dir) if args.panel_dir else None
    if panel_dir:
        panel_dir.mkdir(parents=True, exist_ok=True)

    per_setting: dict[str, list[dict]] = {key: [] for key in map(str, settings)}
    per_pair = []
    for original_path, smoke_path in pairs:
        original = read_bgr(original_path)
        smoked = read_bgr(smoke_path)
        reference = reference_core.infer(original)
        entry = {"original": original_path.name, "smoke": smoke_path.name, "settings": {}}
        for (threshold, edge_width), core in settings.items():
            detected = core.infer(smoked)
            scores = compare(reference.mask, detected.mask, args.tolerance)
            label = f"threshold={threshold:.2f},edge_width={edge_width}"
            entry["settings"][label] = scores
            per_setting[str((threshold, edge_width))].append(scores)
            if panel_dir and (threshold, edge_width) == (thresholds[0], edge_widths[-1]):
                cv2.imwrite(
                    str(panel_dir / f"{original_path.stem}.png"),
                    panel(original, smoked, reference.mask, detected.mask,
                          (args.width, args.height)),
                )
        per_pair.append(entry)

    summary = {}
    for (threshold, edge_width) in settings:
        scores = per_setting[str((threshold, edge_width))]
        label = f"threshold={threshold:.2f},edge_width={edge_width}"
        summary[label] = {
            key: float(np.mean([score[key] for score in scores]))
            for key in ("reference_density", "detected_density", "recall", "precision", "f1")
        }

    payload = {
        "protocol": "firesight_smoke_eval_v1",
        "pairs": len(pairs),
        "tolerance_px": args.tolerance,
        "reference_threshold": args.reference_threshold,
        "core": reference_core.describe(),
        "summary": summary,
        "per_pair": per_pair,
    }
    if args.metrics:
        Path(args.metrics).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print("\n{:<34} {:>8} {:>10} {:>10} {:>8}".format(
        "setting", "density", "recall", "precision", "f1"))
    for label, scores in sorted(summary.items()):
        print("{:<34} {:>8.3f} {:>10.3f} {:>10.3f} {:>8.3f}".format(
            label, scores["detected_density"], scores["recall"],
            scores["precision"], scores["f1"]))
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    try:
        run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
