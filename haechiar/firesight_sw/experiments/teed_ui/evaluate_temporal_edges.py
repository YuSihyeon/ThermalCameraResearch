from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from edge_filter import EdgeFilterConfig, select_edges  # noqa: E402
from hud import HUDConfig, HUDRenderer  # noqa: E402
from teed_core import EdgeResult  # noqa: E402
from temporal_edges import TemporalEdgeConfig, TemporalEdgeStabilizer  # noqa: E402

DATA = Path(__file__).with_name("temporal_stability")


def compare_masks(current, previous, flow):
    h, w = current.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    old = (
        cv2.remap(
            previous.astype(np.uint8),
            xx + flow[..., 0],
            yy + flow[..., 1],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        > 0
    )
    kernel = np.ones((3, 3), np.uint8)
    old_near = cv2.dilate(old.astype(np.uint8), kernel) > 0
    new_near = cv2.dilate(current.astype(np.uint8), kernel) > 0
    unmatched = int(
        np.count_nonzero(current & ~old_near) + np.count_nonzero(old & ~new_near)
    )
    total = int(np.count_nonzero(current) + np.count_nonzero(old))
    return unmatched, total


def header(name):
    canvas = Image.new("RGB", (1440, 540), "#101820")
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    draw.text((22, 8), name + " | Fire360 시간 안정화 비교", font=small, fill="#d7e2ee")
    draw.text((22, 43), "기존 · 프레임별 공간 필터", font=large, fill="white")
    draw.text((742, 43), "개선 · 움직임 보정 + 시간 안정화", font=large, fill="#8bf0a9")
    draw.text(
        (22, 500),
        "동일 TEED / 동일 원본 재생 속도 / 강한 신규 경계 즉시 반영",
        font=small,
        fill="#d7e2ee",
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def evaluate(name, output, weight, render, limit):
    with np.load(DATA / (name + "_inputs.npz"), allow_pickle=False) as z:
        probabilities, sources = z["probability"], z["source"]
        backgrounds, indices, seconds = z["background"], z["indices"], z["seconds"]
        metadata = json.loads(str(z["metadata"]))
    count = min(limit or len(indices), len(indices))
    config = TemporalEdgeConfig(history_weight=weight)
    stabilizer = TemporalEdgeStabilizer(config)
    spatial = EdgeFilterConfig()
    renderer = HUDRenderer(HUDConfig(output_size=(720, 405), temporal_edges=False))
    output.mkdir(parents=True, exist_ok=True)
    writers = []
    base = header(name)
    if render:
        for ext, codec in [("webm", "VP80"), ("mp4", "mp4v")]:
            writer = cv2.VideoWriter(
                str(output / (name + "." + ext)),
                cv2.VideoWriter.fourcc(*codec),
                metadata["fps"],
                (1440, 540),
            )
            assert writer.isOpened(), ext
            writers.append(writer)
        single = cv2.VideoWriter(
            str(output / (name + "_stabilized.webm")),
            cv2.VideoWriter.fourcc(*"VP80"),
            metadata["fps"],
            (720, 406),
        )
        assert single.isOpened()
    rows = []
    old_before = old_after = old_gray = None
    for i in range(count):
        p, source = probabilities[i], sources[i]
        stable, temporal = stabilizer.update(
            p,
            source,
            metadata["regions"],
            frame_index=int(indices[i]),
            timestamp_s=float(seconds[i]),
        )
        before = select_edges(p, source, spatial, metadata["regions"])
        after = select_edges(stable, source, spatial, metadata["regions"])
        b, a = before.mask > 0, after.mask > 0
        near = cv2.dilate(a.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        row = {
            "frame": int(indices[i]),
            "seconds": float(seconds[i]),
            "before_pixels": int(b.sum()),
            "after_pixels": int(a.sum()),
            "baseline_supported_pixels": int((b & near).sum()),
            "temporal": temporal,
        }
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        if old_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                gray, old_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            row["before_unmatched"], row["before_total"] = compare_masks(
                b, old_before, flow
            )
            row["after_unmatched"], row["after_total"] = compare_masks(
                a, old_after, flow
            )
        rows.append(row)
        if render:
            edge = EdgeResult(p, before.mask, source, (p.shape[1], p.shape[0]), {})
            left = renderer.render(
                backgrounds[i], edge, ignore_regions=metadata["regions"]
            ).image
            right = renderer.render(
                backgrounds[i],
                dataclasses.replace(edge, probability=stable),
                ignore_regions=metadata["regions"],
            ).image
            panel = base.copy()
            panel[84:489, :720] = left
            panel[84:489, 720:] = right
            cv2.line(panel, (720, 84), (720, 489), (90, 90, 90), 1)
            cv2.putText(
                panel,
                f"frame {indices[i]} | t={seconds[i]:.2f}s",
                (1100, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            for writer in writers:
                writer.write(panel)
            single.write(cv2.copyMakeBorder(right, 0, 1, 0, 0, cv2.BORDER_CONSTANT))
            if i in {0, count // 3, count // 2, count - 1}:
                cv2.imwrite(str(output / f"{name}_qa_{i:04}.png"), panel)
        old_gray, old_before, old_after = gray, b, a
        if i % 60 == 0:
            print(name, i + 1, "/", count, flush=True)
    for writer in writers:
        writer.release()
    if render:
        single.release()
    before_rate = sum(r.get("before_unmatched", 0) for r in rows) / max(
        1, sum(r.get("before_total", 0) for r in rows)
    )
    after_rate = sum(r.get("after_unmatched", 0) for r in rows) / max(
        1, sum(r.get("after_total", 0) for r in rows)
    )
    result = {
        "name": name,
        "frames": count,
        "fps": metadata["fps"],
        "seconds": count / metadata["fps"],
        "source": metadata,
        "config": dataclasses.asdict(config),
        "motion_aligned_churn_before": before_rate,
        "motion_aligned_churn_after": after_rate,
        "churn_reduction_percent": 100
        * (before_rate - after_rate)
        / max(before_rate, 1e-9),
        "baseline_edge_support_2px": sum(r["baseline_supported_pixels"] for r in rows)
        / max(1, sum(r["before_pixels"] for r in rows)),
        "edge_pixel_ratio": sum(r["after_pixels"] for r in rows)
        / max(1, sum(r["before_pixels"] for r in rows)),
        "limits": [
            "Motion-aligned unmatched edge rate is a stability proxy, not ground-truth obstacle accuracy.",
            "2px baseline support is a model-relative retention proxy.",
            "Videos preserve scene time; encoding FPS is not inference FPS.",
        ],
        "code_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in [
                "edge/orin/temporal_edges.py",
                "edge/orin/edge_filter.py",
                "edge/orin/hud.py",
                "experiments/teed_ui/evaluate_temporal_edges.py",
            ]
        },
        "records": rows,
    }
    (output / (name + ".json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"records", "code_sha256", "source"}
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        nargs="+",
        default=["fire360_ifsi4", "fire360_ifsi8", "fire360_visible"],
    )
    parser.add_argument("--output", default="iteration1")
    parser.add_argument("--history-weight", type=float, default=0.6)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    for name in args.name:
        evaluate(name, DATA / args.output, args.history_weight, args.render, args.limit)


if __name__ == "__main__":
    main()
