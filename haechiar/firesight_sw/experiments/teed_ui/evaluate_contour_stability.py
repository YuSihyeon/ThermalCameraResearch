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
from contour_tracker import ContourConfig  # noqa: E402
from hud import HUDConfig, HUDRenderer  # noqa: E402
from teed_core import EdgeResult  # noqa: E402
from temporal_edges import TemporalEdgeConfig  # noqa: E402

DATA = Path(__file__).with_name("temporal_stability")


def load_previous_temporal_pixels(name: str, count: int) -> list[int]:
    path = DATA / "final" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    if len(records) < count:
        raise ValueError(
            f"previous temporal baseline has only {len(records)} records: {path}"
        )
    return [int(record["after_pixels"]) for record in records[:count]]


def compare_masks(
    current: np.ndarray, previous: np.ndarray, flow: np.ndarray
) -> tuple[int, int]:
    height, width = current.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
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


def compact_components(mask: np.ndarray, max_extent: float = 20.0) -> int:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    compact = 0
    for label in range(1, count):
        width = int(stats[label, cv2.CC_STAT_WIDTH]) - 1
        height = int(stats[label, cv2.CC_STAT_HEIGHT]) - 1
        if np.hypot(width, height) <= max_extent:
            compact += 1
    return compact


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def header(name: str) -> np.ndarray:
    canvas = Image.new("RGB", (1440, 540), "#101820")
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    draw.text(
        (22, 8), name + " | Fire360 contour 안정화 비교", font=small, fill="#d7e2ee"
    )
    draw.text((22, 43), "이전 · temporal 확률 안정화", font=large, fill="white")
    draw.text((742, 43), "개선 · contour 추적 + AA 선", font=large, fill="#8bf0a9")
    draw.text(
        (22, 500),
        "동일 TEED 캐시 / 동일 원본 재생 속도 / 실제 HUD 출력 mask 기준",
        font=small,
        fill="#d7e2ee",
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def detail_frame(
    name: str, left: np.ndarray, right: np.ndarray, frame: int
) -> np.ndarray:
    canvas = Image.new("RGB", (960, 720), "#101820")
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18)
    draw.text((22, 14), name + " | 동일 구간 확대", font=large, fill="#d7e2ee")
    draw.text((22, 48), "왼쪽: 이전 / 오른쪽: contour 개선", font=small, fill="#d7e2ee")
    right_hot_area = (450, 95, 720, 360)
    x0, y0, x1, y1 = right_hot_area
    left_crop = left[y0:y1, x0:x1]
    right_crop = right[y0:y1, x0:x1]
    left_zoom = cv2.resize(left_crop, (430, 430), interpolation=cv2.INTER_NEAREST)
    right_zoom = cv2.resize(right_crop, (430, 430), interpolation=cv2.INTER_NEAREST)
    arr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    arr[92:522, 40:470] = left_zoom
    arr[92:522, 490:920] = right_zoom
    cv2.rectangle(arr, (40, 92), (470, 522), (90, 90, 90), 1)
    cv2.rectangle(arr, (490, 92), (920, 522), (90, 90, 90), 1)
    cv2.putText(
        arr,
        f"frame {frame} | actual encoded/composited panel crop",
        (40, 565),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return arr


def evaluate(name: str, output: Path, limit: int | None, render: bool) -> None:
    code_sha256 = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in [
            "edge/orin/contour_tracker.py",
            "edge/orin/contour_render.py",
            "edge/orin/hud.py",
            "experiments/teed_ui/evaluate_contour_stability.py",
        ]
    }
    with np.load(DATA / f"{name}_inputs.npz", allow_pickle=False) as z:
        probabilities = z["probability"]
        sources = z["source"]
        backgrounds = z["background"]
        indices = z["indices"]
        seconds = z["seconds"]
        metadata = json.loads(str(z["metadata"]))
    count = min(limit or len(indices), len(indices))
    expected_baseline_pixels = load_previous_temporal_pixels(name, count)
    temporal = TemporalEdgeConfig(history_weight=0.6)
    contour = ContourConfig(
        high_threshold=0.75,
        min_output_length=HUDConfig().edge_filter.min_length,
        min_output_extent=HUDConfig().edge_filter.min_extent,
    )
    baseline = HUDRenderer(
        HUDConfig(output_size=(720, 405), temporal_edges=True, temporal_config=temporal)
    )
    improved = HUDRenderer(
        HUDConfig(
            output_size=(720, 405),
            temporal_edges=True,
            temporal_config=temporal,
            contour_edges=True,
            contour_config=contour,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    writers = []
    detail = None
    if render:
        for ext, codec in [("webm", "VP80"), ("mp4", "mp4v")]:
            writer = cv2.VideoWriter(
                str(output / f"{name}.{ext}"),
                cv2.VideoWriter.fourcc(*codec),
                float(metadata["fps"]),
                (1440, 540),
            )
            assert writer.isOpened(), ext
            writers.append(writer)
        single = cv2.VideoWriter(
            str(output / f"{name}_contour.webm"),
            cv2.VideoWriter.fourcc(*"VP80"),
            float(metadata["fps"]),
            (720, 406),
        )
        assert single.isOpened()
        if name == "fire360_ifsi8":
            detail = cv2.VideoWriter(
                str(output / f"{name}_detail.webm"),
                cv2.VideoWriter.fourcc(*"VP80"),
                float(metadata["fps"]),
                (960, 720),
            )
            assert detail.isOpened()
    rows = []
    base_panel = header(name)
    old_gray = old_before = old_after = None
    for i in range(count):
        probability = probabilities[i]
        source = sources[i]
        edge = EdgeResult(
            probability,
            (probability >= 0.75).astype(np.uint8) * 255,
            source,
            (probability.shape[1], probability.shape[0]),
            {},
        )
        frame_index = int(indices[i])
        timestamp_s = float(seconds[i])
        regions = metadata["regions"]
        left = baseline.render(
            backgrounds[i],
            edge,
            ignore_regions=regions,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        right = improved.render(
            backgrounds[i],
            edge,
            ignore_regions=regions,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        assert left.edge_mask is not None
        assert right.edge_mask is not None
        before = left.edge_mask > 0
        after = right.edge_mask > 0
        near_after = cv2.dilate(after.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        row = {
            "frame": int(indices[i]),
            "seconds": float(seconds[i]),
            "before_pixels": int(np.count_nonzero(before)),
            "after_pixels": int(np.count_nonzero(after)),
            "baseline_supported_pixels": int(np.count_nonzero(before & near_after)),
            "compact_before": compact_components(before),
            "compact_after": compact_components(after),
            "render_ms_before": float(left.render_ms),
            "render_ms_after": float(right.render_ms),
            "contour": right.edge_metrics.get("contour", {}),
        }
        expected_before_pixels = expected_baseline_pixels[i]
        if row["before_pixels"] != expected_before_pixels:
            raise AssertionError(
                f"{name} frame {i} baseline parity failed: "
                f"{row['before_pixels']} != {expected_before_pixels}"
            )
        row["previous_temporal_after_pixels"] = expected_before_pixels
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        if old_gray is not None:
            assert old_before is not None
            assert old_after is not None
            flow_seed = np.empty((*gray.shape, 2), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                gray, old_gray, flow_seed, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            row["before_unmatched"], row["before_total"] = compare_masks(
                before, old_before, flow
            )
            row["after_unmatched"], row["after_total"] = compare_masks(
                after, old_after, flow
            )
            row["compact_before_delta"] = abs(
                row["compact_before"] - rows[-1]["compact_before"]
            )
            row["compact_after_delta"] = abs(
                row["compact_after"] - rows[-1]["compact_after"]
            )
        rows.append(row)
        if render:
            panel = base_panel.copy()
            panel[84:489, :720] = left.image
            panel[84:489, 720:] = right.image
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
            single.write(
                cv2.copyMakeBorder(right.image, 0, 1, 0, 0, cv2.BORDER_CONSTANT)
            )
            if detail is not None:
                detail.write(
                    detail_frame(name, left.image, right.image, int(indices[i]))
                )
            if i in {0, count // 3, count // 2, count - 1}:
                cv2.imwrite(str(output / f"{name}_qa_{i:04}.png"), panel)
        old_gray, old_before, old_after = gray, before, after
        if i % 60 == 0:
            print(name, i + 1, "/", count, flush=True)
    for writer in writers:
        writer.release()
    if render:
        single.release()
    if detail is not None:
        detail.release()
    before_total = sum(int(row.get("before_total", 0)) for row in rows)
    after_total = sum(int(row.get("after_total", 0)) for row in rows)
    before_rate = sum(int(row.get("before_unmatched", 0)) for row in rows) / max(
        1, before_total
    )
    after_rate = sum(int(row.get("after_unmatched", 0)) for row in rows) / max(
        1, after_total
    )
    result = {
        "name": name,
        "frames": count,
        "fps": float(metadata["fps"]),
        "seconds": count / float(metadata["fps"]),
        "source": metadata,
        "config": {
            "temporal": dataclasses.asdict(temporal),
            "contour": dataclasses.asdict(contour),
        },
        "motion_aligned_churn_before": before_rate,
        "motion_aligned_churn_after": after_rate,
        "churn_reduction_percent": 100.0
        * (before_rate - after_rate)
        / max(before_rate, 1e-9),
        "baseline_edge_support_2px": sum(
            row["baseline_supported_pixels"] for row in rows
        )
        / max(1, sum(row["before_pixels"] for row in rows)),
        "edge_pixel_ratio": sum(row["after_pixels"] for row in rows)
        / max(1, sum(row["before_pixels"] for row in rows)),
        "compact_components_before_mean": float(
            np.mean([row["compact_before"] for row in rows])
        ),
        "compact_components_after_mean": float(
            np.mean([row["compact_after"] for row in rows])
        ),
        "compact_component_churn_before": float(
            np.mean([row.get("compact_before_delta", 0) for row in rows[1:]])
        )
        if len(rows) > 1
        else 0.0,
        "compact_component_churn_after": float(
            np.mean([row.get("compact_after_delta", 0) for row in rows[1:]])
        )
        if len(rows) > 1
        else 0.0,
        "render_ms_before": summarize([row["render_ms_before"] for row in rows]),
        "render_ms_after": summarize([row["render_ms_after"] for row in rows]),
        "baseline_parity": {
            "status": "PASS",
            "checked_frames": count,
            "source": str(DATA / "final" / f"{name}.json"),
        },
        "limits": [
            "Motion-aligned unmatched edge rate is a stability proxy, not obstacle accuracy.",
            "Compact components are small connected skeleton components at model resolution.",
            "Videos preserve source time; render_ms excludes TEED inference because cached probabilities are used.",
        ],
        "code_sha256": code_sha256,
        "records": rows,
    }
    (output / f"{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"records", "source", "code_sha256"}
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        nargs="+",
        default=["fire360_ifsi4", "fire360_ifsi8", "fire360_visible"],
    )
    parser.add_argument("--output", default="video_iteration1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    for name in args.name:
        evaluate(
            name,
            Path(__file__).with_name("contour_stability") / args.output,
            args.limit,
            args.render,
        )


if __name__ == "__main__":
    main()
