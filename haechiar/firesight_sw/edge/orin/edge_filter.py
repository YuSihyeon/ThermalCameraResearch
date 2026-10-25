from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeFilterConfig:
    low_threshold: float = 0.55
    high_threshold: float = 0.75
    min_length: float = 12.0
    gradient_low: float = 2.0
    gradient_high: float = 6.0
    near_m: float = 0.6
    far_m: float = 6.0
    far_alpha: float = 0.25
    min_branch_length: float = 6.0
    min_extent: float = 8.0

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True)
class EdgeSelection:
    mask: np.ndarray
    alpha: np.ndarray
    metrics: dict[str, float | int]


def select_edges(
    probability: np.ndarray,
    frame_bgr: np.ndarray,
    config: EdgeFilterConfig,
    ignore_regions: Sequence[tuple[float, float, float, float]] = (),
    depth_m: np.ndarray | None = None,
) -> EdgeSelection:
    probability = _validate_probability(probability)
    frame_bgr = _validate_frame(frame_bgr, probability.shape)
    _validate_config(config)
    depth = _validate_depth(depth_m, probability.shape)

    active_region = _active_region(probability.shape, ignore_regions)
    candidates_before_osd = probability >= float(config.low_threshold)
    weak, strong, gradient_gate_count = _hysteresis_candidates(
        probability, frame_bgr, config
    )
    weak &= active_region
    strong &= active_region

    hysteresis_mask = _connected_hysteresis(weak, strong)
    centerlines = _zhang_suen_thinning(hysteresis_mask)
    centerlines &= active_region
    pruned, pruned_branches, pruned_pixels = _prune_short_branches(
        centerlines, config.min_branch_length
    )
    filtered, dropped_short = _drop_short_components(pruned, config.min_length)
    filtered, dropped_compact = _drop_compact_components(filtered, config.min_extent)
    filtered &= active_region

    mask = filtered.astype(np.uint8) * 255
    alpha = np.zeros(probability.shape, dtype=np.float32)
    alpha[filtered] = 1.0
    valid_depth_edge_count = 0
    if depth is not None and np.any(filtered):
        valid_depth = filtered & np.isfinite(depth) & (depth > 0.0)
        valid_depth_edge_count = int(np.count_nonzero(valid_depth))
        if valid_depth_edge_count:
            alpha[valid_depth] = _depth_alpha(depth[valid_depth], config)

    metrics: dict[str, float | int] = {
        "input_edge_pixels": int(np.count_nonzero(candidates_before_osd)),
        "ignored_region_pixels": int(
            np.count_nonzero(candidates_before_osd & ~active_region)
        ),
        "weak_edge_pixels": int(np.count_nonzero(weak)),
        "strong_edge_pixels": int(np.count_nonzero(strong)),
        "hysteresis_edge_pixels": int(np.count_nonzero(hysteresis_mask)),
        "centerline_edge_pixels": int(np.count_nonzero(centerlines)),
        "output_edge_pixels": int(np.count_nonzero(filtered)),
        "dropped_short_components": int(dropped_short),
        "pruned_branches": int(pruned_branches),
        "pruned_branch_pixels": int(pruned_pixels),
        "dropped_compact_components": int(dropped_compact),
        "gradient_gate_count": int(gradient_gate_count),
        "valid_depth_edge_count": valid_depth_edge_count,
    }
    return EdgeSelection(mask=mask, alpha=alpha, metrics=metrics)


def _validate_probability(probability: np.ndarray) -> np.ndarray:
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("probability must be a non-empty HxW array")
    if not np.isfinite(array).all():
        raise ValueError("probability must contain only finite values")
    return np.clip(array, 0.0, 1.0)


def _validate_frame(frame_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(frame_bgr)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")
    if array.shape[:2] != shape:
        raise ValueError("frame_bgr and probability must share the same HxW resolution")
    if not np.isfinite(array).all():
        raise ValueError("frame_bgr must contain only finite values")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _validate_config(config: EdgeFilterConfig) -> None:
    values = (
        config.low_threshold,
        config.high_threshold,
        config.min_length,
        config.min_branch_length,
        config.min_extent,
        config.gradient_low,
        config.gradient_high,
        config.near_m,
        config.far_m,
        config.far_alpha,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("edge filter config values must be finite")
    if not 0.0 <= config.low_threshold <= config.high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
    if min(config.min_length, config.min_branch_length, config.min_extent) < 0.0:
        raise ValueError("length and extent thresholds must be non-negative")
    if not 0.0 <= config.gradient_low <= config.gradient_high:
        raise ValueError("gradient thresholds must satisfy 0 <= low <= high")
    if config.near_m <= 0.0 or config.far_m <= config.near_m:
        raise ValueError("depth fade must satisfy 0 < near_m < far_m")
    if not 0.0 <= config.far_alpha <= 1.0:
        raise ValueError("far_alpha must be within [0, 1]")


def _validate_depth(
    depth_m: np.ndarray | None, shape: tuple[int, int]
) -> np.ndarray | None:
    if depth_m is None:
        return None
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != shape:
        raise ValueError("depth_m must have the same HxW resolution as probability")
    return depth


def _active_region(
    shape: tuple[int, int],
    regions: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    height, width = shape
    active = np.ones(shape, dtype=bool)
    for region in regions:
        if len(region) != 4:
            raise ValueError(
                "ignore_regions entries must be (left, top, right, bottom)"
            )
        left, top, right, bottom = (float(value) for value in region)
        if not all(np.isfinite(value) for value in (left, top, right, bottom)):
            raise ValueError("ignore_regions entries must be finite")
        if not (0 <= left <= right <= 1 and 0 <= top <= bottom <= 1):
            raise ValueError("ignore_regions must be normalized ordered rectangles")
        x0 = int(round(left * width))
        x1 = int(round(np.clip(right, 0.0, 1.0) * width))
        y0 = int(round(np.clip(top, 0.0, 1.0) * height))
        y1 = int(round(np.clip(bottom, 0.0, 1.0) * height))
        active[y0 : y1 + 1, x0 : x1 + 1] = False
    return active


def _hysteresis_candidates(
    probability: np.ndarray, frame_bgr: np.ndarray, config: EdgeFilterConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    prob_weak = probability >= float(config.low_threshold)
    prob_strong = probability >= float(config.high_threshold)
    if config.gradient_high == 0.0:
        return prob_weak, prob_strong, 0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    gradient = cv2.magnitude(dx, dy)

    grad_weak = gradient >= float(config.gradient_low)
    grad_strong = gradient >= float(config.gradient_high)

    weak = prob_weak & (grad_weak | prob_strong)
    strong = prob_strong | (prob_weak & grad_strong)
    gradient_gate_count = int(np.count_nonzero(prob_weak & ~weak))
    return weak, strong, gradient_gate_count


def _connected_hysteresis(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    if not np.any(weak) or not np.any(strong):
        return np.zeros(weak.shape, dtype=bool)
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(weak.shape, dtype=bool)
    strong_labels = np.unique(labels[strong])
    keep = strong_labels[strong_labels > 0]
    if keep.size == 0:
        return np.zeros(weak.shape, dtype=bool)
    return np.isin(labels, keep)


def _zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    image = mask.astype(np.uint8).copy()
    if not np.any(image):
        return image.astype(bool)
    changing = True
    while changing:
        changing = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                removable = (
                    (image == 1)
                    & (neighbours >= 2)
                    & (neighbours <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
            else:
                removable = (
                    (image == 1)
                    & (neighbours >= 2)
                    & (neighbours <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )
            if np.any(removable):
                image[removable] = 0
                changing = True
    return image.astype(bool)


def _prune_short_branches(
    mask: np.ndarray, min_length: float
) -> tuple[np.ndarray, int, int]:
    if min_length <= 0.0 or not np.any(mask):
        return mask.astype(bool), 0, 0
    height, width = mask.shape
    nodes = set(int(value) for value in np.flatnonzero(mask))
    graph: dict[int, list[tuple[int, float]]] = {}
    for node in nodes:
        y, x = divmod(node, width)
        adjacent: list[tuple[int, float]] = []
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            neighbour = ny * width + nx
            if neighbour not in nodes:
                continue
            if dy and dx and (y * width + nx in nodes or ny * width + x in nodes):
                continue
            adjacent.append((neighbour, 2.0**0.5 if dy and dx else 1.0))
        graph[node] = adjacent
    removed: set[int] = set()
    branch_count = 0
    for endpoint in nodes:
        if len(graph[endpoint]) != 1:
            continue
        path = [endpoint]
        previous = -1
        current = endpoint
        length = 0.0
        while True:
            choices = [(node, step) for node, step in graph[current] if node != previous]
            if len(choices) != 1:
                break
            neighbour, step = choices[0]
            previous, current = current, neighbour
            length += step
            if length >= min_length:
                break
            if len(graph[current]) != 2:
                if len(graph[current]) >= 3:
                    removed.update(path)
                    branch_count += 1
                break
            path.append(current)
    result = mask.astype(bool).copy()
    if removed:
        result.ravel()[list(removed)] = False
    return result, branch_count, len(removed)


def _drop_compact_components(
    mask: np.ndarray, min_extent: float
) -> tuple[np.ndarray, int]:
    if min_extent <= 0.0 or not np.any(mask):
        return mask.astype(bool), 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    allowed = np.zeros(count, dtype=bool)
    dropped = 0
    for label in range(1, count):
        width = int(stats[label, cv2.CC_STAT_WIDTH]) - 1
        height = int(stats[label, cv2.CC_STAT_HEIGHT]) - 1
        if np.hypot(width, height) >= min_extent:
            allowed[label] = True
        else:
            dropped += 1
    return allowed[np.asarray(labels, dtype=np.int32)], dropped


def _drop_short_components(
    mask: np.ndarray, min_length: float
) -> tuple[np.ndarray, int]:
    if min_length <= 0.0 or not np.any(mask):
        return mask.astype(bool), 0
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, dtype=bool)
    dropped = 0
    for label in range(1, count):
        component = labels == label
        length = _component_chain_length(component)
        if length >= min_length:
            keep |= component
        else:
            dropped += 1
    return keep, dropped


def _component_chain_length(component: np.ndarray) -> float:
    pixels = int(np.count_nonzero(component))
    if pixels <= 1:
        return float(pixels)
    horizontal = int(np.count_nonzero(component[:, :-1] & component[:, 1:]))
    vertical = int(np.count_nonzero(component[:-1, :] & component[1:, :]))
    diag_down = int(
        np.count_nonzero(
            component[:-1, :-1]
            & component[1:, 1:]
            & ~(component[:-1, 1:] | component[1:, :-1])
        )
    )
    diag_up = int(
        np.count_nonzero(
            component[1:, :-1]
            & component[:-1, 1:]
            & ~(component[:-1, :-1] | component[1:, 1:])
        )
    )
    graph_length = horizontal + vertical + (2.0**0.5) * (diag_down + diag_up)
    return float(max(1.0, graph_length))


def _depth_alpha(depth: np.ndarray, config: EdgeFilterConfig) -> np.ndarray:
    inverse = 1.0 / np.clip(depth.astype(np.float32), config.near_m, config.far_m)
    weight = (inverse - 1.0 / config.far_m) / (1.0 / config.near_m - 1.0 / config.far_m)
    return (config.far_alpha + weight * (1.0 - config.far_alpha)).astype(np.float32)
