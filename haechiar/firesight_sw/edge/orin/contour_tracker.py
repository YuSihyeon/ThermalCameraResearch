from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from edge_filter import _active_region, _drop_compact_components, _drop_short_components
from motion_registration import _estimate_global_motion, _transform_points


@dataclass(frozen=True)
class ContourConfig:
    high_threshold: float = 0.75
    support_threshold: float = 0.25
    weak_short_length: float = 18.0
    min_output_length: float = 0.0
    min_output_extent: float = 0.0
    confirmation_frames: int = 3
    confirmation_window: int = 4
    max_missing_frames: int = 0
    match_distance_px: float = 4.0
    slow_motion_px: float = 1.5
    max_smooth_alpha: float = 0.70
    min_smooth_alpha: float = 0.12
    max_gap_seconds: float = 0.25
    flow_fb_error_px: float = 1.5
    photometric_error: float = 35.0
    min_registration_ratio: float = 0.2
    duplicate_distance_px: float = 1.25
    motion_separation: bool = False
    motion_static_residual_px: float = 0.80
    motion_dynamic_residual_px: float = 2.25
    motion_static_confirm_frames: int = 2
    motion_global_min_samples: int = 24
    motion_global_min_coverage: float = 0.35
    motion_global_min_inlier_ratio: float = 0.55
    motion_global_max_error_px: float = 1.25
    motion_static_max_smooth_alpha: float = 0.86
    motion_static_max_delta_px: float = 1.75

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True)
class ContourPath:
    points: np.ndarray
    alpha: float
    track_id: int
    closed: bool = False


@dataclass(frozen=True)
class ContourResult:
    mask: np.ndarray
    paths: list[ContourPath]
    metrics: dict[str, float | int | str]


@dataclass
class _Track:
    track_id: int
    points: np.ndarray
    closed: bool
    alpha: float
    recent: deque[bool] = field(default_factory=deque)
    missing: int = 0
    displayed: bool = False
    isolated: bool = True
    motion_state: str = "unknown"
    static_evidence: int = 0


@dataclass
class _Candidate:
    points: np.ndarray
    closed: bool
    strong: bool
    weak_short: bool
    isolated: bool
    alpha: float
    track_id: int | None = None


@dataclass(frozen=True)
class _Prediction:
    points: np.ndarray
    motion_state: str = "unknown"
    residual_px: float = 0.0
    support_ratio: float = 0.0


@dataclass(frozen=True)
class _PredictionSet:
    predictions: dict[int, _Prediction]
    registration_ratio: float
    metrics: dict[str, float | int | str]


class ContourTracker:
    def __init__(self, config: ContourConfig | None = None) -> None:
        self.config = config or ContourConfig()
        _validate_config(self.config)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._prev_gray: np.ndarray | None = None
        self._prev_shape: tuple[int, int] | None = None
        self._prev_ignore_regions: (
            tuple[tuple[float, float, float, float], ...] | None
        ) = None
        self._prev_frame_index: int | None = None
        self._prev_timestamp_s: float | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._prev_gray = None
        self._prev_shape = None
        self._prev_ignore_regions = None
        self._prev_frame_index = None
        self._prev_timestamp_s = None

    def update(
        self,
        mask: np.ndarray,
        probability: np.ndarray,
        frame_bgr: np.ndarray,
        ignore_regions: Sequence[tuple[float, float, float, float]] = (),
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ) -> ContourResult:
        current_mask = _validate_mask(mask)
        probability = _validate_probability(probability, current_mask.shape)
        frame_bgr = _validate_frame(frame_bgr, current_mask.shape)
        _validate_optional_time(frame_index, timestamp_s)
        regions = _normalize_ignore_regions(ignore_regions)
        active = _active_region(current_mask.shape, regions)
        current = (current_mask > 0) & active
        gray = _gray_for_registration(frame_bgr, active)
        reset_reason = self._reset_reason(
            current.shape, regions, frame_index, timestamp_s
        )
        first_frame = self._prev_gray is None
        if reset_reason is not None:
            self._tracks.clear()

        prediction_set = self._predict(gray, active)
        predictions = prediction_set.predictions
        registration_ratio = prediction_set.registration_ratio
        if (
            reset_reason is None
            and self._prev_gray is not None
            and registration_ratio < self.config.min_registration_ratio
        ):
            self._tracks.clear()
            predictions = {}
            prediction_set = _PredictionSet(
                {}, registration_ratio, prediction_set.metrics
            )
            reset_reason = "poor_registration"

        candidates = _extract_candidates(current, probability, self.config)
        prediction_points = {
            track_id: prediction.points for track_id, prediction in predictions.items()
        }
        matches = _match_candidates(
            candidates, prediction_points, self._tracks, self.config
        )
        for candidate_index, track_id in matches.items():
            candidates[candidate_index].track_id = track_id

        endpoint_counts: dict[tuple[float, float], int] = {}
        for candidate in candidates:
            if not candidate.closed:
                for point in candidate.points[[0, -1]]:
                    key = (float(point[0]), float(point[1]))
                    endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
        shown: list[ContourPath] = []
        shared_endpoints: dict[
            tuple[float, float], list[tuple[int, int, str, int]]
        ] = {}
        next_tracks: dict[int, _Track] = {}
        pending = 0
        static_paths = 0
        dynamic_paths = 0
        unknown_paths = 0
        for candidate in candidates:
            track = self._tracks.get(candidate.track_id or -1)
            if track is None:
                track = _Track(
                    track_id=self._new_track_id(),
                    points=candidate.points.copy(),
                    closed=candidate.closed,
                    alpha=candidate.alpha,
                    recent=deque(maxlen=self.config.confirmation_window),
                )
            track.isolated = candidate.isolated
            track.recent.append(True)
            confirmed = (
                track.displayed
                or candidate.strong
                or not candidate.weak_short
                or sum(track.recent) >= self.config.confirmation_frames
            )
            if confirmed:
                prediction = predictions.get(track.track_id)
                previous = prediction.points if prediction is not None else track.points
                motion_state = (
                    prediction.motion_state if prediction is not None else "unknown"
                )
                if self.config.motion_separation and prediction is not None:
                    observed_residual = _path_distance(
                        previous, candidate.points, candidate.closed
                    )
                    if observed_residual >= self.config.motion_dynamic_residual_px:
                        motion_state = "dynamic"
                        track.static_evidence = 0
                points = (
                    candidate.points
                    if prediction is None
                    else _smooth_points(
                        previous,
                        candidate.points,
                        candidate.closed,
                        self.config,
                        motion_state=motion_state
                        if self.config.motion_separation
                        else "unknown",
                    )
                )
                if not candidate.closed:
                    for endpoint in (0, -1):
                        point = candidate.points[endpoint]
                        shared = endpoint_counts[(float(point[0]), float(point[1]))] > 1
                        if shared and (
                            not self.config.motion_separation
                            or motion_state != "static"
                        ):
                            points[endpoint] = point
                track.points = points
                track.closed = candidate.closed
                track.alpha = candidate.alpha
                track.missing = 0
                track.displayed = True
                track.motion_state = motion_state
                if motion_state == "static":
                    static_paths += 1
                elif motion_state == "dynamic":
                    dynamic_paths += 1
                else:
                    unknown_paths += 1
                shown.append(
                    ContourPath(
                        points=points.copy(),
                        alpha=float(candidate.alpha),
                        track_id=track.track_id,
                        closed=candidate.closed,
                    )
                )
                if self.config.motion_separation and not candidate.closed:
                    path_index = len(shown) - 1
                    for endpoint in (0, -1):
                        point = candidate.points[endpoint]
                        key = (float(point[0]), float(point[1]))
                        if endpoint_counts[key] > 1:
                            shared_endpoints.setdefault(key, []).append(
                                (path_index, endpoint, motion_state, track.track_id)
                            )
            else:
                pending += 1
                track.points = candidate.points.copy()
                track.closed = candidate.closed
                track.alpha = candidate.alpha
                track.missing = 0
                track.motion_state = "unknown"
            next_tracks[track.track_id] = track

        if self.config.motion_separation:
            _stabilize_shared_endpoints(shown, shared_endpoints, next_tracks)

        current_near = (
            cv2.dilate(current.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        )
        held = 0
        static_held = 0
        cleared = 0
        matched_ids = {track_id for track_id in matches.values()}
        for track_id, track in self._tracks.items():
            if track_id in matched_ids or track_id in next_tracks:
                continue
            track.recent.append(False)
            prediction = predictions.get(track_id)
            predicted = prediction.points if prediction is not None else None
            missing_limit = (
                1
                if self.config.motion_separation
                and track.displayed
                and prediction is not None
                and prediction.motion_state == "static"
                else (
                    self.config.max_missing_frames
                    if track.displayed
                    else self.config.confirmation_window - 1
                )
            )
            if (
                predicted is not None
                and track.isolated
                and track.missing < missing_limit
                and (
                    not track.displayed or _sample_bool(current_near, predicted) < 0.50
                )
                and _path_supported(predicted, probability, active, self.config)
                and (
                    not self.config.motion_separation
                    or not track.displayed
                    or (
                        prediction is not None
                        and prediction.motion_state == "static"
                        and track.motion_state == "static"
                        and prediction.support_ratio >= 0.70
                    )
                )
            ):
                track.points = predicted
                track.missing += 1
                track.motion_state = (
                    prediction.motion_state if prediction is not None else "unknown"
                )
                next_tracks[track_id] = track
                if track.displayed:
                    shown.append(
                        ContourPath(
                            points=predicted.copy(),
                            alpha=float(track.alpha * 0.65),
                            track_id=track.track_id,
                            closed=track.closed,
                        )
                    )
                    held += 1
                    if (
                        self.config.motion_separation
                        and prediction is not None
                        and prediction.motion_state == "static"
                    ):
                        static_held += 1
                else:
                    pending += 1
            else:
                cleared += 1

        protected_ids = {
            key for key, track in next_tracks.items() if not track.isolated
        }
        shown, deduped = _deduplicate(shown, self.config, protected_ids)
        output = _rasterize(current.shape, shown)
        output[~active] = 0
        allowed, dropped_output = _drop_short_components(
            output > 0, self.config.min_output_length
        )
        allowed, dropped_compact = _drop_compact_components(
            allowed, self.config.min_output_extent
        )
        if dropped_output or dropped_compact:
            shown = [
                path for path in shown if _sample_bool(allowed, path.points) >= 0.5
            ]
            output = _rasterize(current.shape, shown)
            output[~active] = 0
        self._tracks = next_tracks
        self._seed(gray, current.shape, regions, frame_index, timestamp_s)
        metrics: dict[str, float | int | str] = {
            "input_edge_pixels": int(np.count_nonzero(current)),
            "candidate_paths": int(len(candidates)),
            "shown_paths": int(len(shown)),
            "pending_paths": int(pending),
            "held_paths": int(held),
            "matched_paths": int(len(matches)),
            "deduped_paths": int(deduped),
            "dropped_output_components": int(dropped_output),
            "dropped_output_compact": int(dropped_compact),
            "min_output_length": float(self.config.min_output_length),
            "cleared_missing_paths": int(cleared),
            "output_edge_pixels": int(np.count_nonzero(output)),
            "registration_ratio": float(registration_ratio),
            "contour_reset": int(first_frame or reset_reason is not None),
            "reset_reason": reset_reason or ("first_frame" if first_frame else ""),
        }
        if self.config.motion_separation:
            metrics.update(prediction_set.metrics)
            metrics.update(
                {
                    "static_paths": int(static_paths),
                    "dynamic_paths": int(dynamic_paths),
                    "unknown_paths": int(unknown_paths),
                    "static_held_paths": int(static_held),
                }
            )
        return ContourResult(
            mask=output,
            paths=shown,
            metrics=metrics,
        )

    def _new_track_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _predict(self, gray: np.ndarray, active: np.ndarray) -> _PredictionSet:
        if self._prev_gray is None:
            return _PredictionSet({}, 0.0, _empty_motion_metrics("first_frame"))
        if not self._tracks:
            return _PredictionSet({}, 1.0, _empty_motion_metrics("no_tracks"))
        flow_seed = np.empty((*gray.shape, 2), dtype=np.float32)
        prev_to_current = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, flow_seed, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        current_to_prev = cv2.calcOpticalFlowFarneback(
            gray, self._prev_gray, flow_seed.copy(), 0.5, 3, 15, 3, 5, 1.2, 0
        )
        height, width = gray.shape
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        map_x = xs + current_to_prev[..., 0]
        map_y = ys + current_to_prev[..., 1]
        sampled_forward = cv2.remap(
            prev_to_current,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        sampled_prev_gray = cv2.remap(
            self._prev_gray,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        in_bounds = (
            (map_x >= 0.0)
            & (map_x <= float(width - 1))
            & (map_y >= 0.0)
            & (map_y <= float(height - 1))
        )
        fb_error = cv2.magnitude(
            map_x + sampled_forward[..., 0] - xs,
            map_y + sampled_forward[..., 1] - ys,
        )
        photo_error = np.abs(sampled_prev_gray - gray)
        confident = (
            in_bounds
            & active
            & (fb_error <= self.config.flow_fb_error_px)
            & (photo_error <= self.config.photometric_error)
        )
        registration_ratio = float(
            np.count_nonzero(confident) / max(1, np.count_nonzero(active))
        )
        if not self.config.motion_separation:
            predictions: dict[int, _Prediction] = {}
            for track_id, track in self._tracks.items():
                predicted = track.points + _sample_flow(prev_to_current, track.points)
                if _sample_bool(confident, predicted) < 0.75:
                    continue
                predictions[track_id] = _Prediction(
                    points=predicted,
                    support_ratio=_sample_bool(confident, predicted),
                )
            return _PredictionSet(predictions, registration_ratio, {})

        global_motion = _estimate_global_motion(
            current_to_prev, confident, active, self._prev_gray, self.config
        )
        residuals: list[float] = []
        predictions = {}
        for track_id, track in self._tracks.items():
            dense_predicted = track.points + _sample_flow(prev_to_current, track.points)
            support_ratio = _sample_bool(confident, dense_predicted)
            if support_ratio < 0.75:
                continue
            motion_state = "unknown"
            residual = 0.0
            predicted = dense_predicted
            if global_motion.status == "ok" and global_motion.affine is not None:
                global_predicted = _transform_points(track.points, global_motion.affine)
                residual = float(
                    np.mean(np.linalg.norm(dense_predicted - global_predicted, axis=1))
                )
                residuals.append(residual)
                if residual >= self.config.motion_dynamic_residual_px:
                    motion_state = "dynamic"
                    track.static_evidence = 0
                    predicted = dense_predicted
                elif residual <= self.config.motion_static_residual_px:
                    track.static_evidence += 1
                    if (
                        track.static_evidence
                        >= self.config.motion_static_confirm_frames
                    ):
                        motion_state = "static"
                        predicted = global_predicted
                else:
                    track.static_evidence = 0
            else:
                track.static_evidence = 0
            predictions[track_id] = _Prediction(
                points=predicted,
                motion_state=motion_state,
                residual_px=residual,
                support_ratio=support_ratio,
            )
        metrics = {
            "global_motion_status": global_motion.status,
            "global_motion_inliers": int(global_motion.inliers),
            "global_motion_coverage": float(global_motion.coverage),
            "global_motion_error_px": float(global_motion.error_px),
            "motion_residual_mean_px": float(np.mean(residuals)) if residuals else 0.0,
            "motion_residual_max_px": float(np.max(residuals)) if residuals else 0.0,
        }
        return _PredictionSet(predictions, registration_ratio, metrics)

    def _reset_reason(
        self,
        shape: tuple[int, int],
        ignore_regions: tuple[tuple[float, float, float, float], ...],
        frame_index: int | None,
        timestamp_s: float | None,
    ) -> str | None:
        if self._prev_shape is not None and shape != self._prev_shape:
            return "shape_changed"
        if (
            self._prev_ignore_regions is not None
            and ignore_regions != self._prev_ignore_regions
        ):
            return "ignore_regions_changed"
        if (
            frame_index is not None
            and self._prev_frame_index is not None
            and frame_index != self._prev_frame_index + 1
        ):
            return "frame_discontinuity"
        if timestamp_s is not None and self._prev_timestamp_s is not None:
            delta_s = timestamp_s - self._prev_timestamp_s
            if delta_s <= 0.0:
                return "timestamp_discontinuity"
            if delta_s > self.config.max_gap_seconds:
                return "time_gap"
        return None

    def _seed(
        self,
        gray: np.ndarray,
        shape: tuple[int, int],
        ignore_regions: tuple[tuple[float, float, float, float], ...],
        frame_index: int | None,
        timestamp_s: float | None,
    ) -> None:
        self._prev_gray = gray.astype(np.float32, copy=True)
        self._prev_shape = shape
        self._prev_ignore_regions = ignore_regions
        self._prev_frame_index = frame_index
        self._prev_timestamp_s = timestamp_s


def _extract_candidates(
    mask: np.ndarray, probability: np.ndarray, config: ContourConfig
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    traced = _trace_paths(mask)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    components = [
        int(labels[int(points[0, 1]), int(points[0, 0])]) for points, _ in traced
    ]
    path_counts = np.bincount(components) if components else np.zeros(1, dtype=int)
    for (raw_points, closed), component in zip(traced, components):
        isolated = path_counts[component] == 1
        points = _resample_by_spacing(_simplify(raw_points, closed), closed, 2.0)
        if len(points) < 2:
            continue
        samples = _sample_probability(probability, raw_points)
        if samples.size == 0:
            continue
        length = _path_length(points, closed)
        max_prob = float(samples.max())
        mean_prob = float(samples.mean())
        strong = max_prob >= config.high_threshold
        candidates.append(
            _Candidate(
                points=points,
                closed=closed,
                strong=strong,
                weak_short=(not strong)
                and isolated
                and length <= config.weak_short_length,
                isolated=bool(isolated),
                alpha=1.0
                if strong
                else float(np.clip(0.45 + 0.55 * mean_prob, 0.45, 0.9)),
            )
        )
    return candidates


def _trace_paths(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    height, width = mask.shape
    nodes = set(int(value) for value in np.flatnonzero(mask))
    graph: dict[int, list[int]] = {}
    for node in nodes:
        y, x = divmod(node, width)
        links: list[int] = []
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            other = ny * width + nx
            if other not in nodes:
                continue
            if dy and dx and (y * width + nx in nodes or ny * width + x in nodes):
                continue
            links.append(other)
        graph[node] = links
    visited: set[tuple[int, int]] = set()
    paths: list[tuple[np.ndarray, bool]] = []
    starts = [node for node in sorted(nodes) if len(graph[node]) != 2]
    for start in starts:
        for nxt in graph[start]:
            edge = _edge_key(start, nxt)
            if edge in visited:
                continue
            path = [start, nxt]
            visited.add(edge)
            previous, current = start, nxt
            while len(graph[current]) == 2:
                choices = [node for node in graph[current] if node != previous]
                if not choices:
                    break
                following = choices[0]
                edge = _edge_key(current, following)
                if edge in visited:
                    break
                visited.add(edge)
                path.append(following)
                previous, current = current, following
            paths.append((_nodes_to_points(path, width), False))
    for start in sorted(nodes):
        if not any(_edge_key(start, node) not in visited for node in graph[start]):
            continue
        path = [start]
        previous = -1
        current = start
        while True:
            choices = [
                node
                for node in graph[current]
                if node != previous and _edge_key(current, node) not in visited
            ]
            if not choices:
                break
            following = choices[0]
            visited.add(_edge_key(current, following))
            previous, current = current, following
            if current == start:
                break
            path.append(current)
        if len(path) >= 3:
            paths.append((_nodes_to_points(path, width), True))
    return paths


def _match_candidates(
    candidates: list[_Candidate],
    predicted: dict[int, np.ndarray],
    tracks: dict[int, _Track],
    config: ContourConfig,
) -> dict[int, int]:
    pairs: list[tuple[float, int, int]] = []
    sample_cache: dict[tuple[int, int], np.ndarray] = {}
    if not predicted:
        return {}
    track_ids = list(predicted)
    centers = np.asarray([_centroid(predicted[key]) for key in track_ids])
    lengths = np.asarray(
        [_path_length(predicted[key], tracks[key].closed) for key in track_ids]
    )
    closed = np.asarray([tracks[key].closed for key in track_ids])
    for candidate_index, candidate in enumerate(candidates):
        length = _path_length(candidate.points, candidate.closed)
        nearby = (
            (
                np.linalg.norm(centers - _centroid(candidate.points), axis=1)
                <= config.match_distance_px * 3
            )
            & (closed == candidate.closed)
            & (
                np.minimum(lengths, length)
                >= 0.45 * np.maximum(np.maximum(lengths, length), 1.0)
            )
        )
        for position in np.flatnonzero(nearby):
            track_id = track_ids[int(position)]
            distance = _path_distance(
                predicted[track_id], candidate.points, candidate.closed, sample_cache
            )
            if distance <= config.match_distance_px:
                pairs.append((distance, candidate_index, track_id))
    pairs.sort()
    matches: dict[int, int] = {}
    used_candidates: set[int] = set()
    used_tracks: set[int] = set()
    for _, candidate_index, track_id in pairs:
        if candidate_index in used_candidates or track_id in used_tracks:
            continue
        matches[candidate_index] = track_id
        used_candidates.add(candidate_index)
        used_tracks.add(track_id)
    return matches


def _smooth_points(
    previous: np.ndarray,
    current: np.ndarray,
    closed: bool,
    config: ContourConfig,
    motion_state: str = "unknown",
) -> np.ndarray:
    prev = _align_previous(previous, current, closed)
    direct = float(np.mean(np.linalg.norm(prev - current, axis=1)))
    max_alpha = (
        config.motion_static_max_smooth_alpha
        if motion_state == "static"
        else config.max_smooth_alpha
    )
    max_delta = config.motion_static_max_delta_px if motion_state == "static" else 1.0
    alpha = np.interp(
        direct,
        [0.0, config.slow_motion_px * 4.0],
        [max_alpha, config.min_smooth_alpha],
    )
    alpha = float(np.clip(alpha, config.min_smooth_alpha, max_alpha))
    smoothed = (alpha * prev + (1.0 - alpha) * current).astype(np.float32)
    delta = smoothed - current
    norm = np.linalg.norm(delta, axis=1)
    too_far = norm > max_delta
    if np.any(too_far):
        smoothed[too_far] = (
            current[too_far] + delta[too_far] / norm[too_far, None] * max_delta
        )
    return smoothed.astype(np.float32)


def _stabilize_shared_endpoints(
    paths: list[ContourPath],
    shared: dict[tuple[float, float], list[tuple[int, int, str, int]]],
    tracks: dict[int, _Track],
) -> None:
    for key, refs in shared.items():
        if len(refs) < 2:
            continue
        if all(motion_state == "static" for _, _, motion_state, _ in refs):
            points = np.asarray(
                [
                    paths[path_index].points[endpoint]
                    for path_index, endpoint, _, _ in refs
                ],
                dtype=np.float32,
            )
            shared_point = np.mean(points, axis=0).astype(np.float32)
        else:
            shared_point = np.asarray(key, dtype=np.float32)
        for path_index, endpoint, _, track_id in refs:
            paths[path_index].points[endpoint] = shared_point
            if track_id in tracks:
                tracks[track_id].points[endpoint] = shared_point


def _empty_motion_metrics(status: str) -> dict[str, float | int | str]:
    return {
        "global_motion_status": status,
        "global_motion_inliers": 0,
        "global_motion_coverage": 0.0,
        "global_motion_error_px": 0.0,
        "motion_residual_mean_px": 0.0,
        "motion_residual_max_px": 0.0,
    }


def _align_previous(
    previous: np.ndarray, current: np.ndarray, closed: bool
) -> np.ndarray:
    prev = _resample_path(previous, len(current), closed)
    choices = [prev, prev[::-1]]
    if closed:
        rolled: list[np.ndarray] = []
        for shift in range(len(prev)):
            forward = np.roll(prev, shift, axis=0)
            rolled.append(forward)
            rolled.append(forward[::-1])
        choices = rolled
    return min(
        choices,
        key=lambda item: float(np.mean(np.linalg.norm(item - current, axis=1))),
    )


def _path_supported(
    points: np.ndarray,
    probability: np.ndarray,
    active: np.ndarray,
    config: ContourConfig,
) -> bool:
    coords = np.rint(points).astype(np.int32)
    height, width = probability.shape
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < height)
    )
    if not np.any(valid):
        return False
    valid_coords = coords[valid]
    samples = probability[valid_coords[:, 1], valid_coords[:, 0]]
    active_samples = active[valid_coords[:, 1], valid_coords[:, 0]]
    supported = active_samples & (samples >= config.support_threshold)
    return bool(np.count_nonzero(supported) >= max(2, int(np.ceil(0.70 * len(points)))))


def _deduplicate(
    paths: list[ContourPath],
    config: ContourConfig,
    protected_ids: set[int] | None = None,
) -> tuple[list[ContourPath], int]:
    protected_ids = protected_ids or set()
    kept: list[ContourPath] = []
    centers: list[np.ndarray] = []
    lengths: list[float] = []
    dropped = 0
    sample_cache: dict[tuple[int, int], np.ndarray] = {}
    descriptors = [
        (path, _centroid(path.points), _path_length(path.points, path.closed))
        for path in paths
    ]
    for path, center, length in sorted(
        descriptors, key=lambda item: (-item[0].alpha, -item[2])
    ):
        duplicate = False
        # Strong current observations are never suppressed merely by proximity.
        if path.alpha < 1.0 and path.track_id not in protected_ids and kept:
            nearby = (
                np.linalg.norm(np.asarray(centers) - center, axis=1)
                <= config.duplicate_distance_px * 2
            )
            for position in np.flatnonzero(nearby):
                index = int(position)
                other = kept[index]
                if path.closed != other.closed:
                    continue
                if min(length, lengths[index]) < 0.70 * max(
                    length, lengths[index], 1.0
                ):
                    continue
                if (
                    _path_distance(path.points, other.points, path.closed, sample_cache)
                    <= config.duplicate_distance_px
                ):
                    duplicate = True
                    break
        if duplicate:
            dropped += 1
        else:
            kept.append(path)
            centers.append(center)
            lengths.append(length)
    return sorted(kept, key=lambda item: item.track_id), dropped


def _centroid(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(2, dtype=np.float32)
    return np.mean(points, axis=0)


def _rasterize(shape: tuple[int, int], paths: Sequence[ContourPath]) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    for path in paths:
        pts = np.rint(path.points).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(output, [pts], path.closed, 255, 1, cv2.LINE_8)
    return output


def _simplify(points: np.ndarray, closed: bool) -> np.ndarray:
    if len(points) < 8:
        return points.astype(np.float32, copy=True)
    simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2), 0.45, closed)
    if len(simplified) < 2:
        return points.astype(np.float32, copy=True)
    return simplified.reshape(-1, 2).astype(np.float32)


def _resample_by_spacing(
    points: np.ndarray, closed: bool, spacing: float
) -> np.ndarray:
    length = _path_length(points, closed)
    count = max(2, int(np.ceil(length / max(spacing, 0.5))) + 1)
    return _resample_path(points, count, closed)


def _path_distance(
    a: np.ndarray,
    b: np.ndarray,
    closed: bool,
    cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> float:
    count = int(np.clip(max(len(a), len(b)), 8, 32))
    if cache is None:
        aa = _resample_path(a, count, closed)
        bb = _resample_path(b, count, closed)
    else:
        for points in (a, b):
            key = (id(points), count)
            if key not in cache:
                cache[key] = _resample_path(points, count, closed)
        aa, bb = cache[(id(a), count)], cache[(id(b), count)]
    distances = [
        float(np.mean(np.linalg.norm(aa - bb, axis=1))),
        float(np.mean(np.linalg.norm(aa[::-1] - bb, axis=1))),
    ]
    if closed:
        for shift in range(1, count):
            rolled = np.roll(aa, shift, axis=0)
            distances.append(float(np.mean(np.linalg.norm(rolled - bb, axis=1))))
            distances.append(float(np.mean(np.linalg.norm(rolled[::-1] - bb, axis=1))))
    return min(distances)


def _resample_path(points: np.ndarray, count: int, closed: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, count, axis=0)
    work = np.vstack([pts, pts[0]]) if closed else pts
    segment = np.linalg.norm(np.diff(work, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(pts[:1], count, axis=0)
    targets = np.linspace(0.0, total, count, endpoint=not closed)
    return np.column_stack(
        [
            np.interp(targets, cumulative, work[:, 0]),
            np.interp(targets, cumulative, work[:, 1]),
        ]
    ).astype(np.float32)


def _path_length(points: np.ndarray, closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    work = np.vstack([points, points[0]]) if closed else points
    return float(np.sum(np.linalg.norm(np.diff(work, axis=0), axis=1)))


def _sample_probability(probability: np.ndarray, points: np.ndarray) -> np.ndarray:
    coords = np.rint(points).astype(np.int32)
    height, width = probability.shape
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < height)
    )
    if not np.any(valid):
        return np.zeros(0, dtype=np.float32)
    coords = coords[valid]
    return probability[coords[:, 1], coords[:, 0]].astype(np.float32)


def _sample_flow(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    xs = points[:, 0].astype(np.float32).reshape(-1, 1)
    ys = points[:, 1].astype(np.float32).reshape(-1, 1)
    dx = cv2.remap(
        flow[..., 0], xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    dy = cv2.remap(
        flow[..., 1], xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return np.column_stack([dx.ravel(), dy.ravel()]).astype(np.float32)


def _sample_bool(mask: np.ndarray, points: np.ndarray) -> float:
    coords = np.rint(points).astype(np.int32)
    height, width = mask.shape
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < height)
    )
    if not np.any(valid):
        return 0.0
    coords = coords[valid]
    return float(np.count_nonzero(mask[coords[:, 1], coords[:, 0]]) / len(coords))


def _nodes_to_points(nodes: Sequence[int], width: int) -> np.ndarray:
    return np.asarray(
        [(node % width, node // width) for node in nodes], dtype=np.float32
    )


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _gray_for_registration(frame_bgr: np.ndarray, active: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if np.any(~active):
        gray[~active] = float(np.median(gray[active])) if np.any(active) else 0.0
    return gray


def _validate_config(config: ContourConfig) -> None:
    values = (
        config.high_threshold,
        config.support_threshold,
        config.weak_short_length,
        config.min_output_length,
        config.min_output_extent,
        config.confirmation_frames,
        config.confirmation_window,
        config.max_missing_frames,
        config.match_distance_px,
        config.slow_motion_px,
        config.max_smooth_alpha,
        config.min_smooth_alpha,
        config.max_gap_seconds,
        config.flow_fb_error_px,
        config.photometric_error,
        config.min_registration_ratio,
        config.duplicate_distance_px,
        config.motion_static_residual_px,
        config.motion_dynamic_residual_px,
        config.motion_static_confirm_frames,
        config.motion_global_min_samples,
        config.motion_global_min_coverage,
        config.motion_global_min_inlier_ratio,
        config.motion_global_max_error_px,
        config.motion_static_max_smooth_alpha,
        config.motion_static_max_delta_px,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("contour config values must be finite")
    if not 0.0 <= config.support_threshold <= config.high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= support <= high <= 1")
    if (
        config.confirmation_frames < 1
        or config.confirmation_window < config.confirmation_frames
    ):
        raise ValueError("confirmation window must contain the required confirmations")
    if config.max_missing_frames < 0:
        raise ValueError("max_missing_frames must be non-negative")
    if (
        min(
            config.weak_short_length,
            config.min_output_length,
            config.min_output_extent,
            config.match_distance_px,
            config.slow_motion_px,
            config.motion_static_residual_px,
            config.motion_dynamic_residual_px,
            config.motion_global_max_error_px,
            config.motion_static_max_delta_px,
        )
        < 0.0
    ):
        raise ValueError("length and distance values must be non-negative")
    if not 0.0 <= config.min_smooth_alpha <= config.max_smooth_alpha < 1.0:
        raise ValueError("smooth alpha values must satisfy 0 <= min <= max < 1")
    if config.max_gap_seconds <= 0.0:
        raise ValueError("max_gap_seconds must be positive")
    if (
        min(
            config.flow_fb_error_px,
            config.photometric_error,
            config.duplicate_distance_px,
        )
        < 0.0
    ):
        raise ValueError("registration and duplicate gates must be non-negative")
    if not 0.0 <= config.min_registration_ratio <= 1.0:
        raise ValueError("min_registration_ratio must be within [0, 1]")
    if config.motion_static_residual_px > config.motion_dynamic_residual_px:
        raise ValueError("motion residual thresholds must be ordered")
    if config.motion_static_confirm_frames < 1:
        raise ValueError("motion_static_confirm_frames must be positive")
    if config.motion_global_min_samples < 3:
        raise ValueError("motion_global_min_samples must be at least 3")
    if not 0.0 <= config.motion_global_min_coverage <= 1.0:
        raise ValueError("motion_global_min_coverage must be within [0, 1]")
    if not 0.0 <= config.motion_global_min_inlier_ratio <= 1.0:
        raise ValueError("motion_global_min_inlier_ratio must be within [0, 1]")
    if not config.min_smooth_alpha <= config.motion_static_max_smooth_alpha < 1.0:
        raise ValueError("motion static smooth alpha must be within configured bounds")


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("mask must be a non-empty HxW array")
    return (array > 0).astype(np.uint8) * 255


def _validate_probability(
    probability: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    array = np.asarray(probability, dtype=np.float32)
    if array.shape != shape:
        raise ValueError("probability must match mask shape")
    if not np.isfinite(array).all():
        raise ValueError("probability must contain only finite values")
    return np.clip(array, 0.0, 1.0)


def _validate_frame(frame_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(frame_bgr)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[:2] != shape:
        raise ValueError("frame_bgr must be an HxWx3 image matching the mask")
    if array.dtype != np.uint8:
        raise ValueError("frame_bgr must be uint8")
    return array


def _validate_optional_time(frame_index: int | None, timestamp_s: float | None) -> None:
    if frame_index is not None and frame_index < 0:
        raise ValueError("frame_index must be non-negative when provided")
    if timestamp_s is not None and not np.isfinite(timestamp_s):
        raise ValueError("timestamp_s must be finite when provided")


def _normalize_ignore_regions(
    regions: Sequence[tuple[float, float, float, float]],
) -> tuple[tuple[float, float, float, float], ...]:
    normalized: list[tuple[float, float, float, float]] = []
    for region in regions:
        if len(region) != 4:
            raise ValueError(
                "ignore_regions entries must be (left, top, right, bottom)"
            )
        left, top, right, bottom = (float(value) for value in region)
        if not all(np.isfinite(value) for value in (left, top, right, bottom)):
            raise ValueError("ignore_regions entries must be finite")
        if not (0.0 <= left <= right <= 1.0 and 0.0 <= top <= bottom <= 1.0):
            raise ValueError("ignore_regions must be normalized ordered rectangles")
        normalized.append((left, top, right, bottom))
    return tuple(normalized)


_NEIGHBOURS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
