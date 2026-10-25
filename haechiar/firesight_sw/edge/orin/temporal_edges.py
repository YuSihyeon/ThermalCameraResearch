from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from motion_registration import MotionRegistrationSettings, _estimate_global_motion


@dataclass(frozen=True)
class TemporalEdgeConfig:
    history_weight: float = 0.6
    immediate_threshold: float = 0.75
    support_threshold: float = 0.25
    max_gap_seconds: float = 0.25
    flow_fb_error_px: float = 1.5
    photometric_error: float = 35.0
    min_registration_ratio: float = 0.2

    def __post_init__(self) -> None:
        _validate_config(self)


class TemporalEdgeStabilizer:
    def __init__(
        self,
        config: TemporalEdgeConfig | None = None,
        motion_config: MotionRegistrationSettings | None = None,
    ) -> None:
        self.config = config or TemporalEdgeConfig()
        self.motion_config = motion_config
        self._static_mask: np.ndarray | None = None
        self._motion_status = "disabled"
        _validate_config(self.config)
        self._history: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_active_region: np.ndarray | None = None
        self._prev_shape: tuple[int, int] | None = None
        self._prev_ignore_regions: (
            tuple[tuple[float, float, float, float], ...] | None
        ) = None
        self._prev_frame_index: int | None = None
        self._prev_timestamp_s: float | None = None

    def reset(self) -> None:
        self._static_mask = None
        self._motion_status = "reset"
        self._history = None
        self._prev_gray = None
        self._prev_active_region = None
        self._prev_shape = None
        self._prev_ignore_regions = None
        self._prev_frame_index = None
        self._prev_timestamp_s = None

    def update(
        self,
        probability: np.ndarray,
        frame_bgr: np.ndarray,
        ignore_regions: Sequence[tuple[float, float, float, float]] = (),
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        probability = _validate_probability(probability)
        frame_bgr = _validate_frame(frame_bgr, probability.shape)
        _validate_config(self.config)
        _validate_optional_time(frame_index, timestamp_s)

        normalized_regions = _normalize_ignore_regions(ignore_regions)
        active_region = _active_region(probability.shape, normalized_regions)
        current = probability.copy()
        current[~active_region] = 0.0
        gray = _gray_for_registration(frame_bgr, active_region)

        reset_reason = self._reset_reason(
            probability.shape, normalized_regions, frame_index, timestamp_s
        )
        if reset_reason is not None or self._history is None or self._prev_gray is None:
            self._seed(
                current,
                gray,
                active_region,
                normalized_regions,
                frame_index,
                timestamp_s,
            )
            return current.copy(), {
                "temporal_reset": 1,
                "reset_reason": reset_reason or "first_frame",
                "registration_ratio": 0.0,
                "warped_history_pixels": 0,
                "history_supported_pixels": int(
                    np.count_nonzero(current >= self.config.support_threshold)
                ),
                "immediate_pixels": int(
                    np.count_nonzero(current >= self.config.immediate_threshold)
                ),
                "history_weight": 0.0,
                "ignored_region_pixels": int(np.count_nonzero(~active_region)),
            }

        warped, confident, registration_ratio = self._warp_history_to_current(
            self._history,
            self._prev_gray,
            self._prev_active_region,
            gray,
            active_region,
        )
        if registration_ratio < self.config.min_registration_ratio:
            self._seed(
                current,
                gray,
                active_region,
                normalized_regions,
                frame_index,
                timestamp_s,
            )
            return current.copy(), {
                "temporal_reset": 1,
                "reset_reason": "poor_registration",
                "registration_ratio": float(registration_ratio),
                "warped_history_pixels": int(
                    np.count_nonzero(warped >= self.config.support_threshold)
                ),
                "history_supported_pixels": int(
                    np.count_nonzero(current >= self.config.support_threshold)
                ),
                "immediate_pixels": int(
                    np.count_nonzero(current >= self.config.immediate_threshold)
                ),
                "history_weight": 0.0,
                "ignored_region_pixels": int(np.count_nonzero(~active_region)),
            }

        history_weight = self._effective_history_weight(timestamp_s)
        supported = (
            confident & (current >= self.config.support_threshold) & (current > 0.0)
        )
        immediate = current >= self.config.immediate_threshold

        stabilized = current.copy()
        stabilized[supported] = (
            history_weight * warped[supported]
            + (1.0 - history_weight) * current[supported]
        )
        stabilized[immediate] = current[immediate]
        static_supported = np.zeros(current.shape, dtype=bool)
        if self.motion_config is not None and self._static_mask is not None:
            static_supported = (
                supported
                & self._static_mask
                & (warped >= self.config.support_threshold)
            )
            static_weight = self._effective_history_weight(
                timestamp_s, max(self.config.history_weight, 0.8)
            )
            stabilized[static_supported] = (
                static_weight * warped[static_supported]
                + (1.0 - static_weight) * current[static_supported]
            )
            stabilized[immediate & (warped < 0.55)] = current[
                immediate & (warped < 0.55)
            ]
            # Strong new boundaries have no supported predecessor and stay immediate.
            # Current support is mandatory; history cannot resurrect a vanished edge.

        stabilized[current < self.config.support_threshold] = current[
            current < self.config.support_threshold
        ]
        stabilized[~active_region] = 0.0
        stabilized = np.clip(stabilized, 0.0, 1.0).astype(np.float32)

        self._seed(
            stabilized,
            gray,
            active_region,
            normalized_regions,
            frame_index,
            timestamp_s,
        )
        return stabilized, {
            "static_probability_pixels": int(np.count_nonzero(static_supported)),
            "motion_probability_status": self._motion_status,
            "temporal_reset": 0,
            "reset_reason": "",
            "registration_ratio": float(registration_ratio),
            "warped_history_pixels": int(
                np.count_nonzero(warped >= self.config.support_threshold)
            ),
            "history_supported_pixels": int(np.count_nonzero(supported)),
            "immediate_pixels": int(np.count_nonzero(immediate)),
            "history_weight": float(history_weight),
            "ignored_region_pixels": int(np.count_nonzero(~active_region)),
        }

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

    def _effective_history_weight(
        self, timestamp_s: float | None, weight: float | None = None
    ) -> float:
        weight = self.config.history_weight if weight is None else weight
        if timestamp_s is None or self._prev_timestamp_s is None:
            return float(weight)
        delta_s = timestamp_s - self._prev_timestamp_s
        if delta_s <= 0.0:
            return 0.0
        frame_equivalent = max(delta_s * 30.0, 0.0)
        return float(weight**frame_equivalent)

    def _warp_history_to_current(
        self,
        history: np.ndarray,
        prev_gray: np.ndarray,
        prev_active_region: np.ndarray | None,
        gray: np.ndarray,
        active_region: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if prev_active_region is None:
            prev_active_region = np.zeros(history.shape, dtype=bool)
        flow_seed = np.empty((*history.shape, 2), dtype=np.float32)
        prev_to_current = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, flow_seed, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        current_to_prev = cv2.calcOpticalFlowFarneback(
            gray, prev_gray, flow_seed.copy(), 0.5, 3, 15, 3, 5, 1.2, 0
        )
        height, width = history.shape
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        map_x = xs + current_to_prev[..., 0]
        map_y = ys + current_to_prev[..., 1]
        in_bounds = (
            (map_x >= 0.0)
            & (map_x <= float(width - 1))
            & (map_y >= 0.0)
            & (map_y <= float(height - 1))
        )

        warped = cv2.remap(
            history,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.float32)
        sampled_forward = cv2.remap(
            prev_to_current,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        sampled_prev_gray = cv2.remap(
            prev_gray,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        sampled_prev_active = cv2.remap(
            prev_active_region.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        roundtrip_x = map_x + sampled_forward[..., 0]
        roundtrip_y = map_y + sampled_forward[..., 1]
        fb_error = cv2.magnitude(roundtrip_x - xs, roundtrip_y - ys)
        photometric_error = np.abs(sampled_prev_gray - gray)
        confident = (
            in_bounds
            & active_region
            & sampled_prev_active
            & (fb_error <= self.config.flow_fb_error_px)
            & (photometric_error <= self.config.photometric_error)
        )
        active_count = int(np.count_nonzero(active_region))
        if active_count == 0:
            registration_ratio = 0.0
        else:
            registration_ratio = float(np.count_nonzero(confident) / active_count)
        self._static_mask = None
        if self.motion_config is not None:
            motion = _estimate_global_motion(
                current_to_prev, confident, active_region, prev_gray, self.motion_config
            )
            self._motion_status = motion.status
            if motion.status == "ok" and motion.affine is not None:
                inverse = cv2.invertAffineTransform(motion.affine)
                global_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
                global_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
                residual = cv2.magnitude(map_x - global_x, map_y - global_y)
                self._static_mask = confident & (
                    residual <= self.motion_config.motion_static_residual_px
                )
        warped[~confident] = 0.0
        return warped, confident, registration_ratio

    def _seed(
        self,
        probability: np.ndarray,
        gray: np.ndarray,
        active_region: np.ndarray,
        ignore_regions: tuple[tuple[float, float, float, float], ...],
        frame_index: int | None,
        timestamp_s: float | None,
    ) -> None:
        self._history = probability.astype(np.float32, copy=True)
        self._prev_gray = gray.astype(np.float32, copy=True)
        self._prev_active_region = active_region.astype(bool, copy=True)
        self._prev_shape = probability.shape
        self._prev_ignore_regions = ignore_regions
        self._prev_frame_index = frame_index
        self._prev_timestamp_s = timestamp_s


def _validate_config(config: TemporalEdgeConfig) -> None:
    values = (
        config.history_weight,
        config.immediate_threshold,
        config.support_threshold,
        config.max_gap_seconds,
        config.flow_fb_error_px,
        config.photometric_error,
        config.min_registration_ratio,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("temporal edge config values must be finite")
    if not 0.0 <= config.history_weight < 1.0:
        raise ValueError("history_weight must be within [0, 1)")
    if not 0.0 <= config.support_threshold <= config.immediate_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= support <= immediate <= 1")
    if config.max_gap_seconds <= 0.0:
        raise ValueError("max_gap_seconds must be positive")
    if config.flow_fb_error_px < 0.0 or config.photometric_error < 0.0:
        raise ValueError("registration gates must be non-negative")
    if not 0.0 <= config.min_registration_ratio <= 1.0:
        raise ValueError("min_registration_ratio must be within [0, 1]")


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
    if array.dtype != np.uint8:
        raise ValueError("frame_bgr must be uint8")
    return array


def _validate_optional_time(frame_index: int | None, timestamp_s: float | None) -> None:
    if frame_index is not None and frame_index < 0:
        raise ValueError("frame_index must be non-negative when provided")
    if timestamp_s is not None and not np.isfinite(timestamp_s):
        raise ValueError("timestamp_s must be finite when provided")


def _active_region(
    shape: tuple[int, int],
    regions: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    height, width = shape
    active = np.ones(shape, dtype=bool)
    for left, top, right, bottom in regions:
        x0 = int(round(left * width))
        x1 = int(round(np.clip(right, 0.0, 1.0) * width))
        y0 = int(round(top * height))
        y1 = int(round(np.clip(bottom, 0.0, 1.0) * height))
        active[y0 : y1 + 1, x0 : x1 + 1] = False
    return active


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


def _gray_for_registration(
    frame_bgr: np.ndarray, active_region: np.ndarray
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray[~active_region] = 0.0
    return gray
