"""Temporal visible-light flame candidate detector.

This is deliberately not a temperature or semantic fire detector. It only marks
warm, bright, temporally changing visible-light regions so Fire360 replay can show an
operator-facing cue when calibrated Celsius data is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

NormalizedRegion = tuple[float, float, float, float]


@dataclass(frozen=True)
class VisibleFlameConfig:
    min_area_px: int = 18
    min_bright_core_px: int = 4
    min_motion_fraction: float = 0.025
    min_motion_px: int = 3
    motion_threshold: int = 10
    confirm_frames: int = 2
    motion_memory_frames: int = 3
    min_core_luminance: int = 170
    min_core_green: int = 140
    warm_hue_max: int = 38
    min_saturation: int = 65
    min_value: int = 105
    bright_value: int = 205
    min_red: int = 135
    red_green_margin: int = -8
    green_blue_margin: int = 18

    def __post_init__(self) -> None:
        if self.min_area_px < 1:
            raise ValueError("min_area_px must be positive")
        if self.min_bright_core_px < 1:
            raise ValueError("min_bright_core_px must be positive")
        if not 0.0 <= self.min_motion_fraction <= 1.0:
            raise ValueError("min_motion_fraction must be within [0, 1]")
        if self.min_motion_px < 1:
            raise ValueError("min_motion_px must be positive")
        if self.motion_threshold < 1:
            raise ValueError("motion_threshold must be positive")
        if not 1 <= self.confirm_frames <= 255:
            raise ValueError("confirm_frames must be within [1, 255]")
        if not 1 <= self.motion_memory_frames <= 255:
            raise ValueError("motion_memory_frames must be within [1, 255]")


class VisibleFlameDetector:
    def __init__(self, config: VisibleFlameConfig | None = None) -> None:
        self.config = config or VisibleFlameConfig()
        self._previous_gray: np.ndarray | None = None
        self._score: np.ndarray | None = None
        self._motion_age: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_gray = None
        self._score = None
        self._motion_age = None

    def update(
        self, frame_bgr: np.ndarray, ignore_regions: Iterable[NormalizedRegion] = ()
    ) -> tuple[np.ndarray, dict]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a BGR uint8 image")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be uint8")

        height, width = frame_bgr.shape[:2]
        if self._score is None or self._score.shape != (height, width):
            self._score = np.zeros((height, width), np.uint8)
            self._previous_gray = None
            self._motion_age = np.zeros((height, width), np.uint8)

        ignore = _ignore_mask((height, width), ignore_regions)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._score.fill(0)
            return (
                np.zeros((height, width), np.uint8),
                {
                    "status": "warming_up",
                    "candidate_pixels": 0,
                    "component_count": 0,
                    "raw_candidate_pixels": 0,
                    "motion_pixels": 0,
                },
            )

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue = frame_bgr[:, :, 0].astype(np.int16)
        green = frame_bgr[:, :, 1].astype(np.int16)
        red = frame_bgr[:, :, 2].astype(np.int16)

        warm = (
            (hue <= self.config.warm_hue_max)
            & (saturation >= self.config.min_saturation)
            & (value >= self.config.min_value)
            & (red >= self.config.min_red)
            & (red + self.config.red_green_margin >= green)
            & (green >= blue + self.config.green_blue_margin)
            & ~ignore
        )
        bright_core = (
            warm
            & (value >= self.config.bright_value)
            & (gray >= self.config.min_core_luminance)
            & (green >= self.config.min_core_green)
            & (red >= green - 4)
            & (red >= blue + 45)
        )
        motion = (
            cv2.absdiff(gray, self._previous_gray) >= self.config.motion_threshold
        ) & ~ignore
        assert self._motion_age is not None
        self._motion_age = np.maximum(
            self._motion_age.astype(np.int16) - 1, 0
        ).astype(np.uint8)
        self._motion_age[motion] = self.config.motion_memory_frames
        self._motion_age[ignore] = 0
        motion_support = cv2.dilate(
            (self._motion_age > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        ).astype(bool)

        local_core_support = cv2.dilate(
            (bright_core & motion_support).astype(np.uint8),
            np.ones((7, 7), np.uint8),
            iterations=1,
        ).astype(bool)
        raw = np.zeros((height, width), np.uint8)
        component_count = 0
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            warm.astype(np.uint8), connectivity=8
        )
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.config.min_area_px:
                continue
            component = labels == label
            core_pixels = int(np.count_nonzero(bright_core & component))
            if core_pixels < self.config.min_bright_core_px:
                continue
            motion_pixels = int(np.count_nonzero(motion_support & component))
            required_motion = max(
                self.config.min_motion_px,
                int(round(area * self.config.min_motion_fraction)),
            )
            if motion_pixels < required_motion:
                continue
            display_pixels = component & local_core_support
            if not np.any(display_pixels):
                continue
            raw[display_pixels] = 255
            component_count += 1

        assert self._score is not None
        self._score[raw > 0] = np.minimum(
            self._score[raw > 0].astype(np.uint16) + 1,
            self.config.confirm_frames,
        ).astype(np.uint8)
        self._score[raw == 0] = 0
        output = np.where(
            (raw > 0) & (self._score >= self.config.confirm_frames), 255, 0
        ).astype(np.uint8)
        count, confirmed_labels, confirmed_stats, _ = cv2.connectedComponentsWithStats(
            output, connectivity=8
        )
        kept = np.zeros(count, bool)
        kept[1:] = confirmed_stats[1:, cv2.CC_STAT_AREA] >= self.config.min_area_px
        confirmed_labels = np.asarray(confirmed_labels, dtype=np.int32)
        output = np.where(kept[confirmed_labels], 255, 0).astype(np.uint8)
        confirmed_count = int(np.count_nonzero(kept))
        self._previous_gray = gray

        candidate_pixels = int(np.count_nonzero(output))
        return (
            output,
            {
                "status": "visible_flame_candidate"
                if candidate_pixels
                else "no_candidate",
                "candidate_pixels": candidate_pixels,
                "component_count": confirmed_count,
                "raw_component_count": component_count,
                "raw_candidate_pixels": int(np.count_nonzero(raw)),
                "motion_pixels": int(np.count_nonzero(motion)),
            },
        )


def _ignore_mask(
    shape: tuple[int, int], regions: Iterable[NormalizedRegion]
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), bool)
    for left, top, right, bottom in regions:
        coords = np.asarray((left, top, right, bottom), dtype=float)
        if not np.isfinite(coords).all() or not (
            0 <= left < right <= 1 and 0 <= top < bottom <= 1
        ):
            raise ValueError("ignore regions must be finite ordered normalized bounds")
        x0, y0 = round(left * width), round(top * height)
        x1, y1 = round(right * width), round(bottom * height)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask
