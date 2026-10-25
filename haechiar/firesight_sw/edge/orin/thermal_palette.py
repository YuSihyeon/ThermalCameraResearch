# ruff: noqa: INP001
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

BGRImage: TypeAlias = NDArray[np.uint8]
BoolMask: TypeAlias = NDArray[np.bool_]
NormalizedRegion: TypeAlias = tuple[float, float, float, float]

IMAGE_DIMS = 3
BGR_CHANNELS = 3
MAX_NORMALIZED = 1.0
MAX_WARM_HUE = 40
MIN_RED_WRAP_HUE = 175
MIN_VALUE = 80
MIN_SATURATION = 35
MIN_RED = 105
MIN_RED_BLUE_DELTA = 35
MIN_GREEN_BLUE_DELTA = 20
MIN_RED_GREEN_DELTA = 20
REFERENCE_WIDTH = 720.0
REFERENCE_OUTLINE_WIDTH = 3.0
REFERENCE_HALO_RADIUS = 1.5
REFERENCE_MIN_AREA = 16.0
RED_OUTLINE_BGR = (0, 0, 255)


@dataclass(frozen=True)
class ThermalPaletteMetrics:
    status: str
    image_shape_hw: tuple[int, int]
    candidate_pixels: int
    ignored_pixels: int
    hot_region_pixels: int
    outline_pixels: int
    region_count: int


def apply_thermal_palette(
    composite: BGRImage,
    source: BGRImage,
    ignore_regions: tuple[NormalizedRegion, ...] = (),
    *,
    background: BGRImage,
) -> tuple[BoolMask, ThermalPaletteMetrics]:
    _validate_bgr_pair(composite, source, "composite", "source")
    _validate_bgr_pair(composite, background, "composite", "background")

    height, width = _shape_hw(source)
    ignore_mask = _ignore_mask((height, width), ignore_regions)
    candidate = _warm_palette_mask(source)
    ignored_candidates = candidate & ignore_mask
    hot_region, region_count = _filtered_hot_region(
        candidate & ~ignore_mask,
        width,
    )
    hot_region &= ~ignore_mask
    external_fill = _external_fill(hot_region)
    restore_body = external_fill | hot_region
    halo = _dilate(restore_body, _scaled_radius(width, REFERENCE_HALO_RADIUS))
    safe_restore = halo & ~ignore_mask
    composite[safe_restore] = background[safe_restore]

    outline = _external_outline(hot_region, width)
    outline &= ~_dilate(ignore_mask, _outline_guard_radius(width))
    composite[outline] = RED_OUTLINE_BGR

    return outline, ThermalPaletteMetrics(
        status="thermal_palette_contours",
        image_shape_hw=(height, width),
        candidate_pixels=int(np.count_nonzero(candidate)),
        ignored_pixels=int(np.count_nonzero(ignored_candidates)),
        hot_region_pixels=int(np.count_nonzero(hot_region)),
        outline_pixels=int(np.count_nonzero(outline)),
        region_count=region_count,
    )


def _validate_bgr_pair(
    first: BGRImage,
    second: BGRImage,
    first_name: str,
    second_name: str,
) -> None:
    if first.shape != second.shape:
        message = f"{first_name} and {second_name} must have the same shape"
        raise ValueError(message)
    if first.ndim != IMAGE_DIMS or first.shape[2] != BGR_CHANNELS:
        message = f"{first_name} and {second_name} must be HxWx3 BGR images"
        raise ValueError(message)
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        message = f"{first_name} and {second_name} must be uint8 images"
        raise TypeError(message)


def _shape_hw(image: BGRImage) -> tuple[int, int]:
    shape = cast("tuple[int, int, int]", image.shape)
    return shape[0], shape[1]


def _ignore_mask(
    image_shape: tuple[int, int],
    ignore_regions: tuple[NormalizedRegion, ...],
) -> BoolMask:
    height, width = image_shape
    ignored = np.zeros((height, width), dtype=np.bool_)
    for region in ignore_regions:
        left, top, right, bottom = region
        if (
            not all(np.isfinite(value) for value in region)
            or not 0 <= left < right <= MAX_NORMALIZED
            or not 0 <= top < bottom <= MAX_NORMALIZED
        ):
            message = (
                "ignore_regions must contain normalized "
                "(left, top, right, bottom) boxes"
            )
            raise ValueError(message)
        x0 = floor(left * width)
        y0 = floor(top * height)
        x1 = ceil(right * width)
        y1 = ceil(bottom * height)
        ignored[y0:y1, x0:x1] = True
    return ignored


def _filtered_hot_region(hot: BoolMask, width: int) -> tuple[BoolMask, int]:
    region = _morph(hot, cv2.MORPH_CLOSE)
    region = _morph(region, cv2.MORPH_OPEN)
    labels, connected, stats, _ = cv2.connectedComponentsWithStats(
        region.astype(np.uint8),
        connectivity=8,
    )
    filtered = np.zeros_like(region)
    min_area = _scaled_area(width)
    kept = 0
    for label in range(1, labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[connected == label] = True
            kept += 1
    return filtered, kept


def _external_outline(hot_region: BoolMask, width: int) -> BoolMask:
    if not hot_region.any():
        return np.zeros_like(hot_region)
    contours = _external_contours(hot_region)
    outline = np.zeros(hot_region.shape, dtype=np.uint8)
    cv2.drawContours(
        outline,
        contours,
        contourIdx=-1,
        color=255,
        thickness=_scaled_outline_width(width),
    )
    return outline.astype(np.bool_)


def _external_fill(hot_region: BoolMask) -> BoolMask:
    if not hot_region.any():
        return np.zeros_like(hot_region)
    filled = np.zeros(hot_region.shape, dtype=np.uint8)
    cv2.drawContours(
        filled,
        _external_contours(hot_region),
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )
    return filled.astype(np.bool_)


def _external_contours(hot_region: BoolMask) -> tuple[NDArray[np.int32], ...]:
    contours, _ = cv2.findContours(
        hot_region.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return tuple(cast("tuple[NDArray[np.int32], ...]", contours))


def _morph(mask: BoolMask, operation: int) -> BoolMask:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(np.bool_)


def _dilate(mask: BoolMask, radius: int) -> BoolMask:
    if radius <= 0 or not mask.any():
        return mask.copy()
    size = radius * 2 + 1
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(np.bool_)


def _scaled_outline_width(width: int) -> int:
    return max(1, round(width * REFERENCE_OUTLINE_WIDTH / REFERENCE_WIDTH))


def _outline_guard_radius(width: int) -> int:
    return _scaled_outline_width(width) + _scaled_radius(width, REFERENCE_HALO_RADIUS)


def _scaled_radius(width: int, reference_radius: float) -> int:
    return max(1, round(width * reference_radius / REFERENCE_WIDTH))


def _scaled_area(width: int) -> int:
    scale = width / REFERENCE_WIDTH
    return max(1, round(REFERENCE_MIN_AREA * scale * scale))


def _warm_palette_mask(source: BGRImage) -> BoolMask:
    hsv = cast("BGRImage", cv2.cvtColor(source, cv2.COLOR_BGR2HSV))
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    blue = source[:, :, 0].astype(np.int16)
    green = source[:, :, 1].astype(np.int16)
    red = source[:, :, 2].astype(np.int16)

    warm_hue = (hue <= MAX_WARM_HUE) | (hue >= MIN_RED_WRAP_HUE)
    visible_colour = (value >= MIN_VALUE) & (saturation >= MIN_SATURATION)
    warm_channels = (red >= MIN_RED) & (red >= blue + MIN_RED_BLUE_DELTA) & (
        (green >= blue + MIN_GREEN_BLUE_DELTA)
        | (red >= green + MIN_RED_GREEN_DELTA)
    )
    return warm_hue & visible_colour & warm_channels
