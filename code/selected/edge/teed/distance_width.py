"""Distance-dependent raster widths for TEED edge masks.

The rasterizer uses square OpenCV dilation kernels.  A requested width of
``2 * radius + 1`` therefore covers that many pixels along each image axis.
"""

import math
import time

import cv2
import numpy as np


class WidthConfig(object):
    """Validated renderer settings, kept compatible with Python 3.6."""

    def __init__(self, near_m=1.0, far_m=8.0, near_px=5.0, far_px=1.0,
                 tau_s=0.05, max_age_s=0.1, jump_px=0.75):
        values = (near_m, far_m, near_px, far_px, tau_s, max_age_s, jump_px)
        if not all(_is_finite_number(value) for value in values):
            raise ValueError("all width configuration values must be finite")
        if near_m <= 0.0:
            raise ValueError("near_m must be positive")
        if near_m >= far_m:
            raise ValueError("near_m must be less than far_m")
        if not 1.0 <= far_px <= 7.0 or not 1.0 <= near_px <= 7.0:
            raise ValueError("pixel widths must be in the inclusive range 1..7")
        if near_px < far_px:
            raise ValueError("near_px must be greater than or equal to far_px")
        if tau_s < 0.0:
            raise ValueError("tau_s must be non-negative")
        if max_age_s < 0.0:
            raise ValueError("max_age_s must be non-negative")
        if jump_px < 0.0:
            raise ValueError("jump_px must be non-negative")

        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.near_px = float(near_px)
        self.far_px = float(far_px)
        self.tau_s = float(tau_s)
        self.max_age_s = float(max_age_s)
        self.jump_px = float(jump_px)


def _is_finite_number(value):
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if np.issubdtype(array.dtype, np.floating):
        return np.logical_and(np.isfinite(array), array > 0)
    return array != 0


def thin_mask(mask):
    """Return a 0/1 uint8 Guo-Hall centerline without changing ``mask``."""
    binary = _binary_mask(mask)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "thinning"):
        raise RuntimeError("distance width rendering requires OpenCV contrib ximgproc")
    binary = binary.astype(np.uint8)
    if not binary.any():
        return binary

    # Padding gives Guo-Hall a background neighbour beyond every image edge;
    # without it, boundary pixels can remain frozen as a thick strip.
    source = np.pad(binary * 255, 1)
    result = ximgproc.thinning(
        source, thinningType=ximgproc.THINNING_GUOHALL)[1:-1, 1:-1].copy()
    result = (result != 0).astype(np.uint8)

    # Preserve one source pixel if thinning removes an entire compact component.
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    represented = np.zeros(count, dtype=bool)
    represented[np.unique(labels[result != 0])] = True
    represented[0] = True
    missing = np.flatnonzero(~represented)
    if missing.size:
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        for label in missing:
            indices = np.flatnonzero(labels == label)
            chosen = indices[np.argmax(distance.flat[indices])]
            result.flat[chosen] = 1
    return result


def render_width(centerline, width):
    """Rasterize scalar or per-pixel widths from their centerline origins.

    Radius ``r`` receives seed alpha ``clip((width - (2*r - 1))/2, 0, 1)``
    before dilation.  This prevents a wide destination pixel from thickening a
    narrow edge seed across a depth boundary.
    """
    seeds = _binary_mask(centerline)
    if np.isscalar(width):
        if not _is_finite_number(width) or not 1.0 <= float(width) <= 7.0:
            raise ValueError("width must be finite and in the range 1..7")
        widths = np.full(seeds.shape, float(width), dtype=np.float32)
    else:
        widths = np.asarray(width)
        if widths.shape != seeds.shape:
            raise ValueError("width map must have the same shape as centerline")
        if not np.all(np.isfinite(widths)):
            raise ValueError("width map must contain only finite values")
        if np.any(widths < 1.0) or np.any(widths > 7.0):
            raise ValueError("width map values must be in the range 1..7")
        widths = widths.astype(np.float32, copy=False)

    alpha = seeds.astype(np.float32)
    if not seeds.any():
        return alpha

    for radius in range(1, 4):
        weight = np.clip((widths - (2 * radius - 1)) / 2.0, 0.0, 1.0)
        weighted_seeds = np.where(seeds, weight, 0.0).astype(np.float32)
        if not weighted_seeds.any():
            continue
        size = 2 * radius + 1
        dilated = cv2.dilate(weighted_seeds, np.ones((size, size), np.uint8))
        np.maximum(alpha, dilated, out=alpha)
    return alpha


class DistanceWidthRenderer(object):
    """Convert a current TEED mask and aligned depth into an alpha mask.

    Current timestamps must be finite and may repeat.  A timestamp older than
    the previous render resets temporal state and is treated as a fresh frame;
    the resulting status is ``"ok_time_reset"`` when all depth is valid.
    Missing, invalid, stale, or future depth falls back to ``far_px`` and clears
    temporal history at the affected pixels.  ``last_status`` is one of
    ``reset``, ``ok``, ``ok_time_reset``, ``partial_invalid``, or
    ``fallback_missing|invalid|stale|future``.
    """

    def __init__(self, config=None):
        if config is not None and not isinstance(config, WidthConfig):
            raise TypeError("config must be a WidthConfig instance or None")
        self.config = config if config is not None else WidthConfig()
        self.reset()

    def reset(self):
        """Clear temporal state and public diagnostics."""
        self._history_width = None
        self._history_valid = None
        self._last_timestamp_s = None
        self.last_width = None
        self.last_status = "reset"
        self.last_timings_ms = {}

    def render(self, mask, depth_m, timestamp_s, depth_timestamp_s=None):
        """Return an HxW float32 alpha image for the current edge mask.

        ``depth_m`` is a scalar or an aligned HxW map in metres.  Reusing the
        same depth timestamp across newer image frames is supported until it
        exceeds ``max_age_s``; future depth is never sampled.
        """
        total_start = time.perf_counter()
        if not _is_finite_number(timestamp_s):
            self.reset()
            raise ValueError("timestamp_s must be finite")
        timestamp_s = float(timestamp_s)

        time_reset = (self._last_timestamp_s is not None and
                      timestamp_s < self._last_timestamp_s)
        if time_reset:
            self.reset()

        thin_start = time.perf_counter()
        centerline = thin_mask(mask)
        thin_ms = (time.perf_counter() - thin_start) * 1000.0

        width_start = time.perf_counter()
        desired, valid, status = self._desired_width(
            centerline.shape, depth_m, timestamp_s, depth_timestamp_s)
        filtered = self._filter_width(desired, valid, timestamp_s)
        if time_reset and status == "ok":
            status = "ok_time_reset"
        width_ms = (time.perf_counter() - width_start) * 1000.0

        raster_start = time.perf_counter()
        alpha = render_width(centerline, filtered)
        raster_ms = (time.perf_counter() - raster_start) * 1000.0

        self._last_timestamp_s = timestamp_s
        self.last_width = filtered
        self.last_status = status
        self.last_timings_ms = {
            "thin": thin_ms,
            "width": width_ms,
            "raster": raster_ms,
            "total": (time.perf_counter() - total_start) * 1000.0,
        }
        return alpha

    def _desired_width(self, shape, depth_m, timestamp_s, depth_timestamp_s):
        fallback = np.full(shape, self.config.far_px, dtype=np.float32)
        if depth_m is None:
            return fallback, np.zeros(shape, dtype=bool), "fallback_missing"

        if depth_timestamp_s is None:
            depth_timestamp_s = timestamp_s
        if not _is_finite_number(depth_timestamp_s):
            return fallback, np.zeros(shape, dtype=bool), "fallback_invalid"
        depth_timestamp_s = float(depth_timestamp_s)
        if depth_timestamp_s > timestamp_s:
            return fallback, np.zeros(shape, dtype=bool), "fallback_future"
        if timestamp_s - depth_timestamp_s > self.config.max_age_s:
            return fallback, np.zeros(shape, dtype=bool), "fallback_stale"

        depth = np.asarray(depth_m)
        if depth.ndim != 0 and depth.shape != shape:
            raise ValueError("depth map must have the same shape as mask")
        try:
            if depth.ndim == 0:
                scalar = float(depth)
                if not math.isfinite(scalar) or scalar <= 0.0:
                    return fallback, np.zeros(shape, dtype=bool), "fallback_invalid"
                scalar = min(max(scalar, self.config.near_m), self.config.far_m)
                depth_values = np.full(shape, scalar, dtype=np.float32)
                valid = np.ones(shape, dtype=bool)
            else:
                if np.issubdtype(depth.dtype, np.complexfloating):
                    raise ValueError("complex depths are invalid")
                if not np.issubdtype(depth.dtype, np.number):
                    depth_values = depth.astype(np.float64)
                else:
                    depth_values = depth
                valid = np.logical_and(np.isfinite(depth_values), depth_values > 0.0)
                if not valid.any():
                    return fallback, valid, "fallback_invalid"
        except (TypeError, ValueError, OverflowError):
            return fallback, np.zeros(shape, dtype=bool), "fallback_invalid"

        # Replace invalid values before arithmetic and clamp before float32
        # conversion, avoiding NaN propagation and overflow warnings.
        safe_depth = np.where(valid, depth_values, self.config.far_m)
        safe_depth = np.clip(
            safe_depth, self.config.near_m, self.config.far_m)
        fraction = ((safe_depth - self.config.near_m) /
                    (self.config.far_m - self.config.near_m))
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        desired = (self.config.near_px +
                   (self.config.far_px - self.config.near_px) * smooth)
        desired = desired.astype(np.float32, copy=False)
        if not valid.all():
            desired = desired.copy()
            desired[~valid] = self.config.far_px
            return desired, valid, "partial_invalid"
        return desired, valid, "ok"

    def _filter_width(self, desired, valid, timestamp_s):
        if (self._history_width is None or
                self._history_width.shape != desired.shape):
            filtered = desired.copy()
        else:
            dt = max(0.0, timestamp_s - self._last_timestamp_s)
            if self.config.tau_s == 0.0:
                alpha = 1.0
            else:
                alpha = -math.expm1(-dt / self.config.tau_s)
            delta = np.abs(desired - self._history_width)
            immediate = delta > self.config.jump_px
            ema = self._history_width + alpha * (desired - self._history_width)
            use_history = np.logical_and(valid, self._history_valid)
            filtered = np.where(
                use_history,
                np.where(immediate, desired, ema),
                desired).astype(np.float32, copy=False)

        self._history_width = filtered.copy()
        self._history_valid = valid.copy()
        return filtered
