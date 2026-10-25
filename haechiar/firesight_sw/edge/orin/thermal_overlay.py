"""Absolute-temperature rendering. Display brightness is never a thermometer."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ThermalConfig:
    mode: str = "contours"
    levels_c: tuple[float, float, float] = (150.0, 260.0, 400.0)
    fill_alpha: float = 0.22

    def __post_init__(self) -> None:
        if self.mode not in {"off", "contours", "overlay"}:
            raise ValueError("thermal mode must be off, contours or overlay")
        levels = np.asarray(self.levels_c, dtype=float)
        if (
            levels.shape != (3,)
            or not np.isfinite(levels).all()
            or not np.all(np.diff(levels) > 0)
        ):
            raise ValueError("three finite, increasing temperature levels are required")
        if not np.isfinite(self.fill_alpha) or not 0 <= self.fill_alpha <= 1:
            raise ValueError("thermal fill_alpha must be within [0, 1]")


def draw_temperature(
    view: np.ndarray,
    temperature_c: np.ndarray | None,
    config: ThermalConfig,
    ignore_regions: tuple = (),
) -> dict[str, object]:
    """Draw into view using a registered Celsius map, never an AGC/display frame.

    Levels are configurable engineering parameters, not NFPA temperature limits
    or a validated fire detector. NaN/Inf pixels remain unknown.
    """
    if config.mode == "off":
        return {"status": "disabled", "hot_pixels": 0, "contour_pixels": 0}
    if temperature_c is None:
        return {
            "status": "temperature_unavailable",
            "hot_pixels": 0,
            "contour_pixels": 0,
        }
    temperature = np.asarray(temperature_c)
    if (
        temperature.ndim != 2
        or temperature.shape != view.shape[:2]
        or temperature.dtype.kind != "f"
    ):
        raise ValueError(
            "temperature_c must be a float Celsius map registered to the view"
        )
    valid = np.isfinite(temperature)
    height, width = valid.shape
    for left, top, right, bottom in ignore_regions:
        valid[
            round(top * height) : round(bottom * height) + 1,
            round(left * width) : round(right * width) + 1,
        ] = False
    if not valid.any():
        return {
            "status": "temperature_unavailable",
            "hot_pixels": 0,
            "contour_pixels": 0,
        }
    hot = valid & (temperature >= config.levels_c[0])
    colours = np.array(((0, 255, 255), (0, 140, 255), (0, 0, 255)), np.float32)
    if config.mode == "overlay" and hot.any():
        values = temperature[hot]
        heat_colours = np.stack(
            [
                np.interp(values, config.levels_c, colours[:, channel])
                for channel in range(3)
            ],
            axis=1,
        )
        view[hot] = np.rint(
            view[hot] * (1 - config.fill_alpha) + heat_colours * config.fill_alpha
        ).astype(np.uint8)
    contours_all = np.zeros(valid.shape, np.uint8)
    for level, colour in zip(config.levels_c, colours):
        region = (valid & (temperature >= level)).astype(np.uint8)
        # Restrict strokes to measured hot pixels; do not paint outside thresholds.
        eroded = cv2.erode(
            region,
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        boundary = (region > 0) & (eroded == 0)
        view[boundary] = colour.astype(np.uint8)
        contours_all[boundary] = 255
    return {
        "status": "calibrated_temperature",
        "hot_pixels": int(np.count_nonzero(hot)),
        "contour_pixels": int(np.count_nonzero(contours_all)),
        "valid_temperature_pixels": int(np.count_nonzero(valid)),
        "levels_c": list(config.levels_c),
    }
