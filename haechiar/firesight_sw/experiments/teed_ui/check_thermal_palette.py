# ruff: noqa: INP001
from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

BGRImage: TypeAlias = NDArray[np.uint8]
BoolMask: TypeAlias = NDArray[np.bool_]
NormalizedRegion: TypeAlias = tuple[float, float, float, float]

SYNTHETIC_SHAPE = (96, 128, 3)
SMALL_SHAPE = (12, 16, 3)
REAL_SHAPE = (540, 960, 3)
GREEN_EDGE_BGR = (0, 255, 0)
BACKGROUND_BGR = (37, 41, 43)
RED_OUTLINE_BGR = (0, 0, 255)
EXPECTED_SYNTHETIC_CANDIDATES = 201
EXPECTED_SYNTHETIC_IGNORED = 56
EXPECTED_SYNTHETIC_REGION_PIXELS = 144
MIN_SYNTHETIC_OUTLINE_PIXELS = 45
IFSI10_MIN_HOT_PIXELS = 1_000
IFSI10_MIN_OUTLINE_PIXELS = 100
COLOR_BAR_LEFT = 0.78
LOGO_RIGHT = 0.26
LOGO_TOP = 0.72


@dataclass(frozen=True)
class MetricsView:
    status: str
    image_shape_hw: tuple[int, int]
    candidate_pixels: int
    ignored_pixels: int
    hot_region_pixels: int
    outline_pixels: int
    region_count: int


class ApplyThermalPalette(Protocol):
    def __call__(
        self,
        composite: BGRImage,
        source: BGRImage,
        ignore_regions: tuple[NormalizedRegion, ...] = (),
        *,
        background: BGRImage,
    ) -> tuple[BoolMask, MetricsView]:
        ...


class CheckError(RuntimeError):
    pass


def _load_apply_thermal_palette() -> ApplyThermalPalette:
    root = Path(__file__).resolve().parents[2]
    module_path = root / "edge/orin/thermal_palette.py"
    spec = importlib.util.spec_from_file_location("thermal_palette", module_path)
    if spec is None or spec.loader is None:
        message = "thermal_palette module spec unavailable"
        raise CheckError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("ApplyThermalPalette", getattr(module, "apply_thermal_palette"))


apply_thermal_palette = _load_apply_thermal_palette()


def _require(condition: object, message: str) -> None:
    if not bool(condition):
        raise CheckError(message)


def _require_array_equal(
    actual: NDArray[np.uint8],
    expected: NDArray[np.uint8] | tuple[int, int, int],
    message: str,
) -> None:
    if not np.array_equal(actual, expected):
        raise CheckError(message)


def _require_raises(
    expected: type[Exception],
    call: Callable[[], object],
) -> None:
    try:
        _ = call()
    except expected:
        return
    message = f"expected {expected.__name__}"
    raise CheckError(message)


def _make_source() -> BGRImage:
    source: BGRImage = np.zeros(SYNTHETIC_SHAPE, np.uint8)
    source[:] = (128, 128, 128)
    source[20:28, 20:28] = (0, 255, 255)
    source[22:26, 24:32] = (0, 165, 255)
    source[36:44, 48:56] = (8, 0, 255)
    source[4, 4] = (0, 255, 255)
    source[8:16, 108:115] = (0, 0, 255)
    source[50:58, 20:28] = (255, 0, 255)
    source[50:58, 36:44] = (0, 255, 0)
    source[50:58, 52:60] = (255, 0, 0)
    source[50:58, 68:76] = (255, 255, 255)
    return source


def _synthetic_outline_checks() -> list[str]:
    checks: list[str] = []
    source = _make_source()
    background: BGRImage = np.zeros_like(source)
    background[:] = BACKGROUND_BGR
    composite = np.full_like(source, GREEN_EDGE_BGR)
    original_source = source.copy()
    original_background = background.copy()
    outline, metrics = apply_thermal_palette(
        composite,
        source,
        ((0.82, 0.0, 1.0, 0.25),),
        background=background,
    )
    _require(outline.dtype == np.bool_, "outline mask should be boolean")
    _require(metrics.status == "thermal_palette_contours", "status mismatch")
    _require(metrics.image_shape_hw == SYNTHETIC_SHAPE[:2], "shape mismatch")
    _require(
        metrics.candidate_pixels == EXPECTED_SYNTHETIC_CANDIDATES,
        "candidate count mismatch",
    )
    _require(
        metrics.ignored_pixels == EXPECTED_SYNTHETIC_IGNORED,
        "ignored count mismatch",
    )
    _require(
        metrics.hot_region_pixels == EXPECTED_SYNTHETIC_REGION_PIXELS,
        "hot region count mismatch",
    )
    _require(metrics.region_count == 2, "region count mismatch")
    _require(
        metrics.outline_pixels >= MIN_SYNTHETIC_OUTLINE_PIXELS,
        "outline too small",
    )
    _require_array_equal(source, original_source, "source mutated")
    _require_array_equal(background, original_background, "background mutated")
    _require_array_equal(composite[24, 24], BACKGROUND_BGR, "hot interior filled")
    _require_array_equal(composite[23, 20], RED_OUTLINE_BGR, "yellow outline missing")
    _require_array_equal(composite[37, 48], RED_OUTLINE_BGR, "red wrap outline missing")
    _require_array_equal(composite[4, 4], GREEN_EDGE_BGR, "single-pixel noise survived")
    _require_array_equal(composite[10, 110], GREEN_EDGE_BGR, "ignored OSD changed")
    _require_array_equal(composite[70, 100], GREEN_EDGE_BGR, "outside area changed")
    _require(not outline[10, 110], "ignored OSD outline leaked")
    checks.append("synthetic hot regions render red outlines without fill")
    return checks


def _empty_and_validation_checks() -> list[str]:
    checks: list[str] = []
    source: BGRImage = np.zeros(SMALL_SHAPE, np.uint8)
    background: BGRImage = np.zeros_like(source)
    background[:] = BACKGROUND_BGR
    composite = np.full_like(source, GREEN_EDGE_BGR)
    outline, metrics = apply_thermal_palette(
        composite,
        source,
        (),
        background=background,
    )
    _require(not outline.any(), "empty outline should have no pixels")
    _require(metrics.hot_region_pixels == 0, "empty input hot pixels")
    _require(metrics.outline_pixels == 0, "empty input outline pixels")
    _require_array_equal(composite, np.full_like(source, GREEN_EDGE_BGR), "empty changed")

    _require_raises(
        ValueError,
        lambda: apply_thermal_palette(
            np.zeros((2, 2, 3), np.uint8),
            source,
            (),
            background=background,
        ),
    )
    _require_raises(
        TypeError,
        lambda: apply_thermal_palette(
            cast("BGRImage", np.zeros(SMALL_SHAPE, np.float32)),
            source,
            (),
            background=background,
        ),
    )
    _require_raises(
        ValueError,
        lambda: apply_thermal_palette(
            composite,
            source,
            ((0.5, 0.5, 0.4, 0.8),),
            background=background,
        ),
    )
    _require_raises(
        ValueError,
        lambda: apply_thermal_palette(
            composite,
            source,
            (),
            background=np.zeros((2, 2, 3), np.uint8),
        ),
    )
    checks.append("empty inputs and validation contracts hold")
    return checks


def _ignored_cut_border_check() -> list[str]:
    source: BGRImage = np.zeros(SYNTHETIC_SHAPE, np.uint8)
    source[:] = (128, 128, 128)
    source[30:70, 20:80] = (0, 165, 255)
    background: BGRImage = np.zeros_like(source)
    background[:] = BACKGROUND_BGR
    composite = np.full_like(source, GREEN_EDGE_BGR)
    outline, _ = apply_thermal_palette(
        composite,
        source,
        ((0.0, 0.5, 1.0, 1.0),),
        background=background,
    )
    _require(not outline[42:50, 25:75].any(), "ignored cut border produced outline")
    _require(outline[42:48, 20].any(), "left side outline missing")
    _require(outline[42:48, 79].any(), "right side outline missing")
    _require(outline[30:42, 20:80].any(), "remaining hot region lost outline")
    return ["ignored ROI cuts do not create artificial red border"]


def _real_snapshot_checks() -> list[str]:
    checks: list[str] = []
    image_dir = Path(__file__).with_name("additional_thermal_sources")
    ignore_regions = (
        (0.0, LOGO_TOP, LOGO_RIGHT, 1.0),
        (COLOR_BAR_LEFT, 0.0, 1.0, 1.0),
    )
    for name in ("ifsi10_source_5s.jpg", "ifsi5_source_5s.jpg"):
        loaded = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
        if loaded is None:
            raise CheckError(name)
        source = cast("BGRImage", loaded)
        _require(source.shape == REAL_SHAPE, "real snapshot shape mismatch")
        background = source.copy()
        composite = np.full_like(source, GREEN_EDGE_BGR)
        outline, metrics = apply_thermal_palette(
            composite,
            source,
            ignore_regions,
            background=background,
        )
        _require(metrics.image_shape_hw == REAL_SHAPE[:2], "real shape metric mismatch")
        _require(metrics.ignored_pixels > 0, "OSD/logo did not exclude candidates")
        if name == "ifsi10_source_5s.jpg":
            _require(
                metrics.hot_region_pixels > IFSI10_MIN_HOT_PIXELS,
                "IFSI10 warm region not retained",
            )
            _require(
                metrics.outline_pixels > IFSI10_MIN_OUTLINE_PIXELS,
                "IFSI10 outline missing",
            )
        else:
            _require(metrics.hot_region_pixels == 0, "IFSI5 scene warm pixels leaked")
            _require(metrics.outline_pixels == 0, "IFSI5 scene outline leaked")
        red_pixels = np.array(RED_OUTLINE_BGR, dtype=np.uint8)
        _require(
            bool(np.all(composite[outline] == red_pixels)),
            "red copy mismatch",
        )
        _require(
            not outline[:, int(source.shape[1] * COLOR_BAR_LEFT) :].any(),
            "color bar leaked",
        )
        logo_y0 = int(source.shape[0] * LOGO_TOP)
        logo_x1 = int(source.shape[1] * LOGO_RIGHT)
        _require(not outline[logo_y0:, :logo_x1].any(), "logo leaked")
        checks.append(f"{name} renders red outline and excludes OSD/logo regions")
    return checks


def main() -> None:
    checks = [
        *_synthetic_outline_checks(),
        *_empty_and_validation_checks(),
        *_ignored_cut_border_check(),
        *_real_snapshot_checks(),
    ]
    for check in checks:
        _ = sys.stdout.write(f"ok - {check}\n")


if __name__ == "__main__":
    main()
