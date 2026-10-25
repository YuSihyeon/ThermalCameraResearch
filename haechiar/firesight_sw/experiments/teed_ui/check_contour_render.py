from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from contour_render import render_contour_alpha  # noqa: E402
from contour_tracker import ContourPath  # noqa: E402
from edge_filter import EdgeFilterConfig  # noqa: E402
from hud import HUDConfig, HUDRenderer  # noqa: E402
from run_pipeline import build_parser  # noqa: E402
from teed_core import EdgeResult  # noqa: E402

OUT = Path(__file__).with_name("contour_stability")


def curve(x: float, opacity: float = 1.0) -> ContourPath:
    return ContourPath(
        points=np.array([[x, 10.0], [x, 50.0]], np.float32),
        alpha=opacity,
        track_id=1,
        closed=False,
    )


def main():
    cv2.setNumThreads(1)
    checks = []
    a = render_contour_alpha([curve(30.0)], (64, 96), (128, 192), (0, 0, 192, 128))
    b = render_contour_alpha([curve(30.25)], (64, 96), (128, 192), (0, 0, 192, 128))

    def center(value):
        weights = value.sum(axis=0)
        return float(weights @ np.arange(len(weights)) / weights.sum())

    assert 0.3 < center(b) - center(a) < 0.7
    checks.append("fractional model motion yields fractional display motion")
    assert np.any((a > 0) & (a < 1))
    checks.append("antialiased coverage is not a binary nearest enlargement")
    c = render_contour_alpha([curve(30)], (64, 96), (148, 212), (10, 10, 192, 128))
    assert (
        np.allclose(c[10:138, 10:202], a)
        and not np.any(c[:10])
        and not np.any(c[:, :10])
    )
    checks.append("letterbox registration preserves source pixel centers")
    blocked = render_contour_alpha(
        [curve(30)],
        (64, 96),
        (128, 192),
        (0, 0, 192, 128),
        ignore_regions=((0.25, 0, 0.40, 1),),
    )
    assert not blocked.any()
    checks.append("OSD mask clips line width and antialias fringe")
    weak = render_contour_alpha(
        [curve(30, 0.4)], (64, 96), (128, 192), (0, 0, 192, 128)
    )
    overlap = render_contour_alpha(
        [curve(30), curve(30, 0.4)], (64, 96), (128, 192), (0, 0, 192, 128)
    )
    assert float(weak.max()) < 0.45 and float(overlap.max()) >= float(a.max()) - 0.01
    checks.append("weak alpha does not erase a coincident strong line")
    empty = render_contour_alpha([], (64, 96), (128, 192), (0, 0, 192, 128))
    assert not empty.any()
    checks.append("empty frame has no residual drawing")
    spec = importlib.util.spec_from_file_location(
        "baseline_hud",
        Path(__file__).with_name("fixtures") / "previous_temporal_hud.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = module.HUDRenderer(
        module.HUDConfig(temporal_edges=True, output_size=(256, 192))
    )
    new = HUDRenderer(
        HUDConfig(temporal_edges=True, contour_edges=False, output_size=(256, 192))
    )
    source = np.zeros((96, 128, 3), np.uint8)
    source[:, 32:80] = 180
    for index in range(6):
        p = np.zeros((96, 128), np.float32)
        p[20:80, 31 + index % 2 : 34 + index % 2] = 0.8
        edge = EdgeResult(p, (p > 0.75).astype(np.uint8) * 255, source, (128, 96), {})
        assert np.array_equal(
            old.render(source, edge, frame_index=index, timestamp_s=index / 30).image,
            new.render(source, edge, frame_index=index, timestamp_s=index / 30).image,
        )
    checks.append("disabled contour path pixel-identical to previous HUD for 6 frames")

    def render_depth(depth):
        renderer = HUDRenderer(
            HUDConfig(
                contour_edges=True,
                background_scale=0,
                output_size=(256, 192),
                edge_filter=EdgeFilterConfig(gradient_low=0, gradient_high=0),
            )
        )
        return renderer.render(
            source, edge, depth_m=depth, frame_index=0, timestamp_s=0
        ).image[:150, :, 1]

    near = render_depth(np.full((96, 128), 0.5, np.float32))
    far = render_depth(np.full((96, 128), 8, np.float32))
    unknown = render_depth(np.full((96, 128), np.nan, np.float32))
    assert (
        near.max() > 200
        and 0.20 < far.sum() / near.sum() < 0.30
        and np.array_equal(near, unknown)
    )
    checks.append(
        "valid far depth dims contours; absent depth does not invent distance"
    )
    for high, value in [(0.60, 0.65), (0.20, 0.22)]:
        custom = HUDRenderer(
            HUDConfig(
                contour_edges=True,
                edge_filter=EdgeFilterConfig(
                    low_threshold=0.1,
                    high_threshold=high,
                    gradient_low=0,
                    gradient_high=0,
                ),
            )
        )
        probability = np.zeros((96, 128), np.float32)
        probability[20:37, 33] = value
        custom_edge = EdgeResult(probability, edge.mask, source, (128, 96), {})
        current = custom.render(source, custom_edge, frame_index=0)
        assert current.edge_mask is not None and current.edge_mask.any()
    checks.append("custom low thresholds retain immediate contour display")
    parser = build_parser()
    assert not parser.parse_args([]).contour_edges
    assert parser.parse_args(["--contour-edges"]).contour_edges
    assert not parser.parse_args(["--no-contour-edges"]).contour_edges
    checks.append("CLI keeps existing default and offers explicit contour and baseline flags")
    (OUT / "render_checks.json").write_text(
        json.dumps({"status": "PASS", "checks": checks}, indent=2)
    )
    print("PASS:", len(checks), "contour render and integration contracts")


if __name__ == "__main__":
    main()
