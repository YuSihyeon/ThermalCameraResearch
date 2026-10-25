#!/usr/bin/env python3
"""Replay -> TEED -> HUD -> Display.

Wires the four modules together and records honest metrics. Every number in the
metrics file is measured or read from the source; nothing about the camera is
assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from aligned_maps import AlignedMaps
from contour_tracker import ContourConfig
from display import Display, list_displays
from edge_filter import EdgeFilterConfig
from hud import HUDConfig, HUDRenderer
from replay import ReplaySource
from teed_core import TEEDCore
from temporal_edges import TemporalEdgeConfig
from thermal_overlay import ThermalConfig

PROTOCOL = "firesight_edge_pipeline_v1"
REPO_ROOT = Path(__file__).resolve().parent


def default_checkpoint() -> str:
    """Prefer the checkpoint the teed package already ships, so edge/orin and
    edge/teed share one file instead of each keeping a copy."""
    shared = REPO_ROOT.parent / "teed" / "models" / "5_model.pth"
    local = REPO_ROOT / "models" / "teed" / "5_model.pth"
    return str(shared if shared.is_file() else local)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_argument_group("source")
    source.add_argument(
        "--source", help="video, image, directory, camera index, or /dev/videoN"
    )
    source.add_argument(
        "--input-format", choices=("auto", "bgr", "gray8", "y16"), default="auto"
    )
    source.add_argument("--capture-width", type=int, default=0)
    source.add_argument("--capture-height", type=int, default=0)
    source.add_argument("--capture-fps", type=float, default=0.0)
    source.add_argument(
        "--fourcc",
        default="",
        help="camera codec FOURCC, e.g. MJPG. Uncompressed YUYV throttles USB webcams",
    )
    source.add_argument(
        "--loop", action="store_true", help="restart video sources at the end"
    )
    source.add_argument(
        "--pace", action="store_true", help="replay at the source frame rate"
    )
    source.add_argument("--start-frame", type=int, default=0)
    source.add_argument("--max-frames", type=int)
    source.add_argument(
        "--stereo-half",
        choices=("left", "right"),
        help="take one eye from a side-by-side stereo camera",
    )
    source.add_argument("--profiles", help="Fire360 video_profiles.json")
    source.add_argument(
        "--video-key", help="profile key when the filename does not match"
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--checkpoint",
        default=default_checkpoint(),
        help="TEED checkpoint; defaults to the shared edge/teed/models/5_model.pth",
    )
    model.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    model.add_argument("--half", action="store_true", help="FP16, CUDA only")
    model.add_argument("--width", type=int, default=320)
    model.add_argument("--height", type=int, default=256)
    model.add_argument("--threshold", type=float, default=0.75)
    model.add_argument("--edge-width", type=int, default=3)
    model.add_argument(
        "--contrast-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE strength for low-visibility model input; 0 disables",
    )
    model.add_argument("--allow-unknown-checkpoint", action="store_true")
    model.add_argument("--warmup", type=int, default=5)
    model.add_argument(
        "--legacy-edges", action="store_true", help="unfiltered main baseline"
    )
    model.add_argument("--temporal-edges", dest="temporal_edges", action="store_true",
                       help="motion-compensated TEED probability stabilization (default)")
    model.add_argument("--no-temporal-edges", dest="temporal_edges", action="store_false",
                       help="disable probability accumulation; contour tracking is a separate switch")
    model.add_argument("--temporal-history-weight", type=float, default=0.6,
                       help="previous evidence weight at 30 fps, in [0, 1)")
    parser.set_defaults(temporal_edges=True, contour_edges=False)
    model.add_argument("--contour-edges", dest="contour_edges", action="store_true",
                       help="enable experimental contour tracking and antialiased drawing")
    model.add_argument("--no-contour-edges", dest="contour_edges", action="store_false",
                       help="use previous probability-only stabilization")
    model.add_argument("--motion-separated-edges", action="store_true",
                       help="enable contour tracking with separate static/dynamic motion policies")
    model.add_argument("--contour-line-width", type=float, default=1.5,
                       help="antialiased contour width in output pixels, [0.5, 8]")
    model.add_argument("--edge-low", type=float, default=0.55)
    model.add_argument(
        "--min-edge-length",
        type=float,
        default=12.0,
        help="skeleton length in model pixels",
    )
    model.add_argument(
        "--gradient-low",
        type=float,
        default=2.0,
        help="display-intensity gradient, not Celsius",
    )
    model.add_argument("--gradient-high", type=float, default=6.0)
    model.add_argument("--min-branch-length", type=float, default=6.0,
                       help="remove endpoint-to-junction branches shorter than this; 0 disables")
    model.add_argument("--min-edge-extent", type=float, default=8.0,
                       help="minimum component bounding-box diagonal in model pixels; 0 disables")
    model.add_argument("--near-m", type=float, default=0.6)
    model.add_argument("--far-m", type=float, default=6.0)
    model.add_argument("--far-alpha", type=float, default=0.25)
    source.add_argument(
        "--aligned-maps", help="registered Celsius/depth NPZ tied to source SHA-256"
    )

    hud = parser.add_argument_group("hud")
    hud.add_argument("--display-width", type=int, help="HUD output width, e.g. 1920")
    hud.add_argument("--display-height", type=int, help="HUD output height, e.g. 1080")
    hud.add_argument("--background-scale", type=float, default=0.24)
    hud.add_argument(
        "--thermal-mode", choices=("off", "contours", "overlay"), default="contours"
    )
    hud.add_argument(
        "--heat-levels-c",
        type=float,
        nargs=3,
        default=(150.0, 260.0, 400.0),
        help="engineering display levels, not validated safety thresholds",
    )
    hud.add_argument(
        "--thermal-palette-overlay", action="store_true",
        help="opt in for known thermal-camera display video: preserve warm colors and outline them in red, "
        "exclude supplied OSD regions, never infer temperature",
    )
    hud.add_argument(
        "--visible-flame-candidates", action="store_true",
        help="opt in for true visible-light video only (not false-colour thermal); "
        "mark temporal flame candidates, never infer temperature",
    )
    hud.add_argument(
        "--no-fire-highlight", action="store_true",
        help="disable temperature overlays, camera palette contours, and visible flame candidates",
    )
    hud.add_argument(
        "--show-panel", action="store_true", help="show developer diagnostics"
    )
    hud.add_argument("--no-panel", dest="show_panel", action="store_false")
    parser.set_defaults(show_panel=False)
    hud.add_argument(
        "--see-through",
        action="store_true",
        help="black background and no panel: on an optical see-through display "
        "black emits no light, so only the edges appear over the real world",
    )
    hud.add_argument(
        "--edge-color", choices=("green", "cyan", "white", "amber"), default="green"
    )
    hud.add_argument(
        "--soft-background",
        dest="full_res_background",
        action="store_false",
        help="draw the background from the model-size frame (faster, blurrier)",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--headless", action="store_true")
    output.add_argument(
        "--fullscreen", action="store_true", help="for the nreal Light screen"
    )
    output.add_argument(
        "--list-displays", action="store_true", help="print screens and exit"
    )
    output.add_argument(
        "--screen",
        help="screen index from --list-displays, or 'auto' for the first "
        "non-main screen; sets window position and size automatically",
    )
    output.add_argument(
        "--window-x", type=int, help="explicit window origin; --screen is easier"
    )
    output.add_argument("--window-y", type=int)
    output.add_argument("--record", action="store_true")
    output.add_argument("--record-path", default="firesight_hud.mp4")
    output.add_argument(
        "--record-fps", type=float, help="default: the source frame rate"
    )
    output.add_argument(
        "--metrics", default="firesight_metrics.json", help="empty string disables"
    )
    return parser


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


def print_displays() -> None:
    displays = list_displays()
    if not displays:
        print("display enumeration is unavailable here; use --window-x / --window-y")
        return
    print("index  origin           size          ")
    for entry in displays:
        note = (
            "  <- main"
            if entry["is_main"]
            else "  <- use --screen {}".format(entry["index"])
        )
        print(
            "{:>5}  ({:>5}, {:>5})   {:>5} x {:<5}{}".format(
                entry["index"],
                entry["x"],
                entry["y"],
                entry["width"],
                entry["height"],
                note,
            )
        )


def run(args: argparse.Namespace) -> dict:
    if not args.source:
        raise ValueError("--source is required (or use --list-displays)")
    if args.no_fire_highlight:
        args.thermal_mode = "off"
        args.visible_flame_candidates = False
        args.thermal_palette_overlay = False
    if args.thermal_palette_overlay and args.visible_flame_candidates:
        raise ValueError("thermal palette and visible-flame modes are mutually exclusive")
    if args.visible_flame_candidates and (
        args.input_format in {"gray8", "y16"} or args.loop or args.aligned_maps
    ):
        raise ValueError(
            "visible flame candidates require continuous visible-light BGR video "
            "without --loop or --aligned-maps"
        )
    display_size = None
    if args.display_width and args.display_height:
        display_size = (args.display_width, args.display_height)
    elif args.display_width or args.display_height:
        raise ValueError("--display-width and --display-height must be given together")

    edge_filter = EdgeFilterConfig(
        low_threshold=args.edge_low,
        high_threshold=args.threshold,
        min_length=args.min_edge_length,
        min_branch_length=args.min_branch_length,
        min_extent=args.min_edge_extent,
        gradient_low=args.gradient_low,
        gradient_high=args.gradient_high,
        near_m=args.near_m,
        far_m=args.far_m,
        far_alpha=args.far_alpha,
    )
    temporal = TemporalEdgeConfig(
        history_weight=args.temporal_history_weight,
        immediate_threshold=args.threshold,
        support_threshold=min(0.25, args.threshold),
    )
    contour = ContourConfig(
        high_threshold=args.threshold, support_threshold=min(0.25, args.threshold),
        min_output_length=args.min_edge_length, min_output_extent=args.min_edge_extent,
        motion_separation=args.motion_separated_edges,
    )
    thermal = ThermalConfig(mode=args.thermal_mode, levels_c=tuple(args.heat_levels_c))
    if args.aligned_maps and (args.stereo_half or args.loop or args.start_frame != 0):
        raise ValueError("aligned replay maps do not support --stereo-half, --loop or --start-frame")
    maps = AlignedMaps(args.aligned_maps, args.source) if args.aligned_maps else None
    core = TEEDCore(
        checkpoint=Path(args.checkpoint),
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        edge_width=args.edge_width,
        contrast_clip_limit=args.contrast_clip_limit,
        device=args.device,
        half=args.half,
        verify_checkpoint=not args.allow_unknown_checkpoint,
    )
    edge_colors = {
        "green": (0, 255, 0),
        "cyan": (255, 255, 0),
        "white": (255, 255, 255),
        "amber": (0, 200, 255),
    }
    renderer = HUDRenderer(
        HUDConfig(
            output_size=display_size,
            background_scale=0.0 if args.see_through else args.background_scale,
            edge_color=edge_colors[args.edge_color],
            edge_filter=edge_filter,
            thermal=thermal,
            visible_flame_candidates=args.visible_flame_candidates,
            thermal_palette_overlay=args.thermal_palette_overlay,
            filter_edges=not args.legacy_edges,
            temporal_edges=args.temporal_edges,
            temporal_config=temporal,
            contour_edges=args.contour_edges or args.motion_separated_edges,
            contour_config=contour,
            contour_line_width=args.contour_line_width,
            show_panel=args.show_panel and not args.see_through,
            full_res_background=args.full_res_background,
        )
    )

    source = ReplaySource(
        uri=args.source,
        input_format=args.input_format,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_fps=args.capture_fps,
        fourcc=args.fourcc,
        loop=args.loop,
        pace=args.pace,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        profile_path=args.profiles,
        video_key=args.video_key,
        stereo_half=args.stereo_half,
    )

    window_position = None
    window_size = None
    if args.screen is not None:
        displays = list_displays()
        if str(args.screen).lower() == "auto":
            match = next((d for d in displays if not d["is_main"]), None)
            if match is None:
                raise ValueError(
                    "no external screen is attached. If the glasses are plugged "
                    "in, their display may have slept - put them on to wake it, "
                    "then check with --list-displays."
                )
        else:
            try:
                wanted = int(args.screen)
            except ValueError:
                raise ValueError("--screen takes a number or 'auto'") from None
            match = next((d for d in displays if d["index"] == wanted), None)
        if match is None:
            available = (
                ", ".join(
                    "{} ({}x{}{})".format(
                        d["index"],
                        d["width"],
                        d["height"],
                        ", main" if d["is_main"] else "",
                    )
                    for d in displays
                )
                or "none"
            )
            raise ValueError(
                f"screen {args.screen} not found. Attached: {available}. "
                "If the glasses are plugged in, their display may have slept - "
                "put them on to wake it."
            )
        window_position = (match["x"], match["y"])
        window_size = (match["width"], match["height"])
        if display_size is None:
            # Render the HUD at the screen's own resolution unless told otherwise.
            display_size = window_size
            renderer.config.output_size = display_size
        print(
            "[display] screen {} at ({}, {}) {}x{}".format(
                args.screen, match["x"], match["y"], match["width"], match["height"]
            )
        )
    elif args.window_x is not None and args.window_y is not None:
        window_position = (args.window_x, args.window_y)

    display = Display(
        title="FireSight Edge",
        headless=args.headless,
        fullscreen=args.fullscreen,
        window_position=window_position,
        window_size=window_size,
        record_path=args.record_path,
        record_fps=args.record_fps or source.playback_fps,
        recording=args.record,
    )

    if args.fullscreen and sys.platform == "darwin":
        print(
            "[display] warning: OpenCV fullscreen renders black on some macOS "
            "setups. If the screen stays black, drop --fullscreen; a window "
            "sized to the screen covers it apart from the title bar."
        )
    print("[core]   {}".format(json.dumps(core.describe(), ensure_ascii=False)))
    print("[source] {}".format(json.dumps(source.info.as_dict(), ensure_ascii=False)))
    if source.info.osd_regions:
        print(
            "[source] masking {} burned-in OSD regions".format(
                len(source.info.osd_regions)
            )
        )

    history: dict[str, list[float]] = {
        key: []
        for key in (
            "preprocess",
            "encoder",
            "edge_head",
            "model",
            "postprocess",
            "hud",
            "e2e",
        )
    }
    edge_records: list[dict] = []
    heat_records: list[dict] = []
    flame_records: list[dict] = []
    palette_records: list[dict] = []
    recent: list[float] = []
    # The on-screen p95 is refreshed periodically over a bounded window.
    # Recomputing it from the full history every frame makes the profiler
    # itself grow into the bottleneck it is meant to measure.
    rolling_p95 = {"model": 0.0, "e2e": 0.0}
    p95_window = 300
    p95_interval = 15
    processed = 0
    warmed = False
    started_at = time.perf_counter()

    try:
        with source, display:
            for frame in source.frames():
                if not warmed:
                    for _ in range(max(0, args.warmup)):
                        core.infer(frame.image, source.info.osd_regions)
                    warmed = True
                    started_at = time.perf_counter()

                frame_started = time.perf_counter()
                if frame.discontinuity:
                    renderer.reset_temporal()
                edge = core.infer(frame.image, source.info.osd_regions)

                stats: dict[str, object] = dict(edge.timings_ms)
                stats.update(
                    {
                        "frame": frame.index,
                        "source": "{} {}x{}".format(
                            source.info.kind, source.info.width, source.info.height
                        ),
                        "fps": (len(recent) - 1) / (recent[-1] - recent[0])
                        if len(recent) > 1 and recent[-1] > recent[0]
                        else 0.0,
                        "p95_model": rolling_p95["model"],
                        "p95_e2e": rolling_p95["e2e"],
                        "recording": display.recording,
                    }
                )
                temperature_c, depth_m = (
                    (None, None)
                    if maps is None
                    else maps.at(frame.index, frame.image.shape[:2])
                )
                hud_frame = renderer.render(
                    frame.image,
                    edge,
                    stats,
                    pixel_origin=frame.pixel_origin,
                    temperature_c=temperature_c,
                    depth_m=depth_m,
                    ignore_regions=source.info.osd_regions,
                    frame_index=frame.index,
                    timestamp_s=(
                        frame_started if source.info.kind in {"camera", "v4l2"}
                        else frame.source_seconds if source.info.fps > 0 else None
                    ),
                )
                e2e_ms = (time.perf_counter() - frame_started) * 1000.0

                for key in (
                    "preprocess",
                    "encoder",
                    "edge_head",
                    "model",
                    "postprocess",
                ):
                    history[key].append(edge.timings_ms[key])
                history["hud"].append(hud_frame.render_ms)
                history["e2e"].append(e2e_ms)
                edge_records.append(hud_frame.edge_metrics)
                heat_records.append(hud_frame.thermal_metrics)
                flame_records.append(hud_frame.flame_metrics)
                palette_records.append(hud_frame.palette_metrics)
                processed += 1
                if processed % p95_interval == 0:
                    for key in rolling_p95:
                        window = history[key][-p95_window:]
                        rolling_p95[key] = float(np.percentile(window, 95))

                now = time.perf_counter()
                recent.append(now)
                if len(recent) > 30:
                    del recent[0]

                key = display.show(hud_frame.image)
                if key in (27, ord("q")):
                    break
                if key == ord("f"):
                    print("fullscreen:", display.toggle_fullscreen())
                if key == ord("r"):
                    state = display.toggle_recording()
                    print(
                        "recording {}: {}".format(
                            "started" if state else "stopped", args.record_path
                        )
                    )
                if args.max_frames is not None and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted")

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    payload = {
        "protocol": PROTOCOL,
        "core": core.describe(),
        "source": source.info.as_dict(),
        "hud": {
            "output_size": list(display_size)
            if display_size
            else list(edge.size)
            if processed
            else None,
            "thermal_mode": args.thermal_mode,
            "heat_levels_c": list(args.heat_levels_c),
            "temperature_provenance": maps.provenance if maps else {},
            "hot_pixels": sum(int(record["hot_pixels"]) for record in heat_records),
            "thermal_statuses": sorted({record["status"] for record in heat_records}),
            "edge_filter_enabled": not args.legacy_edges,
            "edge_filter": vars(edge_filter),
            "temporal_edges_enabled": args.temporal_edges and not args.legacy_edges,
            "temporal_config": vars(temporal),
            "contour_edges_enabled": (args.contour_edges or args.motion_separated_edges) and not args.legacy_edges,
            "motion_separation_enabled": args.motion_separated_edges and not args.legacy_edges,
            "contour_config": vars(contour),
            "contour_line_width": args.contour_line_width,
            "edge_frames": edge_records,
            "thermal_frames": heat_records,
            "thermal_palette_overlay_enabled": args.thermal_palette_overlay,
            "thermal_palette_frames": palette_records,
            "camera_hot_region_pixels": sum(int(record["hot_region_pixels"]) for record in palette_records),
            "camera_hot_outline_pixels": sum(int(record["outline_pixels"]) for record in palette_records),
            "visible_flame_candidates_enabled": args.visible_flame_candidates,
            "visible_flame_frames": flame_records,
        },
        "processed_frames": processed,
        "effective_fps": processed / elapsed,
        "latency_ms": {key: summarize(values) for key, values in history.items()},
    }
    if args.metrics:
        Path(args.metrics).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    try:
        parsed = build_parser().parse_args(argv)
        if parsed.list_displays:
            print_displays()
            return 0
        run(parsed)
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write("error: {}\n".format(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
