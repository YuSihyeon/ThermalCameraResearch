from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import yaml

from .config import verify_configs
from .extrinsic import calibrate_extrinsic
from .io import camera_from_yaml, transform_from_yaml, load_yaml
from .projection import project_points
from .synthetic import run_self_check
from .synthetic_thermal import run_image_check
from .target import Target, load_target
from .workflow import calibrate_dataset


def _target_options(parser):
    parser.add_argument("--target", help="target YAML; default: quoted 600x400 mm, 27-circle board")
    parser.add_argument("--quality", help="quality YAML thresholds (CLI flags override)")
    parser.add_argument("--min-views", type=int, default=15)
    parser.add_argument("--max-rms", type=float, default=0.8)


def _source_options(parser):
    parser.add_argument("--source", choices=["usb", "ros"])
    parser.add_argument("--device", default="0", help="USB index on Windows; index or /dev/videoN on Linux")
    parser.add_argument("--topic", default="/boson/image_raw")
    parser.add_argument("--telemetry-rows", type=int, choices=[0, 2], default=0)


def _open_source(args):
    from .sources import USBSource, ROSSource

    return (
        ROSSource(args.topic, args.telemetry_rows)
        if args.source == "ros"
        else USBSource(args.device, args.telemetry_rows)
    )


def _wizard_options(parser):
    _source_options(parser)
    _target_options(parser)
    parser.set_defaults(source="usb" if sys.platform == "win32" else "ros")
    parser.add_argument("--images", help="native image folder/glob; otherwise ask for input")
    parser.add_argument("--output", help="new directory retaining all attempts")
    parser.add_argument("--no-browser", action="store_true", help="open review.html manually")
    parser.add_argument("--headless", action="store_true", help="ROS only; terminal consent still required")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--rectified-topic", default="/boson/image_rect")
    parser.add_argument("--camera-info-topic", default="/boson/camera_info_rect")
    parser.add_argument("--frame-timeout", type=float, default=8.0, help="stop application after stream loss (seconds)")
    parser.add_argument("--apply-frames", type=int, default=0, help="stop after N corrected frames; 0 runs until quit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firesight-calib", description="FireSight calibration tools")
    sub = parser.add_subparsers(dest="command", required=True)
    from .experiments.cli import register
    register(sub)

    wizard = sub.add_parser("wizard", help="input -> calibrate -> visual report -> ask before live application")
    _wizard_options(wizard)
    review = sub.add_parser("review", help="review a saved result and ask before live application")
    _wizard_options(review)
    review.add_argument("--calibration", required=True, help="v2 YAML with matching review report")

    check = sub.add_parser("self-check", help="run a hardware-free synthetic calibration")
    check.add_argument(
        "--output", default=None, help="new output directory (default: timestamped calibration_data session)"
    )

    doctor = sub.add_parser("doctor", help="check environment; optionally read one USB/ROS frame")
    _source_options(doctor)

    capture = sub.add_parser("capture", help="live Boson capture: SPACE save, A auto, C calibrate, Q stop")
    _source_options(capture)
    _target_options(capture)
    capture.add_argument("--output", help="new session directory")
    capture.add_argument("--auto", action="store_true")
    capture.add_argument("--headless", action="store_true")
    capture.add_argument("--calibrate", action="store_true", help="calibrate when requested count is reached")
    capture.add_argument("--count", type=int, default=24)
    capture.add_argument("--interval", type=float, default=1.0)
    capture.add_argument("--timeout", type=float, default=600.0)

    intrinsic = sub.add_parser(
        "intrinsic", help="calibrate native Boson thermal images; default is 27-circle target"
    )
    intrinsic.add_argument("--images", required=True, help="quoted glob, for example '/data/*.png'")
    intrinsic.add_argument("--output", required=True)
    _target_options(intrinsic)
    intrinsic.add_argument("--pattern", choices=["asymmetric_circles", "checkerboard"])
    intrinsic.add_argument("--corners-x", type=int)
    intrinsic.add_argument("--corners-y", type=int)
    intrinsic.add_argument("--square-m", type=float)

    extrinsic = sub.add_parser(
        "extrinsic", help="solve a camera-to-Boson transform from saved correspondences"
    )
    extrinsic.add_argument("--correspondences", required=True)
    extrinsic.add_argument("--boson-intrinsic", required=True)
    extrinsic.add_argument("--sensor-intrinsic", required=True)
    extrinsic.add_argument("--sensor-frame", required=True)
    extrinsic.add_argument("--output", required=True)

    project = sub.add_parser("project", help="project source-frame 3D points into the Boson image")
    project.add_argument("--points", required=True, help=".npy or CSV with x,y,z columns")
    project.add_argument("--intrinsic", required=True)
    project.add_argument("--extrinsic", required=True)
    project.add_argument("--output", required=True)

    verify = sub.add_parser("verify-config", help="validate sensors, target and quality YAML")
    verify.add_argument("--sensors", required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument("--quality", required=True)
    return parser


def _load_points(path: str) -> np.ndarray:
    return np.load(path) if path.lower().endswith(".npy") else np.loadtxt(path, delimiter=",", ndmin=2)


def main(argv: list[str] | None = None) -> int:
    arguments = (list(argv) if argv is not None else sys.argv[1:]) or ["wizard"]
    args = _parser().parse_args(arguments)
    try:
        exit_code = 0
        if args.command in ("intrinsic", "capture", "wizard", "review"):
            if args.quality:
                thresholds = load_yaml(args.quality).get("thresholds", {})
                if not any(v == "--min-views" or v.startswith("--min-views=") for v in arguments):
                    args.min_views = int(thresholds.get("min_intrinsic_views", 15))
                if not any(v == "--max-rms" or v.startswith("--max-rms=") for v in arguments):
                    args.max_rms = float(thresholds.get("boson_intrinsic_rms_px", 0.8))
            target = load_target(args.target)
        if args.command in ("wizard", "review"):
            from .wizard import run_wizard, review_and_apply

            if args.command == "review":
                decision = review_and_apply(args.calibration, args)
                args.images = None  # A rejection always asks for fresh input.
                result = (run_wizard(args) if decision == "retry" else
                          {"status": "applied" if decision == "applied" else "cancelled"})
            else:
                result = run_wizard(args)
        elif args.command == "experiment":
            from .experiments.cli import dispatch
            result = dispatch(args)
        elif args.command == "self-check":
            directory = Path(args.output or _session_path("self_check"))
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f"self-check output must be empty: {directory}")
            result = run_self_check(directory / "point_check")
            result["image_pipeline"] = run_image_check(directory / "thermal_check")
            (directory / "self_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "doctor":
            result = {
                "python": sys.version.split()[0],
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "native_image_size": [320, 256],
                "target": load_target().metadata(),
                "hardware_checked": False,
            }
            if args.source:
                source = _open_source(args)
                try:
                    import time

                    deadline = time.monotonic() + 8
                    frame = None
                    while frame is None and time.monotonic() < deadline:
                        frame = source.read(0.5)
                    if frame is None:
                        raise RuntimeError(
                            "no frames within 8 seconds; check USB device or ROS publisher/topic/QoS"
                        )
                    result.update(
                        hardware_checked=True,
                        shape=list(frame.shape),
                        dtype=str(frame.dtype),
                        pixel_min=int(frame.min()),
                        pixel_max=int(frame.max()),
                    )
                finally:
                    source.close()
        elif args.command == "capture":
            from .capture import run_capture

            if not args.source:
                raise ValueError("capture requires --source usb or --source ros")
            result = run_capture(
                _open_source(args),
                args.output or _session_path("boson"),
                target,
                headless=args.headless,
                auto=args.auto,
                timeout=args.timeout,
                interval=args.interval,
                count=args.count,
                min_views=args.min_views,
                calibrate=args.calibrate,
                max_rms=args.max_rms,
                source_info={
                    "source": args.source,
                    "device": args.device,
                    "topic": args.topic,
                    "telemetry_rows": args.telemetry_rows,
                },
            )
            if result.get("output"):
                quality = load_yaml(result["output"])["quality"]
                result["quality"] = quality
                exit_code = 0 if quality["passed"] else 3
        elif args.command == "intrinsic":
            legacy = any(v is not None for v in (args.corners_x, args.corners_y, args.square_m))
            if legacy or args.pattern == "checkerboard":
                if args.target or args.pattern == "asymmetric_circles":
                    raise ValueError(
                        "checkerboard flags cannot be combined with --target or asymmetric_circles"
                    )
                target = Target(
                    pattern_type="checkerboard",
                    pattern=(args.corners_x or 9, args.corners_y or 5),
                    spacing_m=args.square_m if args.square_m is not None else 0.05,
                )
            input_path = Path(args.images)
            paths = (
                sorted(str(p) for p in input_path.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff"))
                if input_path.is_dir()
                else sorted(glob.glob(args.images))
            )
            if not paths:
                raise ValueError(f"image glob matched no files: {args.images}")
            value = calibrate_dataset(paths, args.output, target, args.min_views, args.max_rms)
            quality = load_yaml(args.output)["quality"]
            result = {
                "rms_px": value.rms_px,
                "accepted_views": len(value.accepted_images),
                "output": args.output,
                "quality": quality,
            }
            exit_code = 0 if quality["passed"] else 3
        elif args.command == "extrinsic":
            result = calibrate_extrinsic(
                args.correspondences,
                args.boson_intrinsic,
                args.sensor_intrinsic,
                args.sensor_frame,
                args.output,
            )
        elif args.command == "project":
            matrix, distortion, size = camera_from_yaml(args.intrinsic)
            transform = transform_from_yaml(args.extrinsic)
            points = _load_points(args.points)
            pixels, valid = project_points(points, transform, matrix, distortion, size)
            output = np.column_stack([points, pixels, valid.astype(np.uint8)])
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(args.output, output, delimiter=",", header="x,y,z,u,v,valid", comments="")
            result = {"input_points": len(points), "valid_points": int(valid.sum()), "output": args.output}
        else:
            result = verify_configs(args.sensors, args.target, args.quality)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return exit_code
    except KeyboardInterrupt:
        print("Interrupted; saved capture samples remain available.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, cv2.error, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def home() -> int:
    print("FireSight Calibration 0.3.0 | Boson 320, 27-circle thermal target")
    print("USB: firesight-calib wizard --source usb")
    print("ROS: firesight-calib wizard --source ros")
    print("SPACE save | A auto | C calibrate | Q stop. Run firesight-calib --help for all commands.")
    return 0


def _session_path(prefix: str) -> str:
    return str(Path("calibration_data") / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))


if __name__ == "__main__":
    raise SystemExit(main())
