from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import save_yaml
from .calibration import Registration, fit_rigid
from .config import MODES, load_profile, readiness
from .session import read_session


def register(subparsers):
    experiment = subparsers.add_parser("experiment", help="multi-sensor collection, calibration and replay")
    commands = experiment.add_subparsers(dest="experiment_command", required=True)
    status = commands.add_parser("status", help="list configured and pending prerequisites")
    status.add_argument("--profile", required=True, help="built-in mode name or profile YAML")
    demo = commands.add_parser("demo", help="generated data; no device or AI checkpoint required")
    demo.add_argument("--mode", choices=MODES, required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--frames", type=int, default=12)
    capture = commands.add_parser("capture", help="record device plugins concurrently")
    capture.add_argument("--profile", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--duration", type=float, default=60.)
    capture.add_argument("--max-samples", type=int, default=0)
    verify = commands.add_parser("verify-session", help="verify checksums, types and timestamps")
    verify.add_argument("--session", required=True)
    run = commands.add_parser("run", help="process saved synchronized data")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--session", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--registration")
    run.add_argument("--tolerance-ms", type=float, default=30., help="software trial setting, validate on rig")
    run.add_argument("--queue-size", type=int, default=32)
    run.add_argument("--edge-model", help="module:factory; default is explicitly non-AI Canny")
    run.add_argument("--model-options", help="JSON file passed to edge model factory")
    dataset = commands.add_parser("dataset", help="export group-disjoint unlabeled thermal manifest")
    dataset.add_argument("--sessions", nargs="+", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--seed", type=int, default=42)
    rigid = commands.add_parser("fit-rigid", help="fit sensor-to-thermal 3D target correspondences")
    rigid.add_argument("--pairs", required=True, help="NPZ: source_xyz_m and target_xyz_m")
    rigid.add_argument("--source-frame", required=True)
    rigid.add_argument("--target-frame", required=True)
    rigid.add_argument("--output", required=True)


def dispatch(args):
    command = args.experiment_command
    if command == "status":
        return readiness(load_profile(args.profile))
    if command == "demo":
        from .demo import run_demo
        return run_demo(args.mode, args.output, args.frames)
    if command == "capture":
        from .capture import capture_sources
        profile = load_profile(args.profile)
        report = readiness(profile)
        if not report["can_capture"]:
            raise ValueError("; ".join(report["capture_blockers"]))
        metadata = {k: v for k, v in profile.items() if not k.startswith("_")}
        return capture_sources(profile["sources"], args.output, metadata, args.duration, args.max_samples)
    if command == "verify-session":
        counts = {}
        for sample in read_session(args.session):
            counts[sample.stream] = counts.get(sample.stream, 0) + 1
        return {"verified": True, "sample_counts": counts, "hardware_verified": False}
    if command == "run":
        from .pipeline import run_session, load_edge_model
        registration = Registration.load(args.registration) if args.registration else None
        options = json.loads(Path(args.model_options).read_text(encoding="utf-8")) if args.model_options else {}
        if options and not args.edge_model:
            raise ValueError("--model-options requires --edge-model")
        model = load_edge_model(args.edge_model, options) if args.edge_model else None
        try:
            return run_session(args.session, args.output, args.mode, registration,
                               args.tolerance_ms, args.queue_size, model)
        finally:
            if callable(getattr(model, "close", None)):
                model.close()
    if command == "dataset":
        from .dataset import export_dataset
        return export_dataset(args.sessions, args.output, args.seed)
    if command == "fit-rigid":
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        with np.load(args.pairs, allow_pickle=False) as data:
            result = fit_rigid(data["source_xyz_m"], data["target_xyz_m"])
        result.update(schema="firesight.rig_transform/v1", source_frame=args.source_frame,
                      target_frame=args.target_frame, correspondence_file=str(Path(args.pairs).resolve()))
        save_yaml(args.output, result)
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    raise ValueError("unknown experiment command")
