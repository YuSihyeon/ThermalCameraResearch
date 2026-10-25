from __future__ import annotations

import importlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import MODES
from .fusion import fuse
from .session import read_session, write_json, sha256
from .sync import Synchronizer


def load_edge_model(spec, options=None):
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("edge model must be module:factory")
    module, name = spec.split(":")
    model = getattr(importlib.import_module(module), name)(options or {})
    if not callable(getattr(model, "predict", None)) or not isinstance(getattr(model, "name", None), str):
        raise ValueError("edge model requires name and predict(native_thermal) -> probabilities")
    return model


def run_session(session, output, mode, registration=None, tolerance_ms=30., capacity=32, edge_model=None):
    if mode not in MODES or not np.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("invalid experiment mode/tolerance")
    if mode != "thermal_ai" and registration is None:
        raise ValueError("fusion requires registration; capture can run before calibration")
    session, output = Path(session), Path(output)
    document = json.loads((session / "session.json").read_text(encoding="utf-8"))
    if document.get("metadata", {}).get("mode", mode) != mode:
        raise ValueError("requested mode differs from session mode")
    sync = Synchronizer("thermal", MODES[mode], round(tolerance_ms * 1e6), capacity)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "firesight.experiment_result/v1", "mode": mode, "frames": 0,
              "status": "running", "hardware_verified": False, "provenance": None,
              "session": str(session.resolve()), "index_sha256": sha256(session / "samples.jsonl"),
              "tolerance_ms": tolerance_ms, "capacity": capacity,
              "calibration": registration.document() if registration else None,
              "metrics_meaning": "coverage/timing only; no ground-truth accuracy or device FPS"}
    write_json(output / "summary.json", report)
    timings, skews, coverages, counts = [], [], [], {}
    try:
        with (output / "frames.jsonl").open("x", encoding="utf-8") as metrics:
            for sample in read_session(session):
                counts[sample.stream] = counts.get(sample.stream, 0) + 1
                report["provenance"] = sample.provenance
                for bundle in sync.push(sample):
                    start = time.perf_counter()
                    result = fuse(bundle, mode, registration, edge_model)
                    elapsed = (time.perf_counter() - start) * 1000
                    number = report["frames"]
                    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
                    np.savez_compressed(output / f"frame_{number:06d}.npz", **arrays)
                    stamps = [item.timestamp_ns for item in bundle.values()]
                    skew = (max(stamps) - min(stamps)) / 1e6
                    coverage = float(np.isfinite(result["depth_m"]).mean()) if "depth_m" in result else None
                    row = {"index": number, "timestamp_ns": bundle["thermal"].timestamp_ns,
                           "stream_timestamps_ns": {k: v.timestamp_ns for k, v in bundle.items()},
                           "skew_ms": skew, "processing_ms": elapsed, "depth_coverage": coverage,
                           "edge_pixels": int(np.count_nonzero(result["edges"])),
                           "edge_method": result["edge_method"]}
                    metrics.write(json.dumps(row, allow_nan=False) + "\n")
                    if number == 0:
                        preview = cv2.cvtColor(result["view"], cv2.COLOR_GRAY2BGR)
                        preview[result["edges"] > 0] = [0, 255, 0]
                        if not cv2.imwrite(str(output / "preview.png"), preview):
                            raise OSError("cannot save preview")
                    timings.append(elapsed)
                    skews.append(skew)
                    if coverage is not None:
                        coverages.append(coverage)
                    report["frames"] += 1
        report.update(status="complete" if report["frames"] else "no_matches", input_samples=counts,
                      dropped_samples=sync.dropped, unmatched_samples=sum(len(q) for q in sync.queues.values()),
                      processing_ms_mean=float(np.mean(timings)) if timings else None,
                      skew_ms_max=max(skews) if skews else None,
                      depth_coverage_mean=float(np.mean(coverages)) if coverages else None)
        write_json(output / "summary.json", report)
        if not report["frames"]:
            raise ValueError("no synchronized frames; check required streams, clocks and tolerance")
        return report
    except BaseException as exc:
        report.update(status="interrupted", error=str(exc))
        write_json(output / "summary.json", report)
        raise
