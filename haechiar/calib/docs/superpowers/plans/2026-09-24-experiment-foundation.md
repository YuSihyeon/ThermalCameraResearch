# Experiment foundation implementation plan

**Goal:** Hardware-independent, resumable workflows for the three requested sensor experiments.
**Architecture:** Extend `calib/firesight_calib/experiments`; reuse projection/io/geometry.
**Stack:** Python >=3.10, NumPy, OpenCV, PyYAML; optional pyrealsense2 only on device.

## Work sequence

1. `test_experiment_data.py`: failing tests for Sample validation, bounded timestamp matching,
   lossless sessions, corrupted payload rejection and group-disjoint dataset export.
   Implement `model.py`, `sync.py`, `session.py`, `dataset.py`.
2. `test_experiment_fusion.py`: failing numerical tests for depth deprojection, transform
   direction, Z-buffering, rigid fit residuals/degeneracy and synthetic/measured separation.
   Implement `calibration.py`, `fusion.py`; keep existing projection API unchanged.
3. `test_experiment_workflow.py`: failing end-to-end tests for three demo modes, config
   readiness, bounded capture cleanup, replay summaries, inference plugin contract.
   Implement `config.py`, `pipeline.py`, `adapters.py`, `capture.py`, `demo.py`, `cli.py`.
4. Register `firesight-calib experiment` in existing CLI; add profiles/inventory, Korean
   runbook, schemas and continuation checklist. Run existing + new pytest suite.
5. Independently review changes; fix actionable findings, verify clean diff, commit and
   push only the requested `YSH` branch to `HaechiAR/calib`.

## Contracts

`Sample(stream, timestamp_ns, clock, frame, kind, data, provenance, metadata)` carries an
explicit shared clock domain; image data is native, depth is floating metres, radar is
N×3 metres, IMU is acceleration(m/s²) + angular velocity(rad/s). No implicit conversion.
`SessionWriter(path, metadata).append(sample)` and `read_session(path)` round-trip arrays
without pickle. `Synchronizer(anchor, required, tolerance_ns, capacity).push(sample)`
returns complete bundles only, without reusing samples. `run_session` emits per-frame
NPZ results plus JSON metrics; no ground-truth accuracy is inferred from coverage.

## Verification

From repository root: `.venv/Scripts/python.exe -m pytest -q`.
From Linux/ROS environment: `python3 -m pytest -q` (ROS integration where available).
Run each of `experiment demo --mode rgbd_thermal|thermal_ai|radar_thermal --output <new>`.
Inspect generated reports, unsigned synthetic calibrations, raw dtypes and split assignments.
Run `git diff --check` before commit. Hardware results remain explicitly unverified.
