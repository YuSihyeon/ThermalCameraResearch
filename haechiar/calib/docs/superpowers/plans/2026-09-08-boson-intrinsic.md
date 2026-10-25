# Boson intrinsic implementation plan

**Goal:** Clone, install and capture/calibrate Boson 320 using the quoted 27-circle board on Windows USB or Ubuntu ROS 2.
**Architecture:** Shared target/detection/calibration/session modules; optional source adapters; CLI and platform launchers.
**Tech stack:** Python >=3.10, NumPy, OpenCV >=4.10,<5 (pip), PyYAML; Ubuntu system OpenCV for ROS Humble.

## Constraints

Original 320×256 mono8/mono16 samples remain unchanged. Board coordinates are meters and remain provisional. Never infer camera intrinsics from nominal lens/FOV. No external-calibration feature expansion. Hardware-free tests do not establish hardware capture success.

## Tasks

- [x] 1. Target and detector: add `target.py`, `test_thermal_target.py`; connect `config/target.yaml`. Assert `(27,3)` object points match x=.06+.06i,y=.05+.06(2j+i%2). Render 320×256 circles in 16-bit, inverted, rotated and noisy views. Run failing tests, implement validated target loading and circle detection, rerun.
- [x] 2. Intrinsics and quality: extend `intrinsic.py`, add `quality.py`, `test_intrinsic_workflow.py`. Exercise real synthetic image detection through YAML output, image-size rejection, duplicates, degenerate observations and threshold failures. Export target metadata, per-frame report, coverage and undistorted previews.
- [x] 3. Capture: add `capture.py`, `sources.py`, `test_capture.py`. Test padded/endian ROS Image decoding, original uint16 PNG persistence, duplicate gating and shutdown with fake source boundaries. Implement USB DSHOW Y16, ROS sensor-data QoS adapter, interactive capture and headless timed capture. Real source acquisition remains hardware-dependent.
- [x] 4. CLI/install: extend `cli.py`, add root packaging, `run.ps1`, `run.sh`, `test_cli.py`. Test defaults and actionable errors, install from root into a clean venv, execute the Windows launcher. Document apt/ROS prerequisites and both hardware commands.
- [x] 5. Verification: run `python -m pytest calib/test -q`, synthetic image self-check, root wheel install smoke test, config verification, diff check and code review. Update README with commands and actual test limits; keep all work on `codex/boson-intrinsic`.

Run each test suite with `.venv/Scripts/python.exe -m pytest calib/test/<test_file> -q` on this host. Baseline: `.venv/Scripts/python.exe -m pip install -e ./calib`, then full tests. Review code and test outputs after each task; record completed items here.
