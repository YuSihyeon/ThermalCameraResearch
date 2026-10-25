# Calibration review and application plan

**Goal:** Put all Boson intrinsic work on main and provide an input → calibration → visual review → explicit application or fresh input workflow.

**Architecture:** Keep batch calibration commands. Add a terminal wizard as the launcher default, a standalone HTML report produced by the existing image solver, and a reusable rectifier with USB preview and ROS Image/CameraInfo output. The approval belongs to the running program's user; calculations never start application by themselves.

**Tech stack:** Python 3.10+, existing NumPy/OpenCV/PyYAML, Python standard library HTML/SVG, optional ROS 2 Humble.

## Constraints

- Preserve 320×256 mono8/mono16 input and native depth during rectification.
- Explicit yes after review is required each time; no unattended yes flag. No returns to fresh input; EOF/cancel exits without application.
- Reports display numerical quality failures and provisional target status; heuristics inform the user's decision. Invalid matrices cannot be applied.
- Apply to host video processing, never claim firmware or EEPROM changes. USB shows live corrected frames; ROS publishes corrected native images and matching CameraInfo with the original stamp/frame_id.
- Retain raw data, rejected attempts, existing commands, README and historical development documents.
- Keep output topics distinct from input and each other; reject malformed or mismatched calibration before opening a device.
- Use tests for consent, retry, real numerical correction, bit depth, timeout and ROS transport; distinguish synthetic tests from hardware validation.

## Tasks

1. Visual report: create `review.py` and tests, add `write_review(result, observations, target, quality, debug, records)` at the end of `workflow.calibrate_dataset`. Generate `review.html`, per-view error graph, coverage heatmap and actual residual heatmap, with units, no-data cells, K/D, status and original/corrected previews.
2. Runtime: create `application.py` and tests. Load/validate a frozen calibration, cache OpenCV maps, preserve native pixels, expose `run_application(calibration, args)` returning processed count/status. Add ROS publication helpers and transport tests. Open sources only inside the approved application call.
3. Wizard: create `wizard.py`, wire `wizard` and `review` CLI commands and default launchers. Show report and destination, accept explicit yes/no/quit, retain decision+calibration digest audit, return to input after no or recoverable failure. Saved image mode asks for paths; capture mode releases source before review and reopens only after consent.
4. Document usage, output semantics and limits in README/design. Generate and visually inspect a synthetic report. Run full tests and self-check, review the final changes, push branch, wait for CI, merge PR #1 and delete the merged branch. Verify remote has only main and contains every prior branch commit.

## Progress

- Baseline: 31 passed, 1 ROS test skipped locally (Windows, ROS unavailable).
- Repository freshly cloned to `FiresightAR/calib`; other local repositories are untouched.
- Tasks 1–3 and documentation complete. Review findings resolved: Linux implicit ROS selection, heatmap color legend scale.
- Final local validation: 66 passed, 3 ROS transport cases skipped on Windows; Ruff and compileall passed.
- Real Windows launcher bootstrap and self-check passed in a fresh repository `.venv`; synthetic 24-view image RMS 0.0187 px. Report visually inspected. These are synthetic results, not physical-camera accuracy measurements.
- Required GitHub Windows/Ubuntu and Humble transport CI runs precede the authorized merge and merged-branch deletion.
