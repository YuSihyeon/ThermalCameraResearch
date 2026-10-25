# TEED thermal HUD experiments

These scripts reproduce edge filtering, temporal/contour stabilization, thermal palette preservation, and opt-in motion separation. Generated videos, probability caches and run metrics remain local and are ignored by Git. The latest pre-TEED feature tracking experiment is not part of this publication.

Install the runtime dependencies from `edge/orin/requirements.txt`. Plot/render utilities also use Pillow, already used by the earlier experiment environment. Some Korean-label renderers require Windows Malgun Gothic fonts at `C:/Windows/Fonts`; this does not affect the runtime or synthetic checks.

## Self-contained checks

Run from the repository root with Python 3.11 (do not use `python -O`):

```sh
python experiments/teed_ui/check_edge_cleanup.py
python experiments/teed_ui/check_temporal_edges.py
python experiments/teed_ui/check_temporal_integration.py
python experiments/teed_ui/check_contour_tracker.py
python experiments/teed_ui/check_contour_behavior.py
python experiments/teed_ui/check_contour_render.py
python experiments/teed_ui/check_motion_separation.py
python experiments/teed_ui/check_motion_probability.py
python experiments/teed_ui/check_thermal_palette.py
python experiments/teed_ui/check_thermal_palette_integration.py
python experiments/teed_ui/check_thermal_color_integration.py
```

CLI integration tests use the tracked TEED checkpoint. `check_thermal_palette.py` uses two small reference frames extracted from the [Fire360 thermal collection](https://uofi.app.box.com/v/fire360dataset); these retain the source IFSI watermark. Full dataset videos are not included.

## Local dataset checks

Place the thermal clips at `../data/fire360/raw/IFSI Video N (2).mp4`, where N is 4 through 10, then run:

```sh
python experiments/teed_ui/check_contour_pipeline.py
python experiments/teed_ui/check_motion_pipeline.py
python experiments/teed_ui/check_thermal_color_cli.py
```

The replay/render tools use the same local paths. Reports refer to the generated output directories, not downloadable files in this repository. Tests exercise recorded videos; Jetson Orin display latency and real obstacle-avoidance performance require separate measurement.

Run `python experiments/teed_ui/check_evaluate_git.py` to verify PATH-based and explicit Git executables, missing-executable handling, and the pinned baseline. This check requires Git and the baseline commit below.

The original HUD comparison in `evaluate.py` is pinned to commit `481569b7412522f9f6f90cde24b320538dce1e13` so committing the new HUD does not silently change the baseline. Fetch that commit if using a shallow clone.
