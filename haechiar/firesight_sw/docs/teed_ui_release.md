# TEED UI GitHub publication notes

This note defines the documentation set to publish for the TEED UI work before the later feature-tracking experiments. It intentionally covers edge filtering, temporal/contour stabilization, thermal color preservation, red fire contours, and opt-in motion-separated edges. It does not publish feature-tracking plans, generated videos, generated metrics, or dataset video files.

## Publish these docs

Operational and research reports:

- `docs/mvp_technical_overview.md`
- `docs/teed_ui_improvement.md`
- `docs/fire360_thermal_camera_research.md`
- `docs/additional_thermal_sources.md`
- `docs/edge_residual_cleanup.md`
- `docs/temporal_stability.md`
- `docs/contour_stability.md`
- `docs/full_thermal_videos.md`
- `docs/thermal_realtime_results.md`
- `docs/thermal_hot_contours.md`
- `docs/thermal_color_runtime_results.md`
- `docs/motion_separation.md`
- `docs/visible_flame_candidates.md`

Local plan files such as `docs/teed_ui_plan.md`, `docs/edge_residual_plan.md`, `docs/temporal_stability_plan.md`, `docs/contour_stability_plan.md`, `docs/full_thermal_video_plan.md`, `docs/thermal_realtime_plan.md`, and `docs/motion_split_plan.md` can remain unpublished unless a review needs the planning history.

Do not publish `docs/feature_tracking_plan.md` or `docs/thermal_feature_tracking.md` with this release.

## Publish these small assets

- `docs/fire360_teed_processing_stages.png`
- `docs/teed_ui_validation/teed_algorithm_status.png`
- `docs/teed_ui_validation/temperature_contract.png`
- `docs/teed_ui_validation/pyrightconfig.json`
- `experiments/teed_ui/additional_thermal_sources/ifsi10_source_5s.jpg`
- `experiments/teed_ui/additional_thermal_sources/ifsi5_source_5s.jpg`

The two JPEG fixtures come from the public Fire360 source videos and should be attributed in the experiment README. Generated WebM/MP4 videos, generated PNG comparisons, generated metrics JSON, logs, model outputs, and dataset video files should stay local.

## Reproducible tools

The top-level `experiments/teed_ui/*.py` scripts are suitable to publish as reproducibility tools. Generated subdirectories under `experiments/teed_ui/` are local evidence paths referenced by the docs and are not required for GitHub browsing.

Rendering helper scripts in `docs/` use the Windows Korean Malgun Gothic fonts at `C:/Windows/Fonts/malgun.ttf` and `C:/Windows/Fonts/malgunbd.ttf`. On another platform, replace those font paths or install compatible fonts before regenerating the PNGs.

## Evidence boundaries

The published docs should keep these limits clear:

- Generated videos and metrics are local validation artifacts, not shipped assets.
- Output video FPS is playback timing, not model or end-to-end processing FPS.
- Reported edge and contour metrics are visual stability or preservation checks, not obstacle-avoidance accuracy.
- Fire360 videos do not provide calibrated per-pixel Celsius maps; high-temperature color preservation keeps the original thermal palette where available.
- Motion-separated edges are opt-in and do not replace the default TEED path.
