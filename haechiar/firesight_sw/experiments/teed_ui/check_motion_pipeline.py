"""Exercise opt-in motion separation through the real CLI entry point."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from run_pipeline import build_parser, run  # noqa: E402


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(__file__).with_name("motion_split") / "pipeline_checks"
    out.mkdir(parents=True, exist_ok=True)
    parser = build_parser()
    assert not parser.parse_args([]).motion_separated_edges
    base = [
        "--device",
        "cpu",
        "--headless",
        "--warmup",
        "0",
        "--max-frames",
        "8",
        "--thermal-palette-overlay",
        "--display-width",
        "720",
        "--display-height",
        "405",
        "--profiles",
        str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"),
    ]
    checks = []
    for name, number, start, extra in [
        ("baseline", 10, 467, ["--contour-edges"]),
        ("separated", 10, 467, ["--motion-separated-edges"]),
        ("legacy", 10, 467, ["--motion-separated-edges", "--legacy-edges"]),
        ("loop", 8, 659, ["--motion-separated-edges", "--loop"]),
    ]:
        source = ROOT.parent / f"data/fire360/raw/IFSI Video {number} (2).mp4"
        args = parser.parse_args(
            base
            + [
                "--source",
                str(source),
                "--start-frame",
                str(start),
                "--metrics",
                str(out / f"{name}.json"),
            ]
            + extra
        )
        with (
            (out / f"{name}.log").open("w", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
        ):
            result = run(args)
        assert result["processed_frames"] == 8
        assert result["source"]["osd_regions"] == 2
        hud = result["hud"]
        assert hud["thermal_palette_overlay_enabled"]
        if number == 10:
            assert hud["camera_hot_outline_pixels"] > 0
        enabled = name in {"separated", "loop"}
        assert hud["motion_separation_enabled"] == enabled
        if enabled:
            assert hud["contour_edges_enabled"]
            assert all(
                "global_motion_status" in row["contour"] for row in hud["edge_frames"]
            )
            if name == "loop":
                assert hud["edge_frames"][2]["contour"]["contour_reset"] == 1
        if name == "legacy":
            assert not hud["contour_edges_enabled"]
        checks.append(name)
    (out / "checks.json").write_text(
        json.dumps({"status": "PASS", "actual_cli_cases": checks}, indent=2),
        encoding="utf-8",
    )
    print(
        "PASS: default off, implicit contour enablement, legacy override, thermal colors, OSD and loop reset"
    )


if __name__ == "__main__":
    main()
