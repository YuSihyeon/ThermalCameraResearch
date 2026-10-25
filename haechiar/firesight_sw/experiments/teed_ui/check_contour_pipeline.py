from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("contour_stability") / "pipeline_checks"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable,
        str(ROOT / "edge/orin/run_pipeline.py"),
        "--source",
        str(ROOT.parent / "data/fire360/raw/IFSI Video 4 (2).mp4"),
        "--profiles",
        str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"),
        "--device",
        "cpu",
        "--headless",
        "--warmup",
        "0",
        "--max-frames",
        "4",
    ]
    checks = []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name, extra in [
        ("enabled", ["--start-frame", "300", "--contour-edges"]),
        ("previous", ["--start-frame", "300", "--no-contour-edges"]),
        ("loop", ["--start-frame", "623", "--loop", "--contour-edges"]),
        ("legacy", ["--start-frame", "300", "--contour-edges", "--legacy-edges"]),
    ]:
        with (OUT / (name + ".log")).open("w") as log:
            result = subprocess.run(
                base + extra + ["--metrics", str(OUT / (name + ".json"))],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        assert result.returncode == 0, name
        data = json.loads((OUT / (name + ".json")).read_text())
        assert data["processed_frames"] == 4
        hud = data["hud"]
        frames = hud["edge_frames"]
        if name == "enabled":
            assert (
                hud["contour_edges_enabled"]
                and frames[0]["contour"]["contour_reset"] == 1
            )
            assert frames[1]["contour"]["contour_reset"] == 0
        elif name == "previous":
            assert not hud["contour_edges_enabled"]
            assert all(f["contour"]["status"] == "disabled" for f in frames)
            previous = json.loads(
                (
                    ROOT
                    / "experiments/teed_ui/fixtures/previous_temporal_ifsi4_counts.json"
                ).read_text()
            )
            assert [f["output_edge_pixels"] for f in frames] == previous["after_pixels"]
        elif name == "loop":
            assert frames[2]["contour"]["contour_reset"] == 1
        else:
            assert (
                not hud["contour_edges_enabled"] and not hud["temporal_edges_enabled"]
            )
        checks.append("actual production CLI " + name)
    (OUT / "checks.json").write_text(
        json.dumps({"status": "PASS", "checks": checks}, indent=2)
    )
    print("PASS:", len(checks), "production contour CLI contracts")


if __name__ == "__main__":
    main()
