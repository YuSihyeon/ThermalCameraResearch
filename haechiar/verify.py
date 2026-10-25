"""Verify imported sources and run upstream tests that need no physical sensors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECKS = (
    "check_edge_cleanup.py",
    "check_temporal_edges.py",
    "check_contour_tracker.py",
    "check_contour_behavior.py",
    "check_contour_render.py",
    "check_motion_separation.py",
    "check_motion_probability.py",
    "check_thermal_palette.py",
    "check_thermal_palette_integration.py",
    "check_thermal_color_integration.py",
)


def matches_snapshot(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    # Git may check UTF-8 text out as CRLF on Windows. Binary assets stay exact.
    if b"\0" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == expected
    return False


def main() -> int:
    if not __debug__:
        raise SystemExit("Run without -O: upstream checks use assert statements.")
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    changed = []
    for row in manifest["files"]:
        path = ROOT.parent / row["destination"]
        if not matches_snapshot(path, row["sha256"]):
            changed.append(row["destination"])
    if changed:
        raise SystemExit("Imported source differs from recorded snapshot:\n" + "\n".join(changed))
    print(f"PASS: {len(manifest['files'])} source files match the upstream snapshot", flush=True)
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = [([sys.executable, "-m", "pytest", "-q"], ROOT / "calib")]
    commands.extend(([sys.executable, str(ROOT / "firesight_sw/experiments/teed_ui" / name)],
                     ROOT / "firesight_sw") for name in CHECKS)
    failures = []
    for command, cwd in commands:
        print("\nRunning: " + " ".join(command[1:]), flush=True)
        result = subprocess.run(command, cwd=cwd, env=env)
        if result.returncode:
            failures.append(" ".join(command[1:]))
    if failures:
        print("\nFAILED:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("\nPASS: calibration suite and 10 standalone edge checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
