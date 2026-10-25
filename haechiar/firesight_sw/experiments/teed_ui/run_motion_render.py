"""Run full exports with bounded retry for Windows atomic metadata replacement."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import render_motion_separated_videos as renderer

ENTRY = Path(__file__).resolve()
ENTRY_HASH = hashlib.sha256(ENTRY.read_bytes()).hexdigest()
original_save = renderer.save


def save(path: Path, value: dict) -> None:
    if "code_sha256" in value:
        assert hashlib.sha256(ENTRY.read_bytes()).hexdigest() == ENTRY_HASH
        value["code_sha256"][str(ENTRY.relative_to(renderer.ROOT))] = ENTRY_HASH
    for attempt in range(20):
        try:
            original_save(path, value)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


if __name__ == "__main__":
    renderer.save = save
    renderer.main()
