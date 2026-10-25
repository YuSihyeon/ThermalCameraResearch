#!/usr/bin/env python3
"""Identify which camera index is which, and how usable each one is.

macOS cameras routinely fail their first read while the sensor spins up, and
the index order changes as devices come and go, so every index is probed with a
warm-up and a measured frame rate rather than trusting the reported one.
"""

import time

import cv2


def probe(index: int, warmup_seconds: float = 4.0, measure_frames: int = 20) -> None:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(f"  [{index}] not available")
        capture.release()
        return

    frame = None
    deadline = time.perf_counter() + warmup_seconds
    attempts = 0
    while time.perf_counter() < deadline:
        attempts += 1
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)

    if frame is None:
        print(f"  [{index}] opened but produced no frame after {attempts} attempts")
        capture.release()
        return

    # The reported FPS is often wrong; time real reads instead.
    started = time.perf_counter()
    captured = 0
    for _ in range(measure_frames):
        ok, _ = capture.read()
        if ok:
            captured += 1
    elapsed = time.perf_counter() - started
    measured = captured / elapsed if elapsed > 0 else 0.0

    height, width = frame.shape[:2]
    reported = capture.get(cv2.CAP_PROP_FPS)
    notes = []
    if width >= height * 2:
        notes.append(f"looks stereo (one eye {width // 2}x{height}) - try --stereo-half")
    if frame.mean() < 30:
        notes.append("very dark - lens covered, or the room needs light")
    if measured < 10:
        notes.append("slow - may be idle or still waking up")

    print(f"  [{index}] {width}x{height}  reported {reported:.0f}fps  "
          f"measured {measured:.1f}fps  brightness {frame.mean():.0f}/255")
    for note in notes:
        print(f"        note: {note}")
    path = f"camera_{index}.png"
    cv2.imwrite(path, frame)
    print(f"        saved {path}")
    capture.release()


def main() -> int:
    print("probing camera indexes 0..3 (a few seconds each)\n")
    for index in range(4):
        probe(index)
    print("\nOpen the saved camera_*.png files to see which index is which,")
    print("then pass that number to run_pipeline.py as --source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
