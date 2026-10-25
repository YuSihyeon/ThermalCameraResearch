"""Capture native frames and guide collection of distinct board poses."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .quality import coverage, duplicate_pose
from .sources import validate_frame
from .target import Target
from .thermal import find_target, normalize_thermal
from .workflow import calibrate_dataset, write_image


class CaptureSession:
    def __init__(self, directory, target: Target, source: dict):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError(f"session directory is not empty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.points = []
        self.dtype = None
        self.manifest = {
            "schema": "firesight.capture/v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "target": target.metadata(),
            "source": source,
            "frames": [],
            "status": "capturing",
        }
        self.save_manifest()

    def save_manifest(self):
        path = self.directory / "session.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        temporary.replace(path)

    def add(self, frame, points=None, method=None):
        frame = validate_frame(frame)
        if self.dtype is not None and frame.dtype != self.dtype:
            return False, "mixed_encoding"
        if points is None:
            points, method = find_target(frame, self.target)
        if points is None:
            return False, "pattern_not_found"
        if duplicate_pose(points, self.points):
            return False, "duplicate_pose"
        name = f"{len(self.points) + 1:04d}.png"
        path = self.directory / "images" / name
        write_image(path, frame)
        self.points.append(points.copy())
        self.dtype = frame.dtype
        self.manifest["frames"].append(
            {
                "path": "images/" + name,
                "captured_utc": datetime.now(timezone.utc).isoformat(),
                "dtype": str(frame.dtype),
                "image_size": [320, 256],
                "method": method,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "points_px": points.tolist(),
            }
        )
        self.save_manifest()
        return True, "saved"


def run_capture(
    source,
    directory,
    target: Target,
    headless=False,
    auto=False,
    timeout=600.0,
    interval=1.0,
    count=24,
    min_views=15,
    calibrate=False,
    max_rms=0.8,
    source_info=None,
):
    session = None
    try:
        if (
            count < min_views
            or min_views < 3
            or timeout <= 0
            or interval <= 0
            or not np.isfinite([timeout, interval, max_rms]).all()
            or max_rms <= 0
        ):
            raise ValueError("require count >= min_views >= 3 and positive finite timeout/interval/max_rms")
        if headless and not auto:
            raise ValueError("--headless requires --auto")
        session = CaptureSession(directory, target, source_info or {})
        start = time.monotonic()
        last_save = start - interval
        last_notice = start
        last_frame = None
        should_calibrate = False
        status = "Move board: center/edges, different tilts and distances"
        print(f"Session: {session.directory}\nSPACE save | A auto | C calibrate | Q stop", flush=True)
        while True:
            now = time.monotonic()
            if now - start >= timeout:
                if last_frame is not None:
                    write_image(session.directory / "last_frame.png", last_frame)
                raise RuntimeError(
                    f"capture timeout: {len(session.points)}/{count} views. Check source, thermal contrast and move/tilt board; saved samples remain in {session.directory}"
                )
            frame = source.read(min(0.25, timeout - (now - start)))
            if frame is None:
                if not headless and cv2.waitKey(1) & 255 == ord("q"):
                    break
                if now - last_notice > 5:
                    print("Waiting for a native Boson frame; check device/topic and publisher.", flush=True)
                    last_notice = now
                time.sleep(0.01)
                continue
            last_frame = frame
            points, method = find_target(frame, target)
            key = -1
            if not headless:
                preview = cv2.cvtColor(normalize_thermal(frame), cv2.COLOR_GRAY2BGR)
                if points is not None:
                    cv2.drawChessboardCorners(preview, target.pattern, points.reshape(-1, 1, 2), True)
                canvas = cv2.resize(preview, (960, 768), interpolation=cv2.INTER_NEAREST)
                area, bins = coverage(session.points, (320, 256))
                lines = [
                    f"BOSON 320 | {len(session.points)}/{count} views | AUTO: {auto}",
                    f"Coverage {area:.0%} | cells {np.count_nonzero(bins)}/9 | "
                    + ("27 points" if points is not None else "No full pattern"),
                    status,
                    "SPACE save | A auto | C calibrate | Q stop",
                ]
                for i, line in enumerate(lines):
                    y = 26 + i * 27
                    cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
                    cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1)
                cv2.imshow("FireSight Boson Intrinsic", canvas)
                key = cv2.waitKey(1) & 255
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
            if key == ord("c"):
                if len(session.points) >= min_views:
                    should_calibrate = True
                    break
                status = f"Need at least {min_views} distinct valid views"
            if key == 32 or (auto and now - last_save >= interval):
                ok, reason = session.add(frame, points, method)
                status = reason
                last_save = now
                if ok:
                    print(f"Saved {len(session.points)}/{count}", flush=True)
                elif key == 32:
                    write_image(session.directory / "last_rejected.png", frame)
                if len(session.points) >= count:
                    should_calibrate = calibrate
                    break
        session.manifest["status"] = "captured"
        session.save_manifest()
    except BaseException:
        if session is not None:
            session.manifest["status"] = "interrupted"
            session.save_manifest()
        raise
    finally:
        source.close()
        if not headless:
            cv2.destroyAllWindows()
    result = {"session": str(session.directory), "accepted_views": len(session.points), "calibrated": False}
    if should_calibrate:
        path = session.directory / "boson_intrinsic.yaml"
        calibrate_dataset(
            sorted((session.directory / "images").glob("*.png")), path, target, min_views, max_rms
        )
        result.update(calibrated=True, output=str(path))
    return result
