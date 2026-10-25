from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).with_name("temporal_stability") / "final"


def main():
    cv2.setNumThreads(1)
    results = []
    for path in sorted([*ROOT.glob("*.webm"), *ROOT.glob("*.mp4")]):
        name = path.stem.removesuffix("_stabilized")
        expected = json.loads((ROOT / (name + ".json")).read_text())
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), path
        fps = cap.get(cv2.CAP_PROP_FPS)
        dimensions = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        assert dimensions == ([720, 406] if "_stabilized" in path.stem else [1440, 540])
        frames, unique, picked = 0, set(), []
        sample = (
            set(range(70, 85, 2)) if expected["frames"] > 120 else set(range(30, 45, 2))
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            unique.add(hashlib.sha256(frame.tobytes()).hexdigest())
            if (
                path.suffix == ".webm"
                and "_stabilized" not in path.stem
                and frames in sample
            ):
                picked.append(cv2.resize(frame, (720, 270)))
            frames += 1
        cap.release()
        assert frames == expected["frames"], (path, frames)
        assert abs(fps - expected["fps"]) < 0.01, (path, fps)
        assert len(unique) > frames * 0.95, (path, len(unique))
        if picked:
            cv2.imwrite(str(ROOT / (name + "_motion_strip.png")), np.vstack(picked))
        results.append(
            {
                "file": path.name,
                "frames": frames,
                "fps": fps,
                "dimensions": dimensions,
                "unique_decoded_frames": len(unique),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    assert len(results) == 9, len(results)
    (ROOT / "video_decode_qa.json").write_text(
        json.dumps({"status": "PASS", "files": results}, indent=2)
    )
    print("PASS: fully decoded 9 videos with expected counts and FPS")


if __name__ == "__main__":
    main()
