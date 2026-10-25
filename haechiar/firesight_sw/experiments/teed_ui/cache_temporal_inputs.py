from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge/orin"))
from replay import load_fire360_osd_regions  # noqa: E402
from teed_core import TEEDCore  # noqa: E402

OUT = Path(__file__).with_name("temporal_stability")
CLIPS = [
    ("fire360_ifsi4", "IFSI Video 4 (2).mp4", 300, 240),
    ("fire360_ifsi8", "IFSI Video 8 (2).mp4", 90, 240),
    ("fire360_visible", "02815 (2).MTS", 0, 120),
]


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    core = TEEDCore(
        ROOT / "edge/teed/models/5_model.pth",
        width=320,
        height=256,
        threshold=0.75,
        edge_width=3,
        contrast_clip_limit=2,
        device="cpu",
    )
    for name, filename, start, requested in CLIPS:
        target = OUT / (name + "_inputs.npz")
        if target.exists():
            print("CACHE EXISTS", name, flush=True)
            continue
        path = ROOT.parent / "data/fire360/raw" / filename
        regions = load_fire360_osd_regions(
            str(ROOT / "edge/teed/profiles/fire360_video_profiles.json"), filename
        )
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = min(requested, total - start)
        assert count > 0 and fps > 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        probabilities, sources, backgrounds = [], [], []
        for i in range(count):
            ok, frame = cap.read()
            assert ok, (filename, start + i)
            result = core.infer(frame, regions)
            probabilities.append(result.probability)
            sources.append(result.source_view)
            backgrounds.append(
                cv2.resize(
                    frame,
                    (720, round(frame.shape[0] * 720 / frame.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
            )
            if i % 60 == 0:
                print(name, i + 1, "/", count, flush=True)
        cap.release()
        metadata = {
            "source": str(path),
            "fps": fps,
            "start": start,
            "count": count,
            "regions": regions,
            "checkpoint": core.describe()["checkpoint_sha256"],
            "width": 320,
            "height": 256,
            "contrast_clip_limit": 2.0,
        }
        np.savez_compressed(
            target,
            probability=np.stack(probabilities),
            source=np.stack(sources),
            background=np.stack(backgrounds),
            indices=np.arange(start, start + count),
            seconds=np.arange(start, start + count) / fps,
            metadata=json.dumps(metadata),
        )
        print(
            "COMPLETE",
            name,
            count,
            "frames",
            target.stat().st_size,
            "bytes",
            flush=True,
        )


if __name__ == "__main__":
    main()
