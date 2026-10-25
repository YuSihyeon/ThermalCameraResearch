"""Replay the user's visible-flame scene through the actual TEED/HUD pipeline."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edge" / "orin"))
from hud import HUDConfig, HUDRenderer  # noqa: E402
from teed_core import TEEDCore  # noqa: E402

OUT = ROOT / "experiments" / "teed_ui" / "visible_flame_results"
SOURCE = ROOT.parent / "data" / "fire360" / "raw" / "02815 (2).MTS"


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("edge/orin/visible_flame.py", "edge/orin/hud.py",
                     "edge/orin/run_pipeline.py")
    }
    core = TEEDCore(
        ROOT / "edge" / "teed" / "models" / "5_model.pth",
        width=320, height=256, threshold=0.75, edge_width=3,
        contrast_clip_limit=2.0, device="cpu", half=False,
    )
    before = HUDRenderer(HUDConfig(output_size=(960, 540)))
    after = HUDRenderer(HUDConfig(
        output_size=(960, 540), visible_flame_candidates=True,
    ))
    cap = cv2.VideoCapture(str(SOURCE))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        str(OUT / "visible_flame_comparison.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (1440, 310),
    )
    assert writer.isOpened()
    records = []
    try:
        for index in range(120):
            ok, frame = cap.read()
            if not ok:
                break
            edge = core.infer(frame)
            old = before.render(frame, edge)
            new = after.render(frame, edge)
            assert old.flame_metrics["status"] == "disabled"
            assert new.thermal_metrics["status"] == "temperature_unavailable"
            assert new.thermal_metrics["hot_pixels"] == 0
            cells = []
            for label, image in (
                ("Source", frame),
                ("Previous: edges only", old.image),
                ("Updated: visible FLAME? candidate", new.image),
            ):
                cell = np.zeros((310, 480, 3), np.uint8)
                cell[40:] = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
                cv2.putText(cell, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (230, 230, 230), 1, cv2.LINE_AA)
                cells.append(cell)
            comparison = cv2.hconcat(cells)
            writer.write(comparison)
            if index in (30, 66, 90):
                cv2.imwrite(str(OUT / f"frame_{index:03d}_comparison.png"), comparison)
                cv2.imwrite(str(OUT / f"frame_{index:03d}_updated.png"), new.image)
            records.append({"frame_index": index, "source_seconds": index / fps,
                            "flame": new.flame_metrics, "temperature": new.thermal_metrics})
    finally:
        writer.release()
        cap.release()
    assert len(records) > 90, len(records)
    assert records[0]["flame"]["candidate_pixels"] == 0
    assert records[66]["flame"]["candidate_pixels"] > 0, records[66]
    assert all(r["temperature"]["hot_pixels"] == 0 for r in records)
    for name, expected in source_hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    payload = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "frames": len(records), "fps": fps,
        "candidate_frames": sum(r["flame"]["candidate_pixels"] > 0 for r in records),
        "target_frame": records[66], "records": records,
        "core": core.describe(),
        "source_code_sha256": source_hashes,
        "limitations": [
            "Heuristic visible flame candidates, not measured temperature or validated fire detection.",
            "Moving warm lights, reflections or objects can cause false positives.",
            "Static, colourless, smoke-obscured flames may be missed.",
            "No labelled detection dataset evaluated; target-frame success is not accuracy."
        ],
    }
    (OUT / "replay_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
