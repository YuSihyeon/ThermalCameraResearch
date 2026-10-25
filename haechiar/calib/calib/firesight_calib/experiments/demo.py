from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import save_yaml
from .calibration import Camera, Registration
from .config import MODES
from .model import Sample
from .pipeline import run_session
from .session import SessionWriter


def run_demo(mode, output, frames=12):
    if mode not in MODES or type(frames) is not int or frames < 1:
        raise ValueError("valid mode and positive frame count required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    size = (320, 256)
    matrix = np.array([[250., 0, 159.5], [0, 250., 127.5], [0, 0, 1]])
    camera = Camera(matrix, np.zeros(5), size)
    transform = np.eye(4)
    transform[0, 3] = 0.04  # Deliberately synthetic test geometry, never a rig default.
    registration = Registration("synthetic-demo-only", "sensor_optical_frame", "boson_optical_frame",
                                camera, transform, "synthetic", camera)
    save_yaml(output / "intrinsic.yaml", camera.document())
    save_yaml(output / "extrinsic.yaml", {"source_frame": registration.source_frame,
              "target_frame": registration.target_frame, "transform": transform})
    save_yaml(output / "registration.yaml", {"schema": "firesight.experiment_registration/v1",
              "calibration_id": registration.calibration_id, "source_frame": registration.source_frame,
              "target_frame": registration.target_frame, "provenance": "synthetic",
              "intrinsic": "intrinsic.yaml", "extrinsic": "extrinsic.yaml", "source_intrinsic": "intrinsic.yaml"})
    yy, xx = np.mgrid[:size[1], :size[0]]
    with SessionWriter(output / "session", {"mode": mode, "group_id": "synthetic_demo",
                                            "description": "generated arrays, not device measurements"}) as writer:
        for i in range(frames):
            thermal = (10000 + xx * 10 + ((xx > 100 + i) & (xx < 200 + i) & (yy > 50)) * 4000).astype(np.uint16)
            stamp = 1_000_000_000 + i * 33_333_333
            writer.append(Sample("thermal", stamp, "synthetic:demo", "boson_optical_frame",
                                 "thermal", thermal, "synthetic"))
            if mode == "rgbd_thermal":
                rgb = np.repeat((thermal / 256).astype(np.uint8)[..., None], 3, axis=2)
                writer.append(Sample("rgb", stamp + 1_000_000, "synthetic:demo", "rgb_optical_frame",
                                     "rgb", rgb, "synthetic"))
                depth = np.where(xx < 160, 1.5, 3.).astype(np.float32)
                depth[50:70, 50:70] = np.nan
                writer.append(Sample("depth", stamp + 2_000_000, "synthetic:demo", "sensor_optical_frame",
                                     "depth", depth, "synthetic"))
            elif mode == "radar_thermal":
                points = np.array([[-.3, -.2, 2.], [0., 0., 1.], [.3, .2, 3.]], np.float32)
                writer.append(Sample("radar", stamp + 3_000_000, "synthetic:demo", "sensor_optical_frame",
                                     "radar", points, "synthetic"))
    return run_session(output / "session", output / "results", mode,
                       None if mode == "thermal_ai" else registration)
