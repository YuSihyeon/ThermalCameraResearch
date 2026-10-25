from types import SimpleNamespace

import numpy as np
import pytest

from firesight_calib.experiments.adapters import BosonUSB, RealSense
from firesight_calib.experiments.calibration import Camera, Registration
from firesight_calib.experiments.fusion import fuse
from firesight_calib.experiments.model import Sample


def test_boson_preserves_raw_native_values_and_releases(monkeypatch):
    from firesight_calib import sources
    released = []
    camera = SimpleNamespace(release=lambda: released.append(True))
    raw = np.full((256, 320), 40000, np.uint16)
    monkeypatch.setattr(sources, "open_usb_camera", lambda device, rows: camera)
    monkeypatch.setattr(sources, "read_usb_frame", lambda device, rows: raw)
    adapter = BosonUSB({"device": "explicit-device"}, "host:test")
    sample = adapter.read(.1)
    raw[:] = 1  # Simulates reuse of an SDK buffer after return.
    assert sample.data[0, 0] == 40000 and sample.data.dtype == np.uint16
    assert sample.timestamp_ns > 0 and sample.clock == "host:test"
    assert sample.metadata["timestamp_basis"] == "host_receipt"
    adapter.close()
    assert released == [True]


def test_realsense_read_converts_depth_and_preserves_raw_sdk_timestamps():
    # Exercise the real read adapter with SDK frame objects; no hardware is claimed.
    raw_depth = np.array([[1000, 0], [2000, 3000]], np.uint16)
    raw_color = np.zeros((2, 2, 3), np.uint8)
    depth = SimpleNamespace(get_data=lambda: raw_depth, get_timestamp=lambda: 12.5,
                            get_frame_timestamp_domain=lambda: "depth_device_clock")
    color = SimpleNamespace(get_data=lambda: raw_color, get_timestamp=lambda: 12.9,
                            get_frame_timestamp_domain=lambda: "color_device_clock")
    frames = SimpleNamespace(get_depth_frame=lambda: depth, get_color_frame=lambda: color)
    adapter = RealSense.__new__(RealSense)
    adapter.pipeline = SimpleNamespace(poll_for_frames=lambda: frames)
    adapter.clock, adapter.depth_frame, adapter.rgb_frame = "host:test", "depth_frame", "rgb_frame"
    adapter.scale = .001
    adapter.intrinsic = SimpleNamespace(fx=2., fy=2., ppx=.5, ppy=.5, coeffs=[0.] * 5,
                                        width=2, height=2, model="none")
    rgb, sample = adapter.read(.1)
    np.testing.assert_allclose(sample.data, [[1, np.nan], [2, 3]])
    assert sample.timestamp_ns == rgb.timestamp_ns
    assert sample.metadata["sdk_depth_timestamp_ms"] == 12.5
    assert sample.metadata["sdk_color_timestamp_ms"] == 12.9
    assert "camera_matrix" not in rgb.metadata  # Depth intrinsics must not describe RGB.


def test_fusion_rejects_depth_intrinsic_different_from_recorded_sdk():
    camera = Camera(np.array([[2., 0, .5], [0, 2., .5], [0, 0, 1]]), np.zeros(5), (2, 2))
    registration = Registration("test", "depth", "thermal", camera, np.eye(4), "measured", camera)
    thermal = Sample("thermal", 1, "host", "thermal", "thermal", np.zeros((2, 2), np.uint16))
    wrong = camera.document()
    wrong["camera_matrix"][0][0] = 5.
    depth = Sample("depth", 1, "host", "depth", "depth", np.ones((2, 2)), metadata={"intrinsic": wrong})
    with pytest.raises(ValueError, match="intrinsic"):
        fuse({"thermal": thermal, "depth": depth}, "rgbd_thermal", registration)
