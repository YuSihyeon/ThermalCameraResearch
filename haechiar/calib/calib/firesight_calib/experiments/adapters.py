"""Optional hardware readers; unknown radar/CSI/TE-EV2/MTi SDKs remain plugins."""
from __future__ import annotations

import time

import numpy as np

from .model import Sample


class BosonUSB:
    def __init__(self, options, clock):
        from ..sources import open_usb_camera
        if "device" not in options:
            raise ValueError("Boson requires explicit device index/path")
        self.clock = clock
        self.frame = options.get("frame", "boson_optical_frame")
        self.telemetry_rows = options.get("telemetry_rows", 0)
        self.camera = open_usb_camera(options["device"], self.telemetry_rows)

    def read(self, timeout_s):
        from ..sources import read_usb_frame
        data = read_usb_frame(self.camera, self.telemetry_rows)
        stamp = time.monotonic_ns()  # Before IPC, host receipt rather than exposure time.
        if data is None:
            time.sleep(min(timeout_s, 0.01))
            return None
        return Sample("thermal", stamp, self.clock, self.frame, "thermal", data,
                      metadata={"timestamp_basis": "host_receipt", "radiometric_conversion": False})

    def close(self):
        self.camera.release()


def boson_usb(options, clock):
    return BosonUSB(options, clock)


class RealSense:
    def __init__(self, options, clock):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("install device-compatible librealsense/pyrealsense2 on the capture host") from exc
        if any(key not in options for key in ("serial", "width", "height", "fps")):
            raise ValueError("RealSense requires explicit serial, width, height, fps")
        self.clock = clock
        self.depth_frame = options.get("depth_frame", "camera_depth_optical_frame")
        self.rgb_frame = options.get("rgb_frame", "camera_color_optical_frame")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(options["serial"]))
        for stream, fmt in ((rs.stream.depth, rs.format.z16), (rs.stream.color, rs.format.rgb8)):
            config.enable_stream(stream, int(options["width"]), int(options["height"]), fmt, int(options["fps"]))
        self.started = False
        try:
            profile = self.pipeline.start(config)
            self.started = True
            self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.intrinsic = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            if self.intrinsic.model not in (rs.distortion.none, rs.distortion.brown_conrady):
                raise ValueError("depth distortion is not OpenCV-compatible; explicit SDK conversion required")
        except BaseException:
            self.close()
            raise

    def read(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frames = self.pipeline.poll_for_frames()
            stamp = time.monotonic_ns()
            if not frames:
                time.sleep(0.002)
                continue
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                continue
            intr = self.intrinsic
            metadata = {"timestamp_basis": "host_frameset_receipt", "depth_scale_m": self.scale,
                        "sdk_depth_timestamp_ms": depth.get_timestamp(),
                        "sdk_color_timestamp_ms": color.get_timestamp(),
                        "sdk_depth_clock": str(depth.get_frame_timestamp_domain()),
                        "sdk_color_clock": str(color.get_frame_timestamp_domain())}
            intrinsic = {"camera_matrix": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                        "distortion_coefficients": list(intr.coeffs), "image_size": [intr.width, intr.height],
                        "distortion_model": str(intr.model)}
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.scale
            depth_m[depth_m == 0] = np.nan
            return [Sample("rgb", stamp, self.clock, self.rgb_frame, "rgb", np.asanyarray(color.get_data()),
                           metadata=metadata),
                    Sample("depth", stamp, self.clock, self.depth_frame, "depth", depth_m,
                           metadata={**metadata, "intrinsic": intrinsic})]
        return None

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


def realsense(options, clock):
    return RealSense(options, clock)
