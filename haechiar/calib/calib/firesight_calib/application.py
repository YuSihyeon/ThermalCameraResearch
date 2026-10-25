"""Apply reviewed intrinsics to host video processing (no firmware writes)."""

from __future__ import annotations

import re
import time

import cv2
import numpy as np

from .sources import ROSSource, USBSource, validate_frame
from .thermal import normalize_thermal


class Rectifier:
    """Reusable native-depth rectification for a single calibrated camera mode."""

    def __init__(self, calibration: dict):
        if calibration.get("schema") != "firesight.camera_calibration/v2":
            raise ValueError("application requires a v2 native Boson calibration")
        if calibration.get("image_size") != [320, 256]:
            raise ValueError("calibration image_size must be [320, 256]")
        self.size = (320, 256)
        self.matrix = np.array(calibration["camera_matrix"], dtype=np.float64)
        self.distortion = np.array(calibration["distortion_coefficients"], dtype=np.float64)
        self.encoding = calibration.get("encoding")
        if self.encoding not in ("uint8", "uint16"):
            raise ValueError("calibration encoding must be uint8 or uint16")
        if (
            self.matrix.shape != (3, 3)
            or not np.isfinite(self.matrix).all()
            or self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0
            or not 0 < self.matrix[0, 2] < 320 or not 0 < self.matrix[1, 2] < 256
            or not np.allclose(self.matrix[2], [0, 0, 1])
            or not np.allclose([self.matrix[0, 1], self.matrix[1, 0]], 0)
        ):
            raise ValueError("invalid calibration camera_matrix")
        if (
            calibration.get("distortion_model") != "plumb_bob"
            or self.distortion.shape != (5,) or not np.isfinite(self.distortion).all()
        ):
            raise ValueError("calibration requires five finite plumb_bob distortion coefficients")
        self.maps = cv2.initUndistortRectifyMap(
            self.matrix, self.distortion, np.eye(3), self.matrix, self.size, cv2.CV_32FC1
        )
        if not all(np.isfinite(m).all() for m in self.maps):
            raise ValueError("calibration produces non-finite rectification maps")

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        frame = validate_frame(frame)
        if str(frame.dtype) != self.encoding:
            raise ValueError(f"live encoding {frame.dtype} differs from calibrated {self.encoding}")
        return cv2.remap(frame, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def camera_info(self) -> dict:
        # These values describe image_rect, which has already been undistorted.
        projection = np.column_stack([self.matrix, np.zeros(3)])
        return dict(width=320, height=256, distortion_model="plumb_bob", d=[0.0] * 5,
                    k=self.matrix.ravel().tolist(), r=np.eye(3).ravel().tolist(),
                    p=projection.ravel().tolist())


def validate_application_options(args):
    if args.source not in ("usb", "ros"):
        raise ValueError("application requires --source usb or --source ros")
    if not np.isfinite(args.frame_timeout) or args.frame_timeout <= 0 or args.apply_frames < 0:
        raise ValueError("frame-timeout must be positive and apply-frames must be >= 0")
    if args.source == "usb" and args.headless:
        raise ValueError("USB application needs a display; use ROS for headless image output")
    if args.source == "ros":
        topics = [args.topic, args.rectified_topic, args.camera_info_topic]
        if any(not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+", topic) for topic in topics):
            raise ValueError("use absolute ROS topic names, for example /boson/image_raw")
        if len(set(topics)) != 3:
            raise ValueError("input, rectified image and CameraInfo topics must all differ")


def open_source(args):
    return (ROSSource(args.topic, args.telemetry_rows) if args.source == "ros"
            else USBSource(args.device, args.telemetry_rows))


class ROSRectifiedPublisher:
    def __init__(self, source, args, rectifier):
        from sensor_msgs.msg import Image, CameraInfo
        from rclpy.qos import qos_profile_sensor_data

        self.image_type, self.info_type = Image, CameraInfo
        self.rectifier = rectifier
        topics = [source.node.resolve_topic_name(t) for t in
                  (args.topic, args.rectified_topic, args.camera_info_topic)]
        if len(set(topics)) != 3:
            raise ValueError("resolved ROS topics overlap; remove conflicting remappings")
        self.image_publisher = source.node.create_publisher(Image, args.rectified_topic, qos_profile_sensor_data)
        self.info_publisher = source.node.create_publisher(CameraInfo, args.camera_info_topic, qos_profile_sensor_data)

    def publish(self, frame, header):
        if header is None:
            raise ValueError("ROS input frame has no acquisition header")
        image = self.image_type()
        image.header = header
        image.width, image.height = self.rectifier.size
        image.encoding = "mono16" if frame.dtype == np.uint16 else "mono8"
        image.is_bigendian = 0
        image.step = image.width * frame.dtype.itemsize
        image.data = frame.astype("<u2" if frame.dtype == np.uint16 else "u1", copy=False).tobytes()
        info = self.info_type()
        info.header = header
        for key, value in self.rectifier.camera_info().items():
            setattr(info, key, value)
        self.image_publisher.publish(image)
        self.info_publisher.publish(info)


def run_application(calibration, args) -> dict:
    """Caller must have obtained explicit review consent before calling this.

    Only a frame successfully shown/published counts as applied. All device and
    ROS resources close on quit, stream loss, interruption or publication error.
    """
    rectifier = Rectifier(calibration)
    validate_application_options(args)
    source = open_source(args)
    frames = 0
    stopped = "frame_limit"
    window = "FireSight | raw (left) / rectified (right) | Q stop"
    try:
        publisher = ROSRectifiedPublisher(source, args, rectifier) if args.source == "ros" else None
        last_frame_at = time.monotonic()
        print("승인된 보정값으로 영상 처리를 시작합니다. Q 또는 Ctrl+C로 종료합니다.", flush=True)
        while not args.apply_frames or frames < args.apply_frames:
            frame = source.read(min(0.25, args.frame_timeout))
            if frame is None:
                if time.monotonic() - last_frame_at >= args.frame_timeout:
                    raise RuntimeError(f"no live frame for {args.frame_timeout:g} seconds; application stopped")
                if not args.headless and cv2.waitKey(1) & 255 == ord("q"):
                    stopped = "user"
                    break
                time.sleep(0.005)
                continue
            last_frame_at = time.monotonic()
            corrected = rectifier.rectify(frame)
            if publisher:
                publisher.publish(corrected, source.last_header)
            if not args.headless:
                # Shared display normalization keeps before/after comparable.
                preview = normalize_thermal(np.hstack([frame, corrected]))
                cv2.imshow(window, cv2.resize(preview, (1280, 512), interpolation=cv2.INTER_NEAREST))
            frames += 1
            if not args.headless and cv2.waitKey(1) & 255 == ord("q"):
                stopped = "user"
                break
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return {"status": "applied" if frames else "not_applied", "frames": frames,
            "stopped": stopped, "mode": "host_rectification", "source": args.source}
