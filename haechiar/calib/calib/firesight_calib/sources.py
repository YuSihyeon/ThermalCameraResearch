"""Optional live sources. ROS imports are deliberately lazy."""

from __future__ import annotations
import sys
import time
import multiprocessing
import queue
import cv2
import numpy as np


def validate_frame(frame: np.ndarray, telemetry_rows: int = 0) -> np.ndarray:
    if frame.ndim != 2 or frame.dtype not in (np.uint8, np.uint16):
        raise ValueError(
            f"expected mono8/mono16, received shape={frame.shape} dtype={frame.dtype}; select raw Y16, disable RGB conversion"
        )
    if telemetry_rows not in (0, 2):
        raise ValueError("telemetry_rows must be 0 or 2")
    if frame.shape != (256 + telemetry_rows, 320):
        raise ValueError(
            f"expected native 320x256 (+{telemetry_rows} telemetry rows), received {frame.shape[::-1]}; do not resize a colorized stream"
        )
    return np.ascontiguousarray(frame[:256])


def decode_ros_image(message) -> np.ndarray:
    if message.encoding not in ("mono8", "mono16", "8UC1", "16UC1"):
        raise ValueError(f"ROS Image must be mono8/mono16, received {message.encoding}")
    itemsize = 1 if message.encoding in ("mono8", "8UC1") else 2
    if (
        message.width <= 0
        or message.height <= 0
        or message.step < message.width * itemsize
        or len(message.data) != message.step * message.height
    ):
        raise ValueError("invalid ROS Image buffer length or row stride")
    dtype = np.dtype("u1" if itemsize == 1 else (">u2" if message.is_bigendian else "<u2"))
    array = np.ndarray(
        (message.height, message.width),
        dtype=dtype,
        buffer=bytes(message.data),
        strides=(message.step, itemsize),
    )
    return array.astype(np.uint8 if itemsize == 1 else np.uint16, copy=True)


def open_usb_camera(device=0, telemetry_rows=0):
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    camera = cv2.VideoCapture(device, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"cannot open USB camera {device}; check index, UVC driver, cable and other camera apps"
            )
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 256 + telemetry_rows)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
        camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return camera
    except Exception:
        camera.release()
        raise


def read_usb_frame(camera, telemetry_rows):
    ok, frame = camera.read()
    if not ok or frame is None:
        return None
    frame = validate_frame(frame, telemetry_rows)
    if frame.dtype != np.uint16:
        raise ValueError("USB Y16 negotiation failed: received mono8; select Boson raw 16-bit output")
    return frame


def _usb_worker(device, telemetry_rows, frames, stop):
    camera = None
    try:
        camera = open_usb_camera(device, telemetry_rows)
        while not stop.is_set():
            frame = read_usb_frame(camera, telemetry_rows)
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            try:
                frames.put_nowait(("frame", frame))
            except queue.Full:
                pass
    except Exception as exc:
        # Deliver a useful error even if a last good frame is still queued.
        while not stop.is_set():
            try:
                frames.put(("error", str(exc)), timeout=0.1)
                break
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
    finally:
        if camera is not None:
            camera.release()


class USBSource:
    """Isolate potentially uninterruptible UVC calls from GUI and timeout logic."""

    def __init__(self, device=0, telemetry_rows=0):
        context = multiprocessing.get_context("spawn")
        self.frames = context.Queue(maxsize=1)
        self.stop = context.Event()
        self.process = context.Process(
            target=_usb_worker, args=(device, telemetry_rows, self.frames, self.stop), daemon=True
        )
        self.process.start()

    def read(self, timeout=1.0):
        try:
            kind, value = self.frames.get(timeout=max(0, timeout))
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError("USB capture process exited; check driver and camera connection")
            return None
        if kind == "error":
            raise RuntimeError(value)
        return value

    def close(self):
        self.stop.set()
        self.process.join(0.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(0.5)
        self.frames.close()


class ROSSource:
    def __init__(self, topic="/boson/image_raw", telemetry_rows=0):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 unavailable. On Ubuntu: source /opt/ros/humble/setup.bash; use ./run.sh (system Python), install ros-humble-rclpy and ros-humble-sensor-msgs"
            ) from exc
        self.rclpy = rclpy
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = None
        self.executor = None
        try:
            self.node = rclpy.create_node("firesight_intrinsic_capture", context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.frame = None
            self.frame_header = None
            self.last_header = None
            self.error = None
            self.telemetry_rows = telemetry_rows
            self.subscription = self.node.create_subscription(
                Image, topic, self._on_image, qos_profile_sensor_data
            )
        except Exception:
            self.close()
            raise

    def _on_image(self, message):
        try:
            self.frame = validate_frame(decode_ros_image(message), self.telemetry_rows)
            self.frame_header = message.header
        except ValueError as exc:
            self.error = exc

    def read(self, timeout=1.0):
        deadline = time.monotonic() + timeout
        while self.frame is None and self.error is None and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=min(0.1, max(0, deadline - time.monotonic())))
        if self.error:
            raise self.error
        frame = self.frame
        self.last_header = self.frame_header if frame is not None else None
        self.frame = None
        self.frame_header = None
        return frame

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1.0)
        if self.node is not None:
            self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
