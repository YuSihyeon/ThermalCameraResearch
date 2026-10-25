"""Runs in ROS Humble CI; skipped where ROS is not installed."""

import time
import os
import numpy as np
import pytest

from firesight_calib.sources import ROSSource


def test_real_ros_transport_native_u16():
    if os.environ.get("FIRESIGHT_REQUIRE_ROS") == "1":
        import rclpy
    else:
        rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("firesight_test_publisher", context=context)
    topic = "/firesight_test/native_image"
    publisher = node.create_publisher(Image, topic, qos_profile_sensor_data)
    source = ROSSource(topic)
    original = np.arange(320 * 256, dtype=np.uint16).reshape(256, 320)
    message = Image()
    message.height, message.width = 256, 320
    message.encoding = "mono16"
    message.step = 640
    message.is_bigendian = 0
    message.header.frame_id = "boson_optical_frame"
    message.header.stamp.sec = 42
    message.header.stamp.nanosec = 123456
    message.data = original.astype("<u2").tobytes()
    try:
        received = None
        deadline = time.monotonic() + 10
        while received is None and time.monotonic() < deadline:
            publisher.publish(message)
            received = source.read(0.1)
        assert received is not None
        np.testing.assert_array_equal(received, original)
        assert source.last_header == message.header
    finally:
        source.close()
        node.destroy_node()
        context.shutdown()


@pytest.mark.parametrize("encoding", ["uint8", "uint16"])
def test_real_ros_rectified_image_and_camera_info(encoding):
    if os.environ.get("FIRESIGHT_REQUIRE_ROS") == "1":
        import rclpy
    else:
        rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from firesight_calib.application import Rectifier, ROSRectifiedPublisher
    from firesight_calib.sources import decode_ros_image
    from test_application import calibration, options

    prefix = "/firesight_test/rectification_" + encoding
    args = options(source="ros", topic=prefix + "/raw", rectified_topic=prefix + "/rect",
                   camera_info_topic=prefix + "/camera_info")
    value = calibration()
    value["encoding"] = encoding
    rectifier = Rectifier(value)
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("firesight_rectification_test_" + encoding, context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    source = ROSSource(args.topic)
    try:
        output = ROSRectifiedPublisher(source, args, rectifier)
        raw_publisher = node.create_publisher(Image, args.topic, qos_profile_sensor_data)
        images, infos = [], []
        node.create_subscription(Image, args.rectified_topic, images.append, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, args.camera_info_topic, infos.append, qos_profile_sensor_data)
        original = np.arange(320 * 256).astype(encoding).reshape(256, 320)
        message = Image()
        message.header.frame_id = "boson_optical_frame"
        message.header.stamp.sec, message.header.stamp.nanosec = 52, 987654
        message.height, message.width = 256, 320
        message.encoding = "mono16" if encoding == "uint16" else "mono8"
        message.step = 320 * original.dtype.itemsize
        message.data = original.astype("<u2" if encoding == "uint16" else "u1").tobytes()
        deadline = time.monotonic() + 10
        while (not images or not infos) and time.monotonic() < deadline:
            raw_publisher.publish(message)
            frame = source.read(0.1)
            if frame is not None:
                output.publish(rectifier.rectify(frame), source.last_header)
            executor.spin_once(timeout_sec=0.05)
        assert images and infos, "rectified image and CameraInfo must both arrive"
        np.testing.assert_array_equal(decode_ros_image(images[-1]), rectifier.rectify(original))
        assert images[-1].header == infos[-1].header == message.header
        np.testing.assert_array_equal(infos[-1].d, np.zeros(5))
        np.testing.assert_array_equal(np.asarray(infos[-1].k).reshape(3, 3), value["camera_matrix"])
        np.testing.assert_array_equal(np.asarray(infos[-1].p).reshape(3, 4)[:, :3], value["camera_matrix"])
    finally:
        source.close()
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        context.shutdown()
