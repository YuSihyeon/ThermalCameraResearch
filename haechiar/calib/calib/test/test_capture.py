from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize("big", [False, True])
def test_ros_stride_and_endian(big):
    from firesight_calib.sources import decode_ros_image

    values = np.array([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=">u2" if big else "<u2")
    data = b"".join(row.tobytes() + b"xx" for row in values)
    message = SimpleNamespace(encoding="mono16", width=3, height=2, step=8, is_bigendian=int(big), data=data)
    np.testing.assert_array_equal(decode_ros_image(message), values.astype(np.uint16))


def test_reject_color_and_bad_buffer():
    from firesight_calib.sources import decode_ros_image

    m = SimpleNamespace(encoding="rgb8", width=320, height=256, step=960, is_bigendian=0, data=b"")
    with pytest.raises(ValueError, match="mono8/mono16"):
        decode_ros_image(m)
    m.encoding = "mono16"
    with pytest.raises(ValueError, match="buffer"):
        decode_ros_image(m)


def test_telemetry_is_explicit():
    from firesight_calib.sources import validate_frame

    frame = np.arange(258 * 320, dtype=np.uint16).reshape(258, 320)
    with pytest.raises(ValueError, match="320x256"):
        validate_frame(frame)
    np.testing.assert_array_equal(validate_frame(frame, telemetry_rows=2), frame[:256])
    with pytest.raises(ValueError):
        validate_frame(np.zeros((256, 320, 3), np.uint8))


def test_session_preserves_native_frame_and_rejects_duplicate(tmp_path):
    from firesight_calib.capture import CaptureSession
    from firesight_calib.workflow import read_image
    from firesight_calib.target import load_target
    from test_thermal_target import board_image

    session = CaptureSession(tmp_path / "session", load_target(), {"source": "test"})
    frame = board_image()
    assert session.add(frame)[0]
    assert session.add(frame)[1] == "duplicate_pose"
    np.testing.assert_array_equal(read_image(tmp_path / "session" / "images" / "0001.png"), frame)
    assert (tmp_path / "session" / "session.json").exists()


def test_capture_closes_source_on_timeout(tmp_path):
    from firesight_calib.capture import run_capture
    from firesight_calib.target import load_target

    class Empty:
        closed = False

        def read(self, timeout):
            return None

        def close(self):
            self.closed = True

    source = Empty()
    with pytest.raises(RuntimeError, match="time"):
        run_capture(
            source,
            tmp_path / "s",
            load_target(),
            headless=True,
            auto=True,
            timeout=0.03,
            interval=0.001,
            count=15,
        )
    assert source.closed


def test_usb_rejects_silent_color_fallback(monkeypatch):
    from firesight_calib import sources

    class Device:
        released = False

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            return True, np.zeros((512, 640, 3), np.uint8)

        def release(self):
            self.released = True

    dev = Device()
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *args: dev)
    camera = sources.open_usb_camera(0)
    try:
        with pytest.raises(ValueError, match="mono"):
            sources.read_usb_frame(camera, 0)
    finally:
        camera.release()
    assert dev.released


def stalled_usb_worker(device, telemetry_rows, queue, stop):
    import time

    time.sleep(60)


def test_usb_native_stall_has_bounded_read_and_shutdown(monkeypatch):
    import time
    from firesight_calib import sources

    monkeypatch.setattr(sources, "_usb_worker", stalled_usb_worker)
    camera = sources.USBSource(0)
    start = time.monotonic()
    try:
        assert camera.read(0.1) is None
        assert time.monotonic() - start < 1.0
    finally:
        camera.close()
    assert time.monotonic() - start < 3.0
    assert not camera.process.is_alive()


def test_auto_capture_to_calibration(tmp_path):
    from firesight_calib.capture import run_capture
    from firesight_calib.synthetic_thermal import generate_dataset
    from firesight_calib.target import load_target
    from firesight_calib.workflow import read_image

    generate_dataset(tmp_path / "input")

    class Replay:
        closed = False

        def __init__(self):
            self.paths = iter(sorted((tmp_path / "input").glob("*.png")))

        def read(self, timeout):
            path = next(self.paths, None)
            return read_image(path) if path else None

        def close(self):
            self.closed = True

    replay = Replay()
    result = run_capture(
        replay,
        tmp_path / "capture",
        load_target(),
        auto=True,
        headless=True,
        interval=0.00001,
        timeout=15,
        calibrate=True,
    )
    assert result["calibrated"] and result["accepted_views"] == 24
    assert replay.closed
