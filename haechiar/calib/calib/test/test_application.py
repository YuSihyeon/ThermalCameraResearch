from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def calibration():
    return {
        "schema": "firesight.camera_calibration/v2",
        "image_size": [320, 256],
        "camera_matrix": [[280.0, 0, 160], [0, 282.0, 128], [0, 0, 1]],
        "distortion_model": "plumb_bob",
        "distortion_coefficients": [-0.15, 0.02, 0.001, -0.001, 0.0],
        "encoding": "uint16",
        "quality": {"passed": True, "deployment_ready": False, "failures": []},
    }


def options(**kwargs):
    values = dict(source="usb", device="0", topic="/boson/image_raw", telemetry_rows=0,
                  rectified_topic="/boson/image_rect", camera_info_topic="/boson/camera_info_rect",
                  headless=False, apply_frames=2, frame_timeout=1.0)
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def display_boundary(monkeypatch):
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


def test_rectifier_corrects_native_pixels_and_camera_info():
    from firesight_calib.application import Rectifier

    value = calibration()
    rectifier = Rectifier(value)
    y, x = np.indices((256, 320))
    raw = (1000 + y * 32 + x * 2).astype(np.uint16)
    actual = rectifier.rectify(raw)
    expected = cv2.undistort(raw, np.array(value["camera_matrix"]), np.array(value["distortion_coefficients"]))
    assert actual.dtype == np.uint16
    np.testing.assert_allclose(actual, expected, atol=1)
    assert not np.array_equal(actual, raw)
    info = rectifier.camera_info()
    assert info["d"] == [0.0] * 5
    np.testing.assert_array_equal(np.array(info["p"]).reshape(3, 4)[:, :3], value["camera_matrix"])


@pytest.mark.parametrize("key,value", [
    ("image_size", [640, 512]), ("camera_matrix", [[float("nan")]*3]*3),
    ("distortion_coefficients", [0]*4), ("encoding", "rgb8"),
    ("camera_matrix", [[-10, 0, 160], [0, 280, 128], [0, 0, 1]]),
])
def test_bad_calibration_rejected(key, value):
    from firesight_calib.application import Rectifier

    data = calibration()
    data[key] = value
    with pytest.raises(ValueError):
        Rectifier(data)


def test_mismatched_live_encoding_rejected():
    from firesight_calib.application import Rectifier

    with pytest.raises(ValueError, match="encoding"):
        Rectifier(calibration()).rectify(np.zeros((256, 320), np.uint8))


@pytest.mark.parametrize("updates", [
    {"rectified_topic": "/boson/image_raw"}, {"camera_info_topic": "/boson/image_rect"},
    {"rectified_topic": "boson/image_rect"}, {"topic": "/bad//topic"},
])
def test_ros_topics_do_not_feed_back(updates):
    from firesight_calib.application import validate_application_options

    with pytest.raises(ValueError, match="topic"):
        validate_application_options(options(source="ros", **updates))


def test_live_application_processes_frames_and_closes_source(monkeypatch):
    from firesight_calib import application

    class Source:
        closed = False
        def read(self, timeout):
            return np.full((256, 320), 20000, np.uint16)
        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(application, "open_source", lambda args: source)
    result = application.run_application(calibration(), options())
    assert result["frames"] == 2
    assert result["status"] == "applied"
    assert source.closed


def test_no_frames_times_out_and_closes(monkeypatch):
    from firesight_calib import application

    source = SimpleNamespace(read=lambda timeout: None, close=lambda: closed.append(True))
    closed = []
    monkeypatch.setattr(application, "open_source", lambda args: source)
    with pytest.raises(RuntimeError, match="no.*frame"):
        application.run_application(calibration(), options(frame_timeout=0.01))
    assert closed == [True]


def test_invalid_calibration_does_not_open_camera(monkeypatch):
    from firesight_calib import application

    monkeypatch.setattr(application, "open_source", lambda args: pytest.fail("camera was opened"))
    data = calibration()
    data["camera_matrix"][0][0] = 0
    with pytest.raises(ValueError):
        application.run_application(data, options())
