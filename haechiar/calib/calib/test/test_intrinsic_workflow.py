import cv2
import numpy as np
import pytest
from firesight_calib.io import load_yaml
from firesight_calib.target import load_target


def test_circle_image_pipeline(tmp_path):
    from firesight_calib.synthetic_thermal import generate_dataset
    from firesight_calib.workflow import calibrate_dataset

    expected = generate_dataset(tmp_path / "images", views=24)
    result = calibrate_dataset(sorted((tmp_path / "images").glob("*.png")), tmp_path / "result.yaml")
    assert len(result.accepted_images) >= 15
    assert result.rms_px < 0.8
    np.testing.assert_allclose(np.diag(result.camera_matrix)[:2], np.diag(expected["matrix"])[:2], rtol=0.08)
    saved = load_yaml(tmp_path / "result.yaml")
    assert saved["target"]["pattern"] == [3, 9]
    assert saved["target"]["status"] == "provisional"
    assert saved["quality"]["deployment_ready"] is False
    assert saved["quality"]["passed"] is True
    assert (tmp_path / "result_debug" / "report.json").exists()
    assert list((tmp_path / "result_debug").glob("*undistorted.png"))


def test_duplicate_and_wrong_size_report(tmp_path):
    from firesight_calib.workflow import calibrate_dataset
    from test_thermal_target import board_image

    image = board_image()
    paths = []
    for i in range(16):
        p = tmp_path / f"{i:02}.png"
        cv2.imwrite(str(p), image if i < 15 else np.zeros((128, 160), np.uint16))
        paths.append(p)
    with pytest.raises(ValueError, match="valid views"):
        calibrate_dataset(paths, tmp_path / "result.yaml")
    import json

    report = json.loads((tmp_path / "result_debug" / "report.json").read_text())
    assert report["accepted_views"] == 1
    reasons = [r["reason"] for r in report["frames"]]
    assert "duplicate_pose" in reasons and "image_size" in reasons
    assert not (tmp_path / "result.yaml").exists()


def test_degenerate_observations_rejected():
    from firesight_calib.workflow import solve_target

    obj = load_target().object_points()
    matrix = np.array([[343.0, 0, 160], [0, 350, 128], [0, 0, 1]])
    points, _ = cv2.projectPoints(obj, np.zeros(3), np.array([-0.3, -0.2, 1.5]), matrix, np.zeros(5))
    with pytest.raises(ValueError, match="diversity"):
        solve_target([points.reshape(-1, 2)] * 15, load_target())


def test_quality_rejects_bad_fit():
    from firesight_calib.quality import assess_quality
    from firesight_calib.intrinsic import IntrinsicResult

    value = IntrinsicResult(
        2.0,
        np.array([[343.0, 0, 160], [0, 350, 128], [0, 0, 1]]),
        np.zeros(5),
        (320, 256),
        [2.0] * 15,
        [],
        [],
    )
    points = [np.array([[100.0, 100.0], [110.0, 100.0], [100.0, 110.0], [110.0, 110.0]])] * 15
    quality = assess_quality(value, points, load_target(), 0.8)
    assert not quality["passed"]
    assert "rms" in quality["failures"]


def test_frontoparallel_noisy_views_do_not_fake_tilt():
    from firesight_calib.workflow import solve_target

    rng = np.random.default_rng(0)
    matrix = np.array([[343.0, 0, 160], [0, 350.0, 128], [0, 0, 1.0]])
    observations = []
    for _ in range(24):
        tv = np.array([rng.uniform(-0.5, -0.1), rng.uniform(-0.3, -0.05), rng.uniform(1.2, 1.6)])
        points = cv2.projectPoints(load_target().object_points(), np.zeros(3), tv, matrix, np.zeros(5))[
            0
        ].reshape(-1, 2)
        observations.append((points + rng.normal(0, 0.05, points.shape)).astype(np.float32))
    with pytest.raises(ValueError, match="conditioning"):
        solve_target(observations, load_target())
