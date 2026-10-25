import numpy as np
import pytest

from firesight_calib.experiments.calibration import Camera, Registration, fit_rigid
from firesight_calib.experiments.fusion import deproject_depth, registered_depth, fuse
from firesight_calib.experiments.model import Sample


def camera():
    return Camera(np.array([[2., 0, 2], [0, 2, 1], [0, 0, 1]]), np.zeros(5), (5, 3))


def registration(provenance="measured"):
    return Registration("test-calibration", "sensor", "boson", camera(), np.eye(4), provenance)


def test_depth_roundtrip_ignores_missing_and_respects_camera_z():
    depth = np.full((3, 5), np.nan, np.float32)
    depth[1, 2], depth[1, 3] = 2., 4.
    points = deproject_depth(depth, camera())
    np.testing.assert_allclose(points, [[0, 0, 2], [2, 0, 4]])
    output = registered_depth(points, registration())
    np.testing.assert_allclose(output, depth)


def test_projection_nearest_z_no_hole_filling_and_behind_camera_rejected():
    points = np.array([[0., 0, 4], [0, 0, 2], [0, 0, -1], [100, 0, 1]])
    output = registered_depth(points, registration())
    assert output[1, 2] == 2
    assert np.isfinite(output).sum() == 1


@pytest.mark.parametrize("transform", [np.zeros((4, 4)), np.diag([-1., 1, 1, 1]),
                                       np.full((4, 4), np.nan)])
def test_invalid_rigid_transforms_rejected(transform):
    with pytest.raises(ValueError):
        Registration("test", "sensor", "boson", camera(), transform, "measured")


def test_fusion_checks_frames_provenance_and_native_resolution():
    thermal = Sample("thermal", 1, "clock", "boson", "thermal", np.zeros((3, 5), np.uint16))
    radar = Sample("radar", 1, "clock", "sensor", "radar", np.array([[0., 0, 2]]))
    result = fuse({"thermal": thermal, "radar": radar}, "radar_thermal", registration())
    assert result["depth_m"][1, 2] == 2
    bad = Sample("radar", 1, "clock", "different_sensor", "radar", radar.data)
    with pytest.raises(ValueError, match="frame"):
        fuse({"thermal": thermal, "radar": bad}, "radar_thermal", registration())
    with pytest.raises(ValueError, match="provenance"):
        fuse({"thermal": thermal, "radar": radar}, "radar_thermal", registration("synthetic"))
    with pytest.raises(ValueError, match="shape"):
        deproject_depth(np.ones((6, 10)), camera())


def test_rigid_fit_recovers_source_to_target_and_reports_error():
    source = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    target = source @ rotation.T + [0.1, -0.2, 0.3]
    result = fit_rigid(source, target)
    np.testing.assert_allclose(result["transform"][:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(result["transform"][:3, 3], [0.1, -0.2, 0.3])
    assert result["rmse_m"] < 1e-12
    assert result["deployment_ready"] is False


def test_rigid_fit_rejects_collinear_points():
    points = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    with pytest.raises(ValueError, match="collinear"):
        fit_rigid(points, points)


def test_thermal_ai_default_is_explicit_non_ai_baseline():
    thermal = Sample("thermal", 1, "clock", "boson", "thermal", np.zeros((3, 5), np.uint16))
    result = fuse({"thermal": thermal}, "thermal_ai", None)
    assert result["edge_method"] == "canny_baseline_not_ai"
    assert result["edges"].dtype == np.uint8
    assert "depth_m" not in result


def test_plugin_receives_native_values_and_invalid_probabilities_are_rejected():
    thermal = Sample("thermal", 1, "clock", "boson", "thermal", np.full((3, 5), 40000, np.uint16))

    class Model:
        name = "test-model"

        def predict(self, image):
            assert image.dtype == np.uint16 and image[0, 0] == 40000
            return np.full(image.shape, 0.9, dtype=np.float32)

    result = fuse({"thermal": thermal}, "thermal_ai", edge_model=Model())
    assert np.all(result["edges"] == 255)
    assert result["edge_method"] == "test-model"

    class BadModel(Model):
        def predict(self, image):
            return np.full(image.shape, np.nan)

    with pytest.raises(ValueError, match="probabilities"):
        fuse({"thermal": thermal}, "thermal_ai", edge_model=BadModel())


def test_registration_loads_legacy_calib_output_and_checks_direction(tmp_path):
    from firesight_calib.io import save_yaml
    save_yaml(tmp_path / "intrinsic.yaml", camera().document())
    save_yaml(tmp_path / "extrinsic.yaml", {"source_frame": "sensor", "target_frame": "boson",
                                            "T_boson_from_sensor": np.eye(4)})
    manifest = {"schema": "firesight.experiment_registration/v1", "calibration_id": "test",
                "source_frame": "sensor", "target_frame": "boson", "provenance": "measured",
                "intrinsic": "intrinsic.yaml", "extrinsic": "extrinsic.yaml"}
    save_yaml(tmp_path / "registration.yaml", manifest)
    loaded = Registration.load(tmp_path / "registration.yaml")
    np.testing.assert_array_equal(loaded.transform, np.eye(4))
    manifest["source_frame"] = "wrong"
    save_yaml(tmp_path / "registration.yaml", manifest)
    with pytest.raises(ValueError, match="source_frame"):
        Registration.load(tmp_path / "registration.yaml")
