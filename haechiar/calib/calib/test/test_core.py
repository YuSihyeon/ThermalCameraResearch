import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from firesight_calib.extrinsic import calibrate_extrinsic
from firesight_calib.geometry import invert_transform
from firesight_calib.intrinsic import board_points
from firesight_calib.io import save_yaml
from firesight_calib.projection import project_points
from firesight_calib.synthetic import run_self_check


class CoreTests(unittest.TestCase):
    def test_transform_inverse(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.1, -0.2, 0.3]
        self.assertTrue(np.allclose(transform @ invert_transform(transform), np.eye(4)))

    def test_projection_filters_points(self):
        matrix = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        points = np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        pixels, valid = project_points(points, np.eye(4), matrix, np.zeros(5), (100, 80))
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertFalse(valid[2])
        self.assertTrue(np.allclose(pixels[0], [50.0, 40.0]))

    def test_synthetic_self_check(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_self_check(directory)
            self.assertTrue(result["passed"])
            self.assertTrue((Path(directory) / "boson_intrinsic.yaml").exists())
            self.assertTrue((Path(directory) / "self_check.json").exists())

    def test_extrinsic_recovers_known_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = np.asarray([[285.0, 0.0, 160.0], [0.0, 285.0, 128.0], [0.0, 0.0, 1.0]])
            camera = {
                "image_size": [320, 256],
                "camera_matrix": matrix,
                "distortion_coefficients": [0.0] * 5,
            }
            save_yaml(root / "boson.yaml", camera)
            save_yaml(root / "sensor.yaml", camera)
            known = np.eye(4)
            known[:3, 3] = [0.08, -0.015, 0.01]
            obj = board_points((9, 5), 0.035).astype(np.float64)
            objects, boson_pixels, sensor_pixels = [], [], []
            for index in range(6):
                rvec = np.asarray([0.03 * index, -0.02 * index, 0.01 * index])
                tvec_sensor = np.asarray([-0.12 + 0.02 * index, -0.07 + 0.01 * index, 0.9 + 0.04 * index])
                rotation_sensor, _ = cv2.Rodrigues(rvec)
                sensor_from_board = np.eye(4)
                sensor_from_board[:3, :3] = rotation_sensor
                sensor_from_board[:3, 3] = tvec_sensor
                boson_from_board = known @ sensor_from_board
                rvec_boson, _ = cv2.Rodrigues(boson_from_board[:3, :3])
                bp, _ = cv2.projectPoints(obj, rvec_boson, boson_from_board[:3, 3], matrix, np.zeros(5))
                sp, _ = cv2.projectPoints(obj, rvec, tvec_sensor, matrix, np.zeros(5))
                objects.append(obj)
                boson_pixels.append(bp.reshape(-1, 2))
                sensor_pixels.append(sp.reshape(-1, 2))
            np.savez(
                root / "pairs.npz",
                object_points=np.asarray(objects),
                boson_points=np.asarray(boson_pixels),
                sensor_points=np.asarray(sensor_pixels),
            )
            result = calibrate_extrinsic(
                root / "pairs.npz",
                root / "boson.yaml",
                root / "sensor.yaml",
                "d456_color_optical_frame",
                root / "extrinsic.yaml",
            )
            recovered = np.asarray(result["T_boson_from_d456_color"])
            self.assertTrue(np.allclose(recovered, known, atol=1e-5))
            self.assertTrue((root / "extrinsic.yaml").exists())


if __name__ == "__main__":
    unittest.main()
