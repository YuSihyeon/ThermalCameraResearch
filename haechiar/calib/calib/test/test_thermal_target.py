import cv2
import numpy as np
import pytest

from firesight_calib import thermal


def board_image(angle=0, invert=False, noise=0, depth=16):
    canvas = np.full((400, 600), 220, np.uint8)
    for i in range(9):
        for j in range(3):
            cv2.circle(canvas, (60 + 60 * i, 50 + 60 * (2 * j + i % 2)), 30, 40, -1)
    # Native Boson geometry; approximately 1.4 m and 60 mm circles.
    small = cv2.resize(canvas, (150, 100), interpolation=cv2.INTER_AREA)
    image = np.full((256, 320), 220, np.uint8)
    image[78:178, 85:235] = small
    matrix = cv2.getRotationMatrix2D((160, 128), angle, 1)
    image = cv2.warpAffine(image, matrix, (320, 256), borderValue=220)
    if invert:
        image = 255 - image
    rng = np.random.default_rng(8)
    image = np.clip(image.astype(float) + rng.normal(0, noise, image.shape), 0, 255)
    return (image * 180 + 9000).astype(np.uint16) if depth == 16 else image.astype(np.uint8)


def test_quote_coordinates():
    from firesight_calib.target import load_target

    t = load_target()
    assert t.pattern == (3, 9)
    assert t.status == "provisional"
    expected = [[0.06 + 0.06 * i, 0.05 + 0.06 * (2 * j + i % 2), 0] for i in range(9) for j in range(3)]
    np.testing.assert_allclose(t.object_points(), expected, atol=1e-7)


@pytest.mark.parametrize(
    "angle,invert,noise,depth", [(0, False, 0, 16), (30, True, 2, 16), (90, False, 1, 8), (180, True, 1, 16)]
)
def test_native_thermal_detection(angle, invert, noise, depth):
    from firesight_calib.target import load_target

    image = board_image(angle, invert, noise, depth)
    original = image.copy()
    points, method = thermal.find_target(image, load_target())
    assert points is not None and points.shape == (27, 2)
    assert method
    np.testing.assert_array_equal(image, original)
    # Every correspondence must be compatible with the known planar board.
    obj = load_target().object_points()[:, :2]
    homography, _ = cv2.findHomography(obj, points)
    prediction = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), homography).reshape(-1, 2)
    assert np.max(np.linalg.norm(prediction - points, axis=1)) < 0.8


def test_blank_rejected():
    from firesight_calib.target import load_target

    assert thermal.find_target(np.zeros((256, 320), np.uint16), load_target())[0] is None


def test_invalid_target(tmp_path):
    from firesight_calib.target import load_target

    p = tmp_path / "bad.yaml"
    p.write_text("pattern_type: asymmetric_circles\npattern_size: [3, 9]\nspacing_m: -0.06\n")
    with pytest.raises(ValueError):
        load_target(p)
