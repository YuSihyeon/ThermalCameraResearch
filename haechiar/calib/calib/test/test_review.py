import hashlib
import json

import cv2
import numpy as np

from firesight_calib.intrinsic import IntrinsicResult
from firesight_calib.target import Target


def _projection(target, matrix, rvec=(0.08, -0.12, 0.03), tvec=(-0.25, -0.15, 1.3)):
    return cv2.projectPoints(
        target.object_points(), np.asarray(rvec, float), np.asarray(tvec, float), matrix, np.zeros(5)
    )[0].reshape(-1, 2).astype(np.float32)


def test_residual_grid_uses_reprojected_error_and_marks_empty_bins():
    from firesight_calib.review import residual_grid

    target = Target()
    matrix = np.array([[300.0, 0, 160.0], [0, 300.0, 128.0], [0, 0, 1.0]])
    exact = _projection(target, matrix)
    shifted = exact.copy()
    shifted[::2, 0] += 3.0
    shifted[1::2, 1] -= 4.0

    means, counts = residual_grid([shifted], target, matrix, np.zeros(5), (320, 256), bins=(4, 3))

    ok, rvec, tvec = cv2.solvePnP(target.object_points(), shifted, matrix, np.zeros(5))
    assert ok
    fitted = cv2.projectPoints(target.object_points(), rvec, tvec, matrix, np.zeros(5))[0].reshape(-1, 2)
    expected_mean = np.linalg.norm(fitted - shifted, axis=1).mean()
    weighted_mean = np.nansum(means * counts) / counts.sum()
    assert np.isclose(weighted_mean, expected_mean, atol=1e-5)
    assert weighted_mean > 1.0
    assert np.isnan(means[counts == 0]).all()
    assert counts.sum() == len(shifted)


def test_heatmap_legend_matches_zero_based_color_scale():
    from firesight_calib.review import _heatmap

    svg = _heatmap(np.array([[5.0, 10.0]]), np.array([[2, 3]]), " px", "test")

    assert "최솟값 0.00 px" in svg
    assert "최댓값 10.00 px" in svg
    assert "관측 최솟값 5.00 px" in svg
    assert 'viewBox="0 0 576 ' in svg
    assert "rgb(245,238,220)" in svg and "rgb(205,88,50)" in svg
    other = _heatmap(np.array([[5.0, 10.0]]), np.array([[2, 3]]), " px", "other")
    gradient_id = svg.split('<linearGradient id="', 1)[1].split('"', 1)[0]
    assert f'id="{gradient_id}"' not in other


def test_write_review_is_standalone_and_escapes_frame_metadata(tmp_path):
    from firesight_calib.review import write_review

    target = Target()
    matrix = np.array([[300.0, 0, 160.0], [0, 300.0, 128.0], [0, 0, 1.0]])
    image_path = tmp_path / "thermal&.png"
    cv2.imencode(".png", np.full((256, 320), 12000, np.uint16))[1].tofile(image_path)
    observation = _projection(target, matrix)
    result = IntrinsicResult(0.2, matrix, np.zeros(5), (320, 256), [0.2], [str(image_path)], [])
    debug = tmp_path / "debug"
    debug.mkdir()
    quality = {
        "passed": False,
        "deployment_ready": False,
        "failures": ["coverage<&"],
        "max_rms_px": 0.8,
        "coverage_fraction": 0.2,
        "coverage_3x3": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        "normal_spread_deg": 4.0,
        "target_status": "provisional",
    }
    path = write_review(
        result,
        [observation],
        target,
        quality,
        debug,
        [{"path": str(image_path), "reason": "bad<&", "method": "circles<&", "points_px": observation.tolist()}],
    )

    html = path.read_text(encoding="utf-8")
    assert path == debug / "review.html"
    assert "data:image/png;base64," in html
    assert "thermal&amp;.png" in html and "bad&lt;&amp;" in html and "circles&lt;&amp;" not in html
    assert "coverage<&" not in html
    assert "잔차 히트맵" in html and "데이터 없음" in html
    assert "픽셀(px)" in html and "터미널에서 적용 여부" in html
    assert "커버리지 비율" in html and "20.0%" in html
    assert "법선 각도 범위" in html and "4.00°" in html
    assert "호모그래피 조건비" in html
    assert "0.00" in html and "0.80" in html  # numeric y-axis labels
    assert "색상 범위" in html and "최솟값" in html and "최댓값" in html


def test_missing_accepted_image_does_not_shift_observation_or_error_index(tmp_path, monkeypatch):
    import firesight_calib.review as review

    target = Target()
    matrix = np.array([[300.0, 0, 160.0], [0, 300.0, 128.0], [0, 0, 1.0]])
    first = _projection(target, matrix)
    second = _projection(target, matrix, rvec=(-0.1, 0.15, 0.02))
    visible = tmp_path / "visible.png"
    cv2.imencode(".png", np.full((256, 320), 100, np.uint8))[1].tofile(visible)
    captured = []
    real_draw = review.cv2.drawChessboardCorners

    def spy_draw(image, pattern, corners, found):
        captured.append(corners.reshape(-1, 2).copy())
        return real_draw(image, pattern, corners, found)

    monkeypatch.setattr(review.cv2, "drawChessboardCorners", spy_draw)
    result = IntrinsicResult(0.2, matrix, np.zeros(5), (320, 256), [0.11, 0.22], ["missing.png", str(visible)], [])
    quality = {"passed": True, "deployment_ready": False, "max_rms_px": 0.8, "coverage_3x3": [[1] * 3] * 3}
    report = review.write_review(
        result,
        [first, second],
        target,
        quality,
        tmp_path / "debug",
        [{"path": "missing.png", "reason": None}, {"path": str(visible), "reason": None}],
    )

    np.testing.assert_allclose(captured[0], second)
    text = report.read_text(encoding="utf-8")
    assert "재투영 오차: 0.2200 px" in text


def test_rejected_malformed_preview_cannot_abort_report(tmp_path, monkeypatch):
    import firesight_calib.review as review

    target = Target()
    matrix = np.array([[300.0, 0, 160.0], [0, 300.0, 128.0], [0, 0, 1.0]])
    result = IntrinsicResult(0.2, matrix, np.zeros(5), (320, 256), [], [], ["bad.dat"])
    monkeypatch.setattr(review, "_read", lambda path: np.zeros((10, 10, 2), np.uint8))

    path = review.write_review(
        result, [], target, {"failures": [], "coverage_3x3": [[0] * 3] * 3}, tmp_path / "debug",
        [{"path": "bad.dat", "reason": "expected_mono8_or_mono16"}],
    )

    assert path.exists()
    assert "expected_mono8_or_mono16" in path.read_text(encoding="utf-8")


def test_real_synthetic_pipeline_writes_review_artifact(tmp_path):
    from firesight_calib.synthetic_thermal import generate_dataset
    from firesight_calib.workflow import calibrate_dataset

    generate_dataset(tmp_path / "images", views=18)
    output = tmp_path / "result.yaml"
    result = calibrate_dataset(sorted((tmp_path / "images").glob("*.png")), output)

    report = tmp_path / "result_debug" / "review.html"
    text = report.read_text(encoding="utf-8")
    assert result.rms_px < 0.8
    assert report.stat().st_size > 20_000
    assert "카메라 행렬 K" in text
    assert text.count("data:image/png;base64,") >= 3
    manifest = json.loads((tmp_path / "result_debug" / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema": "firesight.review/v1",
        "calibration_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
