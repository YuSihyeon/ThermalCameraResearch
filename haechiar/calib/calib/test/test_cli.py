import json
from firesight_calib.cli import main, _parser


def test_default_target_and_capture_options():
    args = _parser().parse_args(["capture", "--source", "usb", "--device", "1", "--headless", "--auto"])
    assert args.source == "usb" and args.device == "1"
    assert args.min_views == 15


def test_doctor_without_camera(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["native_image_size"] == [320, 256]


def test_config_accepts_circle_target():
    from pathlib import Path

    config = Path(__file__).parents[1] / "config"
    assert (
        main(
            [
                "verify-config",
                "--sensors",
                str(config / "sensors.yaml"),
                "--target",
                str(config / "target.yaml"),
                "--quality",
                str(config / "quality.yaml"),
            ]
        )
        == 0
    )


def test_image_self_check(tmp_path, capsys):
    assert main(["self-check", "--output", str(tmp_path / "self")]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["image_pipeline"]["passed"]
    assert data["image_pipeline"]["accepted_views"] >= 15


def test_intrinsic_empty_glob_is_actionable(tmp_path, capsys):
    assert (
        main(["intrinsic", "--images", str(tmp_path / "*.png"), "--output", str(tmp_path / "result.yaml")])
        == 2
    )
    assert "matched no files" in capsys.readouterr().err


def test_quality_failure_returns_three(tmp_path):
    from firesight_calib.synthetic_thermal import generate_dataset

    generate_dataset(tmp_path / "images")
    assert (
        main(
            [
                "intrinsic",
                "--images",
                str(tmp_path / "images"),
                "--output",
                str(tmp_path / "bad.yaml"),
                "--max-rms",
                "0.00001",
            ]
        )
        == 3
    )
