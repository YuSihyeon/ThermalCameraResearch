import hashlib
import json
from pathlib import Path

import pytest

from firesight_calib.io import save_yaml
from test_application import calibration, options


def candidate(tmp_path, name="result"):
    path = tmp_path / f"{name}.yaml"
    save_yaml(path, calibration())
    debug = tmp_path / f"{name}_debug"
    debug.mkdir()
    report = debug / "review.html"
    report.write_text("<html>review</html>", encoding="utf-8")
    (debug / "review_manifest.json").write_text(json.dumps({
        "schema": "firesight.review/v1",
        "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }))
    return path


def setup_review(monkeypatch, replies):
    from firesight_calib import wizard
    answers = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(wizard.webbrowser, "open", lambda url: True)
    return wizard


def test_no_returns_to_input_without_application(tmp_path, monkeypatch):
    wizard = setup_review(monkeypatch, ["no"])
    path = candidate(tmp_path)
    monkeypatch.setattr(wizard, "run_application", lambda *args: pytest.fail("unapproved application"))
    assert wizard.review_and_apply(path, options(no_browser=True)) == "retry"
    audits = list((tmp_path / "result_debug").glob("decision_*.json"))
    assert json.loads(audits[0].read_text())["decision"] == "no"


def test_only_explicit_yes_applies_frozen_reviewed_values(tmp_path, monkeypatch):
    wizard = setup_review(monkeypatch, ["", "maybe", "yes"])
    path = candidate(tmp_path)
    applied = []
    def apply(value, args):
        applied.append(value)
        return {"status": "applied", "frames": 2}
    monkeypatch.setattr(wizard, "run_application", apply)
    assert wizard.review_and_apply(path, options(no_browser=True)) == "applied"
    assert len(applied) == 1
    assert applied[0]["camera_matrix"] == calibration()["camera_matrix"]


def test_eof_never_implies_approval(tmp_path, monkeypatch):
    wizard = setup_review(monkeypatch, [])
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr(wizard, "run_application", lambda *args: pytest.fail("unapproved"))
    assert wizard.review_and_apply(candidate(tmp_path), options(no_browser=True)) == "quit"


def test_changed_report_or_calibration_is_not_applied(tmp_path, monkeypatch):
    wizard = setup_review(monkeypatch, ["yes"])
    path = candidate(tmp_path)
    changed = calibration()
    changed["camera_matrix"][0][0] = 200
    save_yaml(path, changed)
    monkeypatch.setattr(wizard, "run_application", lambda *args: pytest.fail("stale report"))
    with pytest.raises(ValueError, match="report.*match"):
        wizard.review_and_apply(path, options(no_browser=True))


def test_application_failure_is_recorded_without_success(tmp_path, monkeypatch):
    wizard = setup_review(monkeypatch, ["yes"])
    path = candidate(tmp_path)
    def fail(*args):
        raise RuntimeError("camera disconnected")
    monkeypatch.setattr(wizard, "run_application", fail)
    assert wizard.review_and_apply(path, options(no_browser=True)) == "retry"
    audit = json.loads(next((tmp_path / "result_debug").glob("decision_*.json")).read_text())
    assert audit["application"]["status"] == "failed"


def test_wizard_no_repeats_input_and_preserves_attempts(tmp_path, monkeypatch):
    from firesight_calib import wizard

    images1, images2 = tmp_path / "images1", tmp_path / "images2"
    images1.mkdir()
    images2.mkdir()
    (images1 / "1.png").touch()
    (images2 / "2.png").touch()
    args = options(images=str(images1), output=str(tmp_path / "session"),
                   target=None, min_views=15, max_rms=0.8, no_browser=True)
    monkeypatch.setattr("builtins.input", lambda prompt: str(images2))
    runs = []
    def calibrate(paths, output, *args):
        runs.append((paths, output))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).touch()
    monkeypatch.setattr(wizard, "calibrate_dataset", calibrate)
    decisions = iter(["retry", "applied"])
    monkeypatch.setattr(wizard, "review_and_apply", lambda *args: next(decisions))
    assert wizard.run_wizard(args)["status"] == "applied"
    assert runs[0][0] != runs[1][0]
    assert runs[0][1] != runs[1][1]
    assert all(Path(output).exists() for _, output in runs)


@pytest.mark.parametrize("decision", ["yes", "no"])
def test_cli_images_calculate_review_and_decide_end_to_end(tmp_path, monkeypatch, decision):
    import cv2
    from firesight_calib import application, wizard
    from firesight_calib.cli import main
    from firesight_calib.synthetic_thermal import generate_dataset
    from firesight_calib.workflow import read_image

    images = tmp_path / "images"
    generate_dataset(images)
    answers = iter([decision, "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(wizard, "open_source", lambda args: pytest.fail("offline calibration opened camera"))
    opened = []
    closed = []
    class Source:
        def read(self, timeout):
            return read_image(images / "view_000.png")
        def close(self):
            closed.append(True)
    def open_source(args):
        opened.append(True)
        return Source()
    monkeypatch.setattr(application, "open_source", open_source)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda timeout: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    output = tmp_path / "wizard"
    assert main(["wizard", "--source", "usb", "--images", str(images), "--output", str(output),
                 "--no-browser", "--apply-frames", "1"]) == 0
    debug = output / "attempt_001" / "boson_intrinsic_debug"
    assert (debug / "review.html").exists()
    audit = json.loads(next(debug.glob("decision_*.json")).read_text(encoding="utf-8"))
    assert audit["decision"] == decision
    assert len(opened) == len(closed) == (1 if decision == "yes" else 0)
    assert audit["application"]["status"] == ("applied" if decision == "yes" else "not_started")
