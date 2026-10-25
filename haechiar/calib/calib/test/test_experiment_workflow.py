import json
import time

import numpy as np
import pytest

from firesight_calib.cli import main
from firesight_calib.experiments.config import load_profile, readiness
from firesight_calib.experiments.demo import run_demo
from firesight_calib.experiments.model import Sample
from firesight_calib.experiments.capture import capture_sources
from firesight_calib.experiments.session import read_session


@pytest.mark.parametrize("mode", ["rgbd_thermal", "thermal_ai", "radar_thermal"])
def test_three_modes_roundtrip_without_hardware(tmp_path, mode):
    output = tmp_path / mode
    report = run_demo(mode, output, frames=4)
    assert report["frames"] == 4
    assert report["provenance"] == "synthetic"
    assert report["hardware_verified"] is False
    assert (output / "results" / "preview.png").is_file()
    raw = list(read_session(output / "session"))
    assert raw[0].data.dtype == np.uint16
    assert all(s.provenance == "synthetic" for s in raw)
    with pytest.raises(FileExistsError):
        run_demo(mode, output)


def test_profiles_leave_undecided_fields_empty():
    for mode in ("rgbd_thermal", "thermal_ai", "radar_thermal"):
        profile = load_profile(mode)
        assert profile["mode"] == mode
        assert profile["registration"] is None
        assert profile["sources"] == []
        assert "lidar" not in profile["streams"]
        report = readiness(profile)
        assert report["can_demo"] is True
        assert report["can_capture"] is False
        if mode != "thermal_ai":
            assert report["can_fuse"] is False


def test_cli_demo_and_readiness(tmp_path, capsys):
    assert main(["experiment", "status", "--profile", "radar_thermal"]) == 0
    assert json.loads(capsys.readouterr().out)["can_fuse"] is False
    assert main(["experiment", "demo", "--mode", "thermal_ai", "--frames", "2",
                 "--output", str(tmp_path / "demo")]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 2


class TestSource:
    __test__ = False

    def __init__(self, options, clock):
        self.clock = clock
        self.mode = options.get("mode")
        self.marker = options["marker"]

    def read(self, timeout_s):
        if self.mode == "error":
            raise RuntimeError("device disconnected")
        if self.mode == "hang":
            time.sleep(10)
        time.sleep(0.01)
        return [Sample("thermal", time.monotonic_ns(), self.clock, "boson_optical_frame", "thermal",
                       np.zeros((3, 5), np.uint16))]

    def close(self):
        from pathlib import Path
        Path(self.marker).write_text("closed")


def source_factory(options, clock):
    return TestSource(options, clock)


def test_capture_saves_samples_and_closes_workers(tmp_path):
    marker = tmp_path / "closed"
    report = capture_sources(
        [{"factory": "test_experiment_workflow:source_factory", "options": {"marker": str(marker)}}],
        tmp_path / "capture", {"group_id": "a", "mode": "thermal_ai"}, duration_s=3, max_samples=3)
    assert report["samples"] == 3
    assert marker.exists()
    assert len(list(read_session(tmp_path / "capture"))) == 3


def test_capture_failure_is_not_reported_complete(tmp_path):
    marker = tmp_path / "closed"
    with pytest.raises(RuntimeError, match="disconnected"):
        capture_sources(
            [{"factory": "test_experiment_workflow:source_factory",
              "options": {"marker": str(marker), "mode": "error"}}],
            tmp_path / "capture", {}, duration_s=3)
    doc = json.loads((tmp_path / "capture" / "session.json").read_text())
    assert doc["status"] == "interrupted"
    assert marker.exists()


def test_capture_hung_device_has_bounded_exit(tmp_path):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="no samples"):
        capture_sources(
            [{"factory": "test_experiment_workflow:source_factory",
              "options": {"marker": str(tmp_path / "closed"), "mode": "hang"}}],
            tmp_path / "capture", {}, duration_s=0.5)
    assert time.monotonic() - started < 5


def test_capture_missing_required_radar_is_interrupted(tmp_path):
    with pytest.raises(RuntimeError, match="radar"):
        capture_sources(
            [{"factory": "test_experiment_workflow:source_factory", "options": {"marker": str(tmp_path / "closed")}}],
            tmp_path / "capture", {"mode": "radar_thermal"}, duration_s=3, max_samples=3)
    doc = json.loads((tmp_path / "capture" / "session.json").read_text())
    assert doc["status"] == "interrupted"
