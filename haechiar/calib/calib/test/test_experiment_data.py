import json

import numpy as np
import pytest

from firesight_calib.experiments.model import Sample
from firesight_calib.experiments.sync import Synchronizer
from firesight_calib.experiments.session import SessionWriter, read_session
from firesight_calib.experiments.dataset import export_dataset


def reseal(path):
    from firesight_calib.experiments.session import sha256
    doc = json.loads((path / "session.json").read_text())
    doc["index_sha256"] = sha256(path / "samples.jsonl")
    (path / "session.json").write_text(json.dumps(doc))


def sample(stream="thermal", t=100, clock="host:test", provenance="measured"):
    return Sample(stream, t, clock, "boson_optical_frame", "thermal",
                  np.arange(12, dtype=np.uint16).reshape(3, 4), provenance)


@pytest.mark.parametrize("kwargs", [
    {"timestamp_ns": -1}, {"timestamp_ns": True}, {"clock": ""}, {"stream": "../escape"},
    {"data": np.zeros((3, 4, 3), np.uint8)}, {"data": np.zeros((3, 4), np.float32)},
    {"provenance": "estimated"},
])
def test_sample_rejects_ambiguous_data(kwargs):
    values = vars(sample()).copy()
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        Sample(**values)


def test_depth_requires_float_metres_and_allows_missing_values():
    with pytest.raises(ValueError, match="float"):
        Sample("depth", 1, "host:test", "depth_optical", "depth", np.ones((2, 2), np.uint16))
    Sample("depth", 1, "host:test", "depth_optical", "depth", np.array([[1., np.nan]]))


def test_sync_bounded_no_reuse_and_stale_rejection():
    sync = Synchronizer("thermal", ["thermal", "depth"], tolerance_ns=5, capacity=2)
    assert sync.push(sample(t=10)) == []
    assert sync.push(sample("depth", 30)) == []
    pairs = sync.push(sample(t=31))
    assert len(pairs) == 1
    assert pairs[0]["depth"].timestamp_ns == 30
    assert sync.push(sample(t=32)) == []
    for t in range(40, 50):
        sync.push(sample(t=t))
    assert max(len(q) for q in sync.queues.values()) <= 2
    assert sync.dropped > 0


def test_sync_matches_nearest_available_and_checks_clock_and_order():
    sync = Synchronizer("thermal", ["thermal", "depth"], 10, 4)
    sync.push(sample("depth", 90))
    sync.push(sample("depth", 99))
    assert sync.push(sample(t=100))[0]["depth"].timestamp_ns == 99
    with pytest.raises(ValueError, match="clock"):
        sync.push(sample(t=101, clock="device:unknown"))
    with pytest.raises(ValueError, match="increas"):
        sync.push(sample(t=100))


def test_session_roundtrip_preserves_native_and_rejects_overwrite(tmp_path):
    path = tmp_path / "session"
    with SessionWriter(path, {"group_id": "room1", "mode": "thermal_ai"}) as writer:
        writer.append(sample())
        writer.append(sample(t=200))
    actual = list(read_session(path))
    assert actual[0].data.dtype == np.uint16
    np.testing.assert_array_equal(actual[0].data, sample().data)
    assert actual[1].timestamp_ns == 200
    assert json.loads((path / "session.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        SessionWriter(path, {})


def test_session_checksum_and_path_containment(tmp_path):
    path = tmp_path / "session"
    with SessionWriter(path, {"group_id": "room1"}) as writer:
        writer.append(sample())
    payload = next((path / "samples").glob("*.npz"))
    payload.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        list(read_session(path))
    entry = json.loads((path / "samples.jsonl").read_text())
    entry["file"] = "../elsewhere.npz"
    (path / "samples.jsonl").write_text(json.dumps(entry))
    reseal(path)
    with pytest.raises(ValueError, match="within"):
        list(read_session(path))


def test_session_marks_failed_and_rejects_nonmonotonic(tmp_path):
    path = tmp_path / "session"
    with pytest.raises(ValueError):
        with SessionWriter(path, {}) as writer:
            writer.append(sample())
            writer.append(sample(t=99))
    assert json.loads((path / "session.json").read_text())["status"] == "interrupted"


def test_dataset_splits_entire_groups_and_excludes_synthetic(tmp_path):
    sessions = []
    for i, group in enumerate(["a", "a", "b", "c", "d", "e"]):
        path = tmp_path / str(i)
        with SessionWriter(path, {"group_id": group}) as writer:
            writer.append(sample(provenance="synthetic" if i == 5 else "measured"))
        sessions.append(path)
    output = tmp_path / "dataset"
    report = export_dataset(sessions, output, seed=7)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and report["excluded_synthetic"] == 1
    assert len({r["split"] for r in rows if r["group_id"] == "a"}) == 1
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert all(r["label"] is None for r in rows)
    with pytest.raises(FileExistsError):
        export_dataset(sessions, output)


def test_dataset_requires_enough_independent_groups(tmp_path):
    path = tmp_path / "session"
    with SessionWriter(path, {"group_id": "a"}) as writer:
        writer.append(sample())
    with pytest.raises(ValueError, match="three"):
        export_dataset([path], tmp_path / "dataset")


def test_session_rejects_stream_format_changes(tmp_path):
    with SessionWriter(tmp_path / "session", {}) as writer:
        writer.append(sample())
        changed = Sample("thermal", 200, "host:test", "boson_optical_frame", "thermal",
                         np.zeros((6, 8), np.uint16))
        with pytest.raises(ValueError, match="format"):
            writer.append(changed)


def test_dataset_references_actual_manifest_paths(tmp_path):
    sessions = []
    for i in range(3):
        path = tmp_path / str(i)
        with SessionWriter(path, {"group_id": str(i)}) as writer:
            writer.append(sample())
        entry = json.loads((path / "samples.jsonl").read_text())
        (path / entry["file"]).rename(path / "samples" / "renamed.npz")
        entry["file"] = "samples/renamed.npz"
        (path / "samples.jsonl").write_text(json.dumps(entry))
        reseal(path)
        sessions.append(path)
    export_dataset(sessions, tmp_path / "dataset")
    rows = [json.loads(line) for line in (tmp_path / "dataset" / "manifest.jsonl").read_text().splitlines()]
    assert all(row["file"].endswith("renamed.npz") for row in rows)


def test_session_rejects_provenance_relabeling_even_if_index_rehashed(tmp_path):
    path = tmp_path / "session"
    with SessionWriter(path, {}) as writer:
        writer.append(sample(provenance="synthetic"))
    entry = json.loads((path / "samples.jsonl").read_text())
    entry["provenance"] = "measured"
    (path / "samples.jsonl").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="index checksum"):
        list(read_session(path))
    reseal(path)
    with pytest.raises(ValueError, match="provenance"):
        list(read_session(path))
