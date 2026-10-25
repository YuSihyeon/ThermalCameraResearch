from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import Sample


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("sample file must remain within the session directory")
    return path


def _format(sample):
    shape = sample.data.shape[1:] if sample.kind == "radar" else sample.data.shape
    return sample.kind, sample.frame, str(sample.data.dtype), shape


class SessionWriter:
    def __init__(self, path, metadata):
        json.dumps(metadata, allow_nan=False)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "samples").mkdir()
        self.document = {"schema": "firesight.experiment_session/v1", "status": "recording",
                         "created_utc": datetime.now(timezone.utc).isoformat(), "metadata": metadata,
                         "samples": 0}
        write_json(self.path / "session.json", self.document)
        self.index = (self.path / "samples.jsonl").open("x", encoding="utf-8")
        self.last = {}
        self.formats = {}
        self.clock = None
        self.provenance = None

    def append(self, sample: Sample):
        if self.index.closed:
            raise ValueError("session is closed")
        if sample.timestamp_ns <= self.last.get(sample.stream, -1):
            raise ValueError("sample timestamps must strictly increase within stream")
        if self.clock is not None and self.clock != sample.clock:
            raise ValueError("session requires a shared clock domain")
        if self.provenance is not None and self.provenance != sample.provenance:
            raise ValueError("cannot mix synthetic and measured samples in one session")
        if sample.stream in self.formats and self.formats[sample.stream] != _format(sample):
            raise ValueError("stream format/frame cannot change within a session")
        name = f"samples/{self.document['samples']:08d}.npz"
        path = self.path / name
        np.savez_compressed(path, data=sample.data)
        entry = {**sample.header(), "file": name, "sha256": sha256(path),
                 "shape": list(sample.data.shape), "dtype": str(sample.data.dtype)}
        self.index.write(json.dumps(entry, allow_nan=False) + "\n")
        self.index.flush()
        self.last[sample.stream] = sample.timestamp_ns
        self.formats[sample.stream] = _format(sample)
        self.clock, self.provenance = sample.clock, sample.provenance
        self.document["samples"] += 1

    def close(self, status="complete"):
        if not self.index.closed:
            self.index.close()
            self.document.update(status=status, clock=self.clock, provenance=self.provenance)
            self.document["index_sha256"] = sha256(self.path / "samples.jsonl")
            write_json(self.path / "session.json", self.document)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close("interrupted" if kind else "complete")


def read_session(path):
    for _, sample in read_records(path):
        yield sample


def read_records(path):
    """Stream verified samples, including recoverable entries of interrupted sessions."""
    root = Path(path)
    document = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if document.get("schema") != "firesight.experiment_session/v1":
        raise ValueError("unsupported session schema")
    if document.get("status") not in ("recording", "interrupted", "complete"):
        raise ValueError("invalid session status")
    if document["status"] != "recording":
        if document.get("index_sha256") != sha256(root / "samples.jsonl"):
            raise ValueError("session index checksum mismatch")
    last, clock, provenance, count = {}, None, None, 0
    formats = {}
    with (root / "samples.jsonl").open(encoding="utf-8") as index:
        for line in index:
            entry = json.loads(line)
            record = entry.copy()
            payload = contained_path(root, entry.pop("file"))
            if sha256(payload) != entry.pop("sha256"):
                raise ValueError(f"sample checksum mismatch: {payload}")
            shape, dtype = entry.pop("shape"), entry.pop("dtype")
            with np.load(payload, allow_pickle=False) as data:
                sample = Sample(data=data["data"], **entry)
            if document["status"] != "recording":
                if sample.provenance != document.get("provenance"):
                    raise ValueError("sample provenance differs from session provenance")
                if sample.clock != document.get("clock"):
                    raise ValueError("sample clock differs from session clock")
            if list(sample.data.shape) != shape or str(sample.data.dtype) != dtype:
                raise ValueError("sample shape/dtype does not match manifest")
            if sample.timestamp_ns <= last.get(sample.stream, -1):
                raise ValueError("sample timestamps must strictly increase")
            if clock is not None and sample.clock != clock:
                raise ValueError("session has mixed clock domains")
            if provenance is not None and sample.provenance != provenance:
                raise ValueError("session has mixed provenance")
            if sample.stream in formats and formats[sample.stream] != _format(sample):
                raise ValueError("stream format/frame cannot change within a session")
            formats[sample.stream] = _format(sample)
            last[sample.stream] = sample.timestamp_ns
            clock, provenance = sample.clock, sample.provenance
            count += 1
            yield record, sample
    if document["status"] != "recording" and count != document["samples"]:
        raise ValueError("session sample count mismatch")
