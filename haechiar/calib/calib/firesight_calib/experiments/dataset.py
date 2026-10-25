from __future__ import annotations

import json
from pathlib import Path
import random

from .session import read_records, write_json, contained_path


def export_dataset(sessions, output, seed=42):
    """Reference original thermal arrays; never fabricate labels or split adjacent frames."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, excluded = [], 0
    seen = set()
    for source in sessions:
        source = Path(source).resolve()
        if source in seen:
            raise ValueError("duplicate session in dataset input")
        seen.add(source)
        document = json.loads((source / "session.json").read_text(encoding="utf-8"))
        if document["status"] != "complete":
            raise ValueError("dataset export requires complete sessions")
        group = document["metadata"].get("group_id")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("group_id is required to avoid capture leakage")
        for index, (record, sample) in enumerate(read_records(source)):
            if sample.kind != "thermal":
                continue
            if sample.provenance != "measured":
                excluded += 1
                continue
            payload = contained_path(source, record["file"])
            rows.append({"session": str(source), "sample_index": index, "file": str(payload),
                         "sha256": record["sha256"], "stream": sample.stream,
                         "timestamp_ns": sample.timestamp_ns, "group_id": group,
                         "label": None, "label_status": "unlabeled"})
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < 3:
        raise ValueError("at least three independent measured groups are required for train/val/test")
    random.Random(seed).shuffle(groups)
    holdout = max(1, int(len(groups) * 0.15))
    assignments = {g: "test" if i < holdout else "val" if i < 2 * holdout else "train"
                   for i, g in enumerate(groups)}
    output.mkdir(parents=True)
    with (output / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            row["split"] = assignments[row["group_id"]]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"schema": "firesight.thermal_dataset/v1", "seed": seed, "samples": len(rows),
              "groups": assignments, "excluded_synthetic": excluded,
              "split_policy": "group-level 70/15/15 rounded with minimum one holdout group",
              "training_ready": False, "reason": "labels and model are not selected"}
    write_json(output / "dataset.json", report)
    return report
