"""Explicitly registered replay sidecars; no temperature/depth inference from video."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AlignedMaps:
    """NPZ maps keyed by original source frame index and tied to its SHA-256.

    Each map is N,H,W at source-image resolution. Registration and calibration
    are supplied by the producer and recorded as provenance, not inferred here.
    """

    def __init__(self, path: str, source: str) -> None:
        source_path = Path(source)
        if not source_path.is_file():
            raise ValueError("--aligned-maps requires a video/image file source")
        with np.load(path, allow_pickle=False) as data:
            required = {"registered_to", "source_sha256", "frame_indices"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    "sidecar missing required fields: " + ", ".join(sorted(missing))
                )
            if str(data["registered_to"].item()) != "source_image":
                raise ValueError("sidecar must be registered_to='source_image'")
            if str(data["source_sha256"].item()) != source_sha256(source_path):
                raise ValueError("sidecar source SHA-256 mismatch")
            indices = data["frame_indices"]
            if (
                indices.ndim != 1
                or not len(indices)
                or indices.dtype.kind not in "iu"
                or (indices < 0).any()
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError("frame_indices must be unique nonnegative integers")
            self.indices = {int(index): row for row, index in enumerate(indices)}
            self.maps = {}
            self.provenance = {}
            for key, unit in (("temperature_c", "Celsius"), ("depth_m", "m")):
                if key not in data:
                    continue
                metadata = {key + "_unit", key + "_provenance"}
                missing = metadata.difference(data.files)
                if missing:
                    raise ValueError(
                        "sidecar missing map metadata: " + ", ".join(sorted(missing))
                    )
                if str(data[key + "_unit"].item()) != unit:
                    raise ValueError(key + " has an unsupported unit")
                provenance = str(data[key + "_provenance"].item()).strip()
                if not provenance:
                    raise ValueError(
                        key + " requires calibration/registration provenance"
                    )
                values = data[key]
                if (
                    values.dtype.kind != "f"
                    or values.ndim != 3
                    or values.shape[0] != len(indices)
                    or min(values.shape[1:]) < 1
                ):
                    raise ValueError(key + " must be a float N,H,W array")
                self.maps[key] = values.astype(np.float32, copy=False)
                self.provenance[key] = provenance
            if not self.maps:
                raise ValueError("sidecar contains no temperature_c or depth_m")

    def at(
        self, index: int, shape: tuple[int, int]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if index not in self.indices:
            raise ValueError("aligned maps missing source frame {}".format(index))
        row = self.indices[index]
        for key, values in self.maps.items():
            if values.shape[1:] != shape:
                raise ValueError(key + " shape differs from the original source frame")
        temperature = (
            self.maps["temperature_c"][row] if "temperature_c" in self.maps else None
        )
        depth = self.maps["depth_m"][row] if "depth_m" in self.maps else None
        return temperature, depth
