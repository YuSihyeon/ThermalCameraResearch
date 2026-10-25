from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

import numpy as np


@dataclass(frozen=True)
class Sample:
    stream: str
    timestamp_ns: int
    clock: str
    frame: str
    kind: str
    data: np.ndarray
    provenance: str = "measured"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.stream, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.stream):
            raise ValueError("stream must be a simple identifier")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if not isinstance(self.clock, str) or not self.clock.strip():
            raise ValueError("clock must name a shared clock domain")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must name the physical coordinate frame")
        if self.provenance not in ("measured", "synthetic"):
            raise ValueError("provenance must be measured or synthetic")
        a = np.asarray(self.data)
        if a.dtype.kind not in "uif" or a.dtype.kind == "O":
            raise ValueError("data must be a real numeric array")
        if self.kind == "thermal":
            valid = a.ndim == 2 and a.size > 0 and a.dtype in (np.dtype("uint8"), np.dtype("uint16"))
        elif self.kind == "rgb":
            valid = a.ndim == 3 and a.shape[2] == 3 and a.size > 0 and a.dtype == np.uint8
        elif self.kind == "depth":
            valid = a.ndim == 2 and a.size > 0 and a.dtype.kind == "f"
            if not valid:
                raise ValueError("depth must be a 2D float array in metres; missing values may be NaN")
            if np.isinf(a).any() or np.any(a[np.isfinite(a)] < 0):
                raise ValueError("depth cannot be negative or infinite")
        elif self.kind == "radar":
            valid = a.ndim == 2 and a.shape[1] == 3 and a.dtype.kind == "f" and np.isfinite(a).all()
        elif self.kind == "imu":
            valid = a.shape == (6,) and a.dtype.kind == "f" and np.isfinite(a).all()
        else:
            raise ValueError(f"unsupported kind: {self.kind}")
        if not valid:
            raise ValueError(f"invalid {self.kind} shape/dtype: {a.shape}/{a.dtype}")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        json.dumps(self.metadata, allow_nan=False)
        # SDKs frequently reuse their buffers after the next read.
        object.__setattr__(self, "data", a.copy())

    def header(self):
        return {k: v for k, v in vars(self).items() if k != "data"}
