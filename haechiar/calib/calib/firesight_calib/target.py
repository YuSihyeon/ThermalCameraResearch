"""Board geometry, in meters; ordering matches OpenCV's logical rows."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from .io import load_yaml


@dataclass(frozen=True)
class Target:
    pattern_type: str = "asymmetric_circles"
    pattern: tuple[int, int] = (3, 9)
    spacing_m: float = 0.06
    diameter_m: float = 0.06
    origin_m: tuple[float, float] = (0.06, 0.05)
    outer_size_m: tuple[float, float] = (0.6, 0.4)
    status: str = "provisional"
    revision: str = "boson320_27circles_quote_v1"

    def __post_init__(self):
        if self.pattern_type not in ("asymmetric_circles", "checkerboard"):
            raise ValueError("pattern_type must be asymmetric_circles or checkerboard")
        if len(self.pattern) != 2 or any(type(v) is not int or v < 2 for v in self.pattern):
            raise ValueError("pattern_size must contain two integers >= 2")
        if not np.isfinite(self.spacing_m) or self.spacing_m <= 0:
            raise ValueError("spacing_m must be positive and finite")
        if not np.isfinite(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("diameter_m must be positive and finite")
        if len(self.origin_m) != 2 or len(self.outer_size_m) != 2:
            raise ValueError("origin_m and outer_size_m must have two values")
        if not np.isfinite([*self.origin_m, *self.outer_size_m]).all() or min(self.outer_size_m) <= 0:
            raise ValueError("board dimensions must be finite and positive")
        if self.status not in ("provisional", "verified"):
            raise ValueError("target status must be provisional or verified")
        if self.pattern_type == "asymmetric_circles":
            points = self.object_points()[:, :2]
            radius = self.diameter_m / 2
            if np.any(points.min(axis=0) - radius < -1e-6) or np.any(
                points.max(axis=0) + radius > np.array(self.outer_size_m) + 1e-6
            ):
                raise ValueError("circles extend outside the board")
            if self.diameter_m >= np.sqrt(2) * self.spacing_m:
                raise ValueError("circles overlap or touch; decrease diameter_m")

    def object_points(self) -> np.ndarray:
        cols, rows = self.pattern
        if self.pattern_type == "asymmetric_circles":
            xy = [
                (self.origin_m[0] + i * self.spacing_m, self.origin_m[1] + (2 * j + i % 2) * self.spacing_m)
                for i in range(rows)
                for j in range(cols)
            ]
        else:
            xy = [(j * self.spacing_m, i * self.spacing_m) for i in range(rows) for j in range(cols)]
        return np.column_stack([xy, np.zeros(len(xy))]).astype(np.float32)

    def metadata(self) -> dict:
        value = asdict(self)
        value["object_points_m"] = self.object_points().tolist()
        return value


def load_target(path: str | Path | None = None) -> Target:
    if path is None:
        return Target()
    data = load_yaml(path)
    if "checkerboard" in data:
        board = data["checkerboard"]
        return Target(
            pattern_type="checkerboard",
            pattern=(board["internal_corners_x"], board["internal_corners_y"]),
            spacing_m=float(board["square_length_m"]),
            status=data.get("status", "provisional"),
        )
    for key in ("pattern_type", "pattern_size", "spacing_m"):
        if key not in data:
            raise ValueError(f"target missing {key}")
    return Target(
        pattern_type=data["pattern_type"],
        pattern=tuple(data["pattern_size"]),
        spacing_m=float(data["spacing_m"]),
        diameter_m=float(data.get("diameter_m", 0.06)),
        origin_m=tuple(data.get("origin_m", [0.06, 0.05])),
        outer_size_m=tuple(data.get("outer_size_m", [0.6, 0.4])),
        status=data.get("status", "provisional"),
        revision=data.get("revision", "custom"),
    )
