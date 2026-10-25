from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain(data), handle, allow_unicode=True, sort_keys=False)
    return target


def camera_from_yaml(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = load_yaml(path)
    matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(data.get("distortion_coefficients", []), dtype=np.float64).reshape(-1)
    size = tuple(int(v) for v in data["image_size"])
    if len(size) != 2:
        raise ValueError("image_size must be [width, height]")
    return matrix, distortion, (size[0], size[1])


def transform_from_yaml(path: str | Path) -> np.ndarray:
    data = load_yaml(path)
    key = next((k for k in data if k.startswith("T_boson_from_")), None)
    if key is None:
        key = "transform"
    transform = np.asarray(data[key], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{key} must be a 4x4 matrix")
    return transform

