from __future__ import annotations

from pathlib import Path

from .io import load_yaml
from .target import load_target


def verify_configs(sensors_path: str | Path, target_path: str | Path, quality_path: str | Path) -> dict:
    sensors = load_yaml(sensors_path)
    target = load_yaml(target_path)
    quality = load_yaml(quality_path)
    required_sensors = {"boson", "d456", "livox"}
    missing = required_sensors - set(sensors.get("sensors", {}))
    if missing:
        raise ValueError("missing sensor sections: " + ", ".join(sorted(missing)))
    load_target(target_path)
    thresholds = quality.get("thresholds", {})
    required_quality = {"boson_intrinsic_rms_px", "extrinsic_translation_repeat_mm", "extrinsic_rotation_repeat_deg"}
    if required_quality - set(thresholds):
        raise ValueError("quality thresholds are incomplete")
    return {
        "valid": True,
        "rig_id": sensors.get("rig_id"),
        "target_status": target.get("status", "unknown"),
        "warning": "target is provisional; do not promote results" if target.get("status") != "verified" else None,
    }

