from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..io import load_yaml

MODES = {"rgbd_thermal": ["thermal", "rgb", "depth"], "thermal_ai": ["thermal"],
         "radar_thermal": ["thermal", "radar"]}


def load_profile(value):
    path = files(__package__).joinpath("profiles", f"{value}.yaml") if value in MODES else Path(value)
    profile = load_yaml(path)
    if profile.get("schema") != "firesight.experiment_profile/v1" or profile.get("mode") not in MODES:
        raise ValueError("invalid experiment profile schema/mode")
    if profile.get("streams") != MODES[profile["mode"]]:
        raise ValueError("profile streams must match the experiment's required streams")
    if not isinstance(profile.get("sources"), list):
        raise ValueError("profile sources must be an array")
    for source in profile["sources"]:
        if (not isinstance(source, dict) or not isinstance(source.get("factory"), str)
                or source["factory"].count(":") != 1 or not isinstance(source.get("options", {}), dict)):
            raise ValueError("each source needs module:factory and options object")
    profile["_directory"] = str(Path(str(path)).resolve().parent)
    return profile


def readiness(profile):
    capture_blockers = [] if profile["sources"] else ["device factories/options have not been configured"]
    if not profile.get("group_id"):
        capture_blockers.append("group_id for the capture location/sequence is required")
    fusion_blockers = []
    if profile["mode"] != "thermal_ai":
        if not profile.get("registration"):
            fusion_blockers.append("measured source-to-thermal registration is required")
        else:
            from .calibration import Registration
            try:
                registration = Registration.load(Path(profile["_directory"]) / profile["registration"])
                if registration.provenance != "measured":
                    fusion_blockers.append("profile requires measured registration, not synthetic")
                if profile["mode"] == "rgbd_thermal" and registration.source_camera is None:
                    fusion_blockers.append("depth source_intrinsic is required")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                fusion_blockers.append(str(exc))
    return {"mode": profile["mode"], "can_demo": True, "can_capture": not capture_blockers,
            "can_fuse": not fusion_blockers, "capture_blockers": capture_blockers,
            "fusion_blockers": fusion_blockers, "hardware_verified": False,
            "pending": profile.get("pending", []),
            "ai_model_configured": bool(profile.get("edge_model"))}
