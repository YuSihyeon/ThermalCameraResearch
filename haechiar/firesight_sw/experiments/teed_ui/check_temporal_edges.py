import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "edge/orin"))
from temporal_edges import TemporalEdgeConfig, TemporalEdgeStabilizer  # noqa: E402

checks = []


def check(name, passed):
    if not passed:
        raise AssertionError(name)
    checks.append(name)


def textured_frame(shift_x: float = 0.0, shift_y: float = 0.0, size: tuple[int, int] = (96, 128)):
    height, width = size
    y, x = np.indices((height, width), dtype=np.float32)
    base = (
        96.0
        + 55.0 * np.sin((x + shift_x) / 9.0)
        + 45.0 * np.cos((y + shift_y) / 7.0)
        + 30.0 * np.sin((x + y + shift_x + shift_y) / 13.0)
    )
    base = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def line_probability(x0, y0=45, value=0.8, size=(96, 128)):
    prob = np.zeros(size, np.float32)
    prob[y0, x0 : x0 + 30] = value
    return prob


def new_stabilizer(**kwargs):
    return TemporalEdgeStabilizer(TemporalEdgeConfig(**kwargs))


def main():
    stable = new_stabilizer()
    frame0 = textured_frame()
    first, first_metrics = stable.update(line_probability(30, value=0.8), frame0, frame_index=0, timestamp_s=0.0)
    second, second_metrics = stable.update(line_probability(30, value=0.45), textured_frame(0.4), frame_index=1, timestamp_s=1.0 / 30.0)
    check("first frame returns current", np.allclose(first, line_probability(30, value=0.8)))
    check("first frame reports reset", first_metrics["reset_reason"] == "first_frame")
    check("sustained line survives mild probability oscillation", float(second[45, 40]) > 0.55)
    check("normal update does not reset", second_metrics["temporal_reset"] == 0)

    flash = new_stabilizer()
    flash.update(np.zeros((96, 128), np.float32), frame0, frame_index=0, timestamp_s=0.0)
    moderate, _ = flash.update(line_probability(50, value=0.7), textured_frame(0.2), frame_index=1, timestamp_s=1.0 / 30.0)
    check("one-frame moderate flash suppressed", float(moderate[45, 60]) < 0.55)

    immediate = new_stabilizer()
    immediate.update(np.zeros((96, 128), np.float32), frame0, frame_index=0, timestamp_s=0.0)
    strong, strong_metrics = immediate.update(line_probability(50, value=0.8), textured_frame(0.2), frame_index=1, timestamp_s=1.0 / 30.0)
    check("display-high new edge immediately reflected", np.isclose(strong[45, 60], 0.8))
    check("immediate metric counts strong edge", int(strong_metrics["immediate_pixels"]) == 30)
    custom_immediate = new_stabilizer(immediate_threshold=0.6)
    custom_immediate.update(np.zeros((96, 128), np.float32), frame0, frame_index=0, timestamp_s=0.0)
    custom_strong, custom_metrics = custom_immediate.update(line_probability(50, value=0.65), textured_frame(0.2), frame_index=1, timestamp_s=1.0 / 30.0)
    check("custom immediate threshold reflects matching new edge", np.isclose(custom_strong[45, 60], 0.65))
    check("custom immediate metric counts matching new edge", int(custom_metrics["immediate_pixels"]) == 30)

    vanish = new_stabilizer()
    vanish.update(line_probability(30, value=0.9), frame0, frame_index=0, timestamp_s=0.0)
    gone, _ = vanish.update(np.zeros((96, 128), np.float32), textured_frame(0.2), frame_index=1, timestamp_s=1.0 / 30.0)
    check("disappearance leaves no unsupported ghosts", not np.any(gone >= 0.25))
    zero_support = new_stabilizer(support_threshold=0.0)
    zero_support.update(line_probability(30, value=0.9), frame0, frame_index=0, timestamp_s=0.0)
    zero_gone, _ = zero_support.update(np.zeros((96, 128), np.float32), textured_frame(0.2), frame_index=1, timestamp_s=1.0 / 30.0)
    check("exact current zero clears even when support threshold is zero", not np.any(zero_gone > 0.0))

    shifted = new_stabilizer()
    shifted.update(line_probability(30, value=0.85), textured_frame(0), frame_index=0, timestamp_s=0.0)
    moved, moved_metrics = shifted.update(line_probability(35, value=0.45), textured_frame(-5), frame_index=1, timestamp_s=1.0 / 30.0)
    check("translated textured scene preserves moved edge", float(moved[45, 45]) > 0.55)
    check("translated textured scene clears old location", float(moved[45, 32]) < 0.25)
    check("real flow registration accepted", float(moved_metrics["registration_ratio"]) >= 0.2)

    cut = new_stabilizer()
    cut.update(line_probability(30, value=0.85), frame0, frame_index=0, timestamp_s=0.0)
    cut_out, cut_metrics = cut.update(line_probability(30, value=0.5), np.full_like(frame0, 240), frame_index=1, timestamp_s=1.0 / 30.0)
    check("poor registration returns current exactly", np.allclose(cut_out, line_probability(30, value=0.5)))
    check("poor registration resets history", cut_metrics["reset_reason"] == "poor_registration")

    gap = new_stabilizer()
    gap.update(line_probability(30, value=0.85), frame0, frame_index=0, timestamp_s=0.0)
    gap_out, gap_metrics = gap.update(line_probability(30, value=0.5), textured_frame(), frame_index=1, timestamp_s=1.0)
    check("time gap returns current exactly", np.allclose(gap_out, line_probability(30, value=0.5)))
    check("time gap reset reported", gap_metrics["reset_reason"] == "time_gap")

    discontinuity = new_stabilizer()
    discontinuity.update(line_probability(30, value=0.85), frame0, frame_index=3)
    _, discontinuity_metrics = discontinuity.update(line_probability(30, value=0.5), textured_frame(), frame_index=5)
    check("frame index discontinuity resets", discontinuity_metrics["reset_reason"] == "frame_discontinuity")

    backwards = new_stabilizer()
    backwards.update(line_probability(30, value=0.85), frame0, timestamp_s=2.0)
    _, backwards_metrics = backwards.update(line_probability(30, value=0.5), textured_frame(), timestamp_s=1.5)
    check("backwards timestamp resets", backwards_metrics["reset_reason"] == "timestamp_discontinuity")

    shaped = new_stabilizer()
    shaped.update(np.zeros((96, 128), np.float32), frame0)
    small = np.zeros((64, 64), np.float32)
    _, shape_metrics = shaped.update(small, np.zeros((64, 64, 3), np.uint8))
    check("shape change resets", shape_metrics["reset_reason"] == "shape_changed")

    osd = new_stabilizer()
    ignored = ((0.0, 0.4, 0.5, 0.6),)
    osd_out, osd_metrics = osd.update(line_probability(30, value=0.99), frame0, ignore_regions=ignored)
    check("OSD region masked before history", not np.any(osd_out[38:58, :64]))
    check("OSD metrics report ignored pixels", int(osd_metrics["ignored_region_pixels"]) > 0)
    _, osd_change_metrics = osd.update(
        line_probability(70, value=0.5),
        textured_frame(0.1),
        ignore_regions=((0.5, 0.4, 1.0, 0.6),),
    )
    check("ignore region changes reset history", osd_change_metrics["reset_reason"] == "ignore_regions_changed")
    prev_hole = new_stabilizer()
    prev_hole._history = line_probability(30, value=0.9)
    prev_hole._prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_hole._prev_active_region = np.ones((96, 128), dtype=bool)
    prev_hole._prev_active_region[44:47, 30:60] = False
    prev_hole._prev_shape = (96, 128)
    prev_hole._prev_ignore_regions = ()
    exposed, exposed_metrics = prev_hole.update(
        line_probability(30, value=0.5), textured_frame(0.1), frame_index=1
    )
    check("remapped previous OSD holes do not blend zero history", np.allclose(exposed[45, 35:55], 0.5))
    check("previous OSD holes lower support count", int(exposed_metrics["history_supported_pixels"]) < 30)

    timed = new_stabilizer()
    timed.update(line_probability(30, value=0.8), frame0, timestamp_s=0.0)
    _, fast_metrics = timed.update(line_probability(30, value=0.5), textured_frame(0.1), timestamp_s=1.0 / 60.0)
    check("timestamp weight normalized to 30 fps", float(fast_metrics["history_weight"]) > 0.6)

    for key, value in (
        ("history_weight", -0.1),
        ("history_weight", 1.0),
        ("immediate_threshold", 1.1),
        ("support_threshold", -0.1),
        ("max_gap_seconds", 0.0),
        ("flow_fb_error_px", -1.0),
        ("photometric_error", -1.0),
        ("min_registration_ratio", 1.1),
    ):
        try:
            TemporalEdgeConfig(**{key: value})
        except ValueError:
            checks.append(f"reject {key}={value}")
        else:
            raise AssertionError(f"accepted invalid {key}")

    invalid = new_stabilizer()
    for name, probability, frame in (
        ("bad probability rank", np.zeros((2, 3, 4), np.float32), frame0),
        ("bad frame dtype", np.zeros((96, 128), np.float32), frame0.astype(np.float32)),
        ("bad frame shape", np.zeros((96, 128), np.float32), np.zeros((95, 128, 3), np.uint8)),
    ):
        try:
            invalid.update(probability, frame)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(f"accepted {name}")

    perf = new_stabilizer()
    perf_frame = textured_frame(size=(256, 320))
    perf_prob = np.zeros((256, 320), np.float32)
    perf_prob[120, 60:260] = 0.6
    perf.update(perf_prob, perf_frame, frame_index=0)
    start = time.perf_counter()
    perf.update(perf_prob, textured_frame(0.5, size=(256, 320)), frame_index=1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    check("default 320x256 update stays under 200 ms synthetic budget", elapsed_ms < 200.0)

    print(f"PASS: {len(checks)} temporal edge contracts ({elapsed_ms:.1f} ms 320x256 synthetic update)")
    return checks


if __name__ == "__main__":
    main()
