"""Audit an already composited Boson recording; never infer heat from HUD colours.

Uses the existing YSH thinning experiment and existing RGB heuristic unchanged.
The preview contains only recovered display strokes, not recovered thermal data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/obstacle_edges"))
from contours import thin
sys.path.insert(0, str(ROOT / "edge/orin"))
from replay import to_bgr


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def existing_colour_gate():
    # The historical helper now lives in a torch-free module. It is retained
    # for reproducing this audit, not used as a detector in the new runtime.
    from colour_candidates import fire_candidate_mask
    return fire_candidate_mask


def green_strokes(frame, saturation=100, value=100):
    # Compression makes the rendered green imperfect. These thresholds recover
    # a display mask approximately; they are not TEED confidence thresholds.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (40, saturation, value), (85, 255, 255)) > 0


def edge_image(mask):
    result = np.zeros((*mask.shape, 3), np.uint8)
    result[mask] = (0, 255, 0)
    return result


def card(frame, label):
    # Enlarge both versions equally for review; one source pixel becomes two
    # review pixels. This is explicitly not the proposed final-display renderer.
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    out = np.zeros((frame.shape[0]+42, frame.shape[1], 3), np.uint8)
    out[42:] = frame
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def verify_video(path, expected_frames, expected_fps, expected_size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != expected_size:
            raise RuntimeError(f"Unexpected frame size in {path}")
        count += 1
    cap.release()
    if count != expected_frames or abs(fps-expected_fps) > .001:
        raise RuntimeError(f"Unexpected timing in {path}: {count}, {fps}")
    return {"frames_decoded": count, "fps": fps, "size": list(expected_size),
            "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Source has no valid FPS")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(3)), int(cap.get(4)))
    videos = {name: args.output / (name + ".mp4") for name in
              ("extracted_green_only", "thin_green_only", "comparison")}
    preview_size = (size[0]*4, size[1]*2+42)
    writers = {}
    for name, path in videos.items():
        writers[name] = imageio_ffmpeg.write_frames(
            str(path), preview_size if name == "comparison" else size,
            fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            macro_block_size=1, quality=8, ffmpeg_log_level="error")
        writers[name].send(None)
    rows = []
    review_frames = []
    targets = {0, round(3*fps), round(6*fps), round(9*fps)}
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            original = green_strokes(frame)
            # Padding lets thinning handle strokes touching the image border.
            skeleton = thin(np.pad(original, 1))[1:-1, 1:-1]
            pair = np.hstack([
                card(frame, f"Original HUD / {index/fps:.2f}s"),
                card(edge_image(skeleton), "Thin strokes only / no thermal recovery")])
            writers["extracted_green_only"].send(edge_image(original))
            writers["thin_green_only"].send(edge_image(skeleton))
            writers["comparison"].send(pair)
            if index in targets:
                review_frames.append(pair)
            rows.append({"frame": index, "seconds": index/fps,
                         "green_pixels": int(original.sum()),
                         "thin_pixels": int(skeleton.sum()),
                         "green_pixels_strict": int(green_strokes(frame, 140, 140).sum()),
                         "green_pixels_loose": int(green_strokes(frame, 70, 70).sum())})
            index += 1
    finally:
        cap.release()
        for writer in writers.values():
            writer.close()
    if not rows or len(rows) != count:
        raise RuntimeError(f"Incomplete decode: expected {count}, got {len(rows)}")
    save_png(args.output / "comparison.png", np.vstack(review_frames))

    gate = existing_colour_gate()
    colour_rows, colour_cards = [], []
    for label, colour in [("Brown-like flat patch", (70, 120, 180)),
                          ("Orange flat patch", (0, 140, 255)),
                          ("Gray flat patch", (180, 180, 180))]:
        scene = np.full((120, 160, 3), 20, np.uint8)
        scene[25:95, 30:130] = colour
        edges = np.zeros(scene.shape[:2], np.uint8)
        # Controlled edge input isolates the colour rule; this is not a TEED
        # detection benchmark or a photograph of the reported cardboard box.
        cv2.rectangle(edges, (30, 25), (129, 94), 255, 1)
        highlight = gate(scene, edges)
        colour_rows.append({"case": label, "bgr": colour,
                            "red_pixels": int(np.count_nonzero(highlight))})
        scene[edges > 0] = (0, 255, 0)
        scene[highlight > 0] = (0, 0, 255)
        colour_cards.append(card(scene, label))
    save_png(args.output / "rgb_rule_reproduction.png", np.hstack(colour_cards))
    _, origin = to_bgr(np.full((8, 8, 3), 120, np.uint8), expected="gray8")

    pixels = size[0]*size[1]
    summary = {
        "protocol": "boson_composited_hud_audit_v1",
        "source": {"filename": args.input.name, "sha256": digest(args.input),
                   "size": size, "fps": fps, "decoded_frames": len(rows),
                   "duration_seconds": len(rows)/fps,
                   "interpretation": "already composited HUD; no raw thermal/probability map"},
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                        [Path(__file__), ROOT / "edge/orin/hud.py", ROOT / "edge/orin/colour_candidates.py", ROOT / "edge/orin/replay.py",
                         ROOT / "experiments/obstacle_edges/contours.py"]},
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__,
                     "imageio_ffmpeg": imageio_ffmpeg.__version__, "python": sys.version},
        "mean_display_coverage_percent": {key: float(np.mean([r[key] for r in rows])/pixels*100)
                                          for key in ("green_pixels", "thin_pixels", "green_pixels_strict", "green_pixels_loose")},
        "rgb_existing_rule_synthetic_cases": colour_rows,
        "gray8_request_three_channel_result": origin,
        "limitations": ["No obstacle accuracy or fire sensitivity/specificity evaluation",
                        "Thinning an encoded overlay cannot undo merged/lost edges",
                        "Background under coloured strokes cannot be recovered",
                        "No new TEED inference, Jetson performance or temperature measurement"],
        "videos": {name: verify_video(path, len(rows), fps,
                   preview_size if name == "comparison" else size) for name, path in videos.items()},
        "frames": rows,
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("frames", "code_sha256")}, indent=2))


if __name__ == "__main__":
    main()
