"""Audit local thermal display clips; this is not a fire detector or calibration."""
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data" / "fire360" / "raw"
OUT = Path(__file__).with_name("thermal_research")
SIZE = (960, 540)
# Fixed UI coordinates verified on these two files, measured at SIZE.
EXCLUSIONS = [(0, 0, 121, 540), (839, 0, 960, 540),
              (727, 0, 839, 482), (127, 394, 241, 540)]


def metadata(path):
    result = {"handlers": [], "codecs": [], "xmp": {}}
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl",
                  b"edts", b"udta", b"meta", b"dinf", b"ilst"}
    def walk(stream, start, end):
        pos = start
        while pos + 8 <= end:
            stream.seek(pos)
            size, tag = struct.unpack(">I4s", stream.read(8))
            offset = 8
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
                offset = 16
            if size == 0:
                size = end - pos
            if size < offset or pos + size > end:
                raise ValueError("Malformed MP4 box")
            if tag in containers:
                walk(stream, pos + offset + (4 if tag == b"meta" else 0), pos + size)
            elif tag in {b"hdlr", b"stsd", b"uuid"}:
                stream.seek(pos + offset)
                content = stream.read(size - offset)
                if tag == b"hdlr":
                    result["handlers"].append(content[8:12].decode("latin1"))
                elif tag == b"stsd":
                    result["codecs"].append(content[12:16].decode("latin1"))
                elif b"<x:xmpmeta" in content:
                    begin = content.index(b"<x:xmpmeta")
                    end_xml = content.index(b"</x:xmpmeta>", begin) + len(b"</x:xmpmeta>")
                    xml = ET.fromstring(content[begin:end_xml])
                    for node in xml.iter():
                        for key, value in node.attrib.items():
                            key = key.split("}")[-1]
                            if key in {"CreatorTool", "Make", "Model", "CreateDate", "format", "filePath"}:
                                if key == "filePath":
                                    value = value.replace(chr(92), "/").split("/")[-1]
                                result["xmp"].setdefault(key, []).append(value)
            pos += size
    with path.open("rb") as stream:
        walk(stream, 0, path.stat().st_size)
    with path.open("rb") as stream:
        result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Permissive test for red/orange/yellow display pixels; NOT inferred Celsius.
    warm = (((hsv[:, :, 0] <= 38) | (hsv[:, :, 0] >= 175))
            & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 80))
    valid = np.ones(frame.shape[:2], dtype=bool)
    for x0, y0, x1, y1 in EXCLUSIONS:
        valid[y0:y1, x0:x1] = False
    scene = warm & valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(scene.astype(np.uint8), 8)
    kept = np.zeros(scene.shape, dtype=np.uint8)
    areas = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 8:
            kept[labels == label] = 255
            areas.append(area)
    return warm, valid, scene, kept, areas


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    report = {
        "purpose": "Display-palette feasibility audit, not accuracy validation",
        "analysis_size": SIZE, "excluded_rectangles_xyxy": EXCLUSIONS,
        "candidate_rule": "OpenCV H<=38 or H>=175, S>=60, V>=80; component area>=8",
        "limits": ["No fire/hotspot ground truth", "No calibration",
                   "Excluded UI rectangles hide underlying scene", "Resized to 960x540",
                   "A negative result does not prove absence of fire or high temperature"],
        "clips": [],
    }
    rows = []
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    for filename in ["IFSI Video 4 (2).mp4", "IFSI Video 8 (2).mp4"]:
        path = DATA / filename
        clip = metadata(path)
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip.update(filename=filename, expected_frames=expected,
                    fps=cap.get(cv2.CAP_PROP_FPS), frames=[], source_resolution=[
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
        index = 0
        representative = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
            warm, valid, scene, kept, areas = masks(frame)
            chroma = frame.max(axis=2).astype(np.int16) - frame.min(axis=2).astype(np.int16)
            clip["frames"].append({
                "index": index, "full_frame_warm_pixels": int(warm.sum()),
                "scene_warm_pixels": int(scene.sum()), "kept_pixels": int((kept > 0).sum()),
                "components": len(areas), "max_component_pixels": max(areas, default=0),
                "scene_max_channel_spread": int(chroma[valid].max())})
            if index == expected // 2:
                representative = frame.copy()
            index += 1
        cap.release()
        if index != expected or representative is None:
            raise RuntimeError(f"Incomplete decode {filename}: {index}/{expected}")
        metrics = clip["frames"]
        clip["summary"] = {
            "decoded_frames": index,
            "frames_with_candidates": sum(m["components"] > 0 for m in metrics),
            "max_scene_warm_pixels": max(m["scene_warm_pixels"] for m in metrics),
            "max_kept_pixels": max(m["kept_pixels"] for m in metrics),
            "max_scene_channel_spread": max(m["scene_max_channel_spread"] for m in metrics)}
        report["clips"].append(clip)
        print(filename, json.dumps(clip["summary"]), flush=True)
        warm, valid, scene, kept, areas = masks(representative)
        excluded = representative.copy()
        excluded[~valid] = (35, 25, 22)
        row = Image.new("RGB", (1920, 630), "#101820")
        draw = ImageDraw.Draw(row)
        draw.text((20, 12), f"{filename} | frame {expected // 2} | source display", fill="white", font=font)
        draw.text((980, 12), "Valid scene | warm palette candidates in red", fill="white", font=font)
        row.paste(Image.fromarray(cv2.cvtColor(representative, cv2.COLOR_BGR2RGB)), (0, 50))
        # Full valid scene remains legible; exclusions visibly tinted; only candidates red.
        excluded[kept > 0] = (0, 0, 255)
        row.paste(Image.fromarray(cv2.cvtColor(excluded, cv2.COLOR_BGR2RGB)), (960, 50))
        draw.text((20, 598), f"All {index} frames: {clip['summary']['frames_with_candidates']} with warm scene components (>=8 px). Not detection accuracy.", fill="#cbd5e1", font=small)
        rows.append(row)
        cv2.imwrite(str(OUT / (filename.split(" (")[0].replace(" ", "_") + "_source.png")), representative)
    report["comparison_clip"] = metadata(DATA / "GOPR8356 (2).MP4")
    report["comparison_clip"]["modality"] = "Visible-light scene, verified by manual frame inspection; GoPro metadata"
    (OUT / "thermal_source_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    panel = Image.new("RGB", (1920, 1340), "#101820")
    draw = ImageDraw.Draw(panel)
    draw.text((20, 14), "Fire360 local thermal clips | UI excluded | displayed color is not a calibrated temperature map", fill="white", font=font)
    for i, row in enumerate(rows):
        panel.paste(row, (0, 65 + 630 * i))
    panel.save(OUT / "thermal_feasibility.png")



def render_examples():
    report = json.loads((OUT / "thermal_source_audit.json").read_text(encoding="utf-8"))
    panel = Image.new("RGB", (1920, 1340), "#101820")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw = ImageDraw.Draw(panel)
    draw.text((20, 14), "Thermal display audit | red contour = colorized heat candidate, not confirmed flame", fill="white", font=font)
    for row, clip in enumerate(report["clips"]):
        best = max(clip["frames"], key=lambda item: item["kept_pixels"])
        index = best["index"] if best["kept_pixels"] else clip["expected_frames"] // 2
        cap = cv2.VideoCapture(str(DATA / clip["filename"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Cannot decode representative frame")
        frame = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
        _, valid, _, kept, _ = masks(frame)
        overlay = frame.copy()
        overlay[~valid] = (35, 25, 22)
        selected = kept > 0
        overlay[selected] = (frame[selected].astype(float) * 0.7 + np.array([0, 0, 255]) * 0.3).astype(np.uint8)
        contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        y = 65 + row * 630
        draw.text((20, y + 12), clip["filename"] + f" | frame {index} | source", fill="white", font=font)
        draw.text((980, y + 12), "Palette candidate outline | logo and scale excluded", fill="white", font=font)
        panel.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, y + 50))
        panel.paste(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), (960, y + 50))
        summary = clip["summary"]
        label = f"Full clip: {summary['frames_with_candidates']} / {summary['decoded_frames']} frames contain colorized scene candidates. This is not an accuracy score."
        draw.text((20, y + 598), label, fill="#cbd5e1", font=small)
        cv2.imwrite(str(OUT / f"example_{row}_source.png"), frame)
        cv2.imwrite(str(OUT / f"example_{row}_overlay.png"), overlay)
    panel.save(OUT / "thermal_feasibility.png")


if __name__ == "__main__":
    import sys
    if "--render-only" not in sys.argv:
        main()
    render_examples()
