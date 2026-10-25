"""Draw camera hot-region contours over verified full-length TEED exports."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge/orin'))
from thermal_palette import apply_thermal_palette  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from render_additional_thermal_videos import (  # noqa: E402
    inspect_output,
    make_contact_sheet,
    open_capture,
)

OUT = Path(__file__).with_name('thermal_outline_full')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def render(number: int) -> dict:
    name = f'fire360_ifsi{number}'
    baseline_root = Path(__file__).with_name('contour_stability') / (
        'thermal_full' if number in (4, 8) else 'thermal_additional'
    )
    metadata_path = baseline_root / f'{name}.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    source = ROOT.parent / 'data/fire360/raw' / f'IFSI Video {number} (2).mp4'
    baseline = baseline_root / f'{name}_full_applied.webm'
    count, fps = metadata['frames'], metadata['source']['fps']
    regions = tuple(tuple(region) for region in metadata['osd_regions'])
    assert digest(source) == metadata['source']['sha256']
    input_hash = digest(baseline)
    paths = [Path(__file__), ROOT / 'edge/orin/thermal_palette.py',
             Path(__file__).with_name('render_additional_thermal_videos.py')]
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    output = OUT / f'{name}_full_hot_outline.webm'
    raw = open_capture(source)
    old = open_capture(baseline)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter.fourcc(*'VP80'), fps, (720, 406))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot write {output}')
    records = []
    started = time.perf_counter()
    try:
        for index in range(count):
            ok_raw, frame = raw.read()
            ok_old, canvas = old.read()
            if not ok_raw or not ok_old:
                raise RuntimeError(f'Incomplete input at {name}:{index}')
            assert canvas.shape == (406, 720, 3)
            assert canvas.dtype == frame.dtype == np.uint8
            canvas = np.asarray(canvas, dtype=np.uint8)
            fitted = np.asarray(cv2.resize(frame, (720, 405), interpolation=cv2.INTER_AREA), dtype=np.uint8)
            before = time.perf_counter()
            gray_source = cv2.cvtColor(cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            background = np.clip(gray_source.astype(np.float32) * metadata['config']['background_scale'], 0, 255).astype(np.uint8)
            for left, top, right, bottom in regions:
                x0, y0 = int(np.floor(left * 720)), int(np.floor(top * 405))
                x1, y1 = int(np.ceil(right * 720)), int(np.ceil(bottom * 405))
                canvas[y0:y1, x0:x1] = background[y0:y1, x0:x1]
            mask, metrics = apply_thermal_palette(canvas[:405], fitted, regions, background=background)
            elapsed_ms = (time.perf_counter() - before) * 1000
            assert np.all(canvas[:405][mask] == (0, 0, 255))
            writer.write(canvas)
            records.append({'frame_index': index, 'source_seconds': index / fps,
                            'palette_ms': elapsed_ms, **dataclasses.asdict(metrics)})
            if (index + 1) % 120 == 0 or index + 1 == count:
                save(OUT / f'{name}_progress.json', {'status': 'RUNNING', 'processed': index + 1, 'expected': count})
                print(f'{name}: {index + 1}/{count}', flush=True)
        assert not raw.read()[0] and not old.read()[0], 'Input extends beyond recorded full length'
    finally:
        raw.release()
        old.release()
        writer.release()
    assert digest(source) == metadata['source']['sha256']
    assert digest(baseline) == input_hash
    for relative, expected in code_hashes.items():
        assert digest(ROOT / relative) == expected, relative
    qa = inspect_output(output, count, fps, (720, 406))
    if qa['status'] != 'PASS':
        raise RuntimeError(str(qa))
    contact = make_contact_sheet(output, count, OUT / f'{name}_contact.png')
    result = {'name': name, 'frames': count, 'start_frame': 0, 'end_frame': count - 1,
              'fps': fps, 'duration_s': count / fps, 'source': metadata['source'],
              'input_teed_video': str(baseline), 'input_teed_sha256': input_hash,
              'input_metadata_sha256': digest(metadata_path), 'source_model': metadata['model'],
              'source_hud_config': metadata['config'], 'osd_regions': regions,
              'code_sha256': code_hashes, 'output': qa, 'contact_sheet': contact,
              'status': 'PASS', 'elapsed_s': time.perf_counter() - started,
              'method': 'Frame-aligned recomposition over full-length TEED output; no repeat neural inference. Red external contours mark camera warm-color regions; interiors return to grayscale background and overlapping green edges are suppressed. Temperature is not inferred. VP8 re-encoding is lossy.',
              'hot_region_pixels': sum(row['hot_region_pixels'] for row in records),
              'outline_pixels': sum(row['outline_pixels'] for row in records),
              'frames_with_palette': sum(row['outline_pixels'] > 0 for row in records),
              'records': records}
    save(OUT / f'{name}.json', result)
    save(OUT / f'{name}_progress.json', {'status': 'COMPLETE', 'processed': count, 'expected': count, 'qa': 'PASS'})
    print(f'{name}: COMPLETE QA PASS', flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--numbers', type=int, nargs='+', choices=[4, 5, 6, 7, 8, 9, 10], required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for number in args.numbers:
        render(number)


if __name__ == '__main__':
    main()
