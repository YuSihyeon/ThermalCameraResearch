"""Measure serial decode -> TEED -> HUD latency; no display or recording."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge/orin'))
from hud import HUDConfig, HUDRenderer  # noqa: E402
from replay import load_fire360_osd_regions  # noqa: E402
from teed_core import TEEDCore, sha256_file  # noqa: E402

OUT = Path(__file__).with_name('thermal_outline_runtime')
CLIPS = [('fire360_ifsi10', 'IFSI Video 10 (2).mp4', 467),
         ('fire360_ifsi8', 'IFSI Video 8 (2).mp4', 90)]
WARMUP, COUNT = 20, 120
CODE = ['teed_core.py', 'teed_model.py', 'hud.py', 'edge_filter.py',
        'temporal_edges.py', 'contour_tracker.py', 'contour_render.py',
        'thermal_overlay.py', 'thermal_palette.py', 'replay.py']


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {'mean': float(array.mean()), 'p50': float(np.percentile(array, 50)),
            'p95': float(np.percentile(array, 95)), 'max': float(array.max())}


def measure(core: TEEDCore, name: str, filename: str, start: int,
            threads: int, palette: bool) -> dict:
    source = ROOT.parent / 'data/fire360/raw' / filename
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {source}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - WARMUP)
    regions = load_fire360_osd_regions(str(ROOT / 'edge/teed/profiles/fire360_video_profiles.json'), 'IFSI Video 4 (2).mp4')
    config = HUDConfig(output_size=(720, 405), temporal_edges=True, contour_edges=True, thermal_palette_overlay=palette)
    renderer = HUDRenderer(config)
    records = []
    digest = hashlib.sha256()
    try:
        for offset in range(WARMUP + COUNT):
            index = start - WARMUP + offset
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                raise RuntimeError(f'Short decode at {name}:{index}')
            edge = core.infer(frame, regions)
            t2 = time.perf_counter()
            result = renderer.render(frame, edge, ignore_regions=regions,
                                     frame_index=index, timestamp_s=index / fps)
            t3 = time.perf_counter()
            if offset >= WARMUP:
                records.append({'frame': index, 'decode_ms': (t1-t0)*1000,
                                'teed_ms': (t2-t1)*1000, 'model_ms': edge.timings_ms['model'],
                                'hud_ms': (t3-t2)*1000, 'processing_ms': (t3-t1)*1000,
                                'serial_ms': (t3-t0)*1000,
                                'palette_ms': result.palette_metrics['render_ms']})
                digest.update(result.image.tobytes())
    finally:
        cap.release()
    timing = {key: stats([row[key] for row in records]) for key in records[0] if key != 'frame'}
    serial = np.asarray([row['serial_ms'] for row in records])
    return {'name': name, 'source': str(source), 'source_sha256': sha256_file(source),
            'source_fps': fps, 'source_size': [int(frame.shape[1]), int(frame.shape[0])],
            'first_measured_frame': start, 'last_measured_frame': start + COUNT - 1,
            'warmup_frames': WARMUP, 'measured_frames': COUNT, 'torch_opencv_threads': threads,
            'decoder_threads': 1, 'contour_edges': True, 'thermal_palette_overlay': palette, 'hud_config': dataclasses.asdict(config),
            'model': core.describe(), 'timings_ms': timing,
            'serial_fps': 1000 / float(serial.mean()),
            'deadline_miss_fraction': {str(target): float(np.mean(serial > 1000 / target)) for target in [9, 15, 30]},
            'rendered_sha256': digest.hexdigest(), 'records': records}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_interop_threads(1)
    paths = [ROOT / 'edge/orin' / f for f in CODE] + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    host = {'platform': platform.platform(), 'processor': platform.processor(),
            'torch': torch.__version__, 'opencv': cv2.__version__,
            'cuda_available': torch.cuda.is_available()}
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            host['processor'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    results = []
    for threads in [4]:
        cv2.setNumThreads(threads)
        torch.set_num_threads(threads)
        core = TEEDCore(ROOT / 'edge/teed/models/5_model.pth', device='cpu')
        for name, filename, start in CLIPS:
            for palette in [False, True]:
                result = measure(core, name, filename, start, threads, palette)
                results.append(result)
                key = f'{name}_t{threads}_palette{int(palette)}'
                save(OUT / f'{key}.json', result)
                save(OUT / 'progress.json', {'status': 'RUNNING', 'completed': len(results), 'expected': 4, 'last': key})
                print(f'{key}: {result["serial_fps"]:.2f} FPS; p95={result["timings_ms"]["serial_ms"]["p95"]:.1f} ms', flush=True)
    for relative, digest in hashes.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f'Code changed during benchmark: {relative}')
    save(OUT / 'benchmark.json', {'status': 'COMPLETE', 'hardware': host, 'code_sha256': hashes,
         'method': 'Sequential batch=1, no cached model outputs, 20 pipeline warmup frames + 120 measured frames per case; 4K source decode + TEED 320x256 + HUD 720x405 with contour tracking; thermal palette red-outline off/on matched clips; unpaced; output hashing excluded from timing.',
         'limits': ['CPU-only local PC; not a Jetson measurement', 'No live camera, display scanout or encoding in serial timings', 'Thermal source timestamps drive temporal filters; no live capture backlog simulation', 'Thread counts configure libraries, not process affinity'],
         'results': results})
    save(OUT / 'progress.json', {'status': 'COMPLETE', 'completed': len(results), 'expected': 4})


if __name__ == '__main__':
    main()
