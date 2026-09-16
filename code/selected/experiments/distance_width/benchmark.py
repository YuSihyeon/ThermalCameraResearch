"""Reproducible local CPU comparison. Thermal depth is SYNTHETIC, never measured.

Run from repository root:
python experiments/distance_width/benchmark.py --output ../outputs/distance_width_2026-09-14
Video encoding is outside timed regions. All variants share each TEED mask.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge/teed'))
import run_teed as teed
from distance_width import DistanceWidthRenderer, WidthConfig, thin_mask, render_width

MODES = ('legacy_3', 'thin_3', 'banded', 'continuous', 'smooth')
LABELS = {'legacy_3': 'Original TEED / dilation 3', 'thin_3': 'Thin centerline / fixed 3px',
          'banded': 'Distance / hard 1,3,5px', 'continuous': 'Continuous / no time filter',
          'smooth': 'Continuous / causal 50ms'}
SIZE = (320, 240)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(zip(('mean', 'p50', 'p95', 'p99', 'max'),
                    map(float, (v.mean(), *np.percentile(v, (50, 95, 99)), v.max()))))


def blend(frame, alpha, binary=False):
    background = (frame.astype(np.float32) * .24).astype(np.uint8)
    if binary:
        background[alpha > 0] = (0, 255, 0)
        return background
    return teed.blend_alpha(background, alpha)


def target_width(depth):
    u = np.clip((np.asarray(depth, np.float32) - 1.) / 7., 0., 1.)
    return 5. - 4. * u * u * (3. - 2. * u)


class Variants:
    def __init__(self):
        self.continuous = DistanceWidthRenderer(WidthConfig(tau_s=0.))
        self.smooth = DistanceWidthRenderer(WidthConfig(tau_s=.05))

    def process(self, name, raw, depth, timestamp, depth_timestamp):
        if name == 'legacy_3':
            return cv2.dilate(raw, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.
        if name == 'thin_3':
            return render_width(thin_mask(raw), 3.)
        if name == 'banded':
            if depth is None or timestamp - depth_timestamp > .1:
                width = 1.
            else:
                d = np.asarray(depth, np.float32)
                valid = np.isfinite(d) & (d > 0)
                width = np.where(valid, 1. + 2. * np.floor((target_width(np.where(valid, d, 8.)) - 1.) / 2. + .5), 1.)
                if width.ndim == 0:
                    width = float(width)
            return render_width(thin_mask(raw), width)
        renderer = self.continuous if name == 'continuous' else self.smooth
        return renderer.render(raw, depth, timestamp, depth_timestamp)


def writer(path, size, fps):
    out = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
        pix_fmt_in='bgr24', pix_fmt_out='yuv420p', quality=8,
        macro_block_size=1, ffmpeg_log_level='error')
    out.send(None)
    return out


def card(frame, title, subtitle):
    enlarged = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
    panel = np.full((538, 640, 3), (22, 26, 30), np.uint8)
    panel[58:] = enlarged
    cv2.putText(panel, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .43, (110, 210, 245), 1, cv2.LINE_AA)
    return panel


def save_image(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('image encode failed')
    path.write_bytes(data.tobytes())


def core_infer(model, frame, args):
    started = time.perf_counter()
    resized, tensor = teed.prepare_tensor(frame, *SIZE, torch.device('cpu'), args.contrast_clip_limit)
    model_started = time.perf_counter()
    with torch.no_grad():
        out = model(tensor, single_test=True)
        p = torch.sigmoid(out[-1])[0, 0, :SIZE[1], :SIZE[0]].numpy()
    model_ms = (time.perf_counter() - model_started) * 1000.
    raw = (p >= .75).astype(np.uint8) * 255
    return resized, raw, model_ms, (time.perf_counter() - started) * 1000.


def thermal_frames(source, model, args):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError('cannot open source')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, masks, core_rows = [], [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not frames:
                for _ in range(5):
                    core_infer(model, frame, args)
            view, raw, model_ms, core_ms = core_infer(model, frame, args)
            frames.append(view)
            masks.append(raw)
            core_rows.append({'frame': len(frames) - 1, 'model_ms': model_ms, 'core_ms': core_ms})
    finally:
        cap.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError('source has no frames or valid FPS')
    return frames, masks, core_rows, fps


def render_clip(output, name, frames, masks, depth_frames, fps, core_rows=None, statuses=None):
    variants = Variants()
    comparison = writer(output / (name + '_comparison.mp4'), (1920, 1076), fps)
    chosen = writer(output / (name + '_smooth.mp4'), (640, 538), fps)
    rows = []
    try:
        for i, (frame, raw, depth) in enumerate(zip(frames, masks, depth_frames)):
            now = i / fps
            depth_time = now if statuses is None or statuses[i] != 'stale' else now - .2
            status = 'ASSUMED DEPTH - NOT SENSOR DATA' if statuses is None else statuses[i]
            panels = [card(frame, 'Input / ' + name, status + ' | %.2fs' % now)]
            for mode in MODES:
                started = time.perf_counter()
                alpha = variants.process(mode, raw, depth, now, depth_time)
                rendered = blend(frame, alpha, binary=mode in ('legacy_3', 'thin_3', 'banded'))
                post_ms = (time.perf_counter() - started) * 1000.
                core_ms = 0. if core_rows is None else core_rows[i]['core_ms']
                rows.append({'clip': name, 'frame': i, 'time_s': now, 'mode': mode,
                             'post_ms': post_ms, 'core_ms': core_ms, 'processing_ms': core_ms + post_ms,
                             'alpha_area_px': float(alpha.sum()), 'status': status})
                subtitle = ('Fixed width | post %.2fms' % post_ms if mode in ('legacy_3', 'thin_3')
                            else 'SYNTHETIC DEPTH | post %.2fms | 1m:5px  8m:1px' % post_ms)
                panel = card(rendered, LABELS[mode], subtitle)
                panels.append(panel)
                if mode == 'smooth':
                    chosen.send(np.ascontiguousarray(panel))
            grid = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
            comparison.send(np.ascontiguousarray(grid))
            if i in (0, len(frames) // 2, len(frames) - 1):
                save_image(output / ('%s_frame_%04d.png' % (name, i)), grid)
    finally:
        comparison.close()
        chosen.close()
    return rows


def synthetic_scene():
    frames, masks, depths, labels = [], [], [], []
    rng = np.random.default_rng(914)
    h, w = SIZE[1], SIZE[0]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for i in range(300):
        now = i / 30.
        frame = np.full((h, w, 3), 55, np.uint8)
        raw = np.zeros((h, w), np.uint8)
        for x in (60, 160, 260):
            cv2.line(raw, (x, 25), (x, 215), 255, 3)
        pts = np.column_stack((np.arange(15, 305), 120 + 35 * np.sin(np.arange(15, 305) / 40.)))
        cv2.polylines(raw, [pts.astype(np.int32)], False, 255, 3)
        base = 4.5 + 3.0 * np.sin(now * .9 + xx / 110. + yy / 200.)
        label = 'Spatial + temporal continuous depth'
        if 3 <= now < 5:
            base = np.full((h, w), 4.5 + rng.normal(0, .18), np.float32)
            label = 'Stationary surface / seeded depth noise'
        elif 5 <= now < 6:
            base = np.full((h, w), 1. if now < 5.5 else 8., np.float32)
            label = 'Abrupt near / far change'
        elif 6 <= now < 7:
            label = 'stale'
        elif 7 <= now < 8:
            base[:, 100:220] = np.nan
            label = 'Invalid depth strip -> thin fallback'
        elif now >= 8:
            cv2.rectangle(raw, (int(20 + (now - 8) * 80), 45),
                          (int(70 + (now - 8) * 80), 95), 255, 3)
            label = 'Moving current-frame edges / no trail buffer'
        frame[raw > 0] = (180, 180, 180)
        frames.append(frame)
        masks.append(raw)
        depths.append(base.astype(np.float32))
        labels.append(label)
    return frames, masks, depths, labels


def isolated_benchmark(masks, depths):
    # Repeat interleaved variants over identical masks, without video encoding.
    rows = []
    variants = Variants()
    canvas = np.full((SIZE[1], SIZE[0], 3), 80, np.uint8)
    for i in range(630):
        idx = i % len(masks)
        for mode in MODES[i % 5:] + MODES[:i % 5]:
            start = time.perf_counter()
            alpha = variants.process(mode, masks[idx], depths[idx], i / 30., i / 30.)
            blend(canvas, alpha, binary=mode in ('legacy_3', 'thin_3', 'banded'))
            elapsed = (time.perf_counter() - start) * 1000.
            if i >= 30:
                rows.append({'iteration': i - 30, 'mode': mode, 'post_ms': elapsed})
    return rows


def temporal_diagnostics():
    rng = np.random.default_rng(914)
    raw = np.zeros((64, 64), np.uint8)
    raw[8:56, 32] = 255
    variants = Variants()
    values = {key: [] for key in ('banded', 'continuous', 'smooth')}
    widths = []
    for i in range(240):
        d = 4.5 + rng.normal(0, .18)
        widths.append(float(target_width(d)))
        for key in values:
            alpha = variants.process(key, raw, d, i / 30., i / 30.)
            values[key].append(float(alpha[32].sum()))
    result = {'noise_seed': 914, 'depth_noise_std_m': .18, 'fps': 30,
              'axis_width_std_px': {k: float(np.std(v[30:])) for k, v in values.items()},
              'width_traces': values}
    renderer = DistanceWidthRenderer()
    for i in range(10):
        renderer.render(raw, 8., i / 30., i / 30.)
    response = renderer.render(raw, 1., 10 / 30., 10 / 30.)
    result['large_step_first_frame_width_px'] = float(response[32].sum())
    result['large_step_delay_frames'] = 0 if abs(response[32].sum() - 5.) < .01 else None
    result['filter_time_constant_ms'] = 50
    result['gradual_change_low_frequency_lag_ms_30fps'] = float((1 / 30.) / np.expm1((1 / 30.) / .05) * 1000.)
    variants = Variants()
    boundary = {key: [] for key in values}
    rng = np.random.default_rng(914)
    for i in range(240):
        # Explicit worst boundary case for the coarse 1/3/5px comparison.
        d = 3.28 + rng.normal(0, .18)
        for key in boundary:
            a = variants.process(key, raw, d, i / 30., i / 30.)
            boundary[key].append(float(a[32].sum()))
    result['band_boundary_noise_mean_depth_m'] = 3.28
    result['band_boundary_width_std_px'] = {k: float(np.std(v[30:])) for k, v in boundary.items()}
    variants = Variants()
    ramp = {key: [] for key in values}
    for i, d in enumerate(np.linspace(1., 8., 240)):
        for key in ramp:
            a = variants.process(key, raw, float(d), i / 30., i / 30.)
            ramp[key].append(float(a[32].sum()))
    result['ramp_max_frame_change_px'] = {k: float(np.max(np.abs(np.diff(v)))) for k, v in ramp.items()}
    result['ramp_traces'] = ramp
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as out:
        csv_writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def verify_video(path):
    cap = cv2.VideoCapture(str(path))
    count = 0
    size = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = [frame.shape[1], frame.shape[0]]
        count += 1
    cap.release()
    if not count:
        raise RuntimeError('empty or undecodable video: ' + str(path))
    return {'file': path.name, 'frames': count, 'fps': fps, 'size': size, 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'experiments/obstacle_edges/deliverables/media/smoke_heldout_original.mp4')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    runtime_args = teed.build_parser().parse_args([])
    model = teed.load_model(teed.default_checkpoint_path(), torch.device('cpu'))
    print('Extracting shared CPU TEED masks...', flush=True)
    frames, masks, core_rows, fps = thermal_frames(args.source, model, runtime_args)
    yy, xx = np.mgrid[:SIZE[1], :SIZE[0]].astype(np.float32)
    depths = [np.clip(4.5 + 2.8 * np.sin(i / fps * .7 + xx / 110. + yy / 180.), 1, 8).astype(np.float32)
              for i in range(len(frames))]
    np.save(output / 'thermal_ASSUMED_depth_m.npy', np.stack(depths))
    np.save(output / 'thermal_shared_masks.npy', np.stack(masks))
    print('Rendering thermal comparisons (ASSUMED depth)...', flush=True)
    rows = render_clip(output, 'thermal_assumed', frames, masks, depths, fps, core_rows)
    print('Rendering controlled synthetic sequence...', flush=True)
    sf, sm, sd, sl = synthetic_scene()
    rows.extend(render_clip(output, 'synthetic', sf, sm, sd, 30., statuses=sl))
    print('Benchmarking without encoding...', flush=True)
    isolated = isolated_benchmark(masks, depths)
    write_csv(output / 'frame_metrics.csv', rows)
    write_csv(output / 'core_metrics.csv', core_rows)
    write_csv(output / 'benchmark_raw.csv', isolated)
    summary = {}
    core = np.asarray([r['core_ms'] for r in core_rows])
    for mode in MODES:
        post = [r['post_ms'] for r in isolated if r['mode'] == mode]
        processing = [r['processing_ms'] for r in rows if r['mode'] == mode and r['clip'] == 'thermal_assumed']
        summary[mode] = {'post_ms': stats(post), 'thermal_processing_ms': stats(processing),
                         'thermal_processing_fps': 1000. / float(np.mean(processing)),
                         'over_30fps_budget_pct': float(np.mean(np.asarray(processing) > 1000. / 30.) * 100.),
                         'over_60fps_budget_pct': float(np.mean(np.asarray(processing) > 1000. / 60.) * 100.)}
    summary['environment'] = {'platform': platform.platform(), 'processor': platform.processor(),
        'python': sys.version, 'opencv': cv2.__version__, 'numpy': np.__version__,
        'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device': 'cpu',
        'threads': 2, 'resolution': list(SIZE), 'source_fps': fps, 'source_frames': len(frames),
        'source_sha256': sha256(args.source), 'checkpoint_sha256': sha256(teed.default_checkpoint_path()),
        'depth_provenance': 'synthetic sinusoidal maps, NOT measured or inferred scene distances',
        'source': str(args.source.resolve()), 'warmup_core_frames': 5,
        'benchmark_samples_per_mode': 600, 'latency_scope': 'CPU core + postprocessing; excludes capture/display/encode/depth estimation',
        'core_ms': stats(core)}
    summary['temporal'] = temporal_diagnostics()
    print('Fully decoding all outputs...', flush=True)
    summary['videos'] = [verify_video(p) for p in sorted(output.glob('*_comparison.mp4'))] + [verify_video(p) for p in sorted(output.glob('*_smooth.mp4'))]
    for video in summary['videos']:
        expected = len(frames) if video['file'].startswith('thermal') else 300
        if video['frames'] != expected:
            raise RuntimeError('video frame count mismatch: ' + video['file'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({mode: summary[mode]['post_ms'] for mode in MODES}, indent=2), flush=True)


if __name__ == '__main__':
    main()
