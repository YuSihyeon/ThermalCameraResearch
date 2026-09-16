"""Run current HUD controls on a pre-overlay thermal video and verify outputs.

No smoothing or learned-weight changes. Shared TEED probability maps isolate
postprocessing/display differences. Motion residuals are diagnostics, not accuracy.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'edge/orin'))
from edge_postprocess import DETAIL_PRESETS, select_edges
from hud import HUDConfig, HUDRenderer
from replay import ReplaySource
from teed_core import TEEDCore


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'experiments/obstacle_edges/deliverables/media/smoke_heldout_original.mp4')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = ReplaySource(str(args.source), sensor_modality='thermal_gray')
    if source.info.fps <= 0:
        raise RuntimeError('Video must report FPS')
    fps, expected = source.info.fps, source.info.frame_count
    size = (640, 480)
    labels = {'original': 'Original thermal display / before edge overlay',
              'legacy': 'Legacy .75 / dilation 3',
              'detailed_1px': 'Detailed / 1 output pixel',
              'balanced_1px': 'Balanced / 1 output pixel',
              'balanced_2px': 'Balanced / 2 output pixels',
              'sparse_1px': 'Sparse / 1 output pixel'}
    names = list(labels)
    writers = {}
    paths = {name: args.output/f'{name}.mp4' for name in names+['comparison']}
    comparison_size = (1920, 1016)
    for name, path in paths.items():
        writer = imageio_ffmpeg.write_frames(str(path), comparison_size if name == 'comparison' else size,
            fps=fps, codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p', quality=8,
            macro_block_size=1, ffmpeg_log_level='error')
        writer.send(None)
        writers[name] = writer
    core = TEEDCore(ROOT/'edge/teed/models/5_model.pth', width=320, height=240, device='cpu')
    renderers = {name: HUDRenderer(HUDConfig(output_size=size,
        background_scale=1 if name == 'original' else .24, show_panel=False,
        line_width=2 if name == 'balanced_2px' else 1)) for name in names}
    rows, thumbnails = [], []
    masks = {name: [] for name in ('detailed', 'balanced', 'sparse')}
    gray_frames = []
    try:
        with source:
            for index, frame in enumerate(source.frames()):
                if index == 0:
                    for _ in range(3):
                        core.infer(frame.image)
                edge = core.infer(frame.image)
                selected = {name: select_edges(edge.probability, *thresholds)
                            for name, thresholds in DETAIL_PRESETS.items()}
                for name, mask in selected.items():
                    masks[name].append(mask)
                gray_frames.append(cv2.cvtColor(edge.source_view, cv2.COLOR_BGR2GRAY))
                stats = {'frame': index, 'seconds': index/fps, 'core_ms': edge.timings_ms['total']}
                cards = []
                for name in names:
                    if name == 'original':
                        result = replace(edge, mask=np.zeros_like(edge.mask))
                    elif name == 'legacy':
                        result = edge
                    else:
                        result = replace(edge, mask=selected[name.split('_')[0]], edge_style='thin')
                    rendered = renderers[name].render(frame.image, result,
                        sensor_modality=frame.sensor_modality, pixel_origin=frame.pixel_origin)
                    writers[name].send(rendered.image)
                    stats[name] = {'edge_pixels': rendered.edge_pixels, 'highlight_pixels': rendered.highlight_pixels,
                                   'hud_ms': rendered.render_ms}
                    card = np.zeros((508, 640, 3), np.uint8)
                    card[28:] = rendered.image
                    cv2.putText(card, f'{labels[name]} | {index/fps:.1f}s', (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .47, (230, 230, 230), 1, cv2.LINE_AA)
                    cards.append(card)
                sheet = np.vstack((np.hstack(cards[:3]), np.hstack(cards[3:])))
                writers['comparison'].send(sheet)
                if index % max(1, round(fps)) == 0:
                    thumb = np.hstack((cards[0], cards[3], cards[4]))
                    thumbnails.append(cv2.resize(thumb, (960, 254), interpolation=cv2.INTER_AREA))
                if index in (0, round(fps*6), expected-1):
                    ok, encoded = cv2.imencode('.png', sheet)
                    if not ok:
                        raise RuntimeError('PNG encoding failed')
                    (args.output/f'frame_{index:04d}.png').write_bytes(encoded.tobytes())
                rows.append(stats)
                if index % 30 == 0:
                    print(f'Processed {index+1}/{expected}', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    if not rows or len(rows) != expected:
        raise RuntimeError(f'Incomplete source decode: {len(rows)} of {expected}')
    ok, encoded = cv2.imencode('.png', np.vstack(thumbnails))
    if not ok:
        raise RuntimeError('PNG encoding failed')
    (args.output/'timeline.png').write_bytes(encoded.tobytes())

    # Compare same-width model-space skeletons only, with one-pixel tolerance.
    # Optical flow is from original grayscale frames, never coloured overlays.
    yy, xx = np.mgrid[:240, :320].astype(np.float32)
    temporal = {name: [] for name in masks}
    valid_fractions = []
    for i in range(1, len(rows)):
        current, previous = gray_frames[i], gray_frames[i-1]
        flow = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 3, 15, 3, 5, 1.2, 0)
        mx, my = xx+flow[:, :, 0], yy+flow[:, :, 1]
        old_gray = cv2.remap(previous, mx, my, cv2.INTER_LINEAR)
        valid = (mx >= 0) & (mx < 319) & (my >= 0) & (my < 239) & (np.abs(current.astype(float)-old_gray) < 20)
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        valid_fractions.append(float(valid.mean()))
        for name in masks:
            prior = cv2.remap(masks[name][i-1], mx, my, cv2.INTER_NEAREST)
            present = masks[name][i]
            old_edges, new_edges = (prior > 0) & valid, (present > 0) & valid
            denom = int(old_edges.sum()+new_edges.sum())
            if denom == 0:
                continue
            near_prior = cv2.dilate(prior, np.ones((3, 3), np.uint8)) > 0
            near_present = cv2.dilate(present, np.ones((3, 3), np.uint8)) > 0
            unmatched = int((new_edges & ~near_prior).sum()+(old_edges & ~near_present).sum())
            temporal[name].append({'frame': i, 'seconds': i/fps, 'residual_fraction': unmatched/denom})

    verification = {}
    for name, path in paths.items():
        cap = cv2.VideoCapture(str(path))
        found_fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        output_size = comparison_size if name == 'comparison' else size
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if (image.shape[1], image.shape[0]) != output_size:
                raise RuntimeError(f'Wrong output size: {name}')
            count += 1
        cap.release()
        if count != expected or abs(found_fps-fps) > .001:
            raise RuntimeError(f'Output timing mismatch: {name}')
        verification[name] = {'frames_decoded': count, 'fps': found_fps, 'size': list(output_size),
                              'bytes': path.stat().st_size, 'sha256': sha256(path)}
    variants = {}
    for name in names[1:]:
        coverage = [row[name]['edge_pixels']/(size[0]*size[1])*100 for row in rows]
        variants[name] = {'mean_display_coverage_percent': float(np.mean(coverage)),
                          'min_display_coverage_percent': min(coverage), 'max_display_coverage_percent': max(coverage),
                          'frames_without_edges': sum(row[name]['edge_pixels'] == 0 for row in rows),
                          'highlight_pixels_sum': sum(row[name]['highlight_pixels'] for row in rows)}
    report = {'protocol': 'hud_video_verification_v1',
              'runtime_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source': {'filename': args.source.name, 'sha256': sha256(args.source), 'fps': fps,
                         'frames': expected, 'duration_seconds': expected/fps},
              'settings': {'input_size': [320, 240], 'output_size': list(size), 'contrast_clip_limit': 2,
                           'background_scale': .24, 'detail_presets': DETAIL_PRESETS, 'modality': 'thermal_gray'},
              'variants': variants, 'files': verification,
              'temporal_diagnostic': {name: {'mean_residual_fraction': float(np.mean([p['residual_fraction'] for p in pairs])) if pairs else None,
                  'usable_pairs': len(pairs), 'highest_residual_pairs': sorted(pairs, key=lambda p:p['residual_fraction'], reverse=True)[:3]}
                  for name, pairs in temporal.items()},
              'mean_flow_valid_pixel_fraction': float(np.mean(valid_fractions)),
              'limits': ['SmokeBasement encoded display video, not the supplied Boson HUD or temperature data',
                         'One scene, 12 seconds; no obstacle labels or accuracy/Jetson speed evaluation',
                         'Temporal residual includes occlusion, scene changes and flow error; not pure flicker',
                         'Shared inference comparison; processing throughput is not production pipeline FPS'],
              'frames': rows}
    (args.output/'video_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    credits = ROOT/'experiments/obstacle_edges/deliverables/SOURCE_CREDITS.txt'
    shutil.copyfile(credits, args.output/'SOURCE_CREDITS.txt')
    with (args.output/'SOURCE_CREDITS.txt').open('a', encoding='utf-8') as stream:
        stream.write('\n2026-09-08 HUD verification: current YSH TEED runtime, upscaled to 640x480; '
                     'detail/width variants and labelled comparison. No creator endorsement.\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('frames', 'files')}, indent=2))


if __name__ == '__main__':
    main()
