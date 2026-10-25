"""Exercise the real headless CLI with a warm thermal clip and 4 CPU threads."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge/orin'))
from run_pipeline import build_parser, run  # noqa: E402


def main() -> None:
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(__file__).with_name('thermal_color_runtime')
    out.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args([
        '--source', str(ROOT.parent / 'data/fire360/raw/IFSI Video 10 (2).mp4'),
        '--device', 'cpu', '--contour-edges', '--thermal-palette-overlay',
        '--profiles', str(ROOT / 'edge/teed/profiles/fire360_video_profiles.json'),
        '--headless', '--start-frame', '467', '--max-frames', '120',
        '--display-width', '720', '--display-height', '405',
        '--metrics', str(out / 'cli_smoke.json'),
    ])
    result = run(args)
    assert result['processed_frames'] == 120
    assert result['source']['osd_regions'] == 2
    assert result['hud']['thermal_palette_overlay_enabled']
    assert result['hud']['camera_hot_outline_pixels'] > 0
    assert len(result['hud']['thermal_palette_frames']) == 120
    assert result['hud']['thermal_statuses'] == ['temperature_unavailable']
    (out / 'cli_checks.json').write_text(json.dumps({
        'status': 'PASS', 'processed_frames': 120,
        'source': args.source, 'start_frame': args.start_frame,
        'profiles': args.profiles, 'osd_regions': result['source']['osd_regions'],
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT / 'edge/orin/run_pipeline.py', ROOT / 'edge/orin/hud.py',
                                  ROOT / 'edge/orin/replay.py', Path(args.profiles), Path(__file__).resolve()]},
        'last_source_frame': args.start_frame + result['processed_frames'] - 1,
        'camera_hot_outline_pixels': result['hud']['camera_hot_outline_pixels'],
        'torch_opencv_threads': 4,
        'limits': 'Headless replay; CLI e2e excludes capture and actual display scanout.',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
