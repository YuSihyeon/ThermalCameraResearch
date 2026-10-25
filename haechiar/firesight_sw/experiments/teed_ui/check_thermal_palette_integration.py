from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge/orin'))
from hud import HUDConfig, HUDRenderer  # noqa: E402
from run_pipeline import build_parser, run  # noqa: E402
from teed_core import EdgeResult  # noqa: E402


def main() -> None:
    cv2.setNumThreads(1)
    source = np.full((80, 160, 3), 130, np.uint8)
    source[10:25, 20:40] = (0, 80, 255)
    source[10:25, 130:155] = (0, 0, 255)
    source[30:45, 20:40] = (255, 0, 0)
    base = cv2.resize(source, (80, 64))
    edge = EdgeResult(np.ones((64, 80), np.float32), np.full((64, 80), 255, np.uint8), base, (80, 64), {})
    regions = ((0.8, 0.0, 1.0, 1.0),)
    common = {'filter_edges': False, 'output_size': (320, 240), 'background_scale': 0.0}
    checks = []
    renderer = HUDRenderer(HUDConfig(**common, thermal_palette_overlay=True))
    image = renderer.render(source, edge, ignore_regions=regions)
    red = np.all(image.image == (0, 0, 255), axis=2)
    assert red[57:93, 37:83].any()
    assert not red[:55].any() and not red[95:].any()
    assert np.all(image.image[66:84, 46:74] == (0, 80, 255))
    assert np.all(image.image[62:87, 265:305] == (0, 255, 0))
    assert not image.image[:40].any()
    assert image.palette_metrics['outline_pixels'] > 0
    assert image.thermal_metrics['status'] == 'temperature_unavailable'
    checks.append('Red external heat contours align in see-through letterbox; original camera colors preserved and green suppressed; OSD excluded; no temperature inferred')
    default = HUDRenderer(HUDConfig(**common)).render(source, edge, ignore_regions=regions)
    assert default.palette_metrics['status'] == 'disabled'
    assert not np.any((default.image[:, :, 2] > 240) & (default.image[:, :, 1] < 100))
    checks.append('General RGB/default path does not activate camera heat contours')
    for pixel_origin in ('gray8', 'y16'):
        result = renderer.render(source, edge, pixel_origin=pixel_origin, ignore_regions=regions)
        assert result.palette_metrics['status'] == 'unsupported_input'
        assert result.palette_metrics['outline_pixels'] == 0
    checks.append('Gray and raw16 input do not masquerade as warm camera palette data')
    radiometric = renderer.render(source, edge, temperature_c=np.zeros((80, 160), np.float32), ignore_regions=regions)
    assert radiometric.palette_metrics['status'] == 'radiometric_priority'
    assert np.all(radiometric.image[62:87, 42:77] == (0, 255, 0))
    checks.append('Registered temperature map takes priority over camera display palette')
    small = HUDRenderer(HUDConfig(**common, thermal_palette_overlay=True, full_res_background=False)).render(source, edge, ignore_regions=regions)
    assert np.all(small.image[40:63, 56:76] == (0, 80, 255))
    assert np.any(np.all(small.image[27:79, 44:89] == (0, 0, 255), axis=2))
    checks.append('Model-sized background mode keeps red heat contours registered to its display placement')
    try:
        HUDRenderer(HUDConfig(thermal_palette_overlay=True, visible_flame_candidates=True))
    except ValueError:
        pass
    else:
        raise AssertionError('Mixed visible-light and thermal-palette modes accepted')
    checks.append('Visible-flame detection and camera thermal-palette modes are mutually exclusive')
    parser = build_parser()
    assert not parser.parse_args([]).thermal_palette_overlay
    assert parser.parse_args(['--thermal-palette-overlay']).thermal_palette_overlay
    args = parser.parse_args(['--source', 'missing', '--thermal-palette-overlay', '--no-fire-highlight'])
    with patch('run_pipeline.TEEDCore', side_effect=RuntimeError('validation boundary')):
        try:
            run(args)
        except RuntimeError as error:
            assert str(error) == 'validation boundary'
        else:
            raise AssertionError('Expected model construction boundary')
    assert not args.thermal_palette_overlay
    checks.append('CLI opt-in and existing all-heat-disable switch are wired')
    output = Path(__file__).with_name('thermal_color_full') / 'palette_integration_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'PASS', 'checks': checks}, indent=2), encoding='utf-8')
    print(f'PASS: {len(checks)} thermal palette integration contracts')


if __name__ == '__main__':
    main()
