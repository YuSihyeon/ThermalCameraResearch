"""Compare an actual thermal source and TEED output with red heat contours."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge/orin'))
from render_additional_thermal_videos import read_frame  # noqa: E402
from thermal_palette import apply_thermal_palette  # noqa: E402


def main() -> None:
    cv2.setNumThreads(1)
    folder = Path(__file__).with_name('contour_stability') / 'thermal_additional'
    out = Path(__file__).with_name('thermal_outline_full')
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((folder / 'fire360_ifsi10.json').read_text(encoding='utf-8'))
    index = 467
    source = read_frame(Path(metadata['source']['path']), index)
    fitted = cv2.resize(source, (720, 405), interpolation=cv2.INTER_AREA)
    baseline = read_frame(folder / 'fire360_ifsi10_full_applied.webm', index)[:405]
    gray_source = cv2.cvtColor(cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    background = np.clip(gray_source.astype(np.float32) * metadata['config']['background_scale'], 0, 255).astype(np.uint8)
    result = baseline.copy()
    for left, top, right, bottom in metadata['osd_regions']:
        x0, y0 = int(np.floor(left * 720)), int(np.floor(top * 405))
        x1, y1 = int(np.ceil(right * 720)), int(np.ceil(bottom * 405))
        result[y0:y1, x0:x1] = background[y0:y1, x0:x1]
    mask, metrics = apply_thermal_palette(result, np.asarray(fitted, dtype=np.uint8), tuple(tuple(r) for r in metadata['osd_regions']), background=np.asarray(background, dtype=np.uint8))
    assert np.all(result[mask] == (0, 0, 255))
    panels = []
    for label, frame in [('Thermal camera', fitted), ('TEED structures', baseline), ('Red heat contours + TEED', result)]:
        panel = cv2.copyMakeBorder(frame, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1, cv2.LINE_AA)
        panels.append(panel)
    assert cv2.imwrite(str(out / 'ifsi10_hot_outline_comparison.png'), np.hstack(panels))
    assert cv2.imwrite(str(out / 'ifsi10_hot_outline.png'), result)
    (out / 'preview_metrics.json').write_text(json.dumps({'frame_index': index, **dataclasses.asdict(metrics)}, indent=2), encoding='utf-8')
    print(dataclasses.asdict(metrics))


if __name__ == '__main__':
    main()
