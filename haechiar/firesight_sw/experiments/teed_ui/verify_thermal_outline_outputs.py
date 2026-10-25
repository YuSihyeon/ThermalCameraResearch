"""Audit all complete thermal outline artifacts and their recorded provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('thermal_outline_full')
COUNTS = {4: 625, 5: 1669, 6: 2153, 7: 3488, 8: 661, 9: 1610, 10: 1868}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    rows = []
    for number, expected in COUNTS.items():
        name = f'fire360_ifsi{number}'
        data = json.loads((OUT / f'{name}.json').read_text(encoding='utf-8'))
        progress = json.loads((OUT / f'{name}_progress.json').read_text(encoding='utf-8'))
        assert progress['status'] == 'COMPLETE', name
        assert data['frames'] == expected and data['start_frame'] == 0 and data['end_frame'] == expected - 1
        records = data['records']
        assert [r['frame_index'] for r in records] == list(range(expected))
        fps = data['source']['fps']
        assert abs(data['duration_s'] - expected / fps) < 1e-6
        assert all(abs(r['source_seconds'] - i/fps) < 1e-8 for i, r in enumerate(records))
        assert digest(Path(data['source']['path'])) == data['source']['sha256']
        for relative, sha in data['code_sha256'].items():
            assert digest(ROOT / relative) == sha, relative
        if number in (7, 9):
            qa = json.loads((OUT / f'{name}_qa.json').read_text(encoding='utf-8'))
            assert qa['status'] == 'PASS'
            output = qa['output']
            palette = [r['palette'] for r in records]
            assert data['config']['thermal_palette_overlay']
        else:
            assert data['status'] == 'PASS'
            output = data['output']
            palette = records
            assert digest(Path(data['input_teed_video'])) == data['input_teed_sha256']
        assert output['status'] == 'PASS' and output['frames'] == expected
        assert (output['width'], output['height']) == (720, 406)
        assert abs(output['fps'] - fps) < .01
        assert digest(Path(output['path'])) == output['file_sha256']
        assert all(p['status'] == 'thermal_palette_contours' for p in palette)
        assert all(p['outline_pixels'] >= 0 and p['hot_region_pixels'] >= 0 for p in palette)
        rows.append({'clip':name,'frames':expected,'duration_s':expected/fps,
                     'frames_with_red':sum(p['outline_pixels'] > 0 for p in palette),
                     'red_outline_pixels':sum(p['outline_pixels'] for p in palette),
                     'path':output['path'],'sha256':output['file_sha256'],'status':'PASS'})
    assert sum(r['frames'] for r in rows) == 12074
    result = {'status':'PASS','total_frames':12074,
              'total_duration_s':sum(r['duration_s'] for r in rows),'videos':rows,
              'scope':'Source/code/output hashes, whole-decode QA, original full frame order and timeline. Does not assess fire detection ground truth or obstacle avoidance.'}
    (OUT / 'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
