"""Sequential sustained measurements and a local Korean report/video.

Use --measure cpu,eager,graph for three independent 180s runs. Build artifacts
afterwards with --build. No simultaneous performance tests are launched.
"""
import argparse
import csv
import hashlib
import html
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/obstacle_edges/deliverables/media/smoke_heldout_original.mp4'
sys.path.insert(0, str(ROOT))
CASES = {'cpu': ('cpu_30', 'CPU FP32', ['--device', 'cpu']),
         'eager': ('gpu_eager_30', 'CUDA FP32', ['--device', 'cuda']),
         'graph': ('gpu_graph_30', 'CUDA Graph FP32', ['--device', 'cuda', '--cuda-graph'])}


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def measure(output, names):
    for name in names:
        stem, label, flags = CASES[name]
        command = [sys.executable, str(ROOT / 'edge/teed/run_realtime.py'),
                   '--source', str(SOURCE), '--loop', '--replay-fps', '30',
                   '--target-fps', '30', '--duration', '180', '--synthetic-depth',
                   '--depth-port', '0', '--record', str(output / (stem + '.mp4')),
                   '--metrics', str(output / (stem + '.json')),
                   '--frame-csv', str(output / (stem + '.csv')), '--require-stable'] + flags
        (output / (stem + '_command.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
        print('START ' + label, flush=True)
        with open(output / (stem + '.log'), 'w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        print('END %s exit=%d' % (label, result.returncode), flush=True)
        if result.returncode not in (0, 3):
            raise RuntimeError('runtime error: inspect ' + stem + '.log')


def comparison_video(output):
    import cv2
    import imageio_ffmpeg
    import numpy as np
    import torch
    from edge.teed.realtime_core import RealtimeCore
    from experiments.distance_width.benchmark import Variants, blend
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    graph = RealtimeCore(cuda_graph=True)
    eager = RealtimeCore()
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError('cannot open public video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, masks, max_diff, disagreed, pixels = 0, [], 0., 0, 0
    union, intersection = 0, 0
    yy, xx = np.mgrid[:240, :320].astype(np.float32)
    variants = Variants()
    writer = imageio_ffmpeg.write_frames(str(output / 'comparison_native_10fps.mp4'),
               (640, 556), fps=fps, codec='libx264', pix_fmt_in='bgr24',
               pix_fmt_out='yuv420p', quality=8, macro_block_size=1, ffmpeg_log_level='error')
    writer.send(None)
    def panel(view, title, note):
        result = np.full((278, 320, 3), (22, 26, 30), np.uint8)
        result[38:] = view
        cv2.putText(result, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (235, 240, 245), 1, cv2.LINE_AA)
        cv2.putText(result, note, (5, 31), cv2.FONT_HERSHEY_SIMPLEX, .32, (100, 215, 245), 1, cv2.LINE_AA)
        return result
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            g, e = graph.infer(frame), eager.infer(frame)
            max_diff = max(max_diff, float(np.max(np.abs(g['probability'] - e['probability']))))
            disagreed += int(np.count_nonzero(g['mask'] != e['mask']))
            pixels += g['mask'].size
            union += int(np.count_nonzero((g['mask'] > 0) | (e['mask'] > 0)))
            intersection += int(np.count_nonzero((g['mask'] > 0) & (e['mask'] > 0)))
            masks.append(g['mask'])
            timestamp = frames / fps
            depth = (4.5 + 2.8 * np.sin(timestamp * .7 + xx / 110. + yy / 180.)).astype(np.float32)
            panels = [panel(g['image'], 'Public thermal input', 'Native 10fps | %.1fs | depth not measured' % timestamp)]
            for mode, title in (('thin_3', 'Fixed 3px centerline'), ('banded', 'Distance: hard 1 / 3 / 5px'), ('smooth', 'Distance: continuous + 50ms')):
                alpha = variants.process(mode, g['mask'], depth, timestamp, timestamp)
                note = 'Same TEED mask' if mode == 'thin_3' else 'SYNTHETIC DEPTH | near thick / far thin'
                panels.append(panel(blend(g['image'], alpha, mode != 'smooth'), title, note))
            grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
            writer.send(np.ascontiguousarray(grid))
            if frames == 60:
                encoded, image = cv2.imencode('.png', grid)
                if not encoded:
                    raise RuntimeError('comparison image encode failed')
                (output / 'comparison.png').write_bytes(image.tobytes())
            frames += 1
    finally:
        writer.close()
        cap.release()
    np.save(str(output / 'gpu_shared_masks.npy'), np.stack(masks), allow_pickle=False)
    result = {'frames': frames, 'source_fps': fps, 'max_abs_probability_difference': max_diff,
              'mask_disagreement_pixels': disagreed, 'pixels_compared': pixels,
              'mask_disagreement_pct': 100. * disagreed / pixels,
              'edge_iou': intersection / float(max(union, 1)),
              'compared': 'same-device CUDA eager FP32 vs CUDA Graph FP32; all source frames'}
    (output / 'quality.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def table(headers, rows):
    return '<div class="scroll"><table><tr>' + ''.join('<th>%s</th>' % html.escape(str(x)) for x in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join('<td>%s</td>' % html.escape(str(x)) for x in row) + '</tr>' for row in rows) + '</table></div>'


def build(output):
    comparison_video(output)
    import cv2
    import numpy as np
    import torch
    import imageio_ffmpeg
    quality = json.loads((output / 'quality.json').read_text())
    runs = [(stem, label, json.loads((output / (stem + '.json')).read_text()))
            for stem, label, _ in CASES.values() if (output / (stem + '.json')).exists()]
    headers = ['방식', '측정 초', '프레임', '평균 FPS', '10초 최저 FPS', '처리 P99 ms', '수신→출력 P99 ms',
               '출력 간격 최대 ms', '예산 초과 %', '누락 %', '판정']
    rows = []
    for stem, label, data in runs:
        s = data['stability']
        rows.append([label, '%.3f' % s['elapsed_s'], s['processed'], '%.3f' % s['effective_fps'],
                     '%.2f' % min(w['fps'] for w in s['windows']), '%.3f' % s['processing_ms']['p99'],
                     '%.3f' % s['host_frame_age_ms']['p99'], '%.3f' % s['output_gap_ms']['max'],
                     '%.3f' % s['deadline_miss_pct'], '%.3f' % s['drop_pct'], 'PASS' if s['passed'] else 'FAIL'])
    with open(output / 'comparison.csv', 'w', newline='', encoding='utf-8-sig') as stream:
        csv.writer(stream).writerows([headers] + rows)
    gpu = dict((stem, data) for stem, _, data in runs)['gpu_graph_30']
    s = gpu['stability']
    lead = ('RTX 5060 Ti · CUDA Graph FP32에서 %.1f초 / %d프레임을 처리했다. 평균 %.3ffps, '
            '처리 P99 %.3fms, 수신→출력 P99 %.3fms. 사전에 정한 30fps 지속성 기준 %s.' %
            (s['elapsed_s'], s['processed'], s['effective_fps'], s['processing_ms']['p99'],
             s['host_frame_age_ms']['p99'], '전부 통과' if s['passed'] else '미달'))
    limits = ('측정 장치는 Windows PC다. Jetson Orin Nano와 실제 거리 센서는 연결되지 않았으며 센서는 미정이다. '
              '현재 영상의 거리 맵은 합성값이다. 실제 장치의 FPS·거리 정확도·영상 등록 정확도는 아직 검증되지 않았다.')
    method = ('모든 표 행은 같은320×240 공개 영상,FP32,각2 CPU threads,예열30회,합성 거리 TCP 전송,선 굵기 렌더링,MP4 저장 조건이다. '
              '원본120프레임·10fps를30fps로 가속 반복했다. 동시 벤치마크는 실행하지 않았다. '
              '처리 지연은 GPU 완료를 기다린 후 측정했고 VideoWriter.write 호출을 포함한다. 종료 시 인코더·디스크 flush는 제외한다. 호스트 수신→출력 지연은 카메라 read 반환 이후부터이며 '
              '센서 노출·드라이버 버퍼·물리 디스플레이 스캔아웃은 제외한다. 이 측정의 창 표시는 꺼져 있다.')
    criteria = ('180초 이상,평균≥29.7fps,모든10초 구간≥29.4fps,처리 P99≤33.333ms,예산 초과≤1%,누락≤1%, '
                '수신→출력 P99≤66.667ms,출력 간격 P99≤50ms,최대≤100ms,오류 없음·정상 종료를 모두 요구한다. 시작·종료 공백도 포함한다.')
    quality_text = ('동일 공개 영상 %d프레임에서 CUDA eager FP32와 CUDA Graph FP32를 비교했다. 확률 최대 절대 차이 %.9g, '
                    '이진 경계 불일치 %.6f%%,경계 IoU %.9f. FP16으로 정밀도를 낮추지 않았다.' %
                    (quality['frames'], quality['max_abs_probability_difference'], quality['mask_disagreement_pct'], quality['edge_iou']))
    changes = [
        '센서를 특정하지 않고 같은 호스트 어댑터 → 등록된 미터 깊이 → 인과적 시각 선택 → 거리 굵기 경로를 추가했다.',
        '카메라 최신 프레임1개와 깊이 최근8개만 보관한다. 오래된 영상 누적을 막고 결측 거리에는1px와 상태 표시를 적용한다.',
        '고정 입력 버퍼·CUDA Graph로 모델 실행 비용을 줄였다. 기존 파일 재생과 고정 선 경로도 유지한다.',
        'Windows GetTickCount64의15.625ms 정밀도로 시각이 중복된 실패를 재현했다. perf_counter로 통일하고 재생 대기도 고정밀 sleep으로 수정했다.',
        '독립 검토에서 종료 직전 멈춤 누락과640×480 보정→320×240 입력 불일치를 찾아 회귀 테스트와 함께 수정했다.',
        '실제 보드 검증은 Orin Nano·CUDA·직접 카메라·실측 거리·등록 ID·유효 거리 대응·180초 지속성 조건을 따로 검사한다.',
        '로컬 YSH에 정리했다. GitHub 원격 push와 외부 공유는 하지 않았다.'
    ]
    source = ('Jianzhu Huai, SmokeBasement, Zenodo (2025), CC BY4.0, DOI10.5281/zenodo.15173055. '
              '원본run5 열화상에서 재표본화한 클립에 TEED와 합성 거리 표시를 적용했다. 원저자의 보증을 뜻하지 않는다.')
    windows = [[w['start_s'], *['%.1f' % d['stability']['windows'][i]['fps'] if i < len(d['stability']['windows']) else '—'
                               for _, _, d in runs]] for i, w in enumerate(s['windows'])]
    raw_links = ' · '.join('<a href="%s.json">%s JSON</a> / <a href="%s.csv">프레임 CSV</a>' % (stem, html.escape(label), stem) for stem, label, _ in runs)
    body = ('<p class="eyebrow">FIRESIGHT · YSH · LOCAL DEVELOPMENT</p><h1>거리 기반 선 굵기<br>안정 FPS 검증</h1>'
            '<p class="lead">%s</p><p class="note">%s</p><h2>같은 입력·출력 조건 비교</h2>%s<p>%s</p><p>%s</p>'
            '<h2>재현 영상</h2><p>아래 비교는 원본10fps 속도다. 고정3px / 거리 구간식 / 연속 평활화가 같은 TEED 경계를 공유한다. '
            '거리 측정 정확도가 아닌 선 굵기 표시를 비교한다.</p><video controls preload="metadata" poster="comparison.png" src="comparison_native_10fps.mp4"></video>'
            '<p><a href="gpu_graph_30.mp4">180초 측정 중 저장한 원본 영상</a> · <a href="../distance_width_2026-09-14/REPORT.html">이전 CPU·노이즈·급변·결측 비교 보고서</a></p>'
            '<h2>검출 결과 보존 확인</h2><p>%s</p><h2>장치 연결 준비</h2><p>%s</p>'
            '<p>센서 SDK 어댑터와 실제 보정을 준비한 뒤 <code>edge/orin/run_distance.py --verify-device</code>로 확인한다. '
            '명령·조건은 저장소의 <code>edge/teed/REALTIME.md</code>와 <code>DEVICE_ADAPTER.md</code>에 있다. '
            '보정 JSON 형식 검사만으로 물리 정렬 정확도가 검증되지는 않는다.</p>'
            '<details><summary>10초 구간별 FPS</summary>%s</details><h2>개발 과정</h2><ol>%s</ol>'
            '<h2>원자료와 재현</h2><p>%s</p><p><a href="quality.json">품질 비교</a> · <a href="verification.json">영상 전체 디코드·해시</a> · '
            '<a href="environment.json">환경</a> · <a href="comparison.csv">비교표 CSV</a></p>'
            '<pre>python experiments/distance_width/realtime_compare.py --output OUTPUT --measure cpu,eager,graph\n'
            'python experiments/distance_width/realtime_compare.py --output OUTPUT --build</pre><p>%s '
            '<a href="https://doi.org/10.5281/zenodo.15173055">공개 데이터 출처</a></p>') % (
            html.escape(lead), html.escape(limits), table(headers, rows), html.escape(method), html.escape(criteria),
            html.escape(quality_text), html.escape(limits), table(['시작 초'] + [label for _, label, _ in runs], windows),
            ''.join('<li>%s</li>' % html.escape(x) for x in changes), raw_links, html.escape(source))
    style = 'body{max-width:1160px;margin:48px auto;padding:0 28px;color:#172e35;background:#f5f4ee;font:16px/1.75 system-ui,sans-serif}h1{font-size:46px;line-height:1.2;letter-spacing:-2px}h2{margin-top:48px;font-size:25px}.eyebrow{color:#25756a;font-weight:700;letter-spacing:2px}.lead{font-size:22px;max-width:960px}.note{padding:20px;background:#fff3d8;border-left:5px solid #d39b3a}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:#fff}th{background:#204e50;color:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #d5dddd;white-space:nowrap}video{width:100%;max-width:900px;background:#162629;border-radius:8px}a{color:#176968}pre{overflow:auto;background:#e5ebe7;padding:18px}details{padding:20px;background:#e5ebe7}li{margin:12px 0}'
    (output / 'REPORT.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FireSight 거리 선 FPS 검증</title><style>%s</style><body>%s</body></html>' % (style, body), encoding='utf-8')
    md = '# 거리 기반 선 굵기 · 지속 FPS\n\n' + lead + '\n\n' + limits + '\n\n'
    md += '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])
    md += '\n\n' + method + '\n\n' + criteria + '\n\n' + quality_text + '\n\n' + '\n'.join('- ' + x for x in changes)
    md += '\n\n[영상과 상세 보고서](REPORT.html) · [비교 영상](comparison_native_10fps.mp4)\n\n' + source
    (output / 'REPORT.md').write_text(md, encoding='utf-8')
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'numpy': np.__version__,
                   'opencv': cv2.__version__, 'ffmpeg': imageio_ffmpeg.get_ffmpeg_version(),
                   'gpu': torch.cuda.get_device_name(), 'source_sha256': sha(SOURCE),
                   'checkpoint_sha256': sha(ROOT / 'edge/teed/models/5_model.pth')}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    verification = []
    for video in output.glob('*.mp4'):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError('cannot decode ' + str(video))
        count, fps = 0, cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        expected = quality['frames'] if video.stem == 'comparison_native_10fps' else dict((stem, d['stability']['processed']) for stem, _, d in runs).get(video.stem)
        if expected is not None and count != expected:
            raise RuntimeError('video frame count mismatch: ' + str(video))
        verification.append({'file': video.name, 'decoded_frames': count, 'fps': fps, 'expected_frames': expected, 'sha256': sha(video)})
    (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps({'runs': len(runs), 'quality': quality, 'videos_verified': len(verification)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--measure', help='comma-separated cpu,eager,graph; each180s')
    p.add_argument('--build', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.measure:
        measure(output, args.measure.split(','))
    if args.build:
        build(output)
