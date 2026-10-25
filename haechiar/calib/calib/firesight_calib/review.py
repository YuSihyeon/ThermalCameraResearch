"""Standalone visual review report for intrinsic calibration results."""

from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path

import cv2
import numpy as np

from .thermal import normalize_thermal


def residual_grid(observations, target, matrix, distortion, image_size, bins=(8, 6)):
    """Return mean reprojection residual (px) and sample count for spatial bins."""
    cols, rows = bins
    totals = np.zeros((rows, cols), dtype=float)
    counts = np.zeros((rows, cols), dtype=int)
    obj = target.object_points()
    for observed in observations:
        points = np.asarray(observed, dtype=np.float32).reshape(-1, 2)
        if points.shape != (len(obj), 2) or not np.isfinite(points).all():
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, points, matrix, distortion)
        if not ok:
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2) - points, axis=1)
        for (x, y), error in zip(points, errors):
            col = min(cols - 1, max(0, int(x / image_size[0] * cols)))
            row = min(rows - 1, max(0, int(y / image_size[1] * rows)))
            totals[row, col] += float(error)
            counts[row, col] += 1
    means = np.full((rows, cols), np.nan)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means, counts


def _png_data(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _read(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error):
        return None


def _heatmap(values, counts, unit, title):
    finite = values[np.isfinite(values)]
    maximum = float(finite.max()) if finite.size else 1.0
    rows, cols = values.shape
    plot_width, cell_h = 576, 55
    cell_w = plot_width / cols
    gradient_id = "scale-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:10]
    observed_min = float(finite.min()) if finite.size else 0.0
    legend_max = float(finite.max()) if finite.size else 0.0
    parts = [f'<svg viewBox="0 0 {plot_width} {rows * cell_h + 72}" role="img" aria-label="{html.escape(title)}">']
    for row in range(rows):
        for col in range(cols):
            value, count = values[row, col], int(counts[row, col])
            if np.isfinite(value):
                ratio = min(1.0, float(value) / maximum) if maximum else 0.0
                color = f"rgb({int(245 - 40 * ratio)},{int(238 - 150 * ratio)},{int(220 - 170 * ratio)})"
                label = f"{value:.2f}{unit}"
                sub = f"n={count}"
            else:
                color, label, sub = "#d9dde3", "데이터 없음", "n=0"
            x, y = col * cell_w, row * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{x + cell_w/2}" y="{y + 23}" text-anchor="middle">{label}</text>')
            parts.append(f'<text class="small" x="{x + cell_w/2}" y="{y + 41}" text-anchor="middle">{sub}</text>')
    parts.append(f'<text class="small" x="0" y="{rows * cell_h + 20}">좌→우: 영상 x, 위→아래: 영상 y · 각 칸 값과 표본 수</text>')
    parts.append(f'<rect x="0" y="{rows * cell_h + 30}" width="90" height="10" fill="url(#{gradient_id})"/><defs><linearGradient id="{gradient_id}"><stop stop-color="rgb(245,238,220)"/><stop offset="1" stop-color="rgb(205,88,50)"/></linearGradient></defs>')
    parts.append(f'<text class="small" x="100" y="{rows * cell_h + 39}">색상 범위 · 최솟값 0.00{unit} · 최댓값 {legend_max:.2f}{unit}</text>')
    parts.append(f'<text class="small" x="100" y="{rows * cell_h + 56}">관측 최솟값 {observed_min:.2f}{unit}</text></svg>')
    return "".join(parts)


def _error_chart(errors, threshold):
    values = [float(v) for v in errors]
    width, height, pad = 760, 240, 42
    top = max([threshold, *values, 0.1]) * 1.15
    points = []
    for index, value in enumerate(values):
        x = pad + index * (width - 2 * pad) / max(1, len(values) - 1)
        y = height - pad - value / top * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")
    threshold_y = height - pad - threshold / top * (height - 2 * pad)
    circles = "".join(f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="3"/>' for p in points)
    y_ticks = []
    for tick in np.linspace(0, top, 5):
        y = height - pad - tick / top * (height - 2 * pad)
        y_ticks.append(f'<line class="grid" x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}"/><text x="{pad-6}" y="{y+4:.1f}" text-anchor="end">{tick:.2f}</text>')
    x_labels = []
    stride = max(1, int(np.ceil(len(values) / 20)))
    for index in range(0, len(values), stride):
        x = pad + index * (width - 2 * pad) / max(1, len(values) - 1)
        x_labels.append(f'<text x="{x:.1f}" y="{height-pad+17}" text-anchor="middle">{index+1}</text>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="뷰별 재투영 오차">
{''.join(y_ticks)}<line class="axis" x1="{pad}" y1="8" x2="{pad}" y2="{height-pad}"/><line class="axis" x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}"/>
<line class="limit" x1="{pad}" y1="{threshold_y:.1f}" x2="{width-pad}" y2="{threshold_y:.1f}"/><text x="{width-pad}" y="{threshold_y-5:.1f}" text-anchor="end">기준 {threshold:.2f} px</text>
<polyline points="{' '.join(points)}" fill="none" stroke="#176b87" stroke-width="2"/>{circles}{''.join(x_labels)}
<text x="8" y="18">오차 (px)</text><text x="{width/2}" y="{height-8}" text-anchor="middle">수락 뷰 번호</text></svg>'''


def _matrix_table(matrix):
    return "".join("<tr>" + "".join(f"<td>{float(v):.8g}</td>" for v in row) + "</tr>" for row in matrix)


def write_review(result, observations, target, quality, debug, records) -> Path:
    """Write a self-contained Korean HTML review and return its path."""
    debug = Path(debug)
    debug.mkdir(parents=True, exist_ok=True)
    def esc(value):
        return html.escape(str(value), quote=True)
    failures = quality.get("failures", [])
    status = "수치 휴리스틱 통과" if quality.get("passed") else "검토 필요"
    deploy = "적용 후보" if quality.get("deployment_ready") else "적용 준비 미완료"
    failure_html = "".join(f"<li>{esc(item)}</li>" for item in failures) or "<li>기록된 수치 실패 없음</li>"

    coverage_counts = np.asarray(quality.get("coverage_3x3", np.zeros((3, 3))), dtype=int)
    coverage_values = coverage_counts.astype(float)
    coverage_values[coverage_counts == 0] = np.nan
    residuals, residual_counts = residual_grid(
        observations, target, result.camera_matrix, result.distortion, result.image_size
    )

    cards = []
    accepted_index = 0
    for number, record in enumerate(records, 1):
        path = Path(str(record.get("path", "")))
        reason = record.get("reason")
        decision = "거부" if reason else "수락"
        detail = reason or record.get("method") or "검출 성공"
        observation = None
        view_error = None
        if reason is None:
            if accepted_index < len(observations):
                observation = np.asarray(observations[accepted_index])
            if accepted_index < len(result.per_view_error_px):
                view_error = float(result.per_view_error_px[accepted_index])
            accepted_index += 1
        media = ""
        image = _read(path)
        if image is not None:
            try:
                display = normalize_thermal(image)
            except (ValueError, cv2.error, TypeError):
                display = None
            original = _png_data(display) if display is not None else ""
            if reason is None and display is not None and observation is not None:
                points = observation
                overlay = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(overlay, target.pattern, points.reshape(-1, 1, 2), True)
                corrected = cv2.undistort(display, result.camera_matrix, result.distortion)
                media = f'<figure><img src="{original}" alt="원본"><figcaption>원본(표시용 정규화)</figcaption></figure><figure><img src="{_png_data(corrected)}" alt="보정"><figcaption>보정 미리보기</figcaption></figure><figure><img src="{_png_data(overlay)}" alt="검출 오버레이"><figcaption>검출 오버레이</figcaption></figure>'
            elif display is not None:
                media = f'<figure><img src="{original}" alt="거부 프레임"><figcaption>원본(표시용 정규화)</figcaption></figure>'
        error_text = f"<p>재투영 오차: {view_error:.4f} px</p>" if view_error is not None else ""
        cards.append(f'<article class="frame"><h3>#{number} {decision}</h3><p class="path">{esc(path)}</p><p>판정 사유/방법: <code>{esc(detail)}</code></p>{error_text}<div class="images">{media}</div></article>')

    distortion = ", ".join(f"{float(v):.8g}" for v in np.asarray(result.distortion).ravel())
    coverage_fraction = float(quality.get("coverage_fraction", 0.0))
    normal_spread = float(quality.get("normal_spread_deg", 0.0))
    condition_ratio = float(quality.get("homography_condition_ratio", 0.0))
    condition_min = float(quality.get("min_homography_condition_ratio", 0.01))
    target_meta = target.metadata()
    target_rows = "".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>" for key, value in target_meta.items() if key != "object_points_m")
    notes = "".join(f"<li>{esc(note)}</li>" for note in quality.get("notes", []))
    css = '''body{font-family:system-ui,sans-serif;background:#f3f5f7;color:#17212b;margin:0;line-height:1.55}main{max-width:1120px;margin:auto;padding:32px}header,.panel,.frame{background:white;border:1px solid #dfe4e8;border-radius:12px;padding:20px;margin:0 0 18px;box-shadow:0 2px 10px #17212b0b}h1,h2{margin-top:0;color:#123b4a}.badge{display:inline-block;padding:5px 10px;border-radius:99px;background:#e6f2f5;margin-right:8px}.warn{background:#fff0d7}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.metric{background:#eef4f6;padding:14px;border-radius:8px}.metric strong{display:block;font-size:1.35rem}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #d7dde2;text-align:left;font-variant-numeric:tabular-nums}svg{width:100%;height:auto}.axis{stroke:#777}.grid{stroke:#dce2e6;stroke-width:1}.limit{stroke:#c14435;stroke-dasharray:6 4}circle{fill:#176b87}svg text{font:12px system-ui}.small{font-size:10px;fill:#46515b}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}figure{margin:0}img{width:100%;image-rendering:auto;border-radius:6px;background:#111}figcaption{font-size:.85rem;color:#55616d}.path{overflow-wrap:anywhere}code{background:#edf0f2;padding:2px 5px}footer{border-top:2px solid #123b4a;padding:24px 0;font-weight:650}'''
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FireSight 카메라 보정 검토</title><style>{css}</style></head><body><main>
<header><h1>FireSight 카메라 보정 검토 보고서</h1><span class="badge">{status}</span><span class="badge warn">{deploy}</span><p>이 보고서의 수치는 의사결정을 돕는 휴리스틱입니다. 자동 승인이나 하드웨어 정확도 인증을 뜻하지 않습니다.</p></header>
<section class="panel"><h2>결과 요약</h2><div class="metrics"><div class="metric">RMS 재투영 오차<strong>{result.rms_px:.4f} px</strong>기준 ≤ {float(quality.get('max_rms_px', 0)):.3f} px</div><div class="metric">수락 / 거부<strong>{len(result.accepted_images)} / {len(result.rejected_images)}</strong>프레임</div><div class="metric">영상 크기<strong>{result.image_size[0]} × {result.image_size[1]} px</strong></div><div class="metric">타깃 상태<strong>{esc(target.status)}</strong>provisional이면 실측 확인 필요</div><div class="metric">커버리지 비율<strong>{coverage_fraction:.1%}</strong>권장 기준 ≥ 35%</div><div class="metric">법선 각도 범위<strong>{normal_spread:.2f}°</strong>권장 기준 ≥ 12°</div><div class="metric">호모그래피 조건비<strong>{condition_ratio:.4f}</strong>기준 ≥ {condition_min:.4f}</div></div><p>커버리지와 각도 범위가 기준보다 낮으면 보드 위치와 기울기를 다양하게 하여 다시 촬영하십시오. 조건비가 낮으면 초점거리 추정이 불안정할 수 있습니다.</p><h3>수치 품질 실패</h3><ul>{failure_html}</ul><ul>{notes}</ul></section>
<section class="panel"><h2>내부 파라미터</h2><p>카메라 행렬 K (초점거리와 주점 단위: 픽셀)</p><table>{_matrix_table(result.camera_matrix)}</table><p>왜곡 계수 D (무차원, plumb_bob 순서): <code>{esc(distortion)}</code></p></section>
<section class="panel"><h2>뷰별 재투영 오차</h2><p>관측점과 모델 투영점 사이의 RMS 거리입니다. 낮을수록 모델 적합이 좋지만, 학습 영상 자체의 오차이므로 독립 정확도는 아닙니다.</p>{_error_chart(result.per_view_error_px, float(quality.get('max_rms_px', 0.8)))}</section>
<section class="panel"><h2>검출 커버리지 히트맵</h2><p>영상 영역별 검출점 개수입니다. 빈 칸이 많으면 시야 가장자리의 왜곡을 충분히 구속하지 못할 수 있습니다.</p>{_heatmap(coverage_values, coverage_counts, '', '검출 커버리지')}</section>
<section class="panel"><h2>실제 공간 잔차 히트맵</h2><p>각 뷰를 solvePnP로 자세 추정한 뒤 projectPoints로 다시 투영하여 계산한 평균 잔차입니다. 단위는 픽셀(px)이며 n은 해당 칸의 점 개수입니다.</p>{_heatmap(residuals, residual_counts, ' px', '잔차 히트맵')}</section>
<section class="panel"><h2>타깃 정의</h2><table>{target_rows}</table><p>상태가 provisional이면 치수와 인쇄/가공 결과를 실측하여 확인한 뒤 적용 결정을 내려야 합니다.</p></section>
<section><h2>프레임별 판정과 미리보기</h2>{''.join(cards)}</section>
<footer>오버레이, 보정 영상, 오차 분포와 수치 실패를 검토한 뒤 터미널에서 적용 여부를 yes/no로 명시하십시오.</footer>
</main></body></html>'''
    destination = debug / "review.html"
    destination.write_text(document, encoding="utf-8")
    return destination
