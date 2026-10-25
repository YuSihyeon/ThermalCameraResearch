"""Rebuild explanatory PNGs from the implemented display contract."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge/orin"))

OUT = ROOT / "docs/teed_ui_validation"
OUT.mkdir(exist_ok=True)
FONT = Path("C:/Windows/Fonts/malgun.ttf")
BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")
BG = "#0c1422"
FG = "#edf3fa"
MUTED = "#a8b7cc"


def text(draw, xy, value, size=25, colour=FG, bold=False):
    draw.text(
        xy,
        value,
        font=ImageFont.truetype(str(BOLD if bold else FONT), size),
        fill=colour,
    )


def algorithm():
    image = Image.new("RGB", (1680, 1260), BG)
    d = ImageDraw.Draw(image)
    text(d, (54, 34), "FireSight · TEED 알고리즘 현황과 UI 개선", 44, bold=True)
    text(
        d,
        (56, 101),
        "최신 main 481569b 기준 | 체크포인트는 유지, 후처리와 표시 정책 변경 | 2026-10-08",
        25,
        MUTED,
    )
    columns = [
        (
            54,
            "#f3b16a",
            "기존 main: 단일 임계값 + 팽창",
            [
                (
                    "01  영상 입력",
                    "8비트 BGR / Y16은 백분위 정규화 후 표시 영상으로 변환",
                ),
                (
                    "02  대비 보정 → TEED",
                    "리사이즈 · CLAHE 2.0 · BGR 평균값 제거 · TEED fused 출력",
                ),
                ("03  엣지 선택", "sigmoid 확률 ≥ 0.75 → 3×3 팽창 → OSD 제거"),
                (
                    "04  HUD 합성",
                    "어두운 배경 + 초록 엣지 / 짧은 연결선·거리 필터 없음",
                ),
                (
                    "05  기존 화점 후보 표시",
                    "RGB/HSV 색 조건과 엣지 중첩 → 빨간색 / 온도 측정 아님",
                ),
            ],
        ),
        (
            862,
            "#67ddb5",
            "개선: 경계 선별 + 온도 근거 분리",
            [
                (
                    "01  원본·부가 데이터 분리",
                    "Fire360 영상 / 선택 입력: 프레임 정합 Celsius·미터 depth",
                ),
                (
                    "02  동일한 TEED 추론",
                    "학습·가중치 불변 / 원본 영상의 gradient는 대비 단서로만 사용",
                ),
                (
                    "03  표시할 엣지 선별",
                    "확률·gradient hysteresis → 중심선 → 길이 12px 미만 제거",
                ),
                (
                    "04  거리·화면 표시",
                    "정합 depth가 있으면 근거리 우선·원거리 감쇠 / 미상은 유지",
                ),
                (
                    "05  고온 등온선",
                    "온도맵 있을 때만 150·260·400°C 예시 / 없으면 TEMP N/A",
                ),
            ],
        ),
    ]
    for x, accent, heading, rows in columns:
        text(d, (x, 161), heading, 30, accent, True)
        for i, (title, detail) in enumerate(rows):
            y = 220 + i * 150
            d.rounded_rectangle(
                (x, y, x + 762, y + 119),
                radius=12,
                fill="#17263a",
                outline=accent,
                width=2,
            )
            text(d, (x + 23, y + 18), title, 28, FG, True)
            # The line is intentionally split for legibility in an exported image.
            if len(detail) > 45:
                split = detail.rfind(" / ")
                if split < 0:
                    split = detail.rfind(" · ", 0, 44)
                if split < 0:
                    split = 38
                text(d, (x + 23, y + 60), detail[:split], 21, MUTED)
                text(d, (x + 23, y + 87), detail[split:].lstrip(" /·"), 21, MUTED)
            else:
                text(d, (x + 23, y + 67), detail, 21, MUTED)
            if i < len(rows) - 1:
                d.line((x + 380, y + 124, x + 380, y + 141), fill=accent, width=3)
                d.polygon(
                    [(x + 375, y + 136), (x + 385, y + 136), (x + 380, y + 145)],
                    fill=accent,
                )
    d.rounded_rectangle((54, 1010, 1624, 1206), radius=12, fill="#202e42")
    text(d, (80, 1030), "입력 데이터가 허용하는 검증 범위", 29, bold=True)
    text(
        d,
        (80, 1080),
        "Fire360: 보유 파일은 8비트 영상. 열화상 카메라 모델·픽셀별 온도 보정정보는 확인되지 않음.",
        25,
        MUTED,
    )
    text(
        d,
        (80, 1120),
        "SmokeBasement: 엣지 밀도·기존 경계 중첩은 측정 가능. 장애물 회피 성능 보존은 별도 검증 필요.",
        25,
        MUTED,
    )
    text(
        d,
        (80, 1160),
        "150 / 260 / 400°C는 구현 검증용 설정값이며, NFPA 지정값이나 검증된 안전 기준이 아님.",
        24,
        "#f3b16a",
    )
    image.save(OUT / "teed_algorithm_status.png")


def thermal():
    from thermal_overlay import ThermalConfig, draw_temperature

    image = Image.new("RGB", (1680, 870), BG)
    d = ImageDraw.Draw(image)
    text(d, (54, 32), "고온 표시 계약 · 합성 온도맵 검증", 43, bold=True)
    text(
        d,
        (56, 100),
        "아래 온도는 기능 확인용 합성값입니다. Fire360 실측 온도가 아닙니다.",
        27,
        "#f3b16a",
    )
    yy, xx = np.mgrid[:300, :480]
    frame = np.full((300, 480, 3), 30, np.uint8)
    cv2.rectangle(frame, (70, 70), (410, 240), (230, 230, 230), -1)
    none = frame.copy()
    draw_temperature(none, None, ThermalConfig())
    cool = frame.copy()
    draw_temperature(cool, np.full((300, 480), 149.0, np.float32), ThermalConfig())
    temp = (
        25 + 500 * np.exp(-((xx - 245) ** 2 / 130**2 + (yy - 152) ** 2 / 95**2))
    ).astype(np.float32)
    hot = np.repeat(
        np.clip(temp * 0.35, 0, 150).astype(np.uint8)[:, :, None], 3, axis=2
    )
    draw_temperature(hot, temp, ThermalConfig(mode="overlay"))
    entries = [
        (
            none,
            "밝은 영상 / 온도 정보 없음",
            ["화점 강조 없음", "TEMP N/A: 온도를 알 수 없음"],
        ),
        (
            cool,
            "온도맵 149°C / 기준 150°C",
            ["기준 미만 → 고온 표시 없음", "밝아 보여도 화점으로 칠하지 않음"],
        ),
        (
            hot,
            "온도맵 기준 이상의 영역",
            ["노랑 150 / 주황 260 / 빨강 400°C", "등온선 + 반투명 오버레이"],
        ),
    ]
    for i, (array, title, lines) in enumerate(entries):
        x = 56 + i * 548
        text(d, (x, 177), title, 26, bold=True)
        image.paste(Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)), (x, 235))
        for j, line in enumerate(lines):
            text(d, (x, 573 + j * 44), line, 24, MUTED)
    text(
        d,
        (56, 737),
        "Fire360 실제 결과에서는 보정정보가 없어 고온 표시가 비활성화됩니다.",
        29,
        FG,
        True,
    )
    text(
        d,
        (56, 785),
        "색상은 고온 정도를 표현하며 화재 발생·발화 위치·안전성을 판정하지 않습니다.",
        25,
        MUTED,
    )
    image.save(OUT / "temperature_contract.png")


if __name__ == "__main__":
    algorithm()
    thermal()
    print(OUT)
