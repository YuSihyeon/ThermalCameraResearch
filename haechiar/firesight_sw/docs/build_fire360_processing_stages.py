#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

CANVAS_SIZE = (2000, 1125)
BACKGROUND = "#07172b"
CARD_FILL = "#0d223b"
CARD_BORDER = "#2f567c"
TEXT = "#f7fbff"
MUTED = "#aac0d5"
CYAN = "#38bdf8"
ORANGE = "#ff982e"
GREEN = "#7ee236"


@dataclass(frozen=True)
class CardSpec:
    x: int
    number: int
    title: str
    image: Image.Image
    lines: tuple[str, str]
    accent: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("edge/teed/samples/ifsi_video_4_frame_000454.jpg"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("edge/teed/models/5_model.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/fire360_teed_processing_stages.png"),
    )
    osd_group = parser.add_mutually_exclusive_group()
    osd_group.add_argument(
        "--ignore-fire360-osd",
        dest="ignore_fire360_osd",
        action="store_true",
        help="exclude the fixed badge and temperature colorbar from the final edge mask",
    )
    osd_group.add_argument(
        "--include-fire360-osd",
        dest="ignore_fire360_osd",
        action="store_false",
        help="keep edges from the fixed badge and temperature colorbar",
    )
    parser.set_defaults(ignore_fire360_osd=True)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def bgr_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", size, "#020912")
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    background.paste(fitted, (left, top))
    return background


def draw_arrow(draw: ImageDraw.ImageDraw, start_x: int, end_x: int, y: int) -> None:
    draw.line((start_x, y, end_x - 16, y), fill=TEXT, width=7)
    draw.polygon(
        ((end_x - 16, y - 14), (end_x, y), (end_x - 16, y + 14)),
        fill=TEXT,
    )


def render_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    spec: CardSpec,
) -> None:
    x = spec.x
    card_width = 420
    card_top = 240
    card_bottom = 795
    draw.rounded_rectangle(
        (x, card_top, x + card_width, card_bottom),
        radius=26,
        fill=CARD_FILL,
        outline=spec.accent,
        width=4,
    )
    draw.ellipse((x + 24, card_top + 22, x + 78, card_top + 76), fill=spec.accent)
    number_text = str(spec.number)
    number_font = load_font(28, bold=True)
    bbox = draw.textbbox((0, 0), number_text, font=number_font)
    draw.text(
        (x + 51 - (bbox[2] - bbox[0]) / 2, card_top + 49 - (bbox[3] - bbox[1]) / 2),
        number_text,
        font=number_font,
        fill=BACKGROUND,
        anchor="mm",
    )
    draw.text(
        (x + 94, card_top + 28),
        spec.title,
        font=load_font(32, bold=True),
        fill=TEXT,
    )
    panel = fit_image(spec.image, (376, 282))
    canvas.paste(panel, (x + 22, card_top + 100))
    draw.line(
        (x + 24, card_top + 402, x + card_width - 24, card_top + 402),
        fill=CARD_BORDER,
        width=2,
    )
    draw.text(
        (x + 28, card_top + 430),
        spec.lines[0],
        font=load_font(24, bold=True),
        fill=TEXT,
    )
    draw.text(
        (x + 28, card_top + 474),
        spec.lines[1],
        font=load_font(21),
        fill=MUTED,
    )


def make_stage_images(
    repo_root: Path,
    source_path: Path,
    checkpoint_path: Path,
    ignore_fire360_osd: bool,
) -> tuple[list[Image.Image], float]:
    teed_dir = repo_root / "edge" / "teed"
    sys.path.insert(0, str(teed_dir))
    import run_teed  # noqa: PLC0415

    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        message = f"unable to read Fire360 source: {source_path}"
        raise RuntimeError(message)

    device = torch.device("cpu")
    args = SimpleNamespace(
        width=320,
        height=240,
        threshold=0.75,
        background_scale=0.24,
        contrast_clip_limit=2.0,
        edge_width=3,
    )
    model = run_teed.load_model(str(checkpoint_path), device)
    resized, tensor = run_teed.prepare_tensor(
        frame,
        args.width,
        args.height,
        device,
        args.contrast_clip_limit,
    )
    enhanced = tensor[0, :, : args.height, : args.width].permute(1, 2, 0).numpy()
    enhanced = np.clip(enhanced + run_teed.MEAN_BGR, 0, 255).astype(np.uint8)

    with torch.no_grad():
        outputs = model(tensor, single_test=True)
        probability = torch.sigmoid(outputs[-1])[0, 0, : args.height, : args.width]
    probability_array = probability.numpy()
    probability_u8 = np.clip(probability_array * 255.0, 0, 255).astype(np.uint8)
    probability_bgr = cv2.cvtColor(probability_u8, cv2.COLOR_GRAY2BGR)

    overlay, mask, _, _ = run_teed.infer(
        model,
        frame,
        args,
        device,
        ignore_fire360_osd=ignore_fire360_osd,
    )
    coverage = float(np.count_nonzero(mask)) / float(mask.size) * 100.0
    return (
        [
            bgr_to_image(resized),
            bgr_to_image(enhanced),
            bgr_to_image(probability_bgr),
            bgr_to_image(overlay),
        ],
        coverage,
    )


def save_individual_stages(output_path: Path, stages: list[Image.Image]) -> None:
    stage_dir = output_path.parent / "fire360_stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "01_fire360_input.png",
        "02_low_visibility_preprocessing.png",
        "03_teed_probability.png",
        "04_field_postprocessing.png",
    )
    for name, stage in zip(names, stages, strict=True):
        stage.save(stage_dir / name, format="PNG", optimize=True)


def render_overview(
    output_path: Path,
    stages: list[Image.Image],
    edge_coverage: float,
    source_name: str,
) -> None:
    canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (72, 45),
        "열화상 HUD 기반 TEED 처리 과정",
        font=load_font(56, bold=True),
        fill=TEXT,
    )
    draw.text(
        (74, 122),
        "열화상 HUD 입력부터 고정 화면 표시를 제외한 초록색 경계까지, 현재 시제품의 실제 출력",
        font=load_font(29),
        fill=MUTED,
    )
    draw.rounded_rectangle((1570, 50, 1928, 112), radius=24, fill="#143252")
    draw.text(
        (1749, 81),
        "CURRENT MVP · ACTUAL OUTPUT",
        font=load_font(21, bold=True),
        fill=CYAN,
        anchor="mm",
    )

    cards = (
        CardSpec(
            52,
            1,
            "열화상 HUD 원본",
            stages[0],
            ("Fire360 · IFSI Video 4", "소방관 · 고온부 · 온도 컬러바"),
            ORANGE,
        ),
        CardSpec(
            532,
            2,
            "화면 선명화",
            stages[1],
            ("어두운 부분 선명하게", "형태가 보이도록 대비 보강"),
            CYAN,
        ),
        CardSpec(
            1012,
            3,
            "TEED 처리",
            stages[2],
            ("경계가 될 가능성", "밝을수록 강한 경계"),
            CYAN,
        ),
        CardSpec(
            1492,
            4,
            "현장용 후처리",
            stages[3],
            ("고정 표시 제외 경계", "약한 선 제거 · 경계 3 px"),
            GREEN,
        ),
    )
    for index, card in enumerate(cards):
        render_card(canvas, draw, card)
        if index < len(cards) - 1:
            draw_arrow(draw, card.x + 428, cards[index + 1].x - 10, 515)

    draw.rounded_rectangle(
        (52, 842, 1948, 1024), radius=24, fill="#0a1d34", outline=CARD_BORDER, width=3
    )
    draw.text((82, 870), "처리 원칙", font=load_font(27, bold=True), fill=CYAN)
    draw.text(
        (245, 870),
        "밝기·대비 보정은 모델 입력에만 적용하고, 최종 배경은 열화상 원본을 24% 밝기로 유지",
        font=load_font(25),
        fill=TEXT,
    )
    draw.text((82, 930), "현재 범위", font=load_font(27, bold=True), fill=ORANGE)
    draw.text(
        (245, 930),
        "사전학습된 범용 TEED 모델 사용 · FireSight 전용 파인튜닝 전 · "
        "온도 숫자와 기관 마크는 최종 경계에서 제외",
        font=load_font(24),
        fill=TEXT,
    )
    footer = (
        f"실제 입력: {source_name}   |   "
        f"최종 경계 비율: {edge_coverage:.1f}%   |   PC에서 재현한 이미지"
    )
    draw.text((54, 1072), footer, font=load_font(20), fill=MUTED)
    draw.text(
        (1946, 1072),
        "FireSight Edge MVP",
        font=load_font(20, bold=True),
        fill=GREEN,
        anchor="ra",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_path = args.source if args.source.is_absolute() else repo_root / args.source
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else repo_root / args.checkpoint
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    stages, coverage = make_stage_images(
        repo_root,
        source_path,
        checkpoint_path,
        args.ignore_fire360_osd,
    )
    save_individual_stages(output_path, stages)
    render_overview(output_path, stages, coverage, source_path.name)
    sys.stdout.write(f"created {output_path}\nedge coverage: {coverage:.2f}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
