"""FLIR Boson 320 촬영용 녹화 스크립트 (16-bit raw TIFF) — 2차 촬영: 온도차 스윕·근접 열원·움직임.

실행:  py -3.12 record_session.py [구도코드]
  구도코드를 안 주면 촬영순서.csv에서 아직 안 찍은 다음 구도를 자동 선택한다.
  그다음 콘솔에서 "최대 녹화 시간(분)"을 묻는다 (Enter = 촬영순서의 값, 없으면 5분).
  TEST로 시작하는 코드는 연습 폴더에 저장된다.
  촬영 폴더는 환경변수 BOSON_SHOOT_DIR (촬영시작.bat이 자기 폴더로 설정), 없으면 현재 폴더의 boson_shoot.
  시리얼 포트는 환경변수 BOSON_PORT (기본 COM3).

미리보기 창에 '지금 할 일'과 눌러야 할 키가 표시된다. 키는 미리보기 창을 클릭한 뒤 누른다.
  Enter  FFC 1회 실행 -> FFC 완료 확인 -> FFC 명령 2초 후 녹화 시작
  S      EVENT 기록. T 구도: 긴 비프에 S(시작) → 히터를 프레임 밖으로 치우고 나와서 S(끝)
  Q      두 번 누르면 녹화 종료 (녹화 전이면 한 번에 종료)
  N      (READY 화면) 이 구도 건너뛰기
  + / -  (READY 화면) 최대 녹화 시간 1분씩 조정

EVENT 신호 (촬영순서의 이벤트(S)=예 인 구도, 예: T01~T06): 녹화 시작 20초 후 긴 비프,
  5초 전부터 1초마다 짧은 비프.
  clear     = 녹화 시작 ~ 첫 S 직전 (사람이 들어오기 전), clear_ref = 그 마지막 10초
  hot_ref   = 두 번째 S + 2초부터 10초 (히터·사람이 빠진 직후 = 가장 뜨거운 깨끗한 구간)
카메라 움직임 경고: 고정 구도에서 녹화 3~5초 화면 기준 2px 이상 밀리면 경고 (이동 촬영 M은 끔).
  EVENT 신호 5초 전 ~ 두 번째 S 후 3초는 사람이 프레임에 들어오므로 검사를 쉰다.

저장 위치: <촬영 폴더>/sessions/<구도코드>/  (연습: <촬영 폴더>/연습/<구도코드>/)
  frames/000000.tif ...   16-bit raw 프레임 (변환 없이 카메라 값 그대로, 1장 164,096바이트)
  timestamps.csv          frame, elapsed_s, unix_time
  frozen_frames.csv       직전 프레임과 픽셀이 완전히 같은 프레임 (FFC 정지 등)
  motion.csv              0.5초마다 기준 화면 대비 이동량 (고정 구도)
  meta.json               장면 정보, EVENT, clear / clear_ref, 카메라 움직임, FFC, 총 프레임 수
"""
import csv
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import winsound
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile
from flirpy.camera.boson import (
    Boson,
    FLR_BOSON_AUTO_FFC,
    FLR_BOSON_EXTERNAL_FFC,
    FLR_BOSON_MANUAL_FFC,
    FLR_BOSON_SHUTTER_TEST_FFC,
)
from PIL import Image, ImageDraw, ImageFont

PORT = os.environ.get("BOSON_PORT") or "COM3"   # Boson 시리얼(FFC 제어) 포트
H, W = 256, 320
# 촬영 폴더: 촬영시작.bat이 BOSON_SHOOT_DIR을 자기 폴더로 설정한다. 없으면 현재 폴더의 boson_shoot
SHOOT_DIR = Path(os.environ.get("BOSON_SHOOT_DIR") or Path.cwd() / "boson_shoot")
SESSIONS_DIR = SHOOT_DIR / "sessions"
PRACTICE_DIR = SHOOT_DIR / "연습"
PRACTICE_PREFIX = "TEST"
MOVING_PREFIX = "M"             # M01~M07: 카메라를 머리에 달고 이동하는 촬영
SHOT_LIST_CSV = SHOOT_DIR / "촬영순서.csv"
AUTO_SKIP_ORDERS = set()        # 자동 순서에서 뺄 순서 번호

FFC_MIN_WAIT_SEC = 1.0          # FFC 명령 후 최소 대기
FFC_TIMEOUT_SEC = 5.0           # 이 시간 안에 정지가 안 끝나면 경고 후 진행
FFC_TO_REC_SEC = 2.0            # FFC 명령 -> 녹화 시작
DEFAULT_MAX_MIN = 5.0           # 촬영순서에 값이 없을 때 최대 녹화 시간(분)
MAX_MIN_LIMIT = (0.5, 30.0)     # 입력 허용 범위(분)
EVENT_SIGNAL_SEC = 20.0         # 녹화 시작 -> EVENT 신호음
COUNTDOWN_BEEPS = 5             # 신호 N초 전부터 1초마다 비프
CLEAR_REF_SEC = 10.0            # clear_ref: clear 끝에서 거슬러 올라간 길이
EVENT_MOTION_PAUSE = (5.0, 3.0)   # 신호 몇 초 전 ~ 두 번째 S 후 몇 초 동안 움직임 검사 쉼 (사람이 프레임에 들어옴)
EVENT_PAUSE_MAX_SEC = 60.0        # 두 번째 S를 안 누르면 첫 S(또는 신호) 후 이 시간까지만 쉼
HOT_REF_DELAY_SEC = 2.0           # hot_ref: 두 번째 S 후 이만큼 지나서부터
HOT_REF_SEC = 10.0                # hot_ref 길이 (히터·사람이 빠진 직후 = 가장 뜨거운 깨끗한 구간)
Q_CONFIRM_SEC = 3.0             # 녹화 중 Q 두 번 누르기 허용 간격
NOMINAL_FPS = 8.57
TIFF_BYTES = 164096             # 프레임 1장 TIFF 크기
MOTION_SETTLE_FRAMES = 26       # 녹화 시작 후 이 프레임(약 3초)은 Enter 누른 흔들림이 가라앉게 건너뜀
MOTION_REF_FRAMES = 17          # 그다음 이 프레임 수(약 2초) 평균을 움직임 기준 화면으로
MOTION_CHECK_EVERY = 4          # 이 프레임마다(약 0.5초) 기준과 비교
MOTION_PX = 2.0                 # 이 픽셀 이상 밀리면 움직임 경고 시작
MOTION_CLEAR_PX = 1.2           # 경고 중이면 이 픽셀 이하로 돌아와야 해제 (경계에서 깜빡임 방지)
MOTION_MIN_RESP = 0.5           # 위상상관 신뢰도가 이보다 낮고 이동량도 MOTION_CLEAR_PX를 넘으면 움직임

WINDOW = "Boson recorder"
PREVIEW_SCALE = 2               # 영상 640x512
PANEL_W = 600                   # 오른쪽 안내 패널 폭
FONT_REG = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"
MODE_NAME = {
    FLR_BOSON_MANUAL_FFC: "MANUAL",
    FLR_BOSON_AUTO_FFC: "AUTO",
    FLR_BOSON_EXTERNAL_FFC: "EXTERNAL",
    FLR_BOSON_SHUTTER_TEST_FFC: "SHUTTER_TEST",
}
KEY_ENTER = (13, 10)

# 색은 BGR (OpenCV 배열 순서 그대로 PIL에 넘긴다)
RED, WHITE, GRAY, YELLOW, ORANGE, GREEN, CYAN = (
    (60, 60, 255), (255, 255, 255), (170, 170, 170), (0, 255, 255), (0, 165, 255), (80, 220, 80), (255, 255, 0))


def banner(lines, ch="!"):
    width = max(len(s) for s in lines) + 8
    print("\n" + ch * width)
    for s in lines:
        print(f"{ch*3} {s}")
    print(ch * width + "\n", flush=True)


def check_raw(frame):
    """판정만 한다. 값 변환(astype 등)으로 raw를 만들어내지 않는다."""
    if frame is None:
        return False, "frame is None"
    desc = f"dtype={frame.dtype}, shape={frame.shape}, min={int(frame.min())}, max={int(frame.max())}"
    ok = frame.dtype == np.uint16 and frame.shape == (H, W) and int(frame.max()) > 255
    return ok, desc


def is_moving(code):
    return code.startswith(MOVING_PREFIX)


def est_bytes(max_sec):
    return int(max_sec * NOMINAL_FPS * TIFF_BYTES)


# ---------------------------------------------------------------- 소리

def beep_async(freq=1500, ms=700):
    # winsound.Beep은 블로킹이라 별도 스레드에서 울린다 (프레임 수신이 끊기지 않도록)
    threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()


def beep_countdown():
    beep_async(1000, 120)


def beep_signal():
    beep_async(1500, 700)


def beep_motion():
    def pattern():
        for _ in range(2):
            winsound.Beep(500, 200)
            time.sleep(0.08)
    threading.Thread(target=pattern, daemon=True).start()


def beep_end():
    # 녹화가 끝난 뒤라 블로킹으로 울려도 된다 (프로그램 종료 전에 끝까지 재생)
    for _ in range(3):
        winsound.Beep(800, 250)
        time.sleep(0.1)


# ---------------------------------------------------------------- 카메라 움직임

class MotionChecker:
    """녹화 초반 평균 화면 대비 이동량(위상상관)으로 카메라 움직임을 감지한다."""

    def __init__(self):
        self.buf = []
        self.ref = None
        self.win = None
        self.episodes = []      # {start_frame, start_s, end_frame, end_s, max_shift_px, min_response}
        self.cur = None
        self.last = None        # (dx, dy, resp)
        self.rows = []          # motion.csv 행
        self.last_ok_idx = None  # 첫 움직임 전, 마지막으로 정렬이 확인된 프레임
        self.paused_frames = [None, None]  # 검사를 쉰 첫·마지막 프레임

    @staticmethod
    def _norm(f):
        f = f.astype(np.float32)  # 비교 계산용 (저장 데이터는 건드리지 않음)
        # 히터처럼 아주 뜨거운 부분이 비교를 좌우하지 않도록 1~97% 밝기 범위로 자른다
        lo, hi = np.percentile(f, [1, 97])
        f = np.clip(f, lo, hi)
        return ((f - f.mean()) / (f.std() + 1e-6)).astype(np.float32)

    def update(self, frame, idx, el, paused=False):
        """새 움직임 구간이 시작되면 True. paused면 기준 화면만 잡고 비교는 쉰다."""
        if idx < MOTION_SETTLE_FRAMES:
            return False
        if self.ref is None:
            self.buf.append(self._norm(frame))
            if len(self.buf) >= MOTION_REF_FRAMES:
                self.ref = self._norm(np.mean(self.buf, axis=0))
                self.win = cv2.createHanningWindow((W, H), cv2.CV_32F)
                self.ref_range = (idx - len(self.buf) + 1, idx)
                self.buf = []
            return False
        if idx % MOTION_CHECK_EVERY:
            return False
        if paused:
            if self.paused_frames[0] is None:
                self.paused_frames[0] = idx
            self.paused_frames[1] = idx
            return False
        # OpenCV 5의 phaseCorrelate는 입력 배열에 창 함수를 곱해 덮어쓰므로 기준 화면은 복사본을 넘긴다
        (dx, dy), resp = cv2.phaseCorrelate(self.ref.copy(), self._norm(frame), self.win)
        self.last = (dx, dy, resp)
        shift = float(np.hypot(dx, dy))
        limit = MOTION_PX if self.cur is None else MOTION_CLEAR_PX
        # 히터를 치우는 등 장면이 바뀌면 신뢰도만 떨어지고 이동량은 0에 가깝다 → 신뢰도가 낮아도 이동량이 작으면 정상
        moved = shift > limit or (resp < MOTION_MIN_RESP and shift > MOTION_CLEAR_PX)
        self.rows.append([idx, f"{el:.3f}", f"{dx:.2f}", f"{dy:.2f}", f"{resp:.3f}", int(moved)])
        if moved:
            if self.cur is None:
                self.cur = {"start_frame": idx, "start_s": round(el, 3), "end_frame": None, "end_s": None,
                            "max_shift_px": round(shift, 2), "min_response": round(float(resp), 3)}
                return True
            self.cur["max_shift_px"] = max(self.cur["max_shift_px"], round(shift, 2))
            self.cur["min_response"] = min(self.cur["min_response"], round(float(resp), 3))
        else:
            if not self.episodes and self.cur is None:
                self.last_ok_idx = idx
            if self.cur is not None:
                self.cur["end_frame"], self.cur["end_s"] = idx, round(el, 3)
                self.episodes.append(self.cur)
                self.cur = None
        return False

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "elapsed_s", "dx_px", "dy_px", "response", "moved"])
            w.writerows(self.rows)

    def finish(self):
        if self.cur is not None:
            self.episodes.append(self.cur)  # 끝까지 원위치로 안 돌아옴 (end_frame=None)
            self.cur = None

    @property
    def first_moved(self):
        eps = self.episodes + ([self.cur] if self.cur else [])
        return eps[0] if eps else None

    def summary(self, total_frames):
        first = self.first_moved
        return {
            "checked": self.ref is not None,
            "ref_frames": list(self.ref_range) if self.ref is not None else None,
            "check_every_frames": MOTION_CHECK_EVERY,
            "threshold_px": MOTION_PX,
            "min_response": MOTION_MIN_RESP,
            # EVENT 전후로 검사를 쉰 구간 (이 구간의 움직임은 감지 못 함)
            "paused_frames": self.paused_frames if self.paused_frames[0] is not None else None,
            "episodes": self.episodes,
            "first_moved_frame": first["start_frame"] if first else None,
            # 처음부터 이 프레임까지는 기준 화면과 정렬됨 (움직임 직전 마지막 정상 확인 프레임)
            "aligned_until_frame": (self.last_ok_idx if self.last_ok_idx is not None else -1) if first
            else total_frames - 1,
            "log": "motion.csv",
        }


# ---------------------------------------------------------------- 미리보기

_fonts = {}


def font(px, bold=False):
    key = (px, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, px)
    return _fonts[key]


def image_8bit(frame):
    """화면 표시 전용 8-bit 영상. 저장에는 쓰지 않는다."""
    if frame.dtype == np.uint16:
        lo, hi = np.percentile(frame, (1, 99))
        img = np.clip((frame.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
    else:
        img = frame if frame.dtype == np.uint8 else np.zeros((H, W), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (W * PREVIEW_SCALE, H * PREVIEW_SCALE), interpolation=cv2.INTER_NEAREST)


def render(frame, lines, border=None):
    """lines: (text, px, color[, opts]) 목록. opts: 'bold', 'dot'(빨간 점), 'box'(배경 강조).
    text 대신 None이면 빈 줄(px 높이). 글자가 패널보다 넓으면 자동으로 줄인다."""
    img = image_8bit(frame)
    if border is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), border, 8)
    panel = Image.new("RGB", (PANEL_W, img.shape[0]), (0, 0, 0))
    d = ImageDraw.Draw(panel)
    y = 12
    for item in lines:
        text, px, color = item[:3]
        opts = item[3] if len(item) > 3 else ""
        if text is None:
            y += px
            continue
        if not text:
            continue
        bold = "bold" in opts or px >= 36
        x = 20
        max_w = PANEL_W - 40 - (px if "dot" in opts else 0)
        f = font(px, bold)
        while f.getlength(text) > max_w and px > 12:
            px = int(px * 0.9)
            f = font(px, bold)
        if "box" in opts:
            d.rectangle([8, y - 2, PANEL_W - 8, y + px + 10], fill=(40, 40, 40), outline=color, width=3)
        if "dot" in opts:
            r = px // 3
            d.ellipse([x, y + px // 2 - r + 4, x + 2 * r, y + px // 2 + r + 4], fill=color)
            x += 2 * r + 14
        d.text((x, y), text, font=f, fill=color)
        y += int(px * 1.45) + (8 if "box" in opts else 0)
        if y > img.shape[0] - 10:
            break
    return np.hstack([img, np.asarray(panel)])


def show(img):
    cv2.imshow(WINDOW, img)


def mmss(sec):
    sec = max(0, int(math.ceil(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def fmt_min(m):
    return f"{m:g}분"


# ---------------------------------------------------------------- 입력

def session_root(scene_id):
    return PRACTICE_DIR if scene_id.startswith(PRACTICE_PREFIX) else SESSIONS_DIR


def validate_scene_id(scene_id):
    """문제없으면 None, 아니면 이유 문자열."""
    if not re.fullmatch(r"[A-Z0-9_-]{1,32}", scene_id):
        return "영문/숫자/_/- 만 사용 가능합니다."
    d = session_root(scene_id) / scene_id
    if d.exists() and any(d.iterdir()):
        return f"{d} 에 이미 데이터가 있습니다. 덮어쓰지 않으니 다른 코드를 입력하세요."
    return None


def load_shot_list():
    """촬영순서.csv -> [{order, code, type, height, camera, target, light, event, max_min, memo}]."""
    try:
        with open(SHOT_LIST_CSV, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        print(f"  촬영순서.csv를 읽지 못했습니다 ({e}). 코드를 직접 입력하세요.")
        return []
    shots = []
    for r in rows:
        order = (r.get("순서") or "").strip()
        if not order.isdigit() or int(order) in AUTO_SKIP_ORDERS:
            continue
        g = lambda k: (r.get(k) or "").strip()
        try:
            max_min = float(g("최대(분)"))
        except ValueError:
            max_min = None
        shots.append({"order": int(order), "code": g("코드").upper(), "type": g("유형"), "height": g("높이"),
                      "camera": g("카메라"), "target": g("대상·열원"), "light": g("조명"),
                      "event": g("이벤트(S)") == "예", "max_min": max_min, "memo": g("계획 메모")})
    return shots


def is_done(code):
    d = session_root(code) / code
    return d.exists() and any(d.iterdir())


def next_shot(shots, after_code=None):
    """after_code 다음부터 아직 안 찍은 첫 구도. after_code가 없으면 처음부터. 끝까지 없으면 앞에서 다시 찾는다."""
    codes = [s["code"] for s in shots]
    start = codes.index(after_code) + 1 if after_code in codes else 0
    for s in shots[start:] + shots[:start]:
        if s["code"] != after_code and not is_done(s["code"]):
            return s
    return None


def shot_defaults(shot, scene_id):
    """(최대 녹화 분, EVENT 신호 사용 여부). 촬영순서에 없는 코드는 기본 5분, 이동 촬영이 아니면 EVENT 사용."""
    if shot is not None:
        return (shot["max_min"] or DEFAULT_MAX_MIN), shot["event"]
    return DEFAULT_MAX_MIN, not is_moving(scene_id)


def ask_scene_id(initial=None):
    if initial:
        scene_id = initial.strip().upper()
        err = validate_scene_id(scene_id)
        if err is None:
            return scene_id
        print(f"  {scene_id}: {err}")
    while True:
        scene_id = input("구도 코드 입력 (예: T01, 연습은 TEST01): ").strip().upper()
        err = validate_scene_id(scene_id)
        if err is None:
            return scene_id
        print(f"  {err}")


def ask_max_minutes(default):
    """콘솔에서 최대 녹화 시간(분)을 묻는다. Enter·입력 불가(백그라운드 실행)면 기본값."""
    lo, hi = MAX_MIN_LIMIT
    while True:
        try:
            s = input(f"최대 녹화 시간(분, 기본 {default:g}): ").strip()
        except EOFError:
            print(f"(입력 없음 → {default:g}분)")
            return default
        if not s:
            return default
        try:
            v = float(s)
        except ValueError:
            print("  숫자로 입력하세요 (예: 15)")
            continue
        if lo <= v <= hi:
            return v
        print(f"  {lo:g}~{hi:g}분 사이로 입력하세요.")


def ask_continue_nonraw(frame, raw_desc):
    """raw가 아닐 때 창에서 Y/N으로 묻는다. 창을 닫으면 N."""
    lines = [("경고: RAW 아님!", 52, RED, "bold"),
             ("이대로 녹화하면 측정 데이터로 못 씁니다", 24, RED),
             (raw_desc, 18, GRAY),
             ("기대값: uint16, (256, 320), max>255", 18, GRAY),
             (None, 20, None),
             ("케이블을 다시 꽂고 다시 실행하세요", 26, YELLOW),
             (None, 20, None),
             ("N : 중단 (권장)", 34, WHITE, "box"),
             ("Y : 그래도 계속", 26, GRAY)]
    show(render(frame, lines, border=RED))
    print("창에서 Y(계속) / N(중단)을 누르세요.", flush=True)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("y"), ord("Y")):
            return True
        if key in (ord("n"), ord("N"), ord("q"), ord("Q"), 27):
            return False
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False


def iso(t):
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


def disk_free(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


# ---------------------------------------------------------------- 실행

def keep_awake(on):
    """실행 중에는 Windows 절전·화면 꺼짐을 막는다 (전원 설정: 배터리 3분, 충전 10분 뒤 절전).
    덮개를 닫을 때의 절전은 막지 못한다."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    keep_awake(True)
    try:
        _main(argv)
    finally:
        keep_awake(False)


def _main(argv):
    print(f"촬영 폴더: {SHOOT_DIR}", flush=True)
    print(f"{PORT} 연결 중...", flush=True)
    cam = Boson(port=PORT)
    cam.set_ffc_manual()
    mode = cam.get_ffc_mode()
    mode_name = MODE_NAME.get(mode, f"UNKNOWN({mode})")
    mode_ok = mode == FLR_BOSON_MANUAL_FFC
    if mode_ok:
        print(f"FFC 모드: {mode_name}  (확인됨)", flush=True)
    else:
        banner([f"FFC 모드가 MANUAL이 아닙니다: {mode_name}",
                "녹화 중 자동 FFC로 영상이 멈출 수 있습니다."])

    try:
        shots = load_shot_list()
        by_code = {s["code"]: s for s in shots}
        if argv:
            scene_id = ask_scene_id(argv[0])
        else:
            nxt = next_shot(shots)
            if nxt is not None:
                scene_id = nxt["code"]
                print(f"다음 구도 자동 선택: {scene_id} (순서 {nxt['order']})", flush=True)
            else:
                print("촬영순서의 구도를 모두 찍었거나 순서 파일이 없습니다.")
                scene_id = ask_scene_id()
        default_min, _ = shot_defaults(by_code.get(scene_id), scene_id)
        max_min = ask_max_minutes(default_min)
        print(f"최대 녹화 시간: {max_min:g}분 (예상 최대 {est_bytes(max_min * 60) / 1e9:.2f} GB)", flush=True)

        frame = cam.grab()
        raw_ok, raw_desc = check_raw(frame)
        raw_overridden = False
        if raw_ok:
            print(f"RAW 확인: {raw_desc}", flush=True)
        else:
            banner(["경고: 16-bit RAW 프레임이 아닙니다!",
                    raw_desc,
                    f"기대값: dtype=uint16, shape=({H}, {W}), max>255",
                    "이대로 녹화하면 측정용 데이터로 쓸 수 없습니다."])
            if frame is None or not ask_continue_nonraw(frame, raw_desc):
                print("중단합니다.", flush=True)
                return
            raw_overridden = True

        record(cam, scene_id, shots, max_min, mode_name, mode_ok, raw_ok, raw_desc, raw_overridden)
    finally:
        cam.close()
        cv2.destroyAllWindows()


def shot_info_lines(shot):
    """촬영순서.csv의 배치 정보를 READY 화면용 줄로."""
    if shot is None:
        return [("(촬영순서에 없는 코드)", 20, GRAY)]
    out = []
    first = "  ".join(x for x in (shot["height"], shot["camera"]) if x)
    if first:
        out.append((first, 22, WHITE))
    if shot["target"]:
        out.append((shot["target"], 22, ORANGE))
    if shot["light"]:
        out.append((f"조명 {shot['light']}", 18, WHITE))
    if shot["memo"]:
        out.append((f"메모: {shot['memo']}", 18, GRAY))
    return out


def guide_lines(state, tp, *, scene_id, practice, moving, event_on, max_sec, mode_name, mode_ok, t_ffc, t_rec0,
                n, dup, event_signal, event_mark, event_end, q_pending, last_event_text, shot=None, shot_pos="",
                motion_now=False, motion_first=None, motion_ready=False, disk_warn=None):
    """상태별 '지금 할 일' 안내. 반환: (lines, border)"""
    head = (f"{scene_id}" + ("  [연습]" if practice else "") + ("  [이동]" if moving else "")
            + f"   FFC:{mode_name}")
    status = (head, 18, GREEN if mode_ok else RED)
    disk = [(disk_warn, 18, RED)] if disk_warn else []

    if state == "IDLE":
        if moving:
            checks = [("□ 리그 머리에 고정  □ 케이블 몸에 고정", 20, WHITE),
                      ("□ 노트북 백팩  □ 점등", 20, WHITE)]
            start = "Enter → FFC 2초 후 녹화 → 바로 출발"
        elif event_on:
            checks = [("□ 대상 가열 끝  □ 히터 위치 확인", 20, WHITE),
                      ("□ 사람·노트북 프레임 밖  □ 사진 1장", 20, WHITE)]
            start = f"Enter → 녹화 → {EVENT_SIGNAL_SEC:.0f}초 비프에 S → 히터 치우고 나와서 S"
        else:
            checks = [("□ 배치가 위와 같은지  □ 열원 위치 확인", 20, WHITE),
                      ("□ 사람·노트북 프레임 밖  □ 사진 1장", 20, WHITE)]
            start = "Enter → FFC 후 녹화 (자동 종료)"
        return [(f"{scene_id}  {shot_pos}".rstrip(), 44, YELLOW),
                *shot_info_lines(shot),
                (f"최대 {fmt_min(max_sec / 60)}" + (f" · {EVENT_SIGNAL_SEC:.0f}초에 EVENT" if event_on else ""),
                 22, CYAN),
                (None, 6, None),
                *checks,
                *disk,
                (None, 6, None),
                ("이 창을 클릭 → Enter", 32, YELLOW, "box"),
                (start, 18, GRAY),
                ("+/- : 최대 시간 1분 조정   N : 건너뛰기   Q : 종료", 17, GRAY),
                status], None

    if state == "FFC":
        return [("FFC 중...", 52, YELLOW),
                (f"{tp - t_ffc:.1f}초" + ("  (영상 정지)" if dup else ""), 28, YELLOW),
                (None, 20, None),
                ("카메라 건드리지 마세요" if not moving else "아직 출발하지 마세요", 30, WHITE, "box"),
                ("곧 녹화가 시작됩니다", 22, GRAY),
                status], None

    el = tp - t_rec0
    blink = int(tp * 2) % 2 == 0
    lines = [("REC  " + mmss(el), 52, RED, "dot" if blink else ""),
             (f"frame {n - 1}   자동 종료까지 {mmss(max_sec - el)}", 18, GRAY)]
    if motion_now:
        lines += [("카메라 움직임!", 40, RED if blink else YELLOW, "box"),
                  ("삼각대·케이블 건드리지 마세요 · 재촬영 권장", 18, RED)]
    elif motion_first is not None:
        lines.append((f"움직임 있었음: frame {motion_first['start_frame']}부터 (재촬영 권장)", 18, RED))
    elif not moving and not motion_ready:
        lines.append(("움직임 기준 화면 잡는 중", 16, GRAY))

    if q_pending:
        lines += [(None, 16, None),
                  ("정말 중단할까요?", 38, RED),
                  ("Q 한 번 더 = 녹화 종료", 30, RED, "box"),
                  ("(3초 지나면 취소)", 20, GRAY)]
    elif moving:
        lines += [(None, 12, None),
                  ("동선 출발", 40, GREEN),
                  ("멈출 곳에서 좌우를 천천히 3~4초", 22, WHITE),
                  ("끝나면 노트북 꺼내 Q 두 번", 26, WHITE, "box"),
                  (f"최대 {mmss(max_sec)} 후 자동 종료", 18, GRAY)]
    elif event_on and event_mark is None and event_signal is None:
        remain = EVENT_SIGNAL_SEC - el
        if remain > COUNTDOWN_BEEPS:
            lines += [(None, 12, None),
                      ("EVENT까지", 26, ORANGE),
                      (f"{math.ceil(remain)}초", 64, ORANGE),
                      ("프레임 밖 노트북 옆에서 대기", 26, WHITE, "box"),
                      ("긴 비프 → S 먼저 → 히터 치우기", 20, GRAY)]
        else:
            lines += [(None, 12, None),
                      ("EVENT 준비!", 34, RED),
                      (f"{math.ceil(remain)}", 80, RED),
                      ("긴 비프에 S 먼저 누르고 → 히터로", 26, WHITE, "box")]
    elif event_on and event_mark is None:
        lines += [(None, 12, None),
                  ("지금 S!", 56, RED if blink else YELLOW),
                  ("S 누르고 → 히터 치우러 가기", 32, WHITE, "box"),
                  (f"신호 후 {el - event_signal[1]:.0f}초 지남", 20, GRAY)]
    elif event_mark is not None and event_end is None:
        lines += [(None, 12, None),
                  (f"EVENT 시작 (frame {event_mark[0]})", 22, GREEN),
                  ("히터 프레임 밖으로", 40, YELLOW),
                  ("치우고 나와서 S 한 번 더", 30, WHITE, "box"),
                  (f"S 후 {el - event_mark[1]:.0f}초 지남", 20, GRAY)]
    else:
        if event_end is not None:
            lines += [(None, 8, None), (f"EVENT {event_mark[0]}~{event_end[0]} 기록됨", 22, GREEN),
                      ("식는 중", 40, GREEN)]
        else:
            lines += [(None, 8, None), ("녹화 중", 40, GREEN)]
        lines += [("남은 시간", 22, GRAY),
                  (mmss(max_sec - el), 56, WHITE),
                  ("프레임 밖에서 대기", 28, WHITE, "box")]
        if event_end is not None:
            lines.append(("히터는 프레임 밖에서 다음 대상 가열", 18, GRAY))
    if last_event_text:
        lines.append((last_event_text, 17, CYAN))
    lines += [(("S EVENT   Q 두 번: 중단" if not moving else "Q 두 번: 중단"), 17, GRAY), status]
    return lines, RED


def end_screen(frame, scene_id, total, events, end_reason, raw_ok, event_on, nxt=None, camera_motion=None):
    """녹화 종료 요약. 아무 키나 누르거나 30초 지나면 닫힌다."""
    reason = {"auto_max_duration": "최대 시간 자동 종료", "user_q": "Q로 종료",
              "ctrl_c": "Ctrl+C"}.get(end_reason, end_reason)
    has_s = any(e["type"] == "event_start" for e in events)
    has_end = any(e["type"] == "event_end" for e in events)
    has_signal = any(e["type"] == "event_signal" for e in events)
    if has_s and has_end:
        ev = ("EVENT 시작·끝 S 기록 있음", GREEN)
    elif has_s:
        ev = ("두 번째 S 없음 → hot_ref 없음 (로그에 메모)", YELLOW)
    elif has_signal:
        ev = ("S 기록 없음 → 20초 신호를 EVENT로 저장", YELLOW)
    elif event_on:
        ev = ("EVENT 신호 전에 종료", YELLOW)
    else:
        ev = ("", GRAY)
    lines = [("녹화 끝", 52, GREEN),
             (f"{scene_id}   {total}프레임 저장", 26, WHITE),
             (f"종료: {reason}", 22, GRAY),
             (ev[0], 22, ev[1]),
             (None, 12, None),
             ("다음 구도 준비", 30, WHITE, "box"),
             ((f"다음: {nxt['code']}  {nxt['height']}  {nxt['camera']}" if nxt else "촬영순서 모두 완료"),
              20, YELLOW),
             ((nxt["target"] if nxt else ""), 20, ORANGE),
             ("현장로그에 S 초·표면온도·사진 번호 적기", 20, GRAY),
             ("촬영시작 다시 실행 → 다음 구도 자동 선택", 18, GRAY),
             (None, 10, None),
             ("아무 키나 누르면 창이 닫힙니다", 20, GRAY)]
    if not raw_ok:
        lines.insert(3, ("RAW 아님 (경고 무시하고 녹화함)", 22, RED))
    border = GREEN
    if camera_motion and camera_motion["first_moved_frame"] is not None:
        lines.insert(1, (f"카메라 움직임: frame {camera_motion['first_moved_frame']}부터 → 고정 후 재촬영 권장",
                         20, RED, "box"))
        border = RED
    elif camera_motion and camera_motion["checked"]:
        lines.insert(1, ("카메라 움직임 없음", 20, GREEN))
    show(render(frame, lines, border=border))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 30:
        if cv2.waitKey(100) != -1 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break


def record(cam, scene_id, shots, max_min, mode_name, mode_ok, raw_ok, raw_desc, raw_overridden):
    by_code = {s["code"]: s for s in shots}

    def shot_pos(code):
        if code not in by_code:
            return ""
        done = sum(is_done(s["code"]) for s in shots)
        return f"(순서 {by_code[code]['order']} · 완료 {done}/{len(shots)})"

    def disk_warning(sec):
        need = est_bytes(sec) * 1.1
        free = disk_free(SHOOT_DIR)
        return None if free > need else f"디스크 부족: 필요 {need / 1e9:.1f}GB / 남음 {free / 1e9:.1f}GB"

    pos_text = shot_pos(scene_id)
    _, event_on = shot_defaults(by_code.get(scene_id), scene_id)
    max_sec = max_min * 60
    disk_warn = disk_warning(max_sec)
    if disk_warn:
        banner([disk_warn])
    session_dir = None
    state = "IDLE"            # IDLE -> FFC -> REC
    prev = None
    n = 0                     # 저장한 프레임 수 (= 다음 프레임 번호)
    t_ffc = t_rec0 = None
    ffc = {"command_unix_time": None, "confirmed": False, "confirm_after_s": None,
           "freeze_frames": 0, "freeze_seen": False, "timed_out": False}
    event_signal = None       # (frame, time_s)
    event_mark = None         # 첫 S (frame, 녹화 기준 초) = 사람이 들어가기 시작
    event_end = None          # 두 번째 S = 히터 치우고 프레임 밖으로 나옴
    beeped = set()            # 이미 울린 카운트다운 초
    events = []
    frozen_count = 0
    end_reason = None
    q_until = 0.0             # 녹화 중 첫 Q 이후 두 번째 Q를 받는 마감 시각
    last_event_text = ""
    rec_start_unix = rec_end_unix = None
    files = []
    practice = scene_id.startswith(PRACTICE_PREFIX)
    moving = is_moving(scene_id)
    motion = MotionChecker()

    try:
        while True:
            frame = cam.grab()
            tp = time.perf_counter()
            tu = time.time()
            if frame is None:
                continue
            dup = prev is not None and np.array_equal(frame, prev)
            prev = frame.copy()

            if state == "FFC":
                el = tp - t_ffc
                if dup:
                    ffc["freeze_frames"] += 1
                    ffc["freeze_seen"] = True
                if not ffc["confirmed"]:
                    if el >= FFC_MIN_WAIT_SEC and not dup:
                        ffc["confirmed"] = True
                        ffc["confirm_after_s"] = round(el, 3)
                        print(f"FFC 완료 확인 (+{el:.2f}s, 정지 프레임 {ffc['freeze_frames']}개)", flush=True)
                        if not ffc["freeze_seen"]:
                            print("  주의: FFC 정지 프레임이 관찰되지 않았습니다. FFC가 실행되지 않았을 수 있습니다.")
                    elif el >= FFC_TIMEOUT_SEC:
                        ffc["confirmed"] = True
                        ffc["timed_out"] = True
                        ffc["confirm_after_s"] = round(el, 3)
                        banner([f"FFC 후 {FFC_TIMEOUT_SEC:.0f}초가 지나도 영상 정지가 끝나지 않았습니다.",
                                "카메라 상태를 확인하세요. 녹화는 계속 진행합니다."])
                if ffc["confirmed"] and el >= FFC_TO_REC_SEC:
                    state = "REC"
                    t_rec0 = tp
                    rec_start_unix = tu
                    session_dir = session_root(scene_id) / scene_id
                    frames_dir = session_dir / "frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    print(f"세션 폴더: {session_dir}", flush=True)
                    ts_f = open(session_dir / "timestamps.csv", "w", newline="", encoding="utf-8")
                    fz_f = open(session_dir / "frozen_frames.csv", "w", newline="", encoding="utf-8")
                    files = [ts_f, fz_f]
                    ts_w = csv.writer(ts_f)
                    fz_w = csv.writer(fz_f)
                    ts_w.writerow(["frame", "elapsed_s", "unix_time"])
                    fz_w.writerow(["frame", "elapsed_s", "same_as_frame"])
                    print(f"● 녹화 시작 (최대 {fmt_min(max_sec / 60)}"
                          + (f", {EVENT_SIGNAL_SEC:.0f}초에 EVENT" if event_on else "") + ")", flush=True)

            if state == "REC":
                el = tp - t_rec0
                tifffile.imwrite(frames_dir / f"{n:06d}.tif", frame)
                ts_w.writerow([n, f"{el:.4f}", f"{tu:.4f}"])
                if dup:
                    frozen_count += 1
                    fz_w.writerow([n, f"{el:.4f}", n - 1])
                if not moving:
                    # EVENT 신호 5초 전 ~ 두 번째 S 후 3초는 사람이 프레임에 들어오므로 쉰다
                    # (두 번째 S를 안 누르면 첫 S 또는 신호 후 60초까지만)
                    paused = False
                    if event_end is not None:
                        paused = el < event_end[1] + EVENT_MOTION_PAUSE[1]
                    elif event_mark is not None:
                        paused = el < event_mark[1] + EVENT_PAUSE_MAX_SEC
                    elif event_on and event_signal is not None:
                        paused = el < event_signal[1] + EVENT_PAUSE_MAX_SEC
                    elif event_on:
                        paused = el >= EVENT_SIGNAL_SEC - EVENT_MOTION_PAUSE[0]
                    if motion.update(frame, n, el, paused=paused):
                        beep_motion()
                        dx, dy, resp = motion.last
                        print(f"⚠ 카메라 움직임 감지 (frame {n}, {el:.1f}s, dx {dx:+.1f} dy {dy:+.1f} "
                              f"신뢰 {resp:.2f})", flush=True)
                if event_on and event_signal is None:
                    k = math.ceil(EVENT_SIGNAL_SEC - el)
                    if 1 <= k <= COUNTDOWN_BEEPS and k not in beeped and event_mark is None:
                        beeped.add(k)
                        beep_countdown()
                    if el >= EVENT_SIGNAL_SEC:
                        if event_mark is None:
                            beep_signal()
                        event_signal = (n, round(el, 4))
                        events.append({"type": "event_signal", "frame": n, "time_s": round(el, 4),
                                       "unix_time": round(tu, 4)})
                        print(f"♪ EVENT 신호 (frame {n}, {el:.2f}s)", flush=True)
                n += 1
                rec_end_unix = tu
                if el >= max_sec:
                    end_reason = "auto_max_duration"
                    print(f"최대 녹화 시간 {fmt_min(max_sec / 60)} 도달, 자동 종료", flush=True)
                    beep_end()
                    break

            lines, border = guide_lines(
                state, tp, scene_id=scene_id, practice=practice, moving=moving, event_on=event_on,
                max_sec=max_sec, mode_name=mode_name, mode_ok=mode_ok, t_ffc=t_ffc, t_rec0=t_rec0, n=n, dup=dup,
                event_signal=event_signal, event_mark=event_mark, event_end=event_end, q_pending=tp < q_until,
                last_event_text=last_event_text, shot=by_code.get(scene_id), shot_pos=pos_text,
                motion_now=motion.cur is not None, motion_first=motion.first_moved,
                motion_ready=motion.ref is not None, disk_warn=disk_warn)
            if not raw_ok:
                lines.insert(1, ("RAW 아님!", 22, RED))
            show(render(frame, lines, border))

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                # 창을 닫으면 확인 없이 종료
                if state == "REC":
                    end_reason = "window_closed"
                print("창 닫힘, 종료", flush=True)
                break
            if key in KEY_ENTER and state == "IDLE":
                cam.do_ffc()
                t_ffc = time.perf_counter()
                ffc["command_unix_time"] = round(time.time(), 4)
                state = "FFC"
                print("FFC 실행", flush=True)
            elif key in (ord("+"), ord("="), ord("-"), ord("_")) and state == "IDLE":
                step = 1 if key in (ord("+"), ord("=")) else -1
                lo, hi = MAX_MIN_LIMIT
                max_sec = min(hi, max(lo, max_sec / 60 + step)) * 60
                disk_warn = disk_warning(max_sec)
                print(f"최대 녹화 시간: {fmt_min(max_sec / 60)}", flush=True)
            elif key in (ord("n"), ord("N")) and state == "IDLE":
                nxt = next_shot(shots, after_code=scene_id)
                if nxt is None:
                    print("건너뛸 다음 구도가 없습니다.", flush=True)
                else:
                    scene_id = nxt["code"]
                    pos_text = shot_pos(scene_id)
                    practice = scene_id.startswith(PRACTICE_PREFIX)
                    moving = is_moving(scene_id)
                    m, event_on = shot_defaults(nxt, scene_id)
                    max_sec = m * 60
                    disk_warn = disk_warning(max_sec)
                    print(f"건너뜀 -> {scene_id} (최대 {fmt_min(m)}, EVENT {'켬' if event_on else '끔'})",
                          flush=True)
            elif key in (ord("s"), ord("S")) and state == "REC" and not moving:
                el = time.perf_counter() - t_rec0
                if event_mark is None:
                    etype, label = "event_start", "EVENT 시작"
                    event_mark = (n - 1, el)
                elif event_end is None:
                    etype, label = "event_end", "EVENT 끝"
                    event_end = (n - 1, el)
                else:
                    etype, label = "event_extra", "추가 S"
                events.append({"type": etype, "frame": n - 1, "time_s": round(el, 4),
                               "unix_time": round(time.time(), 4)})
                last_event_text = f"{label}: frame {n - 1} ({el:.1f}s)"
                print(f"기록: {etype} @ frame {n - 1} ({el:.1f}s)", flush=True)
            elif key in (ord("q"), ord("Q")):
                if state != "REC":
                    print("녹화 전 종료", flush=True)
                    break
                now = time.perf_counter()
                if now < q_until:
                    end_reason = "user_q"
                    print("Q 두 번, 녹화 종료", flush=True)
                    break
                q_until = now + Q_CONFIRM_SEC
    except KeyboardInterrupt:
        end_reason = "ctrl_c"
        print("Ctrl+C, 녹화 종료")
    except Exception as e:
        end_reason = f"error: {type(e).__name__}: {e}"
        print(f"오류로 녹화 종료: {e!r}")
        raise
    finally:
        for f in files:
            f.close()
        camera_motion = None
        if state == "REC":
            if not moving:
                motion.finish()
                motion.write_csv(session_dir / "motion.csv")
                camera_motion = motion.summary(n)
            write_meta(session_dir, scene_id, by_code.get(scene_id), moving, event_on, max_sec, mode_name,
                       mode_ok, raw_ok, raw_desc, raw_overridden, ffc, rec_start_unix, rec_end_unix, n,
                       event_signal, events, frozen_count, end_reason, camera_motion)
    if state == "REC" and prev is not None and end_reason in ("auto_max_duration", "user_q"):
        nxt = next_shot(shots, after_code=scene_id)
        end_screen(prev, scene_id, n, events, end_reason, raw_ok, event_on, nxt, camera_motion)


def compute_clear(times, event_frame):
    """라벨 구간 계산.

    times: 프레임 번호 순서의 elapsed_s 목록 (frame i -> times[i])
    clear:     frame 0 ~ event_frame-1 (EVENT가 없으면 녹화 끝까지)
    clear_ref: clear 끝에서 거슬러 CLEAR_REF_SEC 이내의 프레임 (약 86프레임)
    """
    total = len(times)
    if total == 0:
        return None, None
    end_f, until = (event_frame - 1, "event") if event_frame is not None else (total - 1, "end_of_recording")
    if end_f < 0:
        return None, None
    clear = {"start_frame": 0, "end_frame": end_f, "start_s": times[0], "end_s": times[end_f],
             "n_frames": end_f + 1, "until": until}
    t_lo = times[end_f] - CLEAR_REF_SEC
    ref_start = next(i for i in range(end_f + 1) if times[i] > t_lo)
    clear_ref = {"start_frame": ref_start, "end_frame": end_f,
                 "start_s": times[ref_start], "end_s": times[end_f],
                 "n_frames": end_f - ref_start + 1, "until": until,
                 "target_s": CLEAR_REF_SEC,
                 # clear가 10초보다 짧아서 clear_ref가 덜 채워졌으면 true
                 "shorter_than_target": times[end_f] - times[0] < CLEAR_REF_SEC}
    return clear, clear_ref


def compute_hot_ref(times, end_frame):
    """두 번째 S(히터 치우고 나옴) + HOT_REF_DELAY_SEC 부터 HOT_REF_SEC 길이. 없거나 녹화가 짧으면 None."""
    if end_frame is None or end_frame >= len(times):
        return None
    t0 = times[end_frame] + HOT_REF_DELAY_SEC
    idx = [i for i in range(end_frame + 1, len(times)) if t0 <= times[i] <= t0 + HOT_REF_SEC]
    if not idx:
        return None
    return {"start_frame": idx[0], "end_frame": idx[-1], "start_s": times[idx[0]], "end_s": times[idx[-1]],
            "n_frames": len(idx), "target_s": HOT_REF_SEC,
            "shorter_than_target": times[idx[-1]] - times[idx[0]] < HOT_REF_SEC - 0.5}


def write_meta(session_dir, scene_id, shot, moving, event_on, max_sec, mode_name, mode_ok, raw_ok, raw_desc,
               raw_overridden, ffc, rec_start_unix, rec_end_unix, total_frames, event_signal, events,
               frozen_count, end_reason, camera_motion):
    # event_frame: 첫 S(사람이 들어가기 시작)가 있으면 그것, 없으면 신호 프레임
    start = next((e for e in events if e["type"] == "event_start"), None)
    end = next((e for e in events if e["type"] == "event_end"), None)
    if start:
        event_frame, event_time, event_src = start["frame"], start["time_s"], "manual_S"
    elif event_signal:
        event_frame, event_time, event_src = event_signal[0], event_signal[1], "signal"
    else:
        event_frame = event_time = event_src = None

    with open(session_dir / "timestamps.csv", encoding="utf-8") as f:
        times = [float(r["elapsed_s"]) for r in csv.DictReader(f)]
    clear, clear_ref = compute_clear(times, event_frame)
    hot_ref = compute_hot_ref(times, end["frame"] if end else None)

    duration = (rec_end_unix - rec_start_unix) if rec_start_unix and rec_end_unix else 0.0
    meta = {
        "scene_id": scene_id,
        "practice": scene_id.startswith(PRACTICE_PREFIX),
        "shot": shot,             # 촬영순서.csv의 해당 행 (없으면 null)
        "moving": moving,
        "event_enabled": event_on,
        "total_frames": total_frames,
        "event_frame": event_frame,
        "event_time_s": event_time,
        "event_source": event_src,
        # 두 번째 S = 히터 치우고 프레임 밖으로 나온 시점 (없으면 null)
        "event_end_frame": end["frame"] if end else None,
        "event_end_time_s": end["time_s"] if end else None,
        "event_signal_sec": EVENT_SIGNAL_SEC if event_on else None,
        "event_signal_frame": event_signal[0] if event_signal else None,
        "event_signal_time_s": event_signal[1] if event_signal else None,
        # 라벨 구간: 녹화 시작 ~ 첫 S 직전 (히터는 프레임에 있을 수 있음, 사람 없음). clear_ref는 그 마지막 10초
        "clear": clear,
        "clear_ref": clear_ref,
        # 두 번째 S + 2초부터 10초: 히터·사람이 빠진 직후 (대상이 가장 뜨거운 깨끗한 구간)
        "hot_ref": hot_ref,
        "events": events,
        "recording": {
            "start": iso(rec_start_unix) if rec_start_unix else None,
            "end": iso(rec_end_unix) if rec_end_unix else None,
            "start_unix_time": rec_start_unix,
            "max_s": max_sec,
            "duration_s": round(duration, 3),
            "measured_fps": round((total_frames - 1) / duration, 3) if duration > 0 else None,
            "end_reason": end_reason,
            "frozen_frame_count": frozen_count,
        },
        # 고정 구도만 검사. 이동 촬영은 카메라가 움직이는 게 정상이라 null
        "camera_motion": camera_motion,
        "ffc": ffc,
        "camera": {
            "model": "FLIR Boson 320 (non-radiometric, 9Hz)",
            "port": PORT,
            "ffc_mode_at_start": mode_name,
            "ffc_mode_manual_confirmed": mode_ok,
            "raw_check_ok": raw_ok,
            "raw_check": raw_desc,
            "raw_warning_overridden": raw_overridden,
        },
        "frame_format": {"dtype": "uint16", "shape": [H, W], "file": "frames/NNNNNN.tif"},
    }
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {session_dir}")
    print(f"  총 {total_frames}프레임, {duration:.1f}s, 정지 프레임 {frozen_count}개"
          + (f", event_frame={event_frame} ({event_src})" if event_on else ""), flush=True)
    if clear and event_on:
        print(f"  clear {clear['start_frame']}~{clear['end_frame']} ({clear['until']}), "
              f"clear_ref {clear_ref['start_frame']}~{clear_ref['end_frame']} ({clear_ref['n_frames']}프레임)"
              + (f", hot_ref {hot_ref['start_frame']}~{hot_ref['end_frame']}" if hot_ref else ", hot_ref 없음"))
    if camera_motion and camera_motion["first_moved_frame"] is not None:
        print(f"  ⚠ 카메라 움직임: frame {camera_motion['first_moved_frame']}부터 "
              f"(정렬 정상 ~frame {camera_motion['aligned_until_frame']}), 구간 {len(camera_motion['episodes'])}개")


if __name__ == "__main__":
    sys.exit(main())
