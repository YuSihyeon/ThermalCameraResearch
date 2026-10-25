"""15분 녹화 부하 테스트: 가짜 카메라 + 가짜 시계로 약 7,700프레임을 실제 녹화 루프에 통과시킨다.

실행:  py -3.12 tests/test_load_15min.py   (카메라 불필요, 임시 폴더에 쓰고 끝나면 지운다)
확인:  frames 용량, 프레임당 처리 시간(한도 116.7 ms), 메모리(RSS) 증가 여부, EVENT·hot_ref 기록
"""
import builtins
import json
import shutil
import sys
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import record_session as rs  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="boson_load_"))
rs.SESSIONS_DIR = TMP / "sessions"
rs.PRACTICE_DIR = TMP / "practice"
rs.SHOT_LIST_CSV = TMP / "없음.csv"   # 순서표 없음 → TEST 코드는 EVENT 사용

# 실내 장면 비슷한 합성 기준 프레임 (raw 카운트 범위 22,000~23,500)
yy, xx = np.mgrid[0:rs.H, 0:rs.W]
base = (22500 + 300 * yy / rs.H + 100 * np.sin(xx / 25)).astype(np.float64)
base[60:200, 120:200] += 400        # 문
base[180:240, 30:110] += 800        # 따뜻한 물체
base = base.astype(np.uint16)
rng = np.random.default_rng(0)
proc = psutil.Process()


class Clock:
    t = 1000.0

    def perf_counter(self):
        return self.t

    def time(self):
        return 1.79e9 + self.t

    def sleep(self, s):
        _time.sleep(s)


clock = Clock()
rs.time = clock
for name in ("beep_countdown", "beep_signal", "beep_motion", "beep_end"):
    setattr(rs, name, lambda: None)
rs.end_screen = lambda *a, **k: None
rs.keep_awake = lambda on: None

stats = {"grabs": 0, "loop_ms": [], "rss": [], "last_real": None, "freeze": 0}


class FakeBoson:
    def __init__(self, port=None):
        self.prev = None

    def set_ffc_manual(self):
        pass

    def get_ffc_mode(self):
        return rs.FLR_BOSON_MANUAL_FFC

    def do_ffc(self):
        stats["freeze"] = 3

    def close(self):
        pass

    def grab(self):
        now = _time.perf_counter()
        if stats["last_real"] is not None:
            stats["loop_ms"].append((now - stats["last_real"]) * 1000)
        stats["last_real"] = now
        clock.t += 1 / 8.572
        stats["grabs"] += 1
        if stats["grabs"] % 500 == 0:
            stats["rss"].append((stats["grabs"], proc.memory_info().rss / 2**20))
        if stats["freeze"] and self.prev is not None:
            stats["freeze"] -= 1
            return self.prev.copy()
        f = (base.astype(np.int32) + rng.integers(-8, 9, base.shape)).clip(0, 65535).astype(np.uint16)
        self.prev = f
        return f


rs.Boson = FakeBoson
builtins.input = lambda prompt="": (print(prompt + "15"), "15")[1]
real_waitKey = rs.cv2.waitKey
sent = {"enter": False, "s1": False, "s2": False}


def waitKey(ms):
    real_waitKey(1)
    t = clock.t - 1000.0
    if not sent["enter"] and t > 1.0:
        sent["enter"] = True
        return 13
    if sent["enter"] and not sent["s1"] and t > 26.0:
        sent["s1"] = True
        return ord("s")
    if sent["s1"] and not sent["s2"] and t > 36.0:
        sent["s2"] = True
        return ord("s")
    return -1


rs.cv2.waitKey = waitKey
rss0 = proc.memory_info().rss / 2**20
t_real = _time.perf_counter()
try:
    rs.main(["TEST_15MIN"])
    wall = _time.perf_counter() - t_real
    d = rs.PRACTICE_DIR / "TEST_15MIN"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    size = sum(p.stat().st_size for p in (d / "frames").iterdir())
    lm = np.array(stats["loop_ms"][50:])
    print("\n===== 15분 부하 테스트 결과 =====")
    print(f"프레임: {meta['total_frames']}  (가짜 시계 {meta['recording']['duration_s']:.0f}s, "
          f"종료={meta['recording']['end_reason']})")
    print(f"frames 용량: {size / 1e9:.3f} GB")
    print(f"프레임당 처리 시간: 평균 {lm.mean():.1f} ms, 95% {np.percentile(lm, 95):.1f} ms, "
          f"최대 {lm.max():.1f} ms (한도 116.7 ms), 초과 {int((lm > 116.7).sum())}건")
    print(f"메모리(RSS): 시작 {rss0:.0f} MB → 끝 {stats['rss'][-1][1]:.0f} MB")
    print(f"EVENT {meta['event_frame']}/{meta['event_end_frame']}, hot_ref {meta['hot_ref']['start_frame']}~"
          f"{meta['hot_ref']['end_frame']}, 실제 소요 {wall:.0f}s")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
