r"""저장된 녹화에 카메라 움직임 감지(MotionChecker)를 다시 돌려 본다.

실행:  py -3.12 tests/replay_motion.py <녹화 폴더> [<검사 쉼 시작 frame> <끝 frame>] ...
예:    py -3.12 tests/replay_motion.py D:\boson\sessions\T03 132 352 D:\boson\sessions\H00
  폴더 뒤에 숫자 두 개를 주면 그 구간은 검사를 쉰다 (T 구도의 히터 치우는 구간).
2차 촬영에서 확인한 기대 결과:
  - T03 (히터 치움, 실제 움직임 없음, 쉼 132~352): 움직임 없음
  - 1차 밤 A02 / 오후 A02 (실제로 카메라가 밀림): 각각 frame 184 / 508부터 움직임
"""
import csv
import sys
from pathlib import Path

import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from record_session import MotionChecker  # noqa: E402


def parse(args):
    cases, i = [], 0
    while i < len(args):
        d, pause = Path(args[i]), None
        if i + 2 < len(args) and args[i + 1].isdigit() and args[i + 2].isdigit():
            pause, i = (int(args[i + 1]), int(args[i + 2])), i + 3
        else:
            i += 1
        cases.append((d, pause))
    return cases


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d, pause in parse(sys.argv[1:]):
        with open(d / "timestamps.csv", encoding="utf-8") as f:
            ts = [float(r["elapsed_s"]) for r in csv.DictReader(f)]
        mc = MotionChecker()
        for i, t in enumerate(ts):
            p = pause is not None and pause[0] <= i <= pause[1]
            mc.update(tifffile.imread(d / "frames" / f"{i:06d}.tif"), i, t, paused=p)
        mc.finish()
        s = mc.summary(len(ts))
        eps = [(e["start_frame"], e["max_shift_px"], e["min_response"]) for e in s["episodes"]][:5]
        print(f"{d.name}: 처음 움직임 frame {s['first_moved_frame']}, 정렬 정상 ~frame {s['aligned_until_frame']}, "
              f"구간(시작, 최대 px, 최소 신뢰도) {eps}")


if __name__ == "__main__":
    main()
