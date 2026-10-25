"""녹화 세션 점검: 저장 무결성 + EVENT(첫·두 번째 S) + 카메라 움직임.

실행:  py -3.12 check_sessions.py [코드 또는 폴더 ...]     (생략 시 sessions 폴더 전체)
       py -3.12 check_sessions.py --align H00 H01 H02 H03 (녹화 간 카메라 위치 비교)
촬영 폴더는 환경변수 BOSON_SHOOT_DIR (없으면 record_session의 기본값).

카메라 움직임은 meta.json의 camera_motion(녹화 중 감지)을 쓰고, 없으면 프레임으로 다시 계산한다.
이동 촬영(meta moving=true)은 움직임 검사를 하지 않는다.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_session import SESSIONS_DIR, H, W, MotionChecker  # noqa: E402


def check(d):
    print(f"\n==================== {d.name} ====================")
    problems = []
    mp = d / "meta.json"
    if not mp.exists():
        print("  meta.json 없음 (녹화가 비정상 종료됐을 수 있음)")
        return False
    meta = json.loads(mp.read_text(encoding="utf-8"))
    rec = meta["recording"]
    tifs = sorted((d / "frames").glob("*.tif"))
    with open(d / "timestamps.csv", encoding="utf-8") as f:
        ts = list(csv.DictReader(f))
    with open(d / "frozen_frames.csv", encoding="utf-8") as f:
        frozen = list(csv.DictReader(f))
    total = meta["total_frames"]
    moving = meta.get("moving", False)

    # ---- 저장 무결성
    names_ok = [p.stem for p in tifs] == [f"{i:06d}" for i in range(len(tifs))]
    print(f"  녹화: {rec['start']}  {rec['duration_s']:.1f}s  종료={rec['end_reason']}"
          + ("  [이동 촬영]" if moving else ""))
    print(f"  프레임: meta {total} / TIFF {len(tifs)} / timestamps {len(ts)}  번호 연속={names_ok}")
    if not (total == len(tifs) == len(ts)) or not names_ok:
        problems.append("프레임 수 불일치 또는 번호 누락")
    if len(ts) > 1:
        dt = np.diff([float(r["elapsed_s"]) for r in ts])
        print(f"  간격: 평균 {1000*dt.mean():.1f} ms ({1/dt.mean():.2f} fps), 최대 {1000*dt.max():.0f} ms")
        if dt.max() > 0.3:
            problems.append(f"프레임 간격이 {1000*dt.max():.0f} ms인 곳이 있음 (프레임 누락 가능)")
    print(f"  영상 멈춤(중복) 프레임: {len(frozen)}")
    if frozen:
        problems.append(f"중복 프레임 {len(frozen)}개 (frozen_frames.csv 확인)")
    bad = [i for i in (0, len(tifs) // 2, len(tifs) - 1) if tifs
           and (lambda a: a.dtype != np.uint16 or a.shape != (H, W))(tifffile.imread(tifs[i]))]
    print(f"  RAW: {meta['camera']['raw_check']}  FFC 모드: {meta['camera']['ffc_mode_at_start']}")
    if not meta["camera"]["raw_check_ok"] or bad:
        problems.append("RAW 아님" + (f" (uint16 (256,320)이 아닌 TIFF: {bad})" if bad else ""))
    if rec["end_reason"] not in ("auto_max_duration", "user_q"):
        problems.append(f"비정상 종료: {rec['end_reason']}")

    # ---- EVENT (2차 촬영 T 구도)
    if meta.get("event_enabled"):
        sig, ef, ee = meta.get("event_signal_time_s"), meta.get("event_frame"), meta.get("event_end_frame")
        print(f"  EVENT: 신호 {sig}s, 시작(첫 S) frame {ef} ({meta.get('event_time_s')}s, {meta.get('event_source')}), "
              f"끝(두 번째 S) frame {ee}")
        c, r, hr = meta.get("clear"), meta.get("clear_ref"), meta.get("hot_ref")
        if c:
            print(f"  clear 0~{c['end_frame']}  clear_ref {r['start_frame']}~{r['end_frame']} ({r['n_frames']}f)"
                  + (f"  hot_ref {hr['start_frame']}~{hr['end_frame']} ({hr['n_frames']}f)" if hr else "  hot_ref 없음"))
        if meta.get("event_source") != "manual_S":
            problems.append("첫 S 없음 → 신호 프레임을 EVENT로 사용 (로그의 S 초 확인)")
        if ee is None:
            problems.append("두 번째 S 없음 → hot_ref 없음")
        if sig is not None and meta.get("event_source") == "manual_S" and meta["event_time_s"] - sig > 5:
            problems.append(f"첫 S가 신호보다 {meta['event_time_s'] - sig:.0f}초 늦음")

    # ---- 카메라 움직임
    if moving:
        print("  카메라 움직임: 이동 촬영이라 검사 안 함")
    else:
        cm = meta.get("camera_motion")
        src = "녹화 중 감지"
        if not cm:
            mc = MotionChecker()
            for i, p in enumerate(tifs):
                mc.update(tifffile.imread(p), i, float(ts[i]["elapsed_s"]))
            mc.finish()
            cm = mc.summary(len(tifs))
            src = "사후 계산"
        if not cm["checked"]:
            print("  카메라 움직임: 녹화가 짧아 검사 못 함")
        elif cm["first_moved_frame"] is None:
            print(f"  카메라 움직임: 없음 ({src})")
        else:
            eps = cm["episodes"]
            print(f"  카메라 움직임 ({src}): frame {cm['first_moved_frame']}부터, "
                  f"정렬 정상 ~frame {cm['aligned_until_frame']}, 구간 {len(eps)}개, "
                  f"최대 {max(e['max_shift_px'] for e in eps):.1f}px")
            problems.append(f"카메라 움직임: frame {cm['aligned_until_frame']}까지만 정렬됨 → 재촬영 권장")

    print("  ✔ 문제 없음" if not problems else "  ⚠ 확인 필요:\n    - " + "\n    - ".join(problems))
    return not problems


def align(dirs):
    """여러 녹화의 카메라 위치가 같은지 비교 (예: H00 라벨을 H01~H03에 쓰기 전).
    각 녹화의 3~5초 구간 평균 화면을 첫 녹화와 위상상관으로 비교한다.
    히터처럼 아주 뜨거운 부분이 비교를 망치지 않도록 각 화면을 자기 1~97% 밝기 범위로 자른다."""
    import cv2

    def ref(d):
        fs = [tifffile.imread(d / "frames" / f"{i:06d}.tif").astype(np.float32) for i in range(26, 43)]
        f = np.mean(fs, axis=0)
        lo, hi = np.percentile(f, [1, 97])
        f = np.clip(f, lo, hi)
        return ((f - f.mean()) / (f.std() + 1e-6)).astype(np.float32)

    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    base = ref(dirs[0])
    print(f"\n=== 녹화 간 정렬 (기준 {dirs[0].name}) ===")
    for d in dirs[1:]:
        (dx, dy), resp = cv2.phaseCorrelate(base.copy(), ref(d), win)
        shift = float(np.hypot(dx, dy))
        # 열원이 큰 녹화는 신뢰도가 낮게 나오므로 이동량으로 판정하고, 신뢰도가 아주 낮으면 화면 확인을 권한다
        if shift > 1.0:
            verdict = "⚠ 위치 다름 → 라벨 그대로 적용 불가"
        elif resp < 0.1:
            verdict = "? 신뢰도가 너무 낮음 → 화면을 직접 비교해 확인"
        else:
            verdict = "✔ 같은 위치"
        print(f"  {d.name}: dx {dx:+.2f} dy {dy:+.2f} (이동 {shift:.2f}px, 신뢰 {resp:.2f}) {verdict}")
    print("  (신뢰도는 열원 차이 때문에 낮아질 수 있음. 이동량이 1px 이하인지가 핵심)")


def main():
    # 인자: 코드(sessions 아래) 또는 폴더 경로. --align 코드1 코드2 … : 녹화 간 카메라 위치 비교
    args = sys.argv[1:]
    if args and args[0] == "--align":
        align([Path(a) if Path(a).is_dir() else SESSIONS_DIR / a.upper() for a in args[1:]])
        return
    dirs = ([Path(a) if Path(a).is_dir() else SESSIONS_DIR / a.upper() for a in args] if args
            else sorted(p for p in SESSIONS_DIR.iterdir() if p.is_dir()))
    ok = [check(d) for d in dirs]
    if len(dirs) > 1:
        print(f"\n총 {len(dirs)}개 중 문제 없음 {sum(ok)}개")


if __name__ == "__main__":
    main()
