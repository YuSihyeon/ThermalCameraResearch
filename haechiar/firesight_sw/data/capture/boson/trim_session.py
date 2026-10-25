r"""녹화 뒷부분 잘라내기 (예: 카메라가 밀린 이후 프레임 제외). 잘라낸 프레임은 지우지 않고 보관 폴더로 옮긴다.

실행:  py -3.12 trim_session.py <코드> <남길 마지막 프레임> "<이유>"
예:    py -3.12 trim_session.py P01 850 "frame 851에서 카메라 밀림(약 6.8px)"
- sessions\<코드>: frame 0~마지막 프레임만 남기고 timestamps/motion/meta 갱신
- <촬영 폴더>\재촬영_보관\<코드>_제외_frameA-B\: 잘라낸 TIFF + 그 구간 timestamps/motion + 원래 meta 사본
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_session import SESSIONS_DIR, SHOOT_DIR, compute_clear  # noqa: E402


def split_csv(path, keep_last, dst):
    """frame 열 기준으로 keep_last 이하만 남기고 나머지는 dst로."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    keep = [r for r in body if int(r[0]) <= keep_last]
    drop = [r for r in body if int(r[0]) > keep_last]
    for p, rs in ((path, keep), (dst, drop)):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rs)


def main():
    code, last, reason = sys.argv[1].upper(), int(sys.argv[2]), sys.argv[3]
    d = SESSIONS_DIR / code
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    total = meta["total_frames"]
    if not 0 <= last < total - 1:
        sys.exit(f"남길 마지막 프레임은 0~{total - 2} 사이여야 합니다 (전체 {total}프레임).")
    arch = SHOOT_DIR / "재촬영_보관" / f"{code}_제외_frame{last + 1}-{total - 1}"
    if arch.exists():
        sys.exit(f"이미 있음: {arch}")
    (arch / "frames").mkdir(parents=True)
    shutil.copy2(d / "meta.json", arch / "meta_원본.json")
    for i in range(last + 1, total):
        shutil.move(str(d / "frames" / f"{i:06d}.tif"), str(arch / "frames" / f"{i:06d}.tif"))
    for name in ("timestamps.csv", "motion.csv", "frozen_frames.csv"):
        split_csv(d / name, last, arch / name)

    with open(d / "timestamps.csv", encoding="utf-8") as f:
        times = [float(r["elapsed_s"]) for r in csv.DictReader(f)]
    n = last + 1
    meta["total_frames"] = n
    rec = meta["recording"]
    rec["duration_s"] = round(times[-1], 3)
    with open(d / "frozen_frames.csv", encoding="utf-8") as f:
        rec["frozen_frame_count"] = len(list(csv.DictReader(f)))
    if meta.get("event_frame") is None or meta["event_frame"] > last:
        meta["clear"], meta["clear_ref"] = compute_clear(times, None)
    cm = meta.get("camera_motion")
    if cm:
        cm["episodes"] = [e for e in cm["episodes"] if e["start_frame"] <= last]
        cm["first_moved_frame"] = cm["episodes"][0]["start_frame"] if cm["episodes"] else None
        if not cm["episodes"]:
            cm["aligned_until_frame"] = last
    meta["trimmed"] = {"kept_frames": [0, last], "removed_frames": [last + 1, total - 1],
                       "original_total_frames": total, "reason": reason, "removed_to": str(arch)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{code}: frame 0~{last} ({n}프레임, {times[-1]:.1f}초) 남김, frame {last + 1}~{total - 1} → {arch}")


if __name__ == "__main__":
    main()
