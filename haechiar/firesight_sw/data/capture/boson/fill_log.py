"""현장로그.csv의 자동 칸을 녹화 데이터(meta.json + 점검 결과)로 채운다.

실행:  py -3.12 fill_log.py            (촬영 폴더는 BOSON_SHOOT_DIR, 없으면 record_session 기본값)
채우는 칸: 녹화 시작, S(히터 치움) 초, 유효, 특이사항(자동 부분).
가열 시작·온도·사진 번호는 사람이 적은 값을 그대로 둔다. 특이사항에 사람이 쓴 내용은 "자동:" 앞에 보존한다.
"""
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_session import SESSIONS_DIR, SHOOT_DIR  # noqa: E402
from check_sessions import check  # noqa: E402

LOG_CSV = SHOOT_DIR / "현장로그.csv"
AUTO = "자동:"


def auto_fields(d):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    rec = meta["recording"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ok = check(d)
    problems = out.getvalue().split("⚠ 확인 필요:")[1].strip().replace("\n    - ", " / ").lstrip("- ") if not ok else ""
    s = "—"
    if meta.get("event_enabled"):
        ef = meta.get("event_time_s")
        ee = meta.get("event_end_time_s")
        s = (f"{ef:.1f}" if ef is not None else "없음") + (f" / {ee:.1f}" if ee is not None else " / 두 번째 없음")
    note = f"{AUTO} {rec['duration_s']:.0f}초, {meta['total_frames']}프레임, 종료={rec['end_reason']}"
    note += f", 문제: {problems}" if problems else ", 점검 OK"
    return {"녹화 시작": rec["start"][11:19], "S(히터 치움) 초": s, "유효": "O" if ok else "확인 필요", "note": note}


def main():
    with open(LOG_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    n = 0
    for r in rows:
        d = SESSIONS_DIR / r["코드"]
        if not (d / "meta.json").exists():
            continue
        a = auto_fields(d)
        r["녹화 시작"], r["S(히터 치움) 초"] = a["녹화 시작"], a["S(히터 치움) 초"]
        if r["유효"] in ("", "O", "확인 필요"):
            r["유효"] = a["유효"]
        manual = r["특이사항"].split(AUTO)[0].strip().rstrip("|").strip()
        r["특이사항"] = (manual + " | " if manual else "") + a["note"]
        n += 1
        print(f"{r['코드']}: 녹화 {a['녹화 시작']}  S {a['S(히터 치움) 초']}  유효 {r['유효']}  {r['특이사항']}")
    try:
        with open(LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        print("현장로그.csv가 엑셀에 열려 있어 저장하지 못했습니다. 엑셀을 닫고 다시 실행하세요.")
        return
    print(f"현장로그 {n}개 구도 갱신: {LOG_CSV}")


if __name__ == "__main__":
    main()
