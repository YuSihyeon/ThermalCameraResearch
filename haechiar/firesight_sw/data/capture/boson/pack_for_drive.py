r"""촬영 폴더를 Google Drive 업로드용으로 묶는다 (원본 프레임 포함).

실행:  py -3.12 pack_for_drive.py <촬영 폴더> <출력 폴더> <이름>
예:    py -3.12 pack_for_drive.py D:\Boson_촬영_2차_20261014 D:\업로드 2026-10-14_2차
결과:  <출력 폴더>\<이름>\
         sessions\<코드>.zip      녹화 1개 = zip 1개 (<코드>\frames\*.tif + meta.json, csv)
         보관\<폴더>.zip          sessions 외 하위 폴더 (재촬영_보관, 중단된_녹화, 연습 등)
         *.md, *.csv              문서 (그대로 복사)
         MANIFEST.csv             zip별 파일 수·TIFF 수·크기·SHA-256
- 압축은 deflate 레벨 1 (TIFF가 약 59%로 줄어듦). .bat 파일은 넣지 않는다.
- 만든 뒤 zip을 다시 열어 CRC 검사와 TIFF 개수를 원본과 비교한다.
"""
import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


def zip_dir(src, zpath, arc_root):
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() != ".bat")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files:
            z.write(p, f"{arc_root}/{p.relative_to(src).as_posix()}")
    return files


def verify(zpath, files):
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        names = z.namelist()
    n_tif = sum(1 for p in files if p.suffix.lower() == ".tif")
    z_tif = sum(1 for n in names if n.lower().endswith(".tif"))
    if bad or len(names) != len(files) or z_tif != n_tif:
        raise SystemExit(f"검증 실패: {zpath} (손상={bad}, 파일 {len(names)}/{len(files)}, TIFF {z_tif}/{n_tif})")
    h = hashlib.sha256()
    with open(zpath, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return len(files), n_tif, h.hexdigest()


def main():
    src, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    dst = out / name
    if dst.exists():
        raise SystemExit(f"이미 있음: {dst}")
    (dst / "sessions").mkdir(parents=True)
    rows = []
    jobs = [(d, dst / "sessions" / f"{d.name}.zip", d.name) for d in sorted((src / "sessions").iterdir()) if d.is_dir()]
    jobs += [(d, dst / "보관" / f"{d.name}.zip", d.name) for d in sorted(src.iterdir())
             if d.is_dir() and d.name != "sessions" and any(p.is_file() for p in d.rglob("*"))]
    for d, zpath, arc in jobs:
        zpath.parent.mkdir(exist_ok=True)
        files = zip_dir(d, zpath, arc)
        n, n_tif, sha = verify(zpath, files)
        size = zpath.stat().st_size
        rows.append([zpath.relative_to(dst).as_posix(), n, n_tif, sum(p.stat().st_size for p in files), size, sha])
        print(f"{zpath.relative_to(dst)}: 파일 {n}, TIFF {n_tif}, {size / 1e6:.0f} MB, 검증 OK", flush=True)
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in (".md", ".csv"):
            shutil.copy2(p, dst / p.name)
    with open(dst / "MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["zip", "파일 수", "TIFF 수", "원본 바이트", "zip 바이트", "sha256"])
        w.writerows(rows)
    total = sum(r[4] for r in rows)
    print(f"완료: {dst}  zip {len(rows)}개, {total / 1e9:.2f} GB, TIFF {sum(r[2] for r in rows)}장")


if __name__ == "__main__":
    main()
