"""Interactive input, report review, explicit consent and live application."""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import uuid
import webbrowser

import cv2
import yaml

from .application import Rectifier, open_source, run_application, validate_application_options
from .capture import run_capture
from .target import load_target
from .workflow import calibrate_dataset


def _ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _write_audit(path, audit):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _reviewed_candidate(path):
    path = Path(path).resolve()
    debug = path.parent / (path.stem + "_debug")
    report = debug / "review.html"
    manifest = json.loads((debug / "review_manifest.json").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (manifest.get("schema") != "firesight.review/v1"
            or digest != manifest.get("calibration_sha256")
            or hashlib.sha256(report.read_bytes()).hexdigest() != manifest.get("report_sha256")):
        raise ValueError("review report does not match calibration; generate a new calibration/report")
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("calibration must be a YAML mapping")
    Rectifier(value)  # Reject malformed parameters before offering application.
    return value, digest, report


def review_and_apply(path, args) -> str:
    value, digest, report = _reviewed_candidate(path)
    validate_application_options(args)
    quality = value.get("quality", {})
    print(f"\n검토 보고서: {report}\n보정값: {Path(path).resolve()}")
    print(f"RMS: {value.get('rms_px', 'unknown')} px | 수치 검사: {quality.get('passed', False)}")
    print(f"품질 주의 항목: {quality.get('failures', [])} | 타깃 상태: {quality.get('target_status', 'unknown')}")
    if not quality.get("deployment_ready", False):
        print("타깃 검증 또는 수치 품질 조건이 미충족입니다. 보고서의 경고와 검출 영상을 확인하세요.")
    destination = (f"USB 장치 {args.device}: 실시간 보정 영상 창" if args.source == "usb" else
                   f"ROS {args.topic} → {args.rectified_topic}, CameraInfo: {args.camera_info_topic}")
    print(f"적용 대상: {destination}")
    print("사진을 찍은 동일한 카메라·렌즈·초점·해상도인지 확인하세요. USB 번호는 시리얼 번호가 아닙니다.")
    print("적용은 이 프로그램 실행 중의 영상 왜곡 보정이며 카메라 펌웨어/EEPROM을 변경하지 않습니다.")
    if not getattr(args, "no_browser", False):
        try:
            if not webbrowser.open(report.as_uri()):
                print("브라우저를 열지 못했습니다. 위 review.html을 직접 열어 확인하세요.")
        except (OSError, webbrowser.Error):
            print("위 review.html을 직접 열어 확인하세요.")
    while True:
        answer = _ask("보고서 검토를 마쳤고 이 카메라에 적용하시겠습니까? [yes 적용 / no 새 입력 / quit 종료]: ").lower()
        if answer in ("yes", "y", "예", "적용"):
            decision = "yes"
            break
        if answer in ("no", "n", "아니오", "아니요"):
            decision = "no"
            break
        if answer in ("q", "quit", "종료"):
            decision = "quit"
            break
        print("yes, no 또는 quit를 명시적으로 입력하세요. 빈 입력은 승인하지 않습니다.")
    audit = {
        "schema": "firesight.review_decision/v1", "decided_utc": datetime.now(timezone.utc).isoformat(),
        "calibration": str(Path(path).resolve()), "calibration_sha256": digest,
        "decision": decision, "quality": quality,
        "destination": {key: getattr(args, key) for key in
                        ("source", "device", "topic", "telemetry_rows", "rectified_topic", "camera_info_topic")},
        "application": {"status": "not_started"},
    }
    audit_path = report.parent / f"decision_{_stamp()}_{uuid.uuid4().hex[:8]}.json"
    _write_audit(audit_path, audit)
    if decision != "yes":
        print("적용하지 않았습니다. " + ("새 사진 입력 단계로 돌아갑니다." if decision == "no" else "종료합니다."))
        return "retry" if decision == "no" else "quit"
    try:
        _, current_digest, _ = _reviewed_candidate(path)
        if digest != current_digest:
            raise ValueError("calibration changed during review; review the new result first")
        # The approved in-memory snapshot is passed to runtime, never a mutable file path.
        audit["application"] = {"status": "starting"}
        _write_audit(audit_path, audit)
        audit["application"] = run_application(value, args)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        audit["application"] = {"status": "failed", "error": str(exc)}
        print(f"적용이 완료되지 않았습니다: {exc}\n새 입력 단계로 돌아갑니다.")
        return "retry"
    finally:
        _write_audit(audit_path, audit)
    print(f"적용 기록: {audit_path}")
    return "applied" if audit["application"]["status"] == "applied" else "quit"


def _image_paths(value):
    folder = Path(value)
    paths = (sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".tif", ".tiff"))
             if folder.is_dir() else [Path(p) for p in sorted(glob.glob(value)) if Path(p).is_file()])
    if not paths:
        raise ValueError(f"image input matched no files: {value}")
    return paths


def run_wizard(args) -> dict:
    validate_application_options(args)
    target = load_target(args.target)
    root = Path(args.output or (Path("calibration_data") / ("review_" + _stamp())))
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"wizard output must be a new/empty directory: {root}")
    images = getattr(args, "images", None)
    attempt = 0
    print("FireSight | 사진 입력 → 계산 → 보고서 검토 → 사용자 승인 → 실시간 적용")
    while True:
        if not images:
            answer = _ask("원본 사진 폴더/glob 입력 (Enter: 새 촬영, quit: 종료): ")
            if answer.lower() in ("q", "quit", "종료"):
                return {"status": "cancelled", "session": str(root)}
            images = answer.strip('"')
        attempt += 1
        directory = root / f"attempt_{attempt:03d}"
        output = directory / "boson_intrinsic.yaml"
        try:
            if images:
                paths = _image_paths(images)
                print(f"{len(paths)}개 원본에서 보정 중: {directory}", flush=True)
                calibrate_dataset(paths, output, target, args.min_views, args.max_rms)
            else:
                result = run_capture(
                    open_source(args), directory, target, headless=args.headless, auto=args.auto,
                    timeout=args.timeout, interval=args.interval, count=args.count,
                    min_views=args.min_views, max_rms=args.max_rms, calibrate=True,
                    source_info={key: getattr(args, key) for key in ("source", "device", "topic", "telemetry_rows")},
                )
                if not result.get("output"):
                    return {"status": "cancelled", "session": str(root)}
            decision = review_and_apply(output, args)
            if decision != "retry":
                return {"status": "applied" if decision == "applied" else "cancelled",
                        "output": str(output), "session": str(root)}
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, cv2.error, yaml.YAMLError) as exc:
            print(f"이번 입력을 처리하지 못했습니다: {exc}\n원본과 진단 파일은 유지하며 새 입력을 받습니다.")
        images = None
