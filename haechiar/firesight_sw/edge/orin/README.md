# FireSight Edge — Jetson Orin Nano

Jetson Orin Nano(JetPack 6 / Ubuntu 24.04 / Python 3.12)용 TEED 추론·HUD 런타임.
`edge/teed/`는 원형 Jetson Nano(JetPack 4.6 / Python 3.6)용이며, 이 폴더는
Orin 대상으로 별도 구성한다. TEED 모델과 체크포인트는 공유한다.

## 구조

```text
teed_core.py     추론 코어 (그리기 없음, EdgeResult 반환)
teed_model.py    TEED 정의 + encode/head 분리 (단계별 계측용)
replay.py        입력 계층 (영상·이미지·폴더·카메라·Boson Y16)
hud.py           렌더링 계층 (색·선·해상도 전담)
display.py       출력 계층 (창·전체화면·녹화·디스플레이 열거)
run_pipeline.py  통합 실행기 + 메트릭
smoke_eval.py    농연 성능 평가 (recall/precision)
tools/           orin_verify.sh, find_jetson.sh, camera_check.py, display_check.py
```

작업 문서(RUNBOOK, INTEGRATION, MIGRATION, ORIN_SETUP, PROGRESS)는 `docs/`에
두되 저장소에는 커밋하지 않는다(로컬 작업 노트).

## 체크포인트

`edge/teed/models/5_model.pth`를 공유한다(SHA-256 `0322caf7…6a01ba`). 별도 사본을
두지 않는다. `.gitignore`로 커밋에서 제외되므로 클론 후 그 파일이 있어야 한다.

## 빠른 실행 (Orin)

```bash
cd edge/orin
./tools/orin_verify.sh          # 보드·CUDA·체크포인트·카메라 점검 후 300프레임 실측
```

수동 실행:

```bash
python3 run_pipeline.py --source /dev/video0 --device cuda --half \
  --width 320 --height 240 --headless --max-frames 300 --metrics metrics_orin.json
```

실행 명령·의존 관계·병목 실측, Orin 연결 방법은 로컬 `docs/`의 RUNBOOK,
ORIN_SETUP 문서를 참고한다.

## 개발 (Mac 등)

```bash
python3 run_pipeline.py --source clip.mp4 --device cpu --pace --loop
```

`--device cpu`로 CUDA 없이 검증 가능. 실측은 Orin에서 한다.

## TEED UI 경계 선별 및 고온 등온선

최신 실행기는 짧은 경계를 제거하고 TEED 확률/영상 gradient hysteresis로 표시 경계를 선별한다.
거리 입력이 없으면 거리 효과를 적용하지 않는다. 상대 밝기를 온도로 간주하지 않으며,
정합된 Celsius 온도맵이 있는 경우에만 온도 임계값의 등온선을 표시한다. Fire360 표시 영상만 입력하면 `TEMP N/A`다.
개발자 패널은 `--show-panel`로 켤 수 있다.

[알고리즘 PNG·실행법·검증 한계](../../docs/teed_ui_improvement.md)를 참고한다.

Visible-light replay can opt in to temporal flame candidate contours with --visible-flame-candidates. See [usage and limitations](../../docs/visible_flame_candidates.md). This does not estimate temperature.

## 잔여 픽셀 정리

기본 엣지 필터는 중심선에서 연결 성분 길이 12 px, 끝점에서 분기점까지의
가지 길이 6 px, 연결 성분의 바운딩 박스 대각선 8 px 기준을 함께 적용한다.
단위는 화면 픽셀이 아닌 모델 입력 픽셀(기본 320×256)이다.
짧은 가지는 분기점을 남기고 한 번만 제거한다. 이 공간 필터 뒤의 표시 결과는 아래 시간 안정화와 함께 사용한다.

```bash
python run_pipeline.py --source clip.mp4 --min-edge-length 12 --min-branch-length 6 --min-edge-extent 8
```

추가된 두 필터만 끄려면 `--min-branch-length 0 --min-edge-extent 0`을 사용한다.
작게 보이는 실제 물체도 제거될 수 있으므로 실제 장애물 recall과 Orin 지연은 별도 검증이 필요하다.
[비교 영상·수치·검증 범위](../../docs/edge_residual_cleanup.md)를 참고한다.


## Fire360 영상 시간 안정화

`run_pipeline.py`는 기본으로 이전 TEED 확률을 현재 프레임의 움직임에 맞춰
이동한 뒤 누적하고 공간 필터를 적용한다. 강한 새 경계의 확률은 바로 통과시키며,
정합 실패·영상 루프·프레임/시간 단절에서는 과거 정보를 초기화한다.

```bash
python run_pipeline.py --source clip.mp4 --device cpu --pace --temporal-edges
# 같은 입력으로 시간 안정화만 끄고 비교
python run_pipeline.py --source clip.mp4 --device cpu --pace --no-temporal-edges
```

`--temporal-history-weight` 기본값은 0.6이며 범위는 `[0,1)`이다.
값을 높이면 약한 새 경계의 반영도 늦어질 수 있다. 시간 정보가 있으면
30fps 기준 가중치를 프레임 시간 간격에 맞춰 조정한다.
독립 이미지용 `HUDConfig` API는 기존처럼 기본값이 꺼져 있다.
직접 연속 호출할 때는 `temporal_edges=True`, `frame_index`, `timestamp_s`를 전달하고,
새 입력이나 seek/loop 시 `renderer.reset_temporal()`을 호출한다.

[Fire360 비교 영상·측정값·검증 한계](../../docs/temporal_stability.md)를 참고한다.


## 윤곽선 추적과 출력 해상도 렌더링

`--contour-edges`로 실험용 윤곽 추적을 켠다. 기존 확률 안정화가 기본이며,
CPU 처리 비용 증가 때문에 윤곽 추적은 명시적으로 선택한다. 공간 필터 이후의 윤곽을 추적하고,
강한 새 경계는 즉시 표시하며 약한 조각의 반복 관측을 확인한다.
표시할 선은 모델 마스크를 확대하는 대신 출력 해상도에서 안티앨리어싱해 그린다.

```bash
python run_pipeline.py --source clip.mp4 --device cpu --pace --contour-edges
# 직전 확률 누적 방식과 비교
python run_pipeline.py --source clip.mp4 --device cpu --pace --no-contour-edges
# 모든 시간 처리를 끈 공간 필터 결과
python run_pipeline.py --source clip.mp4 --device cpu --pace --no-contour-edges --no-temporal-edges
```

`--contour-line-width`는 모델 픽셀이 아닌 출력 픽셀 단위이며 기본값은 1.5다.
`--legacy-edges`는 확률 누적과 윤곽 추적을 모두 우회한다.
API에서 `HUDConfig(contour_edges=True)`로 활성화하며, `reset_temporal()`은
확률 누적과 윤곽 추적 상태를 함께 지운다. 기본 API 값은 독립 이미지 호환성을 위해 꺼져 있다.
새 입력·seek·loop에서는 초기화하고 연속 입력에 프레임 번호/타임스탬프를 전달한다.
`HUDFrame.edge_mask`는 검증용 모델 좌표의 표시 중심선이다.

[실제 영상 비교와 검증 범위](../../docs/contour_stability.md)를 참고한다.


## 원본 고온 색상 보존과 빨간 화점 외곽선

Fire360 IFSI4–10과 같이 카메라가 이미 노랑·주황·빨강으로 표시한 열화상 영상은
`--thermal-palette-overlay`로 **원본 고온 색상과 빨간 외곽선**을 함께 표현한다.
TEED용 대비 보정과 단일 채널 확률맵 계산 동안 원본 BGR 색상 레이어를 별도로 유지한다.
영역 내부와 주변의 겹치는 초록 선을 지운 뒤 카메라의 노랑·주황·빨강을 합성한다.
투명 AR 모드에서도 고온 색상은 남으며, 외곽의 빨간 선이 마지막에 그려진다.
화면 너비 720 px에서 빨간 선은 3 px, 구조물 윤곽 기본값은 1.5 px이다.
오른쪽 눈금·왼쪽 로고는 Fire360 OSD 프로필로 제외하고 작은 색 조각도 제거한다.

```bash
python run_pipeline.py --source "IFSI Video 10 (2).mp4" --device cpu   --contour-edges --thermal-palette-overlay --profiles ../teed/profiles/fire360_video_profiles.json --pace
# 투명 AR용 검은 배경
python run_pipeline.py --source "IFSI Video 10 (2).mp4" --device cpu   --contour-edges --thermal-palette-overlay --profiles ../teed/profiles/fire360_video_profiles.json --see-through --pace
```

이 옵션은 확인된 열화상 표시 영상에만 사용한다. 일반 RGB는 기본 비활성이며
밝은 회색/흰색만으로 화점을 만들지 않는다. 실제 온도맵이 있으면 온도맵을 우선한다.
영상 색으로 섭씨 온도나 실제 발화 지점을 확정할 수 없어 `TEMP N/A`는 유지한다.
`--no-fire-highlight`는 이 빨간 외곽선도 끈다. 가시광 화염 후보 모드와 동시 사용하지 않는다.
[전체 영상과 검증 기록](../../docs/thermal_hot_contours.md)을 참고한다.

## 정적·동적 윤곽 분리

`--motion-separated-edges`는 윤곽 추적과 카메라 움직임 보정을 함께 켠다.
공통 움직임을 따르는 선은 더 강하게 안정화하고, 독립적으로 움직이는 선은
현재 위치를 빠르게 반영한다. 불확실한 선은 기존 정책으로 처리한다.
고온 색상은 기존 `--thermal-palette-overlay` 경로를 그대로 사용한다.

```bash
python run_pipeline.py --source "IFSI Video 10 (2).mp4" --device cpu --motion-separated-edges --thermal-palette-overlay --profiles ../teed/profiles/fire360_video_profiles.json --pace
```

[동작·비교 영상·검증 한계](../../docs/motion_separation.md)를 참고한다.
