# Thermal Camera Research

**열화상 입력을 확인하고, 농연 영상의 경계를 덜 가리면서 읽기 쉽게 표시하는 과정을 연구한다.**

[개요](#프로젝트-개요) · [영상 6개](#영상과-설명) · [데이터](#데이터와-선정-과정) · [구현](#구현-과정) · [결과](#결과와-분석) · [다음 실험](#한계와-다음-단계) · [재현](#재현과-자료-안내)

> **핵심 성과** — 카메라 표시·캡처, 밝기 기반 height field, 농연 경계 HUD, 합성 거리의 연속 선 굵기, RTX PC에서의 지속 처리 검증을 연결했다. 실제 거리·절대온도 측정과 Jetson·AR 안경 실장 평가는 아직 검증하지 않았다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 연구 대상 | 열화상 영상의 입력 의미, 경계 선택, HUD 표시, 주어진 거리의 폭 표현 |
| 핵심 문제 | 굵은 초록 윤곽이 배경을 가리는 문제와 거리 구간 경계의 폭 변화 |
| 주요 자료 | 직접 저장 grayscale 52장, PLY 2개, Boson HUD, 공개 SmokeBasement, 합성 거리 |
| 도구 | OpenCV·NumPy·Open3D, TEED/PiDiNet, PyTorch, CUDA Graph |
| 기록 시점 | 초기 카메라 탐색일 미확정 / 후속 보고서 2026-09-06~14 / 보존 검토 2026-09-16 |
| 현재 수준 | 입력·표시·처리 경로의 실험 기록; 현장 장애물 회피 효능과 센서 계측은 후속 과제 |
| 상세 자료 | [기존 README 전문](RESEARCH_REPORT.md) · [결과 문서](docs/03-results.md) · [전체 복원 범위](DATA_AND_RESTORE.md) |

<img src="docs/evidence/obstacle-edges/smoke-comparison.png" width="900" alt="SmokeBasement 열화상 입력에 기존 TEED, 구조 경계 처리와 비교 모델을 적용한 결과">

*위 행 왼쪽부터 공개 입력 / 기존 TEED / 얇은 선·밝은 배경, 아래 행은 구조 경계 / adaptation / PiDiNet이다. 직접 Boson으로 촬영한 농연 실험과 구분하며, 선 점유율 감소를 검출 정확도 향상률로 해석하지 않는다.*

## 영상과 설명

미리보기나 제목을 누르면 MP4를 연다. [MEDIA.md](MEDIA.md)에서 영상별 해상도·FPS와 보존 MP4 다운로드를 확인할 수 있다.

| 영상 | 관찰할 내용과 조건 |
|---|---|
| [<img src="public-media/boson-test.jpg" width="210" alt="Boson HUD 영상 미리보기">](public-media/boson-test.mp4)<br>[Boson HUD 보존 영상](public-media/boson-test.mp4) | **10초 · 320×256 · 9fps.** 직접 제공된 초록 윤곽 HUD. 이미 합성된 표시 결과이며 원시 센서·절대온도 자료는 아님. |
| [<img src="public-media/distance-preview.jpg" width="210" alt="가정 거리별 선 굵기 미리보기">](public-media/distance-preview.mp4)<br>[거리별 굵기 · 가정 거리 시연](public-media/distance-preview.mp4) | **12초 · 1280×736 · 10fps.** 화면 원근에 따라 임의로 지정한 거리. 초기 1/2/3px 표현 시연으로 실제 거리 추정·radar 연동 결과가 아님. |
| [<img src="public-media/smoke-original.jpg" width="210" alt="공개 농연 입력 미리보기">](public-media/smoke-original.mp4)<br>[농연 입력 · SmokeBasement](public-media/smoke-original.mp4) | **12초 · 320×240 · 10fps.** 공개 열화상에서 선정·재표본화한 구간. 전체 센서 원본이나 직접 Boson 촬영본과 구분. |
| [<img src="public-media/smoke-structural.jpg" width="210" alt="TEED 구조 경계 결과 미리보기">](public-media/smoke-structural.mp4)<br>[TEED 구조 경계 후처리](public-media/smoke-structural.mp4) | **12초 · 320×240 · 10fps.** 확률맵 이후 연결 경계 선택과 세선화. 얇아진 선과 남아 있는 구조를 함께 관찰. |
| [<img src="public-media/smoke-balanced-2px.jpg" width="210" alt="Balanced 2px 표시 미리보기">](public-media/smoke-balanced-2px.mp4)<br>[Balanced · 2px 표시 후보](public-media/smoke-balanced-2px.mp4) | **12초 · 640×480 · 10fps.** 같은 확률맵의 표시 폭 비교. 모니터 후보이며 런타임 기본 두께는 1px; 안경 실장 평가는 아님. |
| [<img src="public-media/realtime-comparison.jpg" width="210" alt="연속 선 굵기 비교 미리보기">](public-media/realtime-comparison.mp4)<br>[연속 굵기 · 처리 방식 비교](public-media/realtime-comparison.mp4) | **12초 · 640×556 · 10fps.** 공개 열화상과 합성 거리의 출력 비교. 이 영상의 인코딩 FPS와 본문의 180초 처리 benchmark는 다른 지표. |

**출처:** Boson HUD를 제외한 다섯 영상은 **SmokeBasement — Jianzhu Huai (2025), CC BY 4.0**의 선택·재표본화·resize 또는 처리 파생물이다. [DOI](https://doi.org/10.5281/zenodo.15173055) · [출처·변환 상세](docs/SOURCE_CREDITS.md). 합성 거리는 원 데이터의 LiDAR 실측값이 아니다.

## 연구 질문과 목표

초기에는 열화상 신호를 화면에 띄우고 저장한 뒤 공간적으로 표현할 수 있는지 탐색했다. 밝기를 높이로 바꾸는 방법은 점군 처리 흐름을 확인했지만 실제 깊이를 만들지는 못했다. 후속 연구의 중심은 농연 영상의 벽·문틀·장애물 경계를 읽기 쉽게 표시하는 문제로 구체화되었다.

연구 질문은 세 가지다. **굵은 선의 배경 가림을 줄이면서 필요한 구조를 남길 수 있는가**, **주어진 거리의 변화를 선 굵기로 안정적으로 표현할 수 있는가**, **평균 처리량뿐 아니라 느린 프레임까지 고려해 지속 처리할 수 있는가**다.

검출 모델, 경계 선택, 최종 표시 폭, 센서 거리 입력을 나누어 비교했다. 더 적은 선이 더 정확한 장애물 검출을 뜻하지 않으므로, 표시 성과와 인식·안전 성과를 분리해 평가한다. [연구 방향과 해석](docs/04-analysis.md)

## 수행 내용과 기여 범위

이 저장소는 자체 실험 자료와 FireSight 관련 실험의 직접 근거를 선별한 기록이다. 팀 전체 앱이나 외부 모델을 개인이 모두 개발한 결과로 소개하지 않는다. 개인별 코드 작성 분량을 재구성해 단정할 자료도 여기에는 없다.

| 구분 | 이 자료에서 확인되는 작업 | 외부·팀 범위 |
|---|---|---|
| 초기 입력 탐색 | OpenCV 표시·수동 PNG 저장·height field PLY·합성 점군 분리 | 센서 자체 설계·절대온도 보정과 구분 |
| 농연 모델 비교 | 전처리·경계 후처리 조합, TEED fusion-head adaptation, PiDiNet 비교 | TEED/PiDiNet 핵심 모델·기존 pretrained weight는 외부 기반 |
| HUD 검토 | 기존 초록 stroke 감사, 같은 확률맵에서 상세도와 폭 비교 | 모니터 판단이며 사용자·안경 실험과 구분 |
| 거리 표현 | smoothstep 폭, EMA, invalid/stale/future 처리와 합성 시험 | 실제 거리 센서·전체 장치 adapter는 선별본 범위 밖 |
| 지속성 검증 | CPU/CUDA/CUDA Graph FP32 경로, 프레임 CSV·gate 비교 | FireSight 전체 앱·의존성·목표 장비 배포본이 아님 |
| 보존 검토 | 52장 통계, PLY 전체 검산, 영상 디코드, 출처·해시 대조 | 모델 재학습·카메라 재촬영·현장 성능 재실행은 하지 않음 |

[코드 파일별 역할](code/README.md) · [자료 귀속](docs/SOURCE_CREDITS.md) · [보존 검증 기록](VALIDATION.md)

## 데이터와 선정 과정

### 입력의 성격을 먼저 분리

| 입력 | 확보·선정 내용 | 사용할 수 있는 의미 |
|---|---|---|
| 직접 저장 PNG | `frame_0`~`frame_51`, 52장 모두 640×512·단일 채널 uint8 | 표시값의 분포와 변환 입력; timestamp·intrinsics·온도 보정은 없음 |
| Boson HUD | 90프레임, 초록 윤곽이 이미 합성됨 | 표시 stroke 면적·형태 감사; 가려진 열 배경 복원은 불가 |
| SmokeBasement | run3 중앙 24초의 32장 학습, run5 중앙 12초의 120장 검증 | 서로 다른 run의 비교; 같은 지하실이므로 새 현장 일반화와 구분 |
| Fire360/IFSI | Video 8 보조 학습, Video 4의 625프레임 보조 검증 | 보조 입력이며 농연 강도가 확인된 핵심 시험과 합산하지 않음 |
| 합성 거리·점군 | 임의 원근, sinusoidal 거리, Gaussian noise, step/결측/stale | 표시 동작·실패 정책의 통제 시험; 실측 radar·LiDAR 데이터가 아님 |

초기 캡처 코드는 `s`를 누를 때만 번호를 증가시킨다. 52장의 번호를 일정 주기의 연속 영상으로 해석하지 않는다. 첫 카메라 `VideoCapture(0)`만 기록하므로 파일명·창 제목으로 정확한 장치 part number와 radiometric variant를 확정할 수도 없다. [입력 조사](docs/01-inputs.md)

원래 SmokeBasement PNG는 640×512 uint8 3채널이고 모델 입력은 320×240이다. 검증 120장은 서로 다르며 기록된 시간 재표본화 최대 오차는 40.82ms, 학습은 39.40ms다. 당시 전체 27GB ZIP 대신 152개 PNG를 HTTP Range로 추출하고 CRC/SHA-256을 확인했다는 기록이 있다.

이번 정리에서 그 원 데이터 전체를 다시 받거나 152장을 모두 재추출하지 않았다. 선택 PNG·source manifest·적응 checkpoint 일부는 조사한 Windows 체크아웃에서 확인되지 않았다. 남은 재인코딩 영상은 원 PNG와 정확한 weight의 대체물이 아니다. [현재 복원 한계](DATA_AND_RESTORE.md)

### 대표 입력과 선정 근거

<img src="docs/evidence/inputs/frame_0.png" width="520" alt="밝기 기반 PLY 두 개를 검산하는 데 사용한 실제 저장 grayscale frame 0">

*`frame_0.png`는 PLY 두 개의 계산 입력으로 선정했다. 255는 8bit 표시값의 최댓값이며 센서의 포화 온도나 섭씨 상한을 뜻하지 않는다.*

현장 휴대폰 원본은 장치·모니터 촬영 기록이며 주변 사람·작업 화면을 포함할 수 있어 개인 보존본에 둔다. 공개용 여섯 영상과 로컬 전용 현장 영상의 선정 근거는 [publication review](docs/evidence/publication-review.json)에 남겼다.

## 구현 과정

### 1. 카메라 입력과 밝기 기반 점군

표시 코드는 BGR을 gray로 바꾸고 프레임별 min/max 정규화 후 INFERNO 컬러맵을 적용한다. 캡처는 정규화·컬러맵 **이전 gray 배열**을 저장하지만, 카메라/UVC 단계의 AGC·8bit 변환 여부는 기록되지 않았다. [표시](code/original/flir_view_test.py) · [캡처](code/original/thermal_capture.py)

PLY 변환은 `X=x`, `Y=y`, `Z=float32(I)/255×3000`이다. stride 4에서는 20,480점, stride 2에서는 81,920점이며 조밀한 버전은 `R=G=B=gray`를 저장한다. `3000`은 임의 높이 계수다. 카메라 back-projection과 metric Z가 없으므로 결과는 **intensity height field이며 depth나 절대온도가 아니다**. [변환 코드](code/original/thermal_image_to_pointcloud.py)

별도의 radar 탐색 코드는 바닥 1,000·벽 800·장애물 300점을 난수로 만들고 outlier 제거→RANSAC 두 평면→색 표시를 수행한다. thermal PLY 입력이나 실제 radar 장치 연결은 없으며, 가장 큰 평면을 바닥이라 부르는 가정도 현장 검증이 필요하다. [합성 점군 코드](code/original/firesight_radar_core.py)

### 2. 농연 경계와 제한적 adaptation

| 방식 | 주요 처리 선택 | 비교 목적 |
|---|---|---|
| 기존 TEED | CLAHE 2.0 → threshold .75 → 3px dilation → 배경 .24 | 당시 기준선 |
| 얇은 선·밝은 배경 | 기존 확률맵 → skeletonization → 배경 .55 | 겹침·배경 가림 완화 |
| 구조 경계 TEED | bilateral(5,25,3), CLAHE 없음 → hysteresis .60/.85 → skeletonization | 강한 구조와 연결된 약한 경계 유지 |
| 의사 라벨 adaptation | TEED `block_cat` 480개 파라미터만 학습 | 작은 fusion 변화의 실효성 확인 |
| PiDiNet | 공식 checkpoint·RGB 정규화 → hysteresis .20/.40 → skeletonization | 다른 경계 모델의 비용·연속성 비교 |

hysteresis는 low 이상인 8-connectivity 성분 중 high 이상 응답을 포함한 성분을 남긴다. `thin_only`는 배경 밝기도 바꾸므로 두께만의 순수 ablation은 아니며 모델별 threshold도 같은 precision/recall로 교정한 값이 아니다. [contours 구현](code/selected/obstacle_edges/contours.py)

adaptation은 seed 20260906, 32장×8 epochs=256 step, Adam lr 1e-4다. 320×240/160×120 TEED 자체 응답을 의사 라벨로 사용하고 feature extractor를 고정했다. 사람이 단 장애물 정답 라벨로 새로운 회피 능력을 학습한 실험은 아니다. [학습·평가 코드](code/selected/obstacle_edges/run_smoke.py)

### 3. 기존 HUD 감사와 표시 폭 분리

Boson HUD에는 새 TEED를 다시 적용하지 않고 HSV 초록 범위 H=40~85와 S/V threshold로 이미 그려진 stroke를 추출·세선화했다. 검출 전 원 입력이 없으므로 가려진 배경·누락 경계를 되살리는 절차는 아니다. [감사 코드](code/selected/boson_review/analyze.py)

별도로 합성 전 공개 입력의 동일 확률맵에서 detailed=(.55,.75), balanced=(.60,.85), sparse=(.70,.90)를 비교했다. 최종 640×480에서 Balanced 1px/2px는 선택한 경계가 같고 렌더 폭만 다르다. [표시 비교](code/selected/boson_review/verify_video.py)

### 4. 주어진 거리를 연속 폭으로 표시

거리 모듈은 정렬된 meter 거리 맵 또는 scalar를 받는다. 기본 1m→5px, 8m→1px이며 `t=(clamp(d,1,8)-1)/7`, `w=5-4t²(3-2t)`로 연속 폭을 만든다. 이번 실험의 `d`는 합성값이다. [distance_width](code/selected/edge/teed/distance_width.py)

작은 변화는 `α=1-exp(-dt/0.05)`의 50ms EMA로 필터링하고 폭 차이가 0.75px보다 크면 새 목표값을 즉시 사용한다. 화면 좌표별 history이며 optical flow나 물체 tracking은 사용하지 않는다. Guo-Hall 중심선의 시작 픽셀 폭을 사각 dilation·alpha blending으로 펼친다.

NaN/Inf/0, 거리 없음, 100ms 초과 stale, 미래 timestamp는 기본 1px로 복귀하고 history를 지운다. 이 1px는 먼 물체 판정이 아니라 **거리 결측 표시 정책**이다. [replay 계약](code/selected/edge/teed/depth_replay.py)

### 5. 처리 속도에서 지속성 검증으로

최신 카메라 frame 1개·거리 최근 8개를 보관하고 유효한 과거 거리만 선택한다. FP32 CUDA Graph와 고정 버퍼로 모델 dispatch 비용을 줄인 경로를 CPU/eager와 비교했다. [실행부](code/selected/edge/teed/realtime_core.py) · [판정](code/selected/edge/teed/stability.py)

원 보고서에는 timestamp 중복, 화면 경계의 세선화, 작은 성분 소실, 종료 직전 멈춤 누락, calibration 해상도 불일치 수정 이력이 있다. `perf_counter` 통일과 회귀 검증은 당시 기록이며 이번 README 정리에서 코드를 재수정한 것은 아니다. [처리 과정 상세](docs/02-process.md)

## 결과와 분석

### 입력→PLY 계산 검증

| 점군 | 샘플 간격 | 전체 vertex | 입력 수식 최대 절대 오차 |
|---|---:|---:|---:|
| `thermal_pointcloud.ply` | stride 4 | 20,480 | 0 |
| `thermal_pointcloud_big.ply` | stride 2 | 81,920 | 0 |

전체 X/Y와 `I/255×3000`, 조밀한 파일의 RGB가 입력과 일치했다. 이는 저장·변환 구현의 일치이며 조밀해진 점군이 실제 거리를 더 정확히 복원했다는 결과가 아니다. [전체 검산](docs/evidence/local-audit.json) · [CloudCompare 화면](docs/evidence/pointcloud/floor-view.png)

### 농연 경계의 표시 면적과 비용

**조건:** SmokeBasement run5 120프레임, 320×240, CPU 4 threads. 시간은 첫 10개를 제외한 warm 110개에서 집계하며 전처리·추론·후처리·렌더를 포함하고 읽기·인코딩·저장은 제외한다.

| 방식 | 엣지 점유율 ↓ | 전체 처리 중앙값 | 전체 처리 P95 |
|---|---:|---:|---:|
| 기존 TEED | 12.27% | 68.23ms | 89.37ms |
| 얇은 선·밝은 배경 | 2.36% | 69.15ms | 85.61ms |
| 구조 경계 TEED | 1.51% | 73.54ms | 104.55ms |
| 의사 라벨 adaptation | 1.45% | 69.49ms | 93.68ms |
| PiDiNet | 1.94% | 443.93ms | 556.90ms |

점유율은 출력 경계 mask가 차지한 픽셀 비율의 프레임 평균이다. **12.27%→1.51%는 표시 면적 약 87.7% 감소**이며 boundary 정답이 없어 precision/recall 개선률은 계산하지 않았다. PiDiNet의 비용도 해당 CPU·구현 조건에 한정한다. [원 결과 JSON](docs/evidence/obstacle-edges/results_summary.json)

480개 fusion 파라미터의 weighted soft BCE는 0.7557140→0.7550460, 변경 tensor는 4개다. 학습은 수행되었지만 감소폭과 시각 차이가 작아 기본 checkpoint를 교체할 근거는 부족했다. 원 기록의 checkpoint hash와 실행 조건은 [결과 문서](docs/03-results.md)에 있다.

### 같은 확률맵에서 읽기 쉬운 표시 찾기

| 표시 방식 | 640×480 평균 초록선 점유율 |
|---|---:|
| legacy | 11.4979% |
| detailed / 1px | 1.2701% |
| balanced / 1px | 1.1776% |
| balanced / 2px | 2.6322% |
| sparse / 1px | 1.0272% |

Balanced 2px는 모니터에서 눈에 띄면서 배경을 덜 가리는 후보였고 기본값은 1px로 남았다. 120프레임 모두 경계가 완전히 비지는 않았지만, 개별 얇은 장애물 누락을 검증한 수치는 아니다. 9월 6일 점유율과 이 표는 출력 조건이 달라 연속 개선 곡선으로 합치지 않는다. [판단 원문](docs/evidence/boson-review/RECOMMENDATION.md)

### 합성 거리에서 폭의 안정성

| 통제 입력 | 구간식 | 연속 무필터 | 연속+50ms EMA |
|---|---:|---:|---:|
| 4.5m, 거리 noise σ=.18m의 폭 표준편차 | 0px | .1560px | .0864px |
| 구간 경계 3.28m의 폭 표준편차 | .9963px | .1367px | .0755px |
| 1→8m 완만한 변화의 최대 폭 변화 | 2px/frame | .0251px/frame | .0251px/frame |

연속 폭은 구간 경계의 jump를 줄이고 EMA는 작은 noise를 줄였다. 30fps에서 이론적 저주파 지연은 35.2ms이며 큰 8m→1m step은 첫 frame에 5px로 반영되었다. 경계 검출의 깜빡임이나 실제 거리 오차를 줄인 지표는 아니다. [통제 시험](docs/evidence/distance-width/summary.json)

초기 CPU benchmark의 TEED 포함 P95는 연속+EMA에서 43.583ms, 33.333ms 예산 초과는 23.3%였다. 평균 약 36.6fps만으로 안정적 30fps라 판정하지 않았다. [프레임·후처리 원자료](docs/evidence/distance-width/benchmark_raw.csv)

### PC에서의 180초 지속 처리

**조건:** Windows·RTX 5060 Ti, torch 2.11.0+cu128, FP32, 320×240, 각 2 CPU threads, 예열 30회. 공개 120프레임/10fps 영상을 30fps로 가속 반복하고 합성 거리를 TCP로 전달했다.

| 방식 | 측정 시간 / 프레임 | 평균 FPS | 처리 P99 | 수신→출력 P99 | 예산 초과 | 판정 |
|---|---|---:|---:|---:|---:|---|
| CPU FP32 | 180.003초 / 5,400 | 29.999 | 33.452ms | 34.988ms | 1.056% | FAIL |
| CUDA FP32 | 180.008초 / 5,400 | 29.999 | 11.319ms | 12.021ms | 0% | PASS |
| CUDA Graph FP32 | 180.021초 / 5,401 | 30.002 | 10.981ms | 11.601ms | 0% | PASS |

시간에는 GPU 완료 대기와 `VideoWriter.write`가 포함된다. 종료 flush·창 표시·센서 노출/driver buffer·물리 display scanout은 제외한다. **호스트 수신→출력은 센서 노출→안경 표시 지연이 아니며 Jetson 성능도 아니다.** [환경](docs/evidence/realtime/environment.json) · [프레임 집계](docs/evidence/realtime/comparison.csv)

당시 PASS는 180초 이상, 평균≥29.7fps, 각 10초≥29.4fps, 처리 P99≤33.333ms, 예산 초과·누락≤1%, 수신→출력 P99≤66.667ms, 출력 간격 P99≤50ms·최대≤100ms, 오류 없음·정상 종료를 모두 요구했다. CPU는 평균이 30에 가까워도 FAIL이다. [전체 조건과 판정](docs/evidence/realtime/realtime_results.json)

Graph/eager FP32의 120프레임 확률 최대 절대차는 `4.76837158e-7`, 이진 경계 불일치 0%, IoU 1이다. FP16 정확도 절충으로 얻은 수치가 아니다. 초기 CPU 시험과는 runner·환경이 달라 단순 속도 개선 배수를 계산하지 않는다. [동등성 검증](docs/evidence/realtime/quality.json)

## 한계와 다음 단계

아래는 **후속 실험 제안과 채택 기준**이다. 현재 수행된 현장·장비 검증으로 제시하지 않는다.

| 우선순위 | 다음 실험 | 통과·채택 기준 |
|---|---|---|
| P0 · 입력 계약 | 실제 device ID·part number·encoding·AGC/FFC·timestamp·보정 정보 기록 | HUD 전 입력과 표시 결과를 분리 저장하고 모든 frame의 출처·설정·시각을 추적 |
| P0 · 평가 데이터 | 호스·단차·개구부·빈 장면의 boundary 정답과 현장/세션 분리 | 누락률·precision/recall 및 분할 목록을 고정하고 test를 threshold 선정에 사용하지 않음 |
| P1 · 표시 선택 | 같은 경계에서 1px/2px·배경 대비를 실제 안경으로 비교 | 필요한 경계 누락을 늘리지 않으며 통행 판단·반응시간·가림을 사전 기준으로 개선 |
| P1 · 실측 거리 | 알려진 거리 표적과 등록·가림·누락·stale·time reset 시험 | 거리 오차와 pixel registration 오차를 별도 보고; 무효 입력이 먼 물체로 오해되지 않는 표시 확인 |
| P2 · 목표 장비 | Jetson·카메라·거리 센서·안경 연결 상태의 180초 이상 시험 | 기존 지속성 gate 전체 충족 및 열·전원·drop·재연결 기록; 노출→표시 지연은 별도 계측 |
| P2 · 복원 | 정확한 split·weight·commit·환경을 묶어 새 출력에 재실행 | 해시가 맞는 원 실험과 재수집·재학습한 새 실험을 구분해 비교 |

현재 데이터에는 장애물 경계 정답, 연기 농도/가시거리 GT, 실제 거리 등록·동기화, 안경 사용자 시험이 없다. 정확한 센서 variant도 미확정이다. 표시가 매끈하고 처리량이 높다는 사실만으로 장애물 회피나 온도·거리 계측의 정확성을 주장하지 않는다.

## 재현과 자료 안내

| 목적 | 문서·폴더 |
|---|---|
| 입력과 선택 맥락 | [입력](docs/01-inputs.md) · [처리 과정](docs/02-process.md) |
| 수치와 해석 | [결과](docs/03-results.md) · [분석·후속 연구](docs/04-analysis.md) |
| 변경 전 기록 | [RESEARCH_REPORT.md](RESEARCH_REPORT.md) — 기존 README의 바이트 그대로 보존본 |
| 코드와 실행 범위 | [code/README.md](code/README.md) — 초기 원본과 후속 발췌의 역할 |
| 영상과 출처 | [MEDIA.md](MEDIA.md) · [SOURCE_CREDITS](docs/SOURCE_CREDITS.md) |
| 전체 개인 보존본 | [DATA_AND_RESTORE.md](DATA_AND_RESTORE.md) — 52장·휴대폰 원본·팀 코드·환경·미확보 목록 |
| 확인 범위 | [VALIDATION.md](VALIDATION.md) — 파일·구문·PLY·영상 디코드 감사 |

카메라 없이 확인하려면 별도 빈 작업 폴더에 `frame_0.png`와 PLY 변환 스크립트를 복사해 NumPy/OpenCV로 실행한다. stride 2는 81,920점, backup stride 4는 20,480점이 기준이다. 원본 보존 위치에서 실행하면 같은 이름 PLY를 덮어쓸 수 있으므로 작업 복제를 사용한다.

후속 경계 처리는 OpenCV contrib의 `ximgproc`, 정확한 TEED/PiDiNet 소스·weight·환경이 필요하다. 선별 runner만으로 전체 FireSight 앱이 바로 실행되지는 않는다. 조사한 Windows 체크아웃에서 일부 원 PNG·적응 weight·PiDiNet checkpoint·IFSI 원 영상이 확인되지 않았으며 새 다운로드가 원 hash와 다르면 새 실험으로 기록한다.

공개 clone은 전체 원본 백업이 아니다. 개인 보존본에는 직접 입력·PLY·휴대폰 원본, FireSight 전체 작업 상태와 이전 아카이브가 따로 있다. 로컬 파일 검증과 USB 전송·외부 사본 검증, 새 PC 전체 실행은 서로 다른 단계이며 후자의 완료를 이 README가 주장하지 않는다.
