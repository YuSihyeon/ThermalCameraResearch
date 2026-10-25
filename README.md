# Thermal Camera Research

**열화상 원본 수집부터 경계 표시, 센서 보정과 깊이 추정으로 이어지는 연구.**

농연 영상에서 벽·문틀·장애물의 경계를 읽기 쉽게 표시하는 방법을 탐색합니다. 기존 경계·거리 HUD 실험에 HaechiAR의 최신 Boson 촬영, 윤곽 안정화, 보정·센서 융합 작업을 연결해 정리했습니다.

[실행 코드](#실행-코드) · [최신 진행](#최신-진행) · [그림과 영상](#그림과-영상) · [검증 범위](#검증-범위) · [자료 안내](#자료-안내)

**코드 업데이트: 2026-10-25.** HaechiAR의 `firesight_sw/main` (`339af10`)과 `calib/YSH` (`659dca3`) 소스를 이 저장소에 가져왔습니다. 기존 이미지·영상과 연구 기록은 유지합니다. [설치·실행 안내](haechiar/README.md) · [커밋·반영 범위](UPSTREAM_STATUS.md)

<img src="docs/evidence/obstacle-edges/smoke-comparison.png" width="960" alt="공개 농연 열화상에 기존 TEED, 세선화, 구조 경계 처리, 제한적 adaptation과 PiDiNet을 적용한 비교">

**SmokeBasement 농연 영상의 경계 처리 비교.** 기존 TEED, 세선화, 구조 경계, 제한적 adaptation, PiDiNet의 화면을 비교합니다. 직접 Boson으로 촬영한 농연 실험은 아닙니다.

## 실행 코드

| 목적 | 시작 위치 |
|---|---|
| Boson 320 raw 촬영·세션 점검 | [촬영 도구](haechiar/firesight_sw/data/capture/boson/README.md) |
| TEED 윤곽선·시간 안정화·고온 색상 HUD | [윤곽선 런타임](haechiar/firesight_sw/edge/orin/README.md) |
| MonoTher-Depth·AnyThermal·RIDERS 비교 | [깊이 추정 wrapper](haechiar/firesight_sw/depth/README.md) |
| Boson 내부 보정·RGB-D/레이더 융합 | [보정 패키지](haechiar/calib/README.md) · [센서 실험](haechiar/calib/calib/docs/experiments.md) |

**[처음 실행하기 →](haechiar/README.md)** 가상환경 설치, 카메라 없는 합성 데모,
기존 공개 영상의 CPU 처리와 검사 명령을 한곳에 정리했습니다.
기존 `code/`는 과거 실험 보존본이며, 최신 실행 코드는 `haechiar/`에 있습니다.

## 최신 진행

| 영역 | 최신 저장소에서 확인한 내용 | 현재 범위 |
|---|---|---|
| Boson 원본 촬영 | flirpy 기반 320×256 uint16 TIFF, 약 8.57 FPS 수집과 메타데이터 기록 | 비방사형 raw 카운트, 절대온도 아님 |
| 2차 촬영 · 10/14 | 열원·핫팩·식힘·이동을 포함한 18개 녹화 | 촬영 요약의 사용 가능 판정, 모델 평가 전 |
| 경계·HUD 개선 | 짧은 경계 제거, 시간·윤곽 안정화, 고온 색상 보존·빨간 외곽선 | 최신 CPU 시험의 전체 처리율 2.47–3.48 FPS |
| 깊이 추정 준비 | MonoTher-Depth, AnyThermal, RIDERS wrapper | 비교 실행을 위한 기반, 정량 우열 결과 없음 |
| Boson 내부 보정 | 원본 입력·촬영 → 보정 → 보고서 검토 → 사용자 승인 후 적용 | 실제 타깃·장비 반복 정확도 확인 필요 |
| 센서 융합 개발 | RGB-D+열화상, 열화상 AI, mmWave+열화상 수집·재생 구조 | `calib/YSH` 개발본, 합성 데모·소프트웨어 검사 |

원본 저장, 실제 센서 계측, 모델 평가를 구분합니다. 최신 기능과 이전 GPU benchmark는 장비·처리 경로가 달라 직접적인 성능 전후 비교로 사용하지 않습니다. [최신 내용과 출처](UPSTREAM_STATUS.md)

## 연구 흐름

```text
열화상 입력·원본 보존
  → 경계 검출과 표시: TEED · 세선화 · 안정화 · HUD
  → 보정과 시간·좌표 정합
  → 깊이 추정 / RGB-D·레이더 융합 실험
  → 표시 품질 · 지연 · 장비 조건별 평가
```

| 단계 | 이 저장소에서 볼 수 있는 자료 |
|---|---|
| 초기 입력 | 640×512 uint8 프레임과 표시·저장 코드 |
| 밝기 기반 점군 | 영상 밝기를 높이로 표현한 PLY 계산·검산 |
| 경계 선택 | 엣지 점유율, 구조 경계·세선화 비교 |
| 거리 HUD | 합성 거리의 연속 선 굵기와 EMA 실험 |
| 지속 처리 | 180초 CPU/CUDA 시험의 수치·영상 |
| 후속 개발 | HaechiAR 최신 실행 코드·설치 안내·파일별 원본 해시 |

## 그림과 영상

기존 이미지와 영상은 원래 파일·링크를 유지합니다. 영상 이미지를 누르면 해당 MP4가 열립니다.

### 초기 입력과 밝기 기반 점군

<img src="docs/evidence/inputs/frame_0.png" width="640" alt="두 밝기 기반 PLY 파일의 계산 입력인 640×512 grayscale frame 0">

초기 계산 입력은 640×512 단일 채널 uint8입니다. 최신 Boson 320의 16-bit raw 촬영과 다른 입력입니다.

<img src="docs/evidence/pointcloud/floor-view.png" width="850" alt="밝기 기반 PLY를 CloudCompare에서 열어 관찰한 화면">

밝기를 높이로 바꾼 PLY 시각화입니다. 실제 깊이나 물체 표면의 3D 계측 결과는 아닙니다.

### Boson HUD 표시

[<img src="public-media/boson-test.jpg" width="640" alt="어두운 배경 위에 이미 초록 윤곽이 합성된 Boson HUD의 한 시점">](public-media/boson-test.mp4)

초록 윤곽이 이미 합성된 표시 영상을 관찰한 기록입니다. [영상](public-media/boson-test.mp4)

### 농연 입력과 구조 경계

[<img src="public-media/smoke-original.jpg" width="640" alt="모델 비교와 표시 실험에 공통으로 사용한 SmokeBasement 농연 열화상 입력">](public-media/smoke-original.mp4)

모델 비교에 사용한 SmokeBasement 입력. [영상](public-media/smoke-original.mp4)

[<img src="public-media/smoke-structural.jpg" width="640" alt="SmokeBasement 입력에서 강한 구조와 연결된 경계를 선택하고 세선화한 TEED 출력">](public-media/smoke-structural.mp4)

강한 구조와 연결된 경계를 선택하고 세선화한 TEED 출력. [영상](public-media/smoke-structural.mp4)

### 표시 폭과 거리 HUD

[<img src="public-media/smoke-balanced-2px.jpg" width="760" alt="동일 확률맵에서 선택한 Balanced 경계를 최종 640×480 화면에 2px로 표시한 결과">](public-media/smoke-balanced-2px.mp4)

같은 확률맵에서 선택한 Balanced 경계를 최종 화면에 2px로 표시했습니다. [영상](public-media/smoke-balanced-2px.mp4)

[<img src="public-media/distance-preview.jpg" width="900" alt="같은 Balanced 경계를 고정 2px와 임의 원근에 따른 1·2·3px로 비교한 화면">](public-media/distance-preview.mp4)

고정 2px와 임의 원근에 따른 1·2·3px 비교입니다. 사용한 거리는 실측값이 아닙니다. [영상](public-media/distance-preview.mp4)

[<img src="public-media/realtime-comparison.jpg" width="720" alt="같은 공개 열화상과 TEED 경계에 고정 폭, 합성 거리 구간식, 연속 폭과 EMA를 적용한 네 화면">](public-media/realtime-comparison.mp4)

고정 폭·합성 거리 구간식·연속 폭·EMA를 비교합니다. [영상](public-media/realtime-comparison.mp4)

## 검증 범위

| 기록 | 확인한 결과 | 해석 |
|---|---|---|
| 기존 경계 비교 | TEED 점유율 12.27% → 구조 경계 1.51% | 화면 가림 지표, 검출 정확도 아님 |
| 합성 거리 실험 | 연속 폭 + EMA로 구간 경계의 폭 흔들림 감소 | 거리 입력을 가정한 표시 실험 |
| 9월 GPU 지속 처리 | CUDA / CUDA Graph에서 180초·약 30 FPS 기준 통과 | 해당 RTX PC와 호스트 처리 구간 |
| 10/8 최신 색상·윤곽 처리 | CPU 전체 2.47–3.48 FPS, 15/30 FPS 조건 미충족 | 다른 CPU·후처리·입력 조건 |
| 9/24 센서 융합 개발 | 보존 기록 107 passed, 3 skipped | 합성·개발 환경 검사, 실장비 성능 아님 |

위 표는 기존 연구 기록입니다. 2026-10-25 코드 이관 후의 소프트웨어 검사와 실행 결과는
[별도 검증 기록](haechiar/VALIDATION.md)에 정리했습니다. 실제 카메라·Jetson 성능과 모델 학습은 검증 범위에 포함하지 않습니다.

## 자료 안내

- [최신 HaechiAR 작업과 출처](UPSTREAM_STATUS.md)
- [최신 코드 설치·실행](haechiar/README.md) · [이관 검증](haechiar/VALIDATION.md)
- [상세 연구 기록](RESEARCH_DETAILS.md): 기존 실험의 입력·방법·수치·해석 전체
- [입력](docs/01-inputs.md) · [처리 과정](docs/02-process.md) · [결과](docs/03-results.md) · [분석](docs/04-analysis.md)
- [이전 코드 보존본](code/README.md) · [영상 정보](MEDIA.md) · [기존 검증 기록](VALIDATION.md)
- [데이터·환경 복원](DATA_AND_RESTORE.md) · [이전 보고서](RESEARCH_REPORT.md)

**출처.** Boson HUD를 제외한 공개 영상 5개는 SmokeBasement — Jianzhu Huai (2025), CC BY 4.0의 처리 파생물입니다. [원 데이터](https://doi.org/10.5281/zenodo.15173055) · [출처와 변환 내역](docs/SOURCE_CREDITS.md). TEED·PiDiNet과 각 깊이 모델은 외부 기반이며 팀 시스템과 개인 연구 기록의 범위를 구분합니다.

