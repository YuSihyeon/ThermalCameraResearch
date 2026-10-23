# HaechiAR 최신 작업 반영

확인일: **2026-10-23 (KST)**. 이 문서는 ThermalCameraResearch의 첫 화면을 최신 연구 흐름에 맞추기 위한 출처 기록입니다. 최신 내용을 README와 이 문서에 요약했으며, 팀 저장소 전체의 코드·데이터를 복제한 배포본은 아닙니다.

## 기준 커밋

| 저장소 | 기준 | 커밋 날짜 | 반영 내용 |
|---|---|---|---|
| [HaechiAR/firesight_sw](https://github.com/HaechiAR/firesight_sw) | `main` · [`dbb9a8d`](https://github.com/HaechiAR/firesight_sw/commit/dbb9a8d7d6741edb104107d050c5fab220986a76) | 2026-10-15 | Boson 수집·촬영 기록, 경계·고온 색상 HUD, 깊이 wrapper |
| [HaechiAR/calib](https://github.com/HaechiAR/calib) | `main` · [`897c439`](https://github.com/HaechiAR/calib/commit/897c43902b67fd39a456d7745902299ae1468339) | 2026-09-14 | Boson 내부 보정과 보고서 검토·승인·적용 |
| [HaechiAR/calib](https://github.com/HaechiAR/calib) | `YSH` · [`659dca3`](https://github.com/HaechiAR/calib/commit/659dca33f0aaa964540aa4d5ca78879cc7c07af7) | 2026-09-24 | 다중 센서 수집·시간 정합·깊이 투영·재생 개발본 |

HaechiAR의 두 저장소는 확인 당시 비공개였습니다. 아래 원문 링크는 접근 권한이 있는 계정에서 열립니다. `calib/YSH`는 `main`과 구분한 개발 브랜치입니다.

## 1. Boson 320 raw 촬영

수집 경로는 flirpy의 `Boson.grab()`을 사용합니다. 원본은 **uint16, 256×320 배열의 TIFF**, 측정 수집률은 약 **8.57 FPS**입니다. 프레임, timestamp, 중복 프레임, 카메라 움직임, 이벤트, FFC 정보를 함께 기록합니다. 비방사형 카메라의 raw 카운트이며 섭씨 온도로 해석하지 않습니다.

10월 12일 기록은 16-bit 입력 경로 확인과 포그·카메라 고정 문제를 다뤘습니다. 포그머신 조건에서의 관찰은 실제 화재 연기 전체에 대한 결론으로 확대하지 않습니다. 10월 14일 2차 촬영 요약은 열원·핫팩·식힘·이동을 포함한 **18개 녹화를 사용 가능**으로 기록합니다. 이것은 촬영 정리 결과이며 레이블 완성이나 모델 평가 통과를 의미하지 않습니다.

FFC는 녹화 직전에 실행하고 녹화 중 수동 모드를 유지합니다. 따라서 장시간 raw 오프셋 변화와 상대 대비를 함께 고려해야 합니다. 사람·히터 이동 이벤트, 중복 프레임, 카메라 이동 후 제외 구간도 분석 시 구분합니다.

[수집 도구 원문](https://github.com/HaechiAR/firesight_sw/blob/dbb9a8d7d6741edb104107d050c5fab220986a76/capture/boson/README.md) · [촬영 기록](https://github.com/HaechiAR/firesight_sw/tree/dbb9a8d7d6741edb104107d050c5fab220986a76/docs/data_collection)

## 2. 경계 안정화와 고온 색상 표시

최신 Orin 경로에는 짧은 경계 제거, 시간·윤곽 안정화, 원본 고온 색상 보존과 빨간 외곽선 표시가 포함됩니다. 색상은 열화상 표시 영상의 색을 보존하는 것이며 온도 추정이나 화재 판정 결과가 아닙니다.

2026-10-08 기록은 Intel Core Ultra 7 258V, PyTorch CPU 환경에서 원본 읽기·TEED·안정화·HUD 생성까지 측정했습니다. 예열 20프레임 이후 120프레임의 단일 실행 비교입니다.

| 입력 | 색상·빨간 외곽선 | 전체 처리율 |
|---|---|---:|
| IFSI10 | 꺼짐 | 2.56 FPS |
| IFSI10 | 켜짐 | 2.47 FPS |
| IFSI8 | 꺼짐 | 3.60 FPS |
| IFSI8 | 켜짐 | 3.48 FPS |

이 경로는 15/30 FPS 시간 예산을 충족하지 못했습니다. 실제 카메라 노출, 디스플레이 스캔아웃, 인코딩은 측정 범위 밖입니다. 기존 9월 CUDA 지속 처리 결과는 다른 장비와 경로이므로 이 표와 직접 비교하지 않습니다.

[런타임 안내](https://github.com/HaechiAR/firesight_sw/blob/dbb9a8d7d6741edb104107d050c5fab220986a76/edge/orin/README.md) · [측정 결과](https://github.com/HaechiAR/firesight_sw/blob/dbb9a8d7d6741edb104107d050c5fab220986a76/docs/thermal_color_runtime_results.md)

## 3. 열화상 깊이 추정의 비교 기반

| 후보 | 입력 | 준비된 기능 |
|---|---|---|
| MonoTher-Depth | 열화상 | thermal branch 실행과 dense depth 저장 wrapper |
| AnyThermal + MiDaS | 열화상 | 공식 inference 형식 변환·실행 wrapper |
| RIDERS | 열화상 + 투영된 mmWave 깊이 | 입력 준비와 radar global-scale alignment 실험 |

열화상 단독 방법과 radar 추가 방법은 입력 조건이 다릅니다. wrapper 준비는 실제 거리 정확도·농연 일반화·Jetson 실시간 성능 검증과 별도입니다. 모델에 넣는 16-bit → 8-bit percentile 정규화 역시 온도 변환이 아닙니다.

[깊이 실험 안내](https://github.com/HaechiAR/firesight_sw/blob/dbb9a8d7d6741edb104107d050c5fab220986a76/depth/README.md)

## 4. 내부 보정과 다중 센서 실험

`calib/main`은 원본 입력 또는 촬영, 내부 파라미터 계산, 독립 HTML 보고서 검토, 사용자 승인 후 실시간 왜곡 보정을 연결합니다. 호스트 영상 처리에 보정값을 적용하며 카메라 펌웨어·EEPROM에 값을 기록하는 방식은 아닙니다.

`calib/YSH`의 9월 24일 개발본은 다음 세 실험을 위한 공통 수집·보존·재생 구조를 추가했습니다.

- **RGB-D + 열화상:** 원본 수집, 시간 짝짓기, 깊이 역투영과 열영상 좌표의 Z-map.
- **열화상 단독 + AI:** 그룹 단위 데이터 분할, 추론 플러그인, 경계 비교 기반. 모델 미지정 기본 경계는 Canny입니다.
- **mmWave + 열화상:** XYZ 점군 보존·재생, 대응점 기반 강체변환, 희소 깊이 투영.

깊이 출력의 단위는 m이며 **열화상 카메라 Z축 깊이**입니다. 미관측 픽셀은 NaN으로 남깁니다. 호스트 수신 timestamp는 센서 노출 시각과 구분합니다.

보존된 개발 검증 기록은 **107 passed, 3 skipped**와 세 종류의 합성 CLI demo입니다. 실제 Boson·D456·radar 연결, 외부 보정 정확도, AI 학습, Jetson 성능은 이 기록으로 검증되지 않습니다. 이 문서 업데이트에서 테스트를 다시 실행하지는 않았습니다.

[내부 보정 안내](https://github.com/HaechiAR/calib/blob/897c43902b67fd39a456d7745902299ae1468339/README.md) · [다중 센서 실험](https://github.com/HaechiAR/calib/blob/659dca33f0aaa964540aa4d5ca78879cc7c07af7/calib/docs/experiments.md) · [개발 검증 기록](https://github.com/HaechiAR/calib/blob/659dca33f0aaa964540aa4d5ca78879cc7c07af7/calib/docs/experiment_verification.md)

## 보존과 해석

기존 README 전체는 같은 루트의 [RESEARCH_DETAILS.md](RESEARCH_DETAILS.md)에 보존했습니다. 이미지·영상 파일, 초기 코드와 결과 자료는 그대로 유지합니다. 기존 보고서의 '현재'는 해당 보고서의 2026-09-16 확인 시점을 뜻합니다. 이번 최신 요약과 과거 실험을 하나의 실행 결과로 합치지 않습니다.

이 공개 저장소에는 원본 raw 전체, 조직 내부 저장 위치, 팀 저장소의 전체 코드·checkpoint를 새로 복제하지 않았습니다. 팀 작업은 HaechiAR의 출처로 표시하며 개인 단독 구현으로 귀속하지 않습니다.

