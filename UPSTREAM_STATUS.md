# HaechiAR 최신 작업 반영

확인일: **2026-10-25 (KST)**. 이번에는 README 요약에 더해 HaechiAR의 실제 코드를
`haechiar/`에 가져왔습니다. **212개 파일, Python 소스 126개**이며 실행 코드·설정·테스트·
사용 문서와 TEED 체크포인트가 포함됩니다. [설치·실행](haechiar/README.md) ·
[파일별 원본 커밋과 해시](haechiar/SOURCE_MANIFEST.json) · [이관 검증](haechiar/VALIDATION.md)

## 코드 반영 범위

| 실행 경로 | 내용 |
|---|---|
| `haechiar/firesight_sw/data/capture/boson/` | uint16 raw 녹화·움직임 점검·세션 정리 |
| `haechiar/firesight_sw/edge/` | Nano/Orin TEED 추론·윤곽 안정화·고온 색상 HUD |
| `haechiar/firesight_sw/depth/` | MonoTher-Depth·AnyThermal·RIDERS 입력과 실행 wrapper |
| `haechiar/firesight_sw/experiments/` | UI 비교·재현·회귀 검사 |
| `haechiar/calib/` | 설치 가능한 내부 보정 패키지, RGB-D·열화상 AI·레이더 실험 |

기본 소프트웨어는 가장 최근의 `main`을 사용합니다. `firesight_sw/YSH`의 10월 23일
거리 HUD 개발본(`6271b35`)은 `main`의 최신 윤곽선 구조와 달라 덮어씌우지 않았습니다.
보정은 9월 24일 `YSH`가 가장 최근이며 기본 보정과 다중 센서 개발 코드를 함께 포함합니다.
원본의 폴더 관계와 가져온 파일 내용은 유지하고, 이 저장소에서 실행하는 안내를 별도로 추가했습니다.
실제 촬영 세션·로그·raw 원본과 원본 저장소 CI 설정은 이관 대상에서 제외했습니다.
이전 연구 자료는 기존 위치에 보존합니다.

## 기준 커밋

| 저장소 | 기준 | 커밋 날짜 | 반영 내용 |
|---|---|---|---|
| [HaechiAR/firesight_sw](https://github.com/HaechiAR/firesight_sw) | `main` · [`339af10`](https://github.com/HaechiAR/firesight_sw/commit/339af10729a5be5436c63cc74324585e1f97777e) | 2026-10-25 | 촬영 코드, 경계·고온 색상 HUD, 깊이 wrapper와 검증 스크립트 이관 |
| [HaechiAR/calib](https://github.com/HaechiAR/calib) | `main` · [`897c439`](https://github.com/HaechiAR/calib/commit/897c43902b67fd39a456d7745902299ae1468339) | 2026-09-14 | Boson 내부 보정과 보고서 검토·승인·적용 |
| [HaechiAR/calib](https://github.com/HaechiAR/calib) | `YSH` · [`659dca3`](https://github.com/HaechiAR/calib/commit/659dca33f0aaa964540aa4d5ca78879cc7c07af7) | 2026-09-24 | 다중 센서 수집·시간 정합·깊이 투영·재생 개발본 |

HaechiAR의 두 저장소는 확인 당시 비공개였습니다. 아래 원문 링크는 접근 권한이 있는 계정에서 열립니다. `calib/YSH`는 `main`과 구분한 개발 브랜치입니다.

## 1. Boson 320 raw 촬영

수집 경로는 flirpy의 `Boson.grab()`을 사용합니다. 원본은 **uint16, 256×320 배열의 TIFF**, 측정 수집률은 약 **8.57 FPS**입니다. 프레임, timestamp, 중복 프레임, 카메라 움직임, 이벤트, FFC 정보를 함께 기록합니다. 비방사형 카메라의 raw 카운트이며 섭씨 온도로 해석하지 않습니다.

10월 12일 기록은 16-bit 입력 경로 확인과 포그·카메라 고정 문제를 다뤘습니다. 포그머신 조건에서의 관찰은 실제 화재 연기 전체에 대한 결론으로 확대하지 않습니다. 10월 14일 2차 촬영 요약은 열원·핫팩·식힘·이동을 포함한 **18개 녹화를 사용 가능**으로 기록합니다. 이것은 촬영 정리 결과이며 레이블 완성이나 모델 평가 통과를 의미하지 않습니다.

FFC는 녹화 직전에 실행하고 녹화 중 수동 모드를 유지합니다. 따라서 장시간 raw 오프셋 변화와 상대 대비를 함께 고려해야 합니다. 사람·히터 이동 이벤트, 중복 프레임, 카메라 이동 후 제외 구간도 분석 시 구분합니다.

[수집 도구 원문](https://github.com/HaechiAR/firesight_sw/blob/339af10729a5be5436c63cc74324585e1f97777e/data/capture/boson/README.md) · [촬영 기록](https://github.com/HaechiAR/firesight_sw/tree/339af10729a5be5436c63cc74324585e1f97777e/data/collection)

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

[런타임 안내](https://github.com/HaechiAR/firesight_sw/blob/339af10729a5be5436c63cc74324585e1f97777e/edge/orin/README.md) · [측정 결과](https://github.com/HaechiAR/firesight_sw/blob/339af10729a5be5436c63cc74324585e1f97777e/docs/thermal_color_runtime_results.md)

## 3. 열화상 깊이 추정의 비교 기반

| 후보 | 입력 | 준비된 기능 |
|---|---|---|
| MonoTher-Depth | 열화상 | thermal branch 실행과 dense depth 저장 wrapper |
| AnyThermal + MiDaS | 열화상 | 공식 inference 형식 변환·실행 wrapper |
| RIDERS | 열화상 + 투영된 mmWave 깊이 | 입력 준비와 radar global-scale alignment 실험 |

열화상 단독 방법과 radar 추가 방법은 입력 조건이 다릅니다. wrapper 준비는 실제 거리 정확도·농연 일반화·Jetson 실시간 성능 검증과 별도입니다. 모델에 넣는 16-bit → 8-bit percentile 정규화 역시 온도 변환이 아닙니다.

[깊이 실험 안내](https://github.com/HaechiAR/firesight_sw/blob/339af10729a5be5436c63cc74324585e1f97777e/depth/README.md)

## 4. 내부 보정과 다중 센서 실험

`calib/main`은 원본 입력 또는 촬영, 내부 파라미터 계산, 독립 HTML 보고서 검토, 사용자 승인 후 실시간 왜곡 보정을 연결합니다. 호스트 영상 처리에 보정값을 적용하며 카메라 펌웨어·EEPROM에 값을 기록하는 방식은 아닙니다.

`calib/YSH`의 9월 24일 개발본은 다음 세 실험을 위한 공통 수집·보존·재생 구조를 추가했습니다.

- **RGB-D + 열화상:** 원본 수집, 시간 짝짓기, 깊이 역투영과 열영상 좌표의 Z-map.
- **열화상 단독 + AI:** 그룹 단위 데이터 분할, 추론 플러그인, 경계 비교 기반. 모델 미지정 기본 경계는 Canny입니다.
- **mmWave + 열화상:** XYZ 점군 보존·재생, 대응점 기반 강체변환, 희소 깊이 투영.

깊이 출력의 단위는 m이며 **열화상 카메라 Z축 깊이**입니다. 미관측 픽셀은 NaN으로 남깁니다. 호스트 수신 timestamp는 센서 노출 시각과 구분합니다.

보존된 개발 검증 기록은 **107 passed, 3 skipped**와 세 종류의 합성 CLI demo입니다. 실제 Boson·D456·radar 연결, 외부 보정 정확도, AI 학습, Jetson 성능은 이 기록으로 검증되지 않습니다. 이번 이관에서는 보정 테스트와 합성 데모를 다시 실행했습니다. 실제 재실행 결과는 [이관 검증 기록](haechiar/VALIDATION.md)에 구분해 남겼습니다.

[내부 보정 안내](https://github.com/HaechiAR/calib/blob/897c43902b67fd39a456d7745902299ae1468339/README.md) · [다중 센서 실험](https://github.com/HaechiAR/calib/blob/659dca33f0aaa964540aa4d5ca78879cc7c07af7/calib/docs/experiments.md) · [개발 검증 기록](https://github.com/HaechiAR/calib/blob/659dca33f0aaa964540aa4d5ca78879cc7c07af7/calib/docs/experiment_verification.md)

## 보존과 해석

기존 README 전체는 같은 루트의 [RESEARCH_DETAILS.md](RESEARCH_DETAILS.md)에 보존했습니다. 이미지·영상 파일, 초기 코드와 결과 자료는 그대로 유지합니다. 기존 보고서의 '현재'는 해당 보고서의 2026-09-16 확인 시점을 뜻합니다. 이번 최신 요약과 과거 실험을 하나의 실행 결과로 합치지 않습니다.

이번에는 최신 실행 코드와 함께 TEED 체크포인트를 가져왔습니다. 원본 raw 촬영 데이터와
외부 깊이 모델 전체·가중치는 포함하지 않으며 별도 준비가 필요합니다. 기존 그림·영상과
그 링크는 유지했습니다. HaechiAR 팀 코드와 외부 모델의 출처는 각 원문에 남겼습니다.

