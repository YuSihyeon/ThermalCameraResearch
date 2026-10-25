# 원본 고온 색상 보존 기능의 실제 처리 속도

측정일: 2026-10-08. 최종 운영 HUD에서 원본 고온 색상 보존과 빨간 외곽선을 활성화하면 이 PC의 전체 처리율은 **2.47–3.48 FPS**였다. 15/30 FPS 실시간 조건은 충족하지 못했다. 출력 동영상의 29.97 FPS는 재생 타임라인이며 처리 성능이 아니다.

## 측정 조건

Intel Core Ultra 7 258V, Windows, PyTorch 2.13.0+cpu, OpenCV 4.14.0, CUDA 비활성. Torch/OpenCV 4스레드, 디코더 1스레드, 배치 1이다. 3840×2160 원본 읽기 → TEED 320×256 → 안정화·윤곽 추적·HUD 720×405를 순차 실행했다. 모든 전체 영상 생성 작업이 종료된 뒤 측정했다.

조건마다 전체 파이프라인 20프레임 예열 후 120프레임을 측정했다. IFSI10은 467–586, IFSI8은 90–209프레임이다. 추론 캐시, 프레임 건너뛰기, 재생 속도 제한은 없으며 두 조건 모두 윤곽 추적과 같은 OSD 제외 영역을 사용한다.

## 결과

| 영상 | 원본 고온 색상·빨간 외곽선 | 전체 FPS | 평균 읽기 | 평균 TEED | 평균 HUD | 전체 지연 p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fire360_ifsi10 | 꺼짐 | 2.56 | 33.0 ms | 39.3 ms | 319.0 ms | 476.7 ms |
| fire360_ifsi10 | 켜짐 | 2.47 | 33.1 ms | 39.1 ms | 332.5 ms | 500.3 ms |
| fire360_ifsi8 | 꺼짐 | 3.60 | 32.1 ms | 40.1 ms | 205.6 ms | 360.9 ms |
| fire360_ifsi8 | 켜짐 | 3.48 | 32.5 ms | 39.6 ms | 215.5 ms | 354.6 ms |

색상 합성 구간 자체는 평균 11.2–14.8 ms였고, 기능을 켠 HUD 전체는 215.5–332.5 ms였다. TEED 신경망 연산 자체는 평균 약 22–24 ms였다. 이 결과에서는 윤곽 안정화와 HUD 후처리가 전체 속도에 크게 기여하며, 모델 연산 시간만으로 실시간 성능을 판단할 수 없다. 색상 합성 타이머는 일부 공통 배경 처리를 제외하므로 정확한 추가 비용은 위의 기능 꺼짐/켜짐 전체 결과와 함께 해석해야 한다. 단일 실행 비교이며 반복 측정 오차 범위는 산출하지 않았다.

모든 조건의 측정 프레임이 15 FPS(66.7 ms)와 30 FPS(33.3 ms) 시간 예산을 초과했다.

## 범위

- 영상 읽기, TEED, HUD 버퍼 생성까지 포함한다. 실제 카메라 캡처, 디스플레이 스캔아웃, 영상 저장 인코딩은 제외했다.
- 영상의 원래 프레임 시간으로 안정화 상태를 갱신한다. 실시간 입력의 버퍼 누적이나 오래된 프레임 폐기 정책을 검증한 시험은 아니다.
- 프로세스 CPU affinity는 고정하지 않았다. Jetson/Orin, 라이브 카메라, 실물 AR/HMD는 측정하지 않았다.
- 소스 코드 해시와 프레임별 시간은 로컬 생성 기록 `experiments/teed_ui/thermal_color_runtime/benchmark.json`에 기록했다. 재현 스크립트는 [benchmark_thermal_color_runtime.py](../experiments/teed_ui/benchmark_thermal_color_runtime.py)이다.
- [기능 추가 전 측정](thermal_realtime_results.md)은 별도 역사 기록이며, 다른 영상·시점의 숫자를 직접 성능 개선율로 비교하지 않는다.

## 실제 실행 명령 검증

운영 진입점 `run_pipeline.py`에 `--contour-edges --thermal-palette-overlay --profiles edge/teed/profiles/fire360_video_profiles.json --headless`를 지정해 IFSI10의 467–586프레임을 실행했다. OSD 제외 영역 2개가 적용되고 120프레임 처리가 완료됐다. 처리율은 **2.50 FPS**, 추론+HUD 평균 372.6 ms, p95 442.2 ms였다. 이 CLI의 `e2e`는 영상 읽기·실제 화면 표시를 제외하므로 센서부터 화면까지의 지연이 아니다.

실행 기록과 검증·코드 해시는 로컬 생성 기록 `experiments/teed_ui/thermal_color_runtime/cli_smoke.json`, `experiments/teed_ui/thermal_color_runtime/cli_checks.json`에 저장했다. 재현 스크립트는 [check_thermal_color_cli.py](../experiments/teed_ui/check_thermal_color_cli.py)다. 프로필이 빠진 초기 점검은 `thermal_color_runtime/unprofiled_cli`에 보존했으며 최종 수치로 사용하지 않았다. 이 JSON 산출물은 GitHub 배포 대상이 아니다.
