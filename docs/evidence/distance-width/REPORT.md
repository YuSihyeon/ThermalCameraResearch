# 거리 기반 TEED 선 굵기 — 로컬 비교 보고서



2026-09-14 · YuSihyeon / YSH



연속 굵기 + 50ms 시간 필터를 로컬 기본 후보로 선택했다. 1m에서 5px, 8m에서 1px로 표시하며, 합성 정지면 시험의 작은 굵기 흔들림이 무필터 대비 44.6% 감소했다. 후처리 P95는 6.216ms이며 기존 고정 팽창 방식의 후처리 P95와의 차이는 +5.276ms다. 이번 CPU 비교에서 TEED 포함 P95는 43.6ms이고 30fps 처리 예산 초과율은 23.3%이므로, 안정적인 30fps를 확보했다고 판단하지 않았다.



**실제 열화상 원본에 적용한 거리 맵은 합성 가정값이다. 실제 거리 측정·추정 정확도를 검증한 영상이 아니다. 측정은 아래 환경 기록의 PC / CPU PyTorch 기준이며 Jetson Nano/Orin 또는 센서→디스플레이 지연은 미측정이다.**



## 성능 비교



| 방식 | 후처리 P50 ms | 후처리 P95 ms | 후처리 P99 ms | TEED 포함 P95 ms | 처리 FPS | 30fps 예산 초과 % | 60fps 예산 초과 % |

| --- | --- | --- | --- | --- | --- | --- | --- |

| 기존 TEED 고정 팽창 3 | 0.449 | 0.941 | 1.001 | 41.443 | 40.5 | 15.0 | 100.0 |

| 중심선 고정 3px | 2.298 | 4.324 | 4.796 | 43.856 | 37.2 | 22.5 | 100.0 |

| 거리 구간식 1·3·5px | 2.590 | 4.788 | 5.365 | 43.899 | 37.2 | 23.3 | 100.0 |

| 연속 굵기 / 시간 필터 없음 | 3.206 | 6.026 | 6.644 | 43.450 | 36.6 | 23.3 | 100.0 |

| 연속 굵기 / 50ms 시간 필터 | 3.285 | 6.216 | 6.606 | 43.583 | 36.6 | 23.3 | 100.0 |



320×240, OpenCV·PyTorch 각 2 threads. 실제 열화상 120프레임 / 10.0fps에 동일 TEED 확률맵·threshold 0.75를 사용했다. 모델 예열 5회, 후처리 예열 30회 후 방식별 600회, 처리 순서를 순환하며 측정했다. 후처리 P50/P95/P99는 인코딩 없는 별도 반복 측정으로, 세선화·거리 폭 계산·래스터화·배경 혼합을 포함한다. TEED 포함 수치는 실제 열화상 재생에서 공통 core 시간과 각 후처리를 합한 값이며, 측정 구간 사이 영상 저장에 따른 캐시/스케줄링 영향은 남는다. 캡처·깊이 추론·거리 로딩·화면 표시·인코딩 시간은 해당 수치에 포함하지 않는다. 30/60fps 예산 초과율은 처리 시간 기준이다. 별도 CLI 통합 재생은 120프레임을 처리했으며 처리시간 P95 49.6ms, 디코드·출력 저장 등을 포함한 실행 루프 처리율 27.8fps였다. 수치는 실행 시 PC 부하에 따라 변하며 비교표와 동일한 측정 구간은 아니다.



## 부드러움과 응답



| 방식 | 4.5m 노이즈 폭 표준편차 px | 3.28m 구간경계 노이즈 표준편차 px | 완만한 거리 변화 최대 폭 변화 px/frame |

| --- | --- | --- | --- |

| 거리 구간식 1·3·5px | 0.0000 | 0.9963 | 2.0000 |

| 연속 굵기 / 시간 필터 없음 | 0.1560 | 0.1367 | 0.0251 |

| 연속 굵기 / 50ms 시간 필터 | 0.0864 | 0.0755 | 0.0251 |



노이즈 시험은 30fps, 거리 노이즈 σ=0.18m, seed=914, 240프레임 중 초기 30프레임 제외다. 구간식은 구간 중앙(4.5m)에서는 흔들림이 0이지만, 구간 경계(3.28m)에서는 2px 점프를 만든다. 연속 방식은 경계 점프를 줄이고, 시간 필터는 작은 흔들림을 더 줄인다. 완만한 변화 시험은 1→8m를 240프레임에 걸쳐 변화시켰다. 표준편차는 거리 정확도가 아닌 표시 안정성 지표다.



시간 평활화는 미래 프레임을 기다리지 않는 EMA다. 30fps에서 완만한 변화의 저주파 응답 지연은 이론상 35.2ms다. 폭 변화가 0.75px보다 크면 즉시 적용하여 8m→1m 합성 step의 첫 프레임에서 5.0px를 확인했다. 반대로 큰 변화·결측·장면 전환은 부드러운 전환보다 최신 상태 반영을 우선한다.



## 구현과 입력



거리 맵은 미터 단위로 영상과 정렬된 float NPY 또는 timestamped scalar JSON을 받는다. 중심선에 지정된 폭을 사각 팽창과 알파 혼합으로 확장하므로 물체 바깥 깊이값이 선의 굵기를 결정하지 않는다. 깊이 불연속을 공간 블러로 섞지 않으며 현재 프레임의 선만 렌더링한다. 거리 결측·NaN/Inf/0·100ms 초과 입력·미래 입력은 far-px(기본 1px)로 복귀한다. far-px 복귀는 거리 판단이 아니라 입력 결측 표시 정책이다. NPY/JSON CLI는 파일 재생용이고 실제 센서 수집·시간 동기화·등록은 별도 통합이 필요하다.



가로·세로 선의 알파 합 기준 픽셀 폭이다. 사각 커널이므로 대각선·교차점의 기하학적 폭은 다르다. 기본 320×240 화면을 2배 확대하면 표시 폭도 2배가 된다.



## 영상



실제 열화상 + 가정 거리 6분할 (원본 프로젝트 참조: `thermal_assumed_comparison.mp4`; 선별본에는 미포함) · 합성 거리/노이즈/결측 시험 (원본 프로젝트 참조: `synthetic_comparison.mp4`; 선별본에는 미포함) · 최종 방식 단독 (원본 프로젝트 참조: `thermal_assumed_smooth.mp4`; 선별본에는 미포함)



| 파일 | 프레임 | FPS | 해상도 |

| --- | --- | --- | --- |

| synthetic_comparison.mp4 | 300 | 30.0 | 1920×1076 |

| thermal_assumed_comparison.mp4 | 120 | 10.0 | 1920×1076 |

| synthetic_smooth.mp4 | 300 | 30.0 | 640×538 |

| thermal_assumed_smooth.mp4 | 120 | 10.0 | 640×538 |



합성 영상: 0–3초 공간·시간 연속 변화, 3–5초 노이즈, 5–6초 급격한 근거리/원거리 변화, 6–7초 stale, 7–8초 결측 띠, 8–10초 움직이는 선.



## 개발 과정



1. GitHub 계정 YuSihyeon, 기존 YSH 및 작성자 이메일을 확인했다. 기존 이력을 보존한 로컬 YSH에 main 481569b를 병합했다.

2. 기존 edge/teed 고정 폭 경로를 유지하고 거리 모듈과 재생 입력을 별도 파일로 추가했다. 기존 Orin 코드는 수정하지 않았다.

3. 거리 단조성, 분수 폭, stale/future/invalid 입력, 작은 노이즈, 급격한 변화, 공간 깊이 경계, 영상 FPS 재생 계약을 검증했다.

4. 영상 생성 중 scalar 형태 처리 문제를 고쳤고, 코드 검토에서 발견한 화면 경계 세선화와 작은 연결 성분 보존을 보완했다.

5. RTSP/HTTP/장치 입력과 offline 거리 재생의 혼용을 차단했다. 알파 혼합은 현재 선이 있는 픽셀만 계산하도록 바꾸고 전체 픽셀 방식과 출력 일치를 검증했다.

6. 최종 코드로 벤치마크와 영상을 다시 생성하고 모든 출력 영상 전체 디코드·프레임 수·해시를 확인했다.

7. 개발 코드·과정·결과는 로컬에 정리했다. GitHub main 변경, push, PR, 외부 공유는 하지 않았다.



## 재현 및 후속 검증



저장소 루트에서 기존 Python 환경으로 실행한다.



```bash

python experiments/distance_width/benchmark.py --output ../outputs/distance_width_2026-09-14

python experiments/distance_width/build_report.py ../outputs/distance_width_2026-09-14

python -m unittest discover -s edge/teed/tests -v

python edge/teed/run_teed.py --source experiments/obstacle_edges/deliverables/media/smoke_heldout_original.mp4 --device cpu --width-mode distance --depth-replay ../outputs/distance_width_2026-09-14/thermal_ASSUMED_depth_m.npy --headless --output ../outputs/distance_width_2026-09-14/runtime_replay.mp4 --metrics ../outputs/distance_width_2026-09-14/runtime_metrics.json

```



장비 검증에는 영상과 정렬된 실측 거리 녹화가 필요하다. Jetson에서 후처리/모델 P95·열 스로틀링·프레임 드롭·센서→디스플레이 지연을 측정하고, 가까운 물체가 먼 배경을 가리는 경계의 거리 등록 오류를 별도로 평가한다.



## 환경·원자료·출처



Jianzhu Huai, SmokeBasement, Zenodo (2025), DOI: https://doi.org/10.5281/zenodo.15173055, CC BY 4.0. 데이터 readme는 Dezhong Chen의 수집 기여도 기록한다. 기존 run5 열화상 원본을 320×240·10fps로 재표본화한 로컬 클립에 TEED와 합성 거리 굵기를 적용했다. 원저자의 보증을 의미하지 않는다.



원자료: [summary.json](summary.json), [프레임별 측정 CSV](frame_metrics.csv), [후처리 600회 원자료](benchmark_raw.csv), [공통 TEED 측정](core_metrics.csv).



```json

{

  "platform": "Windows-11-10.0.26200-SP0",

  "processor": "Intel64 Family 6 Model 198 Stepping 2, GenuineIntel",

  "python": "3.13.10 (tags/v3.13.10:4fd8843, Dec  2 2025, 15:08:18) [MSC v.1944 64 bit (AMD64)]",

  "opencv": "4.13.0",

  "numpy": "2.5.3",

  "torch": "2.14.0+cpu",

  "cuda_available": false,

  "device": "cpu",

  "threads": 2,

  "resolution": [

    320,

    240

  ],

  "source_fps": 10.0,

  "source_frames": 120,

  "source_sha256": "3bb826c914c96572ad796664a3e1941dff09803523f63671229f615a546f3aa2",

  "checkpoint_sha256": "0322caf70f588355aaaf59c2bf5872b21a4b7e9f679971a7a3bb1f69b56a01ba",

  "depth_provenance": "synthetic sinusoidal maps, NOT measured or inferred scene distances",

  "source": "[local source path; see source-map.json]",

  "warmup_core_frames": 5,

  "benchmark_samples_per_mode": 600,

  "latency_scope": "CPU core + postprocessing; excludes capture/display/encode/depth estimation",

  "core_ms": {

    "mean": 24.00763333280338,

    "p50": 19.75885000138078,

    "p95": 40.66826501075411,

    "p99": 46.28454299963778,

    "max": 50.16099999193102

  }

}

```

