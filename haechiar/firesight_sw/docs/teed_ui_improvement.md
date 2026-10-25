# TEED UI 개선 및 고온 등온선

기준: `origin/main` `481569b7412522f9f6f90cde24b320538dce1e13` (2026-10-08).
작업 폴더 `firesight_sw_ui`, 브랜치 `codex/teed-ui-high-temperature`.
기존 작업 폴더의 미커밋 수정은 유지했다. 이번 수정 대상은 최신 **Orin 실행기 `edge/orin/run_pipeline.py`**이며 원형 Nano `edge/teed/run_teed.py`는 변경하지 않았다.

## 현황

- main의 `TEEDCore.infer`: 리사이즈/CLAHE → TEED fused logits → sigmoid → 확률 0.75 임계값 → 기본 3×3 팽창 → OSD 제거. 짧은 연결선·거리 필터는 없다.
- main의 `HUDRenderer`: RGB/HSV 색 조건과 엣지 중첩을 빨간색으로 표시했다. 실제 온도 판단이 아니다.
- 별도 로컬 `firesight_sw_adaptive_hud/edge/orin/adaptive_display.py`: `max(210, median+25)` 밝기 또는 warm-hue/neutral-bright 조건으로 열 후보를 고른다. 지적된 상대 밝기 문제에 해당한다. 해당 폴더는 다른 변경이 있어 직접 수정하지 않았다.

## 구현

1. TEED 확률 low/high 0.55/0.75와 grayscale Sobel gradient 2/6 intensity/pixel를 결합한 연결 hysteresis. 강한 확률은 저대비에서도 보존한다. gradient는 화면 대비이며 온도차가 아니다.
2. 추가 의존성 없이 Zhang–Suen 중심선 추출 후 cardinal/diagonal 연결 길이 12 model-pixel 미만의 성분 제거. 큰 구조에 붙은 짧은 가지는 별도 가지치기하지 않는다. OSD를 연결 판단 전과 후에 제외한다.
3. 정합된 미터 단위 depth가 있으면 역거리 불투명도: 0.6m 이내 1.0, 6m 이상 0.25. `--far-alpha 0`이면 먼 선을 숨길 수 있다. depth 없음·0·음수·NaN·Inf는 미상으로 유지해 불투명도 1.0. 실제 영상에서 깊이를 새로 추정하지 않는다.
4. 색/밝기 기반 화점 경로 제거. Celsius 온도맵이 있을 때만 기준 이상의 영역에 등온선을 표시한다. `contours`가 기본, `overlay`는 노랑→주황→빨강 반투명 채움 추가. 온도 없으면 `TEMP N/A`. 개발자 패널은 기본 숨김이며 `--show-panel`로 표시한다.

150/260/400°C는 기능 확인용 초기 설정이며 **NFPA 지정값이나 검증된 안전 기준이 아니다**. 고온 표시와 발화점·화재 판정은 다르다.

## Fire360 카메라·온도 형식

[Fire360 논문 §3](https://arxiv.org/html/2506.02167v1#S3)은 360° 촬영 장비를 Ricoh Theta V, 3840×1920/60fps로 설명한다. 이는 로컬 IFSI 영상의 열화상 화면을 생성한 카메라 모델을 특정하는 근거가 아니다. 해당 열화상 카메라 모델은 확인되지 않았다.

보유한 `data/fire360/raw`의 7개 MP4/MTS는 uint8 3채널 BGR 표시 영상으로 읽힌다. 색상표·숫자가 있어도 픽셀별 온도 변환식, gain, palette, 방사측정 보정 메타데이터는 확인되지 않았다. 밝기나 화면 최저/최고 숫자만으로 Celsius를 만들지 않았다. **현재 Fire360에서 실제 고온만 표시하는 검증은 온도 원본/보정정보 없이는 불가능하다.**

Y16도 그 자체로 온도를 뜻하지 않는다. 이번 구현은 명시적인 Celsius sidecar만 받으며 Y16 백분위 정규화 영상을 온도맵으로 재사용하지 않는다. SmokeBasement의 16-bit temperature PNG도 단위 보정 근거가 확인되지 않아 Celsius로 사용하지 않았다.

## 실행

저장소 루트에서 기존 NumPy/OpenCV/PyTorch 환경을 사용한다.

~~~bash
python edge/orin/run_pipeline.py --source "../data/fire360/raw/IFSI Video 4 (2).mp4" --profiles edge/teed/profiles/fire360_video_profiles.json --device cpu --width 320 --height 240 --headless --max-frames 60 --record --record-path filtered.mp4 --metrics filtered.json
~~~

보드에서는 환경에 맞게 `--device cuda --half`. 이번 실측은 CPU이며 Orin 성능을 의미하지 않는다. `--legacy-edges`는 기존 팽창 엣지를 표시하지만 색 기반 화점 경로를 다시 켜지는 않는다. 완전한 기존 화면 비교는 평가 스크립트가 기준 커밋의 HUD 코드를 읽어 수행한다.

~~~bash
python edge/orin/run_pipeline.py --source calibrated.mp4 --aligned-maps calibrated_maps.npz --thermal-mode overlay --heat-levels-c 150 260 400 --headless
~~~

## 부가 입력 스키마

NPZ, pickle 불허. 온도/거리 중 하나 이상이 필요하다.

| 키 | 형식 |
|---|---|
| `registered_to` | scalar string `source_image` |
| `source_sha256` | 해당 원본 파일의 SHA-256 |
| `frame_indices` | 중복 없는 원본 프레임 번호, 0 이상 정수 N개 |
| `temperature_c` | 선택 float N,H,W, 원본 영상 좌표계 섭씨 온도 |
| `temperature_c_unit` | scalar string `Celsius` |
| `temperature_c_provenance` | 센서/온도 보정/프레임 정합 근거 문자열 |
| `depth_m` | 선택 float N,H,W, 원본 좌표로 정합된 미터 거리 |
| `depth_m_unit` | scalar string `m` |
| `depth_m_provenance` | 거리 측정/정합 근거 문자열 |

파일 해시·누락 프레임·단위·shape 불일치는 오류다. 보정과 정합은 부가 데이터 생산자가 보증해야 하며 문자열로 물리적 보정이 검증되지는 않는다. sidecar는 파일 재생용이며 live camera/loop/stereo-half/시작 프레임 건너뛰기와 함께 사용하지 않는다.

## 산출물·검증 한계

- [알고리즘 현황 PNG](teed_ui_validation/teed_algorithm_status.png)
- [고온 표시 설명 PNG](teed_ui_validation/temperature_contract.png): 합성 온도맵이며 Fire360 실측이 아니다.
- 35개 동작 계약 확인, 실제 Fire360 CLI 결과, 합성 온도맵 CLI 결과는 로컬 생성 기록 `docs/teed_ui_validation/contract_checks.json`, `docs/teed_ui_validation/fire360_cli.json`, `docs/teed_ui_validation/synthetic_heat_cli.json`에 보존했다. 이 JSON은 GitHub 배포 대상이 아니다.
- `experiments/teed_ui/evaluate.py`: 실제 Fire360/SmokeBasement 재생과 비교 영상·지표.

기존 모델 경계와의 중첩은 **모델 기준 보존 proxy**이며 장애물 정답 recall이 아니다. 라벨과 사용자 회피 실험 없이 회피 성능 유지로 결론내리지 않는다. 정합 metric depth가 없는 실제 클립에서는 거리 감쇠도 실데이터로 검증하지 못했으며 코드 경계조건으로만 확인했다.

## 실측 결과

CPU 1개 스레드, TEED 320×256, 총 163프레임. 녹화는 비교용 샘플 재생 속도이며 실시간 처리 성능과 구분한다.

| 입력 | 프레임 | 기존 엣지 표시 면적 | 개선 표시 면적 | 기존 경계 중첩 proxy (2px) |
|---|---:|---:|---:|---:|
| Fire360 IFSI 4 | 40 | 20.14% | 4.29% | 83.28% |
| Fire360 IFSI 8 | 40 | 16.49% | 3.53% | 76.68% |
| Fire360 02815 | 23 | 9.24% | 1.88% | 81.57% |
| SmokeBasement run5 | 60 | 13.85% | 2.96% | 89.46% |
| 전체 평균 | 163 | 15.39% | 3.27% | 83.69% |

표시 면적 감소에는 3px 팽창 제거와 중심선 가늘게 만들기 효과가 함께 포함된다. 따라서 엣지 객체 개수가 같은 비율로 줄었다는 뜻이 아니다. 일부 기존 경계가 소실되므로 라벨 기반 경계 검증 전 회피 성능 유지의 근거로 쓰지 않는다. 모든 실데이터 프레임의 고온 강조 픽셀은 0이며 TEMP N/A였다.

비교 지표 원본과 대표 장면 비교 PNG는 로컬 생성 산출물 `experiments/teed_ui/results/teed_ui_metrics.json`, `experiments/teed_ui/results/teed_ui_compact_overview.png`, `experiments/teed_ui/results/teed_ui_multiscene_comparison.png`에 보존했다. 이 결과 파일은 GitHub 배포 대상이 아니다.

최종 비교는 양쪽에 동일한 OSD 제외 영역을 적용했다. 초기 중간 실행은 재구성한 기존 마스크에 OSD 제외를 빠뜨려 전체 기존 표시 면적 19.15%, proxy 72.07%로 잘못 집계했다. 이 값은 폐기하고 최종 15.39%, 83.69%를 사용한다. 개선 표시 면적 3.27%와 SmokeBasement 결과는 변하지 않았다.

후속 구현: [온도맵 없는 가시광 불꽃 후보 표시](visible_flame_candidates.md). 이전 절의 163프레임 비교는 불꽃 후보 기능을 끈 기준 결과로 유지한다.
