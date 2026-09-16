# 2. 처리 과정과 선택 이유

## 2.1 카메라 표시와 저장

[표시 코드](../code/original/flir_view_test.py)는 첫 카메라를 열어 BGR이면 gray로 변환한 뒤, 각 프레임을 최소/최대값 기준으로 0~255 정규화하고 INFERNO 컬러맵을 씌운다. ESC로 종료한다. 장치 열기 실패와 frame read 실패를 출력하는 분기는 있지만, 당시 실패가 실제 발생했는지 확인할 로그는 없다.

[캡처 코드](../code/original/thermal_capture.py)는 같은 표시 과정을 사용하고 `s` 키에서 **정규화·컬러맵 전의 gray 배열**을 PNG로 저장한다. 따라서 display 컬러와 저장된 PNG 값은 같은 배열이 아니다. 다만 카메라나 UVC driver 단계에서 이미 AGC 또는 8bit 변환했는지는 이 코드에 기록되어 있지 않다. 저장값을 센서 raw라고 부르지 않는다.

초기 선택의 장점은 OpenCV 하나로 연결·표시·저장을 빠르게 확인할 수 있다는 점이다. 이후 정량 실험을 하려면 device ID, pixel encoding/bit depth, FPS, exposure/AGC/FFC, timestamp, radiometric 변환 정보를 함께 저장해야 한다.

## 2.2 밝기 → PLY: 계산은 성공했으나 깊이 측정은 아니다

두 변환 스크립트는 동일 `frame_0.png`를 8bit 회색으로 읽는다. 구현한 식은 다음과 같다.

```text
I = gray[y, x] ∈ {0, ..., 255}
X = x                        # 이미지 픽셀 좌표
Y = y                        # 이미지 픽셀 좌표
Z = (float32(I) / 255) × 3000 # 밝기를 임의 높이로 확대
```

| 구현 | 간격 | 출력 | 색 |
|---|---:|---|---|
| [backup](../code/original/thermal_image_to_pointcloud_backup.py) | stride 4 | 20,480점 `thermal_pointcloud.ply` | 없음 |
| [현재본](../code/original/thermal_image_to_pointcloud.py) | stride 2 | 81,920점 `thermal_pointcloud_big.ply` | RGB 모두 원래 gray 값 |

stride를 반으로 줄이면 가로·세로 샘플 수가 각각 2배가 되어 점은 4배가 된다. `3000`은 구현상의 z 확대 계수이며 3m/3000mm/3000℃가 아니다. 실제 카메라 back-projection에 필요한 초점거리·주점·metric Z가 없고, thermal intensity→depth를 추정하는 모델도 없다.

그래서 결과는 한 장의 밝기를 높이로 세운 **height field**다. CloudCompare에서 보는 방향에 따라 벽·바닥처럼 느껴질 수 있지만, 기하학적으로 실제 벽과 바닥을 복원했다는 증거는 아니다. 서로 다른 물체가 같은 밝기를 가지면 같은 Z가 되고, 같은 평면에서도 밝기가 다르면 기복이 생긴다.

## 2.3 별개 탐색: 합성 radar 평면 분리

[FireSight radar 코드](../code/original/firesight_radar_core.py)의 실제 `__main__`은 바닥 1,000점, 벽 800점, 장애물 300점을 난수로 만든다. `real_points`라는 변수 이름과 무관하게 실제 radar 입력이 아니다. 별도로 정의된 `generate_mock_radar_data`는 호출되지 않는다.

처리는 Open3D point cloud → 통계적 outlier 제거(`nb_neighbors=20`, `std_ratio=1.5`) → RANSAC 첫 평면(`distance_threshold=0.15`, 1,000 iteration) → 나머지의 두 번째 평면(`0.2`, 1,000 iteration) → 초록/빨강/파랑 표시 순서다. 첫 최대 평면을 바닥, 두 번째를 벽이라고 이름 붙이지만 normal 방향과 중력, 센서 좌표계 제약을 확인하지 않는다. 실제 장면에 적용하면 가장 큰 벽이 바닥으로 분류될 수 있다.

이 코드는 초기 공간 분리 아이디어로 보존했다. thermal PLY를 입력하는 경로도, radar 장치 연결·단위·timestamp를 읽는 경로도 없다. 실측 radar-camera 융합 결과로 묶지 않는다.

## 2.4 TEED의 초록선이 너무 굵은 문제

농연 실험은 새 모델 학습부터 시작하지 않고, 같은 TEED의 확률맵 이후 처리와 화면 배경을 비교했다. 굵은 3px dilation이 장면을 많이 가리는 문제를 먼저 줄이려는 선택이었다.

| 방식 | 입력·확률맵 이후 처리 | 의도 |
|---|---|---|
| 기존 TEED | CLAHE 2.0 → TEED → threshold .75 → 3px dilation → 배경 .24 | 당시 표시 기준선 |
| 얇은 선/밝은 배경 | 동일 TEED 입력 → skeletonization → 배경 .55 | 선의 중첩과 배경 소실 완화 |
| 구조 경계 TEED | bilateral(5,25,3), CLAHE 없음 → TEED → hysteresis .60/.85 → skeletonization → 배경 .55 | 강한 구조와 연결된 약한 경계를 남김 |
| 의사 라벨 adaptation | fusion head만 학습 → 같은 구조 경계 처리 | 다중 scale에서도 남는 응답을 강화할 수 있는지 확인 |
| PiDiNet | 공식 checkpoint·RGB 정규화 → hysteresis .20/.40 → skeletonization → 배경 .55 | 다른 경계 모델의 연속성과 비용 비교 |

[contours 코드](../code/selected/obstacle_edges/contours.py)는 low 이상인 픽셀의 8-connectivity 연결 성분을 만들고, 그 성분 안에 high 이상 응답이 하나라도 있으면 유지한다. 짧다는 이유만으로 강한 component를 없애지는 않는다. `thin_only`는 배경 밝기도 바꾸므로 “선 두께 하나만”의 순수 ablation이 아니다. 서로 다른 threshold는 같은 precision/recall 기준으로 교정한 값이 아니다.

### 제한적 학습의 실제 범위

seed 20260906, 32장×8 epochs=256 step, Adam lr 1e-4로 TEED의 `block_cat` 480개 파라미터만 학습했다. 320×240/160×120 TEED 자체 응답을 의사 라벨로 사용하고 feature extractor는 고정했다. 새로 사람 손으로 장애물 의미를 라벨링해 가르친 실험이 아니다. [학습·평가 실행 코드](../code/selected/obstacle_edges/run_smoke.py), [기록 JSON](evidence/obstacle-edges/results_summary.json)

## 2.5 Boson HUD 감사와 표시 폭 분리

이미 초록선이 있는 `boson_test.mp4`에는 새 TEED를 다시 적용하지 않았다. [감사 코드](../code/selected/boson_review/analyze.py)는 HSV의 초록 범위 H=40~85와 S/V threshold로 표시된 stroke를 근사 추출하고 세선화했다. 이는 화면에 그려진 선을 다루는 것이므로 원래 가려진 thermal 배경이나 사라진 경계를 복원하지 못한다.

별도로 HUD 합성 전 SmokeBasement 입력을 TEED에 넣어 같은 확률맵에서 상세도와 출력 폭을 비교했다. detailed=(.55,.75), balanced=(.60,.85), sparse=(.70,.90)이며, 선 폭은 최종 출력 640×480의 픽셀 단위다. Balanced 1px와 2px는 선택된 경계가 같고 렌더 폭만 다르다. 2px가 새 장애물을 검출한 것이 아니다.

이 과정에서 예전 RGB 색상 규칙도 갈색/주황색 평면 패치에 반응함을 재현했다. BGR (70,120,180)과 (0,140,255)는 각각 빨간 후보 1,004픽셀을 만들고 회색 패치는 0이었다. 이는 “화재 감지 AI가 물체를 오인했다”는 실험이 아니라, 색상 휴리스틱이 유사 색 물체에도 반응하는 구조를 통제 입력으로 확인한 것이다. 이후 thermal/rgb 입력 의미를 분리하고 RGB 후보는 기본에서 끈 기록이 있다. [감사 결과](evidence/boson-review/evidence/audit.json)

## 2.6 거리별 선 굵기: 깊이를 추정하는 코드가 아니라 주어진 거리를 표시하는 코드

[distance_width.py](../code/selected/edge/teed/distance_width.py)는 영상과 정렬된 meter 단위 거리 맵 또는 scalar를 입력으로 받는다. 기본값은 near=1m/5px, far=8m/1px다. 거리 `d`를 [1,8]로 제한하고 다음 식으로 폭을 만든다.

```text
t = (clamp(d, 1, 8) - 1) / 7
s = t²(3 - 2t)
w = 5 + (1 - 5)s
```

이 smoothstep은 가까울수록 굵게, 멀수록 가늘게 이어지게 한다. 실제 거리로 입력한 경우에만 물리 거리와 연결된다. 이번 화면의 `d`는 합성값이므로 “가정된 meter 값에 대한 renderer 반응”을 검증한 것이다.

작은 폭 변화에는 `α=1-exp(-dt/0.05)`인 인과적 EMA를 적용하고, 직전 폭과 차이가 0.75px보다 크면 최신 목표 폭을 즉시 사용한다. 미래 frame을 기다리지 않지만 작은 변화에는 응답 지연이 생긴다. 이 필터는 화면 좌표별 폭 history이며 물체를 tracking하거나 optical flow로 정렬한 필터가 아니다. 경계 검출의 깜빡임 자체를 해결하지 않는다.

Guo-Hall 세선화로 중심선을 만든 후, 각 중심선 시작 픽셀에 부여된 폭을 사각 dilation과 alpha blending으로 펼친다. 폭 1~7px에서 반경 r의 seed alpha는 `clip((w-(2r-1))/2,0,1)`이다. destination의 배경 거리로 선이 갑자기 굵어지는 것을 줄이는 구성이다. 사각 커널이므로 대각선·교차점 폭은 가로/세로 alpha 합으로 정의한 폭과 다를 수 있다.

NaN/Inf/0·거리 없음·100ms 초과 stale·미래 timestamp는 기본 1px로 돌아가고 해당 history를 지운다. **1px 복귀는 먼 물체라고 판정한 것이 아니라 결측 표시 정책**이다. 거리 불연속을 spatial blur로 섞지 않으며 현재 frame의 경계만 그린다. [파일 재생 계약](../code/selected/edge/teed/depth_replay.py)

## 2.7 지속 처리 시험

후속 실행은 최신 카메라 frame 1개와 거리 최근 8개만 보관하고, 유효한 과거 거리만 선택하며, 고정 버퍼의 CUDA Graph FP32로 model dispatch 비용을 줄이는 방향이었다. 이 아카이브에는 [모델 실행부](../code/selected/edge/teed/realtime_core.py), [지속성 판정](../code/selected/edge/teed/stability.py), [계측 runner](../code/selected/experiments/distance_width/realtime_compare.py)를 선별했다. 실제 장치 adapter 전체와 팀 앱 전체는 포함하지 않았다.

시행착오에는 Windows의 낮은 정밀도 clock으로 timestamp가 중복되는 실패, 화면 경계에서 세선화가 두껍게 남는 문제, 작은 연결 성분 소실, 종료 직전 멈춤이 집계에서 빠지는 문제, 640×480 calibration을 320×240 입력에 적용하는 불일치가 기록되었다. 최종 보고서는 clock을 `perf_counter`로 통일하고 해당 회귀 검증을 추가했다고 적는다. 이는 원 보고서에 남은 수정 이력이며 이번 아카이브가 다시 수정한 코드가 아니다.
