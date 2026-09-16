# 거리 기반 선 굵기 — 로컬 실험

기존 `run_teed.py`의 고정 굵기 동작은 기본값이다. `--width-mode distance`로
거리 기반 표시를 켠다. 이 기능은 거리 입력으로 **표시 폭만** 바꾸며 TEED 모델이나
경계 선택 threshold를 바꾸지 않는다.

## 실행

```bash
python edge/teed/run_teed.py --source input.mp4 --device cpu \
  --width 320 --height 240 --width-mode distance \
  --depth-replay aligned_depth_m.npy \
  --near-m 1 --far-m 8 --near-px 5 --far-px 1 \
  --width-tau-ms 50 --depth-max-age-ms 100 \
  --headless --output overlay.mp4 --metrics metrics.json
```

거리 모드의 중심선 세선화에는 `cv2.ximgproc.thinning`을 제공하는 OpenCV contrib가
필요하다. PC 실험 환경은 `opencv-contrib-python`, `numpy`, `torch`를 사용한다.
Jetson에서는 해당 JetPack과 호환되는 contrib 빌드를 사용해야 한다. 기존 시스템
OpenCV를 최신 pip wheel로 덮어쓰는 설치 절차는 제공하지 않는다.

Python 3.6 문법을 유지했지만 이번 테스트 실행 환경은 Windows Python 3.13 / CPU다.
원형 Nano나 Orin 보드에서의 속도는 아직 측정하지 않았다.

## 입력 계약

`aligned_depth_m.npy`: float32/float64, `[프레임 수, height, width]`, 미터 단위.
각 깊이 프레임이 영상의 같은 인덱스와 시점에 대응해야 한다. H×W는 TEED 출력
크기와 같아야 하며, 카메라 보정·투영·가림 처리까지 외부에서 완료해야 한다.
정수 mm 또는 임의 깊이 색상 이미지를 자동으로 미터로 해석하지 않는다.
0, 음수, NaN, Inf는 해당 픽셀의 입력 결측이다. 짧은 NPY가 끝나면 결측으로 처리한다.

대안으로 JSON scalar 타임라인을 사용할 수 있다.

```json
{"protocol":"teed_depth_scalar_v1","samples":[
  {"time_s":0.0,"depth_m":1.5},
  {"time_s":0.03333333333333333,"depth_m":1.6},
  {"time_s":0.06666666666666667,"depth_m":null}
]}
```

scalar는 화면 전체에 하나의 거리를 적용하는 시험용 입력이다. 다중 물체의 거리
측정이 아니다. JSON 시점은 strictly increasing이며 미래 값을 미리 사용하지 않는다.
마지막 scalar를 유지할 때 원래 시각도 유지하므로 100ms 초과 시 오래된 입력으로 처리한다.

CLI의 NPY/JSON 재생은 파일 전용이며 실시간 카메라와 혼용하면 오류를 낸다.
센서의 실시간 입력은 `DistanceWidthRenderer.render(mask, depth_m, timestamp_s,
depth_timestamp_s)`에 같은 시간 기준으로 전달하도록 연결한다. 센서 수집/등록 모듈은
이번 변경에 포함하지 않았다. 최신 프레임을 선택하는 수집기와 함께 쓰고 입력 큐를
누적시키지 않는 것은 장비 통합 시 검증해야 한다.

## 표시와 지연

`u=clip((거리-1)/(8-1),0,1)`, `폭=5-4*u²*(3-2u)`가 기본 곡선이다.
near/far 거리와 폭은 옵션으로 바꿀 수 있다. 폭은 **현재 출력 해상도의 픽셀** 단위다.
확대 표시하면 화면에서 그 픽셀도 함께 확대된다.

현재 threshold mask를 Guo-Hall 중심선으로 만든 뒤, 반경별 최대 팽창과 알파
혼합으로 분수 폭을 그린다. 가로/세로 선에서 알파 합으로 정의한 폭이며, 사각 팽창을
사용하므로 대각선/교차점에서는 기하학적 유클리드 폭과 다르다.
거리 구간 경계의 정수 폭 점프는 없지만 입력 경계 자체의 출현/소멸을 평활화하지는 않는다.

시간 필터는 `alpha=1-exp(-dt/0.05)`인 인과적 EMA다. 미래 프레임을 기다리지
않지만 작은 변화에는 응답 지연이 있다. 30fps에서 저주파 변화의 예상 지연은
약 35.2ms이고, 목표 폭 변화가 0.75px보다 크면 필터를 건너뛰어 즉시 적용한다.
따라서 큰 변화의 즉시성과 작은 노이즈의 안정성을 우선하며, 모든 거리 변화가
느리게 이어지도록 강제하지 않는다. `--width-tau-ms 0`으로 시간 필터를 끈다.

입력 결측·오래된 입력·미래 시각은 far-px로 즉시 복귀한다. 기본 far-px는 1px다.
이 fallback 폭은 “안전한 먼 물체” 판정이 아니며, 거리 상태는 metrics에 별도 기록된다.
깊이 불연속 부근에 공간 블러를 넣지 않아 서로 다른 물체의 거리를 섞지 않는다.

## 재현

```bash
python -m unittest discover -s edge/teed/tests -v
python experiments/distance_width/benchmark.py --output experiments/distance_width/outputs
```

비교는 동일 TEED mask로 original dilation 3 / thin fixed 3 / 거리 1·3·5 구간화 /
연속 무필터 / 연속 50ms 필터를 처리한다. 실제 공개 열화상 원본에 합성 거리 맵을
적용한 영상과 완전 합성 시험 장면을 분리한다. 실제 거리 추정 정확도나 구조 현장
유효성을 평가한 결과는 아니다. 벤치마크는 영상 저장 시간을 제외한 CPU 처리 시간이며
센서→디스플레이 종단 지연과 구분한다.
