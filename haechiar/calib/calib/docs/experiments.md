# 세 가지 센서 실험 — 실행과 후속 개발

`HaechiAR/calib`의 `YSH` 브랜치에서 개발한다. 기존 Boson 내부 보정과 검토·적용
명령은 유지하고 `firesight-calib experiment` 명령을 추가했다.
[개발 환경의 검증 결과](experiment_verification.md)도 함께 기록했다.

| 실험 | 지금 실행할 수 있는 기능 | 장비에서 확인할 부분 |
|---|---|---|
| 1. RGB-D + 열화상 | 동시 원본 수집 구조, 시간 짝짓기, 깊이 역투영→열영상 Z-map, 재생 | D456 스트림 모드/SDK, 열화상 타깃, 실제 외부 보정 |
| 2. 열화상 단독 + AI | 원본/메타데이터 수집, 그룹별 데이터 분할, 추론 플러그인, 엣지 비교 기준 | 레이블, 학습 모델/가중치, 학습과 성능 평가 |
| 3. mmWave + 열화상 (2순위) | XYZ 점군 기록/재생, 3D 대응점 강체 보정, 희소 깊이 투영 | AWR6843AOPEVM 펌웨어/패킷, 반사 타깃, 브래킷, 실제 정합 |

라이다는 모델 미정이므로 제외했다. IMX678, TE-EV2, MTi-610은 장비 목록과
공통 데이터/플러그인 계약을 마련했다. 출력 형식·SDK가 확인되지 않은 상태에서
전용 드라이버를 구현했다고 보지 않는다. 장비 표는
[`experiment_hardware.yaml`](../config/experiment_hardware.yaml)에 옮겼다.

## 장비 없이 시작

저장소 루트에서 Python 3.10 이상을 사용한다. `python`은 현재 가상환경의 실행파일이다.

```bash
python -m pip install -e '.[test]'
python -m firesight_calib.cli experiment status --profile rgbd_thermal
python -m firesight_calib.cli experiment demo --mode rgbd_thermal --output calibration_data/demo_rgbd
python -m firesight_calib.cli experiment demo --mode thermal_ai --output calibration_data/demo_ai
python -m firesight_calib.cli experiment demo --mode radar_thermal --output calibration_data/demo_radar
python -m pytest -q
```

출력 폴더는 매번 새 이름을 지정한다. Windows에서는 기존 런처도 사용할 수 있다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 experiment demo --mode rgbd_thermal --output calibration_data/demo_rgbd
```

Ubuntu/ROS 환경에서는 `bash run.sh experiment ...`를 사용할 수 있다. 이 명령들은
ROS 없이도 동작한다. `demo`의 보정값과 데이터는 모두 `synthetic`이다. 데모 통과는
실제 장비의 정확도, 속도, 연기 환경 성능 검증이 아니다.

## 출력 구조와 의미

```text
demo_rgbd/
  intrinsic.yaml, extrinsic.yaml, registration.yaml   합성 보정 예제
  session/
    session.json          모드, 촬영 그룹, 장치 설정, 시계, 완료/중단 상태
    samples.jsonl         각 원본의 timestamp, frame, dtype, SHA-256
    samples/*.npz         pickle 없는 원본 배열(data 키)
  results/
    summary.json          처리 수, 동기화 탈락, 미측정 비율, 보정 스냅샷
    frames.jsonl          프레임별 시간차, 처리시간, 엣지 픽셀, 깊이 coverage
    frame_*.npz           view, edges, depth_m(융합 모드만)
    preview.png           첫 프레임 열영상 + 엣지 미리보기
```

원본 열영상은 uint8/uint16을 보존한다. `view`와 미리보기만 프레임별 1–99 percentile
정규화를 적용한다. 이는 온도 변환이 아니다. `depth_m`는 **열화상 카메라 Z축 깊이(m)**이며
유클리드 거리가 아니다. 같은 픽셀에는 가장 가까운 Z만 남기고 관측이 없는 픽셀은 NaN이다.
레이더의 빈 픽셀을 채워 조밀한 깊이 영상인 것처럼 보이게 하지 않는다.
`preview.png`는 엣지 확인용이며 깊이 정보는 NPZ로 확인한다.

AI 모델을 지정하지 않은 기본 엣지는 `canny_baseline_not_ai`다. 학습된 AI의 결과가 아니다.
coverage는 관측 픽셀 비율이고 정확도 점수가 아니다. 처리시간은 파일 재생 중 융합/엣지
연산만 측정하며 파일 I/O, 센서 노출, IPC 지연을 포함한 Jetson FPS가 아니다.

## 실제 수집 준비

기본 프로필은 패키지의 `experiments/profiles/*.yaml`에 있다. 복사해서 `group_id`,
`sources`를 채운다. 기본 프로필에는 포트, 보드 크기, 학습 모델을 추정해서 넣지 않았다.
`status`는 설정 준비 상태를 보고하며 장치 연결 성공을 보증하지 않는다.
보정이 없어도 원본 수집은 가능하다.

Boson 수집 예시: 아래 장치 인덱스는 **예시**이며 `doctor`로 실제 연결을 확인한 뒤 쓴다.

```yaml
schema: firesight.experiment_profile/v1
mode: thermal_ai
streams: [thermal]
group_id: room01_visit01
registration: null
edge_model: null
sources:
  - factory: firesight_calib.experiments.adapters:boson_usb
    options:
      device: 0
      telemetry_rows: 0
      frame: boson_optical_frame
```

```bash
python -m firesight_calib.cli doctor --source usb --device 0
python -m firesight_calib.cli experiment status --profile my_thermal.yaml
python -m firesight_calib.cli experiment capture --profile my_thermal.yaml --duration 60 --output calibration_data/room01
python -m firesight_calib.cli experiment verify-session --session calibration_data/room01
```

수집은 어댑터별 별도 프로세스를 사용한다. SDK 시작/읽기가 멈춰도 전체 수집 시간 뒤
프로세스를 종료한다. 큐는 128개로 제한하며 탈락 수와 강제 종료 수를 세션에 기록한다.
필수 스트림이 한 건도 없으면 정상 완료로 처리하지 않는다. 강제 종료 수가 0보다 크면
SDK 정리/읽기 상태도 장비에서 확인한다. 종료한 세션은 인덱스 전체 SHA-256을 함께 검사한다.
`--max-samples`는 전체 스트림의 개별 샘플 수 제한이므로 RGB-D bundle 경계와 같지 않다.
Ctrl+C/장치 오류는 이미 저장된 원본을 남기고 interrupted로 기록한다. 중단 세션은
`verify-session`과 재생으로 점검할 수 있지만 학습 데이터셋에는 자동 포함하지 않는다.

RGB-D 수집은 mode를 `rgbd_thermal`, streams를 `[thermal, rgb, depth]`로 바꾸고
sources에 `firesight_calib.experiments.adapters:realsense`를 추가한다.
options의 `serial`, `width`, `height`, `fps`는 **필수**다. 실제 D456에서 지원되는
프로필을 조회해서 지정해야 한다. `pyrealsense2`는 장비 호스트에 별도 설치한다.
SDK 깊이 scale로 정수 깊이를 미터로 변환하며 0은 NaN으로 저장한다.
RGB는 RGB8, 깊이는 native depth frame이고 RGB에 정렬된 깊이가 아니다.
SDK의 OpenCV 호환 왜곡 모델(none/Brown-Conrady)만 허용한다.

## 시간과 좌표계 계약

- timestamp는 정수 ns이며 스트림별로 엄격히 증가한다. 어댑터가 SDK 반환 직후
  `time.monotonic_ns()`를 기록하고 수집기가 준 `clock`을 그대로 사용한다.
- 호스트 수신 시각은 센서 노출 시각과 다르다. D456의 색/깊이는 같은 frameset
  수신 시각을 사용하되 SDK의 개별 timestamp/domain도 metadata에 보관한다.
- 장치 clock끼리 직접 비교하지 않는다. 하드웨어 시각을 사용하려면 offset/drift
  추정 후 공통 시계로 매핑하는 어댑터가 필요하다.
- 열영상을 anchor로 설정한 nearest-available 방식이다. 도착한 후보 중 가장
  가까운 것을 한 번만 사용하며 더 나은 미래 샘플을 기다리지 않는다.
- `--tolerance-ms 30`, `--queue-size 32`는 소프트웨어 시험 기본값이다. 장비 표의
  확정 사양이 아니며 실제 시간 오차를 보고 조정한다. tolerance는 각 스트림과
  anchor 사이의 차이에 적용된다. bundle 전체 span은 그 두 배까지 가능하다.
- 영상 해상도와 센서 frame이 보정값과 달라지면 멈춘다. 자동 resize나 축 추정을 하지 않는다.

## RGB-D + 열화상 보정 연결

1. Boson 내부 보정은 기존 `intrinsic`, `review`, `apply` 흐름을 사용한다.
2. 보드 실측 뒤 checkerboard/circle 타깃을 명시해서 두 카메라에서 같은 점을 관측한다.
   기존 target.yaml은 잠정 원형 보드이며 첨부 이미지의 보드 규격으로 확정하지 않는다.
3. 기존 `extrinsic`으로 camera-to-Boson 변환을 구한다. RGB에서 보정한 변환을
   native depth 영상에 바로 적용하면 안 된다. SDK의 depth→color 변환을 합성한다:
   `T_boson_from_depth = T_boson_from_color @ T_color_from_depth`.
4. 깊이 원본과 동일한 해상도·왜곡 모델의 SDK intrinsic을 별도 YAML로 저장한다.
5. 다음 manifest를 작성한다. 경로는 manifest 파일 기준이다.

```yaml
schema: firesight.experiment_registration/v1
calibration_id: rig01_measured_session01
source_frame: camera_depth_optical_frame
target_frame: boson_optical_frame
provenance: measured
intrinsic: boson_intrinsic.yaml
source_intrinsic: depth_intrinsic.yaml
extrinsic: depth_to_boson.yaml
```

intrinsic YAML에는 `camera_matrix`, `distortion_coefficients`, `image_size: [width,height]`가
필요하다. extrinsic YAML에는 `source_frame`, `target_frame`, `transform`(4×4)가 필요하며
기존 `T_boson_from_*` 키도 읽는다. 변환 방향은 반드시 source→thermal, translation은 m이다.
manifest의 `measured`는 사용자가 관리하는 데이터 출처 표시이며 물리적 보정 품질 승인과 다르다.
변환행렬의 형식 검사만으로 실제 보정 정확도를 보증할 수 없다.

```bash
python -m firesight_calib.cli experiment run --mode rgbd_thermal --session calibration_data/room01_rgbd --registration rig01/registration.yaml --output calibration_data/room01_fused
```

RGB는 동기화/수집 대상으로 포함한다. 현재 출력은 열영상과 열영상 좌표의 깊이이며
RGB 색상을 열영상에 워핑하는 기능은 포함하지 않는다.

## 레이더 + 열화상

펌웨어와 출력 형식이 확정된 후 아래 어댑터 계약에 맞춰 XYZ(m)를 전달한다. TI 제품명이
같아도 펌웨어별 packet/TLV 형식이 다를 수 있어 UART 파서를 미리 고정하지 않았다.
Passive thermal board가 레이더에서도 같은 점으로 보인다고 가정하지 않는다.

반사 타깃의 중심을 두 좌표계에서 측정하고 대응점을 `source_xyz_m`, `target_xyz_m` 키의
NPZ로 저장한다. 최소 4개이며 공선 배치를 거부한다. 브래킷을 고정한 뒤 거리와 자세를
바꾸어 수집하고, 별도 holdout 위치에서 결과를 확인한다.

```bash
python -m firesight_calib.cli experiment fit-rigid --pairs radar_pairs.npz --source-frame radar_frame --target-frame boson_optical_frame --output radar_to_boson.yaml
python -m firesight_calib.cli experiment run --mode radar_thermal --session calibration_data/radar01 --registration rig01/radar_registration.yaml --output calibration_data/radar01_fused
```

radar manifest는 위 예시에서 source frame/extrinsic을 바꾸고 source_intrinsic을 생략한다.
`fit-rigid`는 최소제곱 강체변환과 잔차를 출력한다. outlier 자동 제거, 반사점 자동 대응,
성능 합격 판정은 하지 않으며 `deployment_ready`는 false다.

## AI 데이터와 모델 연결

같은 장소/촬영 회차의 연속 영상은 같은 `group_id`로 둔다. train/val/test에 같은 그룹이
나뉘지 않도록 최소 3개 독립 그룹을 수집한다. 한 회차를 여러 세션으로 나눠도 group_id는
같게 유지해야 한다.

```bash
python -m firesight_calib.cli experiment dataset --sessions calibration_data/room01 calibration_data/room02 calibration_data/room03 --output calibration_data/dataset01
```

기본 분할은 그룹 단위 약 70/15/15, holdout 그룹 최소 1개씩이다. 원본 NPZ의 절대 경로와
SHA-256을 manifest로 내보내므로 원본 세션을 함께 보관한다. 합성 원본은 제외한다.
레이블은 null/unlabeled이고 `training_ready: false`다. 프레임이 많다는 이유만으로
학습 준비가 끝났다고 판단하지 않는다. `np.load(row['file'], allow_pickle=False)['data']`로
원본을 읽을 수 있다. 레이블 정의와 모델 결정 후 이 manifest 위에 학습기를 연결한다.

추론 플러그인은 `factory(options)`가 `name`과 `predict(native_thermal)`를 가진 객체를
반환한다. 출력은 원본 H×W의 float 확률 [0,1]이다. 기본 후처리 threshold는 0.5다.
전처리·모델 크기 변환·가중치 해시 검증은 모델 어댑터가 명시적으로 담당한다.
기존 TEED 또는 후속 학습 모델을 이 경계에 연결할 수 있다. 옵션 파일은 JSON이다.

```bash
python -m firesight_calib.cli experiment run --mode thermal_ai --session calibration_data/room01 --edge-model my_model:factory --model-options model.json --output calibration_data/room01_ai
```

## 장치 확장 계약

팩토리 `factory(options, clock)` → 객체 `read(timeout_s)` / `close()`.
read는 None, Sample 하나 또는 Sample 리스트를 반환한다. 실제 SDK로 읽은 값만 measured로
반환한다. 테스트용 입력은 `SessionWriter`를 사용하여 synthetic으로 기록한다.

| kind | 배열 | 단위/규약 |
|---|---|---|
| thermal | H×W uint8/uint16 | 원본 강도, 온도라고 가정하지 않음 |
| rgb | H×W×3 uint8 | RGB 채널 순서 |
| depth | H×W float | 카메라 Z(m), NaN 또는 0은 미측정 |
| radar | N×3 float | 명시한 센서 frame의 XYZ(m), N=0 허용 |
| imu | 6 float | ax,ay,az(m/s²), gx,gy,gz(rad/s) |

```python
from firesight_calib.experiments.model import Sample
from firesight_calib.experiments.session import SessionWriter

# vendor_decoder가 확인된 프로토콜에서 만들어낸 xyz_m과 timestamp_ns를 사용한다.
# 공통 시계로 변환하지 않은 장치 timestamp를 host clock이라고 표시하지 않는다.
sample = Sample('radar', timestamp_ns, common_clock, 'radar_frame', 'radar', xyz_m)
with SessionWriter('calibration_data/import01', {'mode': 'radar_thermal', 'group_id': 'room01'}) as writer:
    writer.append(sample)
```

열화상/점군을 모두 기록해야 radar_thermal 융합이 실행된다. 보조 IMU나 두 번째 TE-EV2는
서로 다른 stream 이름으로 저장할 수 있다. IMU 기반 자세 보상, CSI 카메라의 IR 분류,
레이더 Doppler/SNR 필터는 아직 선택하지 않았으며 자동 적용하지 않는다.

## 다음 작업 시작 지점

1. `experiment status`로 pending 목록을 확인한다.
2. 장비의 실제 serial/출력 모드/SDK와 radar firmware 버전을 기록한다.
3. 보드 실측값과 브래킷이 정해지면 내부·외부 보정을 수행한다.
4. 촬영 그룹과 라벨 규칙을 정하고 세 실험의 같은 환경 자료를 수집한다.
5. timestamp 오차·깊이 정합을 별도 검증하고 동일 조건의 결과를 비교한다.

참고 API: [OpenCV calibration/projection](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[RealSense projection/coordinate conventions](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0),
[RealSense Python examples](https://github.com/realsenseai/librealsense/tree/master/wrappers/python/examples).
