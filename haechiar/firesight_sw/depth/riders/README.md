# RIDERS baseline

## 관련 자료 및 오픈소스

- 논문: *RIDERS: Radar-Infrared Depth Estimation for Robust Sensing* (2024)
- 공식 코드: https://github.com/MMOCKING/RIDERS
- 사용 센서: monocular LWIR thermal + mmWave radar

## 모델 내용

RIDERS는 thermal monocular prediction을 radar 거리와 결합해 metric dense depth를 만드는 구조다. 크게 global scale alignment, RC-Net 기반 quasi-dense radar augmentation, Scale Map Learner를 이용한 local refinement로 구성된다.

## 구조

```text
thermal image ──→ monocular dense prediction ─┐
                                               ├─ global scale alignment
projected mmWave sparse depth ────────────────┘
                         ↓
             RC-Net quasi-dense radar
                         ↓
              local scale refinement
                         ↓
                dense metric depth
```

## 선택 이유

FireSight가 thermal camera와 TI mmWave를 함께 사용한다는 점에서 최종 시스템 후보로 가장 직접적이다. 특히 monocular depth가 실내 domain shift 때문에 absolute scale을 잘못 잡더라도 radar의 실제 거리로 보정할 수 있는 구조를 검토할 수 있다.

다만 `MonoTher-Depth`와 `AnyThermal`처럼 thermal 영상 하나만 넣어서 완성된 RIDERS 결과를 얻을 수 있는 모델은 아니다. **radar point를 thermal image plane으로 투영한 sparse depth**, RC-Net/SML weight, sensor calibration이 필요하다. 공식 저장소도 ZJU-Multispectrum 디렉터리와 학습 결과 경로를 기준으로 작성되어 있다.

그래서 이 폴더는 두 부분으로 나눈다.

- `prepare_inputs.py`: FireSight thermal 영상과 이미 projection된 radar depth PNG를 RIDERS 실험 구조로 정리
- `run_stage1_alignment.py`: full RIDERS라고 부르지 않고, RIDERS의 첫 단계인 radar global-scale alignment만 빠르게 확인

full RC-Net + Scale Map Learner는 calibration과 checkpoint가 준비된 뒤 official repo를 FireSight 데이터 구조에 연결한다.

## 설치

```bash
cd depth/riders
pip install -r requirements.txt
bash setup.sh
```

공식 환경을 그대로 만들려면:

```bash
cd ../third_party/RIDERS
conda env create -f environment.yaml
conda activate rc-depth
```

## FireSight 입력 준비

먼저 TI mmWave point/depth를 **Boson thermal camera 좌표계의 pixel 위치로 투영**해야 한다. `radar_png`는 그 projection이 끝난 sparse depth map이어야 한다. 이 스크립트가 extrinsic calibration을 대신하지 않는다.

```bash
python prepare_inputs.py \
  --thermal /data/firesight/thermal.mp4 \
  --radar-dir /data/firesight/radar_projected_png \
  --output ./outputs/riders_input \
  --scene test01
```

## stage-1 scale alignment 빠른 확인

MonoTher-Depth 등에서 저장한 dense depth `.npy`와 동일 프레임의 projected radar depth를 이용한다.

```bash
python run_stage1_alignment.py \
  --mono-depth ../monother_depth/outputs/test01/frames/000000_frame_depth.npy \
  --radar-depth /data/firesight/radar_projected/000000.npy \
  --output ./outputs/aligned_000000.npy \
  --mono-type depth
```

radar PNG가 mm 단위라면 `--radar-scale 0.001`을 사용한다.

현재 stage-1 코드는 유효 radar correspondence에서 `radar_depth / mono_depth` 비율의 median으로 global scale을 잡는다. 논문의 전체 RIDERS가 아니며 RC-Net과 local refinement 효과를 포함하지 않는다.

## full RIDERS로 넘어갈 때 필요한 것

- Boson intrinsic / distortion calibration
- TI mmWave ↔ Boson extrinsic calibration
- radar point를 thermal image plane으로 투영하는 코드
- frame timestamp synchronization
- monocular prediction (`any`에 해당하는 입력)
- RC-Net output 또는 재학습
- Scale Map Learner checkpoint 또는 재학습

이 조건을 먼저 맞춘 뒤 `MMOCKING/RIDERS`의 `run_rcnet_zju.py`, `val_zju.py` 구조를 FireSight 데이터에 맞춰 연결한다.
