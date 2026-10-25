# FireSight Depth Baselines

FireSight 열화상 입력에서 깊이 추정 가능성을 빠르게 비교하기 위한 실험 폴더다. 현재 후보는 `MonoTher-Depth`, `AnyThermal`, `RIDERS` 세 가지다. 세 방법을 먼저 독립적으로 확인하고, 성능이 확인된 뒤 Jetson 최적화나 센서 통합 여부를 결정한다.

## 비교 대상

| 후보 | 입력 | 현재 폴더에서 할 수 있는 것 | 성격 |
| --- | --- | --- | --- |
| MonoTher-Depth | thermal 1장/영상 | FireSight 입력을 바로 thermal branch에 넣고 dense depth 저장 | thermal depth 전용 모델 |
| AnyThermal + MiDaS | thermal 1장/영상 | 공식 AnyThermal depth inference 형식으로 자동 변환 후 실행 | 범용 thermal representation 기반 depth |
| RIDERS | thermal + projected mmWave depth | RIDERS 형식 입력 준비, radar global-scale alignment 실험 | thermal-radar metric depth 방향 |

세 후보는 완전히 같은 입력 조건이 아니다. `MonoTher-Depth`와 `AnyThermal`은 thermal-only 비교가 가능하고, `RIDERS`는 radar가 추가되는 별도 시스템 후보로 본다.

## 입력 형식

공통 wrapper는 다음 입력을 받는다.

- PNG/JPG/BMP/TIFF
- MP4/AVI 등 OpenCV가 읽을 수 있는 영상
- 이미지 폴더
- Boson Y16을 저장한 16-bit PNG/TIFF
- header 없는 `.raw`/`.y16` 파일: `--raw-width`, `--raw-height` 필요

16-bit 열화상은 모델 입력을 만들기 위해 프레임별 1–99 percentile 범위를 8-bit로 변환한다. 이 과정은 depth inference용 영상 정규화이며 radiometric temperature 변환이 아니다. 원본 Y16은 별도로 보존해야 한다.

## 공통 구조

```text
depth/
├── README.md
├── thermal_io.py
├── monother_depth/
│   ├── README.md
│   ├── setup.sh
│   ├── run.py
│   └── requirements.txt
├── anythermal/
│   ├── README.md
│   ├── setup.sh
│   ├── run.py
│   └── requirements.txt
└── riders/
    ├── README.md
    ├── setup.sh
    ├── prepare_inputs.py
    ├── run_stage1_alignment.py
    └── requirements.txt
```

원 논문 코드 전체를 이 저장소에 복사하지 않는다. `setup.sh`가 각 공식 저장소를 `depth/third_party/` 아래에 clone하고 FireSight wrapper가 그 코드를 호출한다. 모델 라이선스와 checkpoint 배포 조건은 각 upstream 저장소를 따른다.

## 빠른 비교 방법

같은 thermal 영상으로 먼저 아래 두 개를 병렬 실행한다.

```bash
cd depth/monother_depth
bash setup.sh
python run.py --input /path/to/thermal.mp4 \
  --weights /path/to/ZoeDepthDualv1_05-Aug_17-28-23b9f4b49f50_latest.pt \
  --depth-anything-weights /path/to/depth_anything_vitl14.pth
```

```bash
cd depth/anythermal
bash setup.sh
python run.py --input /path/to/thermal.mp4
```

비교할 때는 한 프레임의 예쁜 결과보다 다음 항목을 본다.

- near / mid / far 순서가 반복적으로 맞는지
- 문, 벽, 사람, 장애물 경계에서 depth가 급격히 무너지지 않는지
- 카메라 이동 중 frame-to-frame depth가 얼마나 흔들리는지
- 실제 알려진 거리와 metric scale이 어느 정도 맞는지
- 연기 전/후 같은 구조의 depth ordering이 유지되는지
- PC GPU와 Jetson에서 처리시간이 어느 정도인지

`RIDERS`는 thermal 영상만으로 실행하는 모델이 아니므로 별도로 `riders/README.md`의 radar projection 조건을 맞춘 뒤 비교한다.

## 원 자료

- MonoTher-Depth: https://github.com/ZuoJiaxing/monother_depth
- AnyThermal: https://github.com/castacks/AnyThermal
- AnyThermal depth submodule / BridgeMultiSpectralDepth: https://github.com/castacks/BridgeMultiSpectralDepth
- RIDERS: https://github.com/MMOCKING/RIDERS

## 현재 범위

이 폴더는 우선 **baseline feasibility**를 확인하기 위한 것이다. ROS2 topic 연결, Boson SDK direct capture, TI mmWave 실시간 projection, TensorRT/Jetson 최적화는 baseline 결과가 확인된 다음 단계로 둔다.
