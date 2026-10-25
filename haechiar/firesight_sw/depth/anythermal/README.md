# AnyThermal + MiDaS baseline

## 관련 자료 및 오픈소스

- 논문: *AnyThermal: Towards Learning Universal Representations for Thermal Perception* (ICRA 2026)
- 공식 코드: https://github.com/castacks/AnyThermal
- depth 코드: https://github.com/castacks/BridgeMultiSpectralDepth
- 공개 checkpoint: AnyThermal 공식 README의 Hugging Face checkpoint bundle

## 모델 내용

AnyThermal은 depth 전용 network라기보다 여러 thermal 환경에서 사용할 수 있는 범용 thermal representation을 학습한 backbone이다. depth baseline에서는 frozen AnyThermal DINOv2 backbone 위에 MiDaS depth head를 붙여 MS2 thermal depth를 학습한다.

## 구조

```text
Boson / thermal image or video
        ↓
Y16 → 8-bit inference normalization
        ↓
AnyThermal DINOv2 backbone
        ↓
MiDaS depth head
        ↓
depth visualization
```

FireSight `run.py`는 입력 이미지/영상을 프레임으로 풀어 공식 `inference_depth.py`가 요구하는 `left/*_thr.png` 형식으로 자동 배치한 뒤 upstream inference를 호출한다. 모델 내부 코드는 수정하지 않는다.

## 선택 이유

FireSight는 MS2와 달리 소방 실내 환경이 주 대상이다. AnyThermal은 여러 thermal domain을 대상으로 representation을 학습했기 때문에, thermal depth benchmark 점수뿐 아니라 새로운 Boson+ 실내 영상에서 구조가 유지되는지를 확인할 비교군으로 둔다.

## 설치

```bash
cd depth/anythermal
pip install -r requirements.txt
bash setup.sh
```

`setup.sh`는 공식 AnyThermal 저장소와 depth submodule을 `depth/third_party/AnyThermal` 아래에 준비한다.

AnyThermal 공식 환경은 Python 3.10, PyTorch 2.1.2 / CUDA 12.1 기준이다. depth baseline은 `pytorch3d`가 추가로 필요하다. CUDA/PyTorch 조합에 따라 빌드가 달라지므로 이 부분은 공식 README 절차를 따른다.

checkpoint bundle을 받은 뒤 다음 경로가 있어야 한다.

```text
depth/third_party/AnyThermal/pretrained_checkpoints/
├── backbone/AnyThermal_full/model20.pth
└── depth/Midas_anythermal/ckpt_epoch=28_step=145000.ckpt
```

## 실행

```bash
python run.py \
  --input /data/firesight/thermal.mp4 \
  --output ./outputs/test01
```

checkpoint를 다른 위치에 둔 경우:

```bash
python run.py \
  --input /data/firesight/thermal.mp4 \
  --ckpt /data/weights/ckpt_epoch=28_step=145000.ckpt
```

Boson raw Y16:

```bash
python run.py --input frame.y16 --raw-width 640 --raw-height 512
```

## 출력

```text
outputs/test01/
├── all/
│   ├── 000000_thr.png
│   └── ...
└── preview.mp4
```

현재 upstream demo는 visualization 중심이다. raw float depth가 필요해지면 모델의 `inference_depth()` 출력을 직접 저장하는 FireSight adapter를 다음 단계에서 붙인다.

공식 inference 코드가 `.cuda()`를 사용하므로 현재 wrapper도 CUDA GPU를 전제로 한다. Jetson Orin Nano에서의 PyTorch/CUDA/pytorch3d 호환성과 FPS는 별도 검증이 필요하다.
