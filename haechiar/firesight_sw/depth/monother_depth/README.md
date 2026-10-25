# MonoTher-Depth baseline

## 관련 자료 및 오픈소스

- 논문: *MonoTher-Depth: Enhancing Thermal Depth Estimation via Confidence-Aware Distillation* (2025)
- 공식 코드: https://github.com/ZuoJiaxing/monother_depth
- 공개 weight: 공식 README의 Hugging Face `RockyZuo/Monther-Depth`

## 모델 내용

thermal monocular depth estimation을 목표로 한 모델이다. RGB depth model의 geometry prior를 thermal branch로 distillation해 thermal 영상 한 장에서 metric depth를 예측한다. 공개 MS2 checkpoint는 Depth Anything 기반 dual-branch ZoeDepth 구조로 제공된다.

FireSight wrapper는 공개 MS2 설정에서 cross-branch fusion이 비어 있는 점을 이용해 **thermal output branch만 사용**한다. upstream 함수 시그니처가 두 입력을 요구하므로 두 번째 branch 입력에도 같은 thermal frame을 넣지만, 저장하는 결과는 첫 번째 thermal branch의 `metric_depth`다. 이 실행 방식은 FireSight 영상에 빠르게 적용하기 위한 adapter이며 논문의 공식 MS2 evaluation protocol과는 구분한다.

## 구조

```text
Boson / thermal image or video
        ↓
Y16 → 8-bit inference normalization
        ↓
resize 640 × 256
        ↓
MonoTher-Depth thermal branch
        ↓
dense metric depth
        ↓
.npy raw prediction + color preview
```

## 선택 이유

SupDepth4Thermal보다 최근의 thermal-specific depth 후보이고, 공개 checkpoint가 있어 별도 학습 전에 FireSight 실내 열화상에서 depth 구조가 살아나는지 확인하기 좋다. 우선 AnyThermal과 동일 영상을 넣어 병렬 비교한다.

## 설치

```bash
cd depth/monother_depth
pip install -r requirements.txt
bash setup.sh
```

`setup.sh`는 공식 저장소를 `depth/third_party/monother_depth`에 clone한다. upstream 환경을 그대로 만들고 싶으면 다음을 사용한다.

```bash
cd ../third_party/monother_depth
conda env create -n monother_depth --file environment.yml
conda activate monother_depth
```

두 개의 weight가 필요하다.

- MonoTher MS2 checkpoint: `ZoeDepthDualv1_05-Aug_17-28-23b9f4b49f50_latest.pt`
- Depth Anything base checkpoint: `depth_anything_vitl14.pth`

checkpoint는 repo에 commit하지 않는다.

## 실행

```bash
python run.py \
  --input /data/firesight/thermal.mp4 \
  --weights /data/weights/ZoeDepthDualv1_05-Aug_17-28-23b9f4b49f50_latest.pt \
  --depth-anything-weights /data/weights/depth_anything_vitl14.pth \
  --output ./outputs/test01
```

header 없는 Boson Y16 raw:

```bash
python run.py \
  --input frame.y16 --raw-width 640 --raw-height 512 \
  --weights /data/weights/ZoeDepthDualv1_05-Aug_17-28-23b9f4b49f50_latest.pt \
  --depth-anything-weights /data/weights/depth_anything_vitl14.pth
```

## 출력

```text
outputs/test01/
├── frames/
│   ├── 000000_*_depth.npy
│   ├── 000000_*_preview.png
│   └── ...
└── preview.mp4
```

`.npy`에는 float32 depth prediction을 저장한다. FireSight와 MS2 사이 domain shift가 있으므로 출력값을 즉시 안전거리 판정에 사용하지 말고 알려진 거리의 실내 장면으로 먼저 검증한다.
