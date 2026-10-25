# HaechiAR 실행 코드

2026-10-25에 확인한 소스를 이 저장소 안에 가져왔습니다. 기존 연구 자료는 루트의
`docs/`, `public-media/`, `code/`에 그대로 보존합니다.

| 목적 | 코드 | 기준 |
|---|---|---|
| Boson 320 원본 촬영·점검 | [firesight_sw/data/capture/boson](firesight_sw/data/capture/boson/) | `firesight_sw/main` · `339af10` |
| TEED 윤곽선·시간 안정화·고온 색상 HUD | [firesight_sw/edge/orin](firesight_sw/edge/orin/) | 같은 `main` |
| 열화상 깊이 추정 wrapper | [firesight_sw/depth](firesight_sw/depth/) | 같은 `main` |
| 내부 보정·RGB-D/레이더 융합·재생 | [calib](calib/) | `calib/YSH` · `659dca3` |

소프트웨어는 최신 기본 브랜치, 보정은 `main`의 기능을 포함한 최신 YSH 개발본을 사용합니다.
`firesight_sw/YSH`의 별도 거리 HUD 실행기는 이 기본 브랜치와 구조가 달라 합치지 않았습니다.
이전 거리 HUD 실험의 영상·결과·발췌 코드는 [기존 연구 기록](../RESEARCH_DETAILS.md)에 남아 있습니다.

## PC에서 시작

Python 3.10 이상을 지원하는 원본 코드이며, **Windows에서는 Python 3.13 이상을 권장**합니다.
이번 검사는 Python 3.13.17에서 수행했습니다. 아래 명령은 **ThermalCameraResearch 저장소 루트** 기준입니다.
카메라가 없어도 보정 데모와 기존 공개 영상의 CPU 처리를 실행할 수 있습니다.

```bash
python -m venv .venv
```

Windows PowerShell에서는 `.venv\Scripts\Activate.ps1`, Linux/macOS에서는
`source .venv/bin/activate`로 활성화한 뒤 설치합니다.

```bash
python -m pip install -e "./haechiar/calib[test]"
python -m pip install -r haechiar/firesight_sw/edge/orin/requirements.txt
```

### 보정과 센서 융합 데모

```bash
python -m firesight_calib.cli experiment status --profile rgbd_thermal
python -m firesight_calib.cli experiment demo --mode rgbd_thermal --output local/demo_rgbd
python -m firesight_calib.cli experiment demo --mode thermal_ai --output local/demo_ai
python -m firesight_calib.cli experiment demo --mode radar_thermal --output local/demo_radar
```

결과 폴더는 매번 새 이름으로 지정합니다. 데모 입력·보정값은 합성이며 실제 거리 정확도나
AI 모델 성능을 나타내지 않습니다. 실제 장비 연결은 [보정 안내](calib/README.md)와
[센서별 실험 안내](calib/calib/docs/experiments.md)를 따릅니다.

### 기존 공개 영상으로 윤곽선 처리

```bash
python -c "from pathlib import Path; Path('local').mkdir(exist_ok=True)"
python haechiar/firesight_sw/edge/orin/run_pipeline.py --source public-media/smoke-original.mp4 --device cpu --headless --warmup 0 --max-frames 10 --contour-edges --metrics local/thermal_preview.json
```

이 명령은 기존 MP4를 읽고 새 측정 JSON만 만듭니다. 창으로 보려면 `--headless`를 빼고
`--pace`를 추가합니다. TEED 모델 코드와 원본 저장소의 `5_model.pth`가 함께 들어 있으며,
런타임은 알려진 체크포인트 SHA-256을 검사합니다. [세부 옵션](firesight_sw/edge/orin/README.md)

### Boson raw 촬영

촬영기는 Windows 전용이며 보정·HUD 환경과 다른 OpenCV 버전을 사용합니다.
별도의 가상환경에 [촬영 requirements](firesight_sw/data/capture/boson/requirements.txt)를
설치하고 [촬영 안내](firesight_sw/data/capture/boson/README.md)를 따릅니다.
`BOSON_SHOOT_DIR`에 새 촬영 폴더, `BOSON_PORT`에 실제 COM 포트를 지정합니다.
촬영 순서표의 열 형식과 예제 실행 파일도 해당 폴더에 포함했습니다.

### 깊이 모델과 Jetson

[MonoTher-Depth·AnyThermal·RIDERS 안내](firesight_sw/depth/README.md)에 입력 변환,
공식 코드 설치와 체크포인트 준비 방법이 있습니다. 깊이 모델의 전체 외부 코드와 가중치는
각 모델별 환경에 별도로 설치합니다. Jetson은 JetPack에 맞는 PyTorch와 시스템 OpenCV를 사용합니다.

## 확인과 출처

```bash
python haechiar/verify.py
```

가져온 파일의 무결성, 보정 테스트, 데이터셋 없이 실행 가능한 윤곽선 검사를 순서대로 실행합니다.
외부 Fire360 전체 영상이나 원본 Git 이력을 요구하는 비교 스크립트는 포함하지 않습니다.
필요 조건은 [원본 실험 안내](firesight_sw/experiments/teed_ui/README.md)에 있습니다.

- [이번 이관의 검증 결과](VALIDATION.md)
- [원본 경로·커밋·파일별 SHA-256](SOURCE_MANIFEST.json)
- [상위 저장소와 반영 범위](../UPSTREAM_STATUS.md)

가져온 212개 파일은 기준 커밋의 Git blob과 동일합니다. Windows checkout의 텍스트 줄바꿈
변환은 무결성 검사에서 허용하며 바이너리는 그대로 대조합니다. 촬영 세션·원본 데이터와 원본 저장소의
CI 설정은 제외했습니다. `firesight_sw/data/collection/README.md`는 이 배포본의 데이터 안내로
추가했습니다. 원문 README의 clone·로컬 데이터 경로 예시는 원본 저장소 기준이므로,
여기서는 위 실행 명령부터 사용합니다. 팀 코드와 외부 모델의 출처·배포 조건은 원문을 따릅니다.
