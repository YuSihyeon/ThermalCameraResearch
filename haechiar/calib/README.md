# FireSight Boson 내부 캘리브레이션

## 다중 센서 실험 추가 개발

RGB-D+열화상, 열화상 단독+AI, mmWave+열화상을 위한 공통 수집·보정·재생 구조는
[실험 실행 및 후속 개발 안내](calib/docs/experiments.md)에 정리했다.
`firesight-calib experiment demo --mode rgbd_thermal --output calibration_data/demo_rgbd`로
장비 없이 시작할 수 있다. 장비 표의 미정 항목은 설정에서 비워 두었다.
실제 장비·AI 모델 검증은 후속 단계이며, 데모는 명시적인 합성 입력을 사용한다.

**원본 사진 입력/촬영 → 내부 보정 계산 → 그래프·히트맵 검토 → 사용자 승인 → 실시간 영상 적용**을 한 실행에서 진행합니다. Boson 320의 내부 파라미터 `K`, 왜곡계수 `D`를 계산하며 Windows USB Y16과 Ubuntu 22.04 / ROS 2 Humble을 지원합니다.

기본 실행은 `wizard`입니다. 결과 보고서를 연 다음 터미널에서 마지막으로 **적용할지** 묻습니다. `yes`를 입력한 경우에만 실시간 왜곡 보정을 시작하고, `no`를 입력하면 **새 사진 입력/촬영 단계로 돌아갑니다.** 각 시도의 원본과 결과는 따로 보존합니다.

## 빠른 실행

### Windows + USB

준비: Git, **64-bit Python 3.10 이상**, Boson USB/UVC 연결. 다른 카메라 앱에서 Boson을 사용하고 있다면 닫습니다.

```powershell
git clone https://github.com/HaechiAR/calib.git
cd calib
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

첫 실행은 `.venv`에 필요한 Python 패키지를 설치하고 원본 사진 폴더를 묻습니다. 폴더 경로를 입력하거나 Enter를 눌러 새 촬영을 시작합니다. 이후에는 같은 명령으로 실행합니다. PowerShell 실행 정책 변경은 해당 프로세스에만 적용됩니다. USB 인덱스가 0이 아니면:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 doctor --source usb --device 1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 wizard --source usb --device 1
```

### Ubuntu 22.04 + ROS 2 Humble

준비: ROS 2 Humble과 Boson 드라이버를 설치하고 **320×256 `mono16` 또는 `mono8` 원본 토픽**을 발행합니다. 이 저장소는 카메라 펌웨어나 별도 Boson 드라이버를 설치하지 않습니다.

```bash
sudo apt install python3-opencv python3-numpy python3-yaml ros-humble-rclpy ros-humble-sensor-msgs
git clone https://github.com/HaechiAR/calib.git
cd calib
bash run.sh
```

`run.sh`는 `/opt/ros/humble/setup.bash`를 읽고 시스템 Python을 사용합니다. ROS 모드에서는 pip의 OpenCV/NumPy를 시스템에 덮어쓰지 않습니다. 기본 토픽은 `/boson/image_raw`이며 다른 토픽이면:

```bash
bash run.sh doctor --source ros --topic /thermal/image_raw
bash run.sh wizard --source ros --topic /thermal/image_raw
```

카메라/토픽 없이 프로그램부터 확인할 때:

```powershell
# Windows
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 self-check
```

```bash
# Linux의 ROS 없는 Python 경로: Python 3.10+ 및 python3-venv 필요
bash run.sh self-check
```

명시적으로 결과 경로를 지정할 때는 새로운 폴더를 사용합니다: `self-check --output calibration_data/check01`.

## 촬영 방법

촬영 창에서:

| 키 | 동작 |
|---|---|
| Space | 검출된 타깃의 원본 프레임 저장 |
| A | 자동 저장 켜기/끄기 |
| C | 최소 15개 뷰를 확보한 뒤 보정 실행 |
| Q | 저장한 이미지를 남기고 촬영 종료 |

기본 목표는 **서로 다른 24개 뷰**입니다. `wizard`는 목표 장수 도달 후 자동으로 계산하고 검토 단계로 이동합니다. 기존 `capture` 명령에서는 `--calibrate`를 주면 자동 계산합니다. Q는 보정 없이 종료합니다. 검출 실패와 거의 같은 자세의 중복 촬영은 장수에 포함되지 않습니다. `--auto --count 30 --interval 1.5`로 자동 촬영을 시작할 수 있습니다.

1. Boson을 예열하고 초점과 출력 모드를 고정합니다. 셔터/FFC 동작 중이거나 흔들린 프레임은 피합니다.
2. 견적의 보드는 **Passive Thermal Polymer**입니다. 공급업체가 지정한 열 대비 형성 방법을 적용합니다. 일반 종이 인쇄물이 열영상에서 같은 패턴을 만든다고 가정하지 않습니다.
3. 약 1.5~2 m에서 먼저 원 27개가 모두 검출되는지 확인합니다. 거리는 검출 결과에 맞춥니다. 견적의 10 px 원 크기는 보장값이 아닌 설계 참고치입니다.
4. 전체 타깃이 영상 안에 있도록 유지하면서 중앙·좌우·상하·모서리로 이동하고 거리를 바꿉니다. 보드를 좌우와 상하 두 방향으로 기울여 촬영합니다. 정면 사진만 많이 모으면 정확한 보정이 되지 않습니다.
5. 촬영 화면의 coverage, 결과의 프레임별 오차와 보정 전후 영상을 함께 확인합니다.

GUI가 없는 ROS PC에서는:

```bash
bash run.sh wizard --source ros --auto --headless --no-browser --count 30 --interval 1.5 --timeout 600
```

`--headless`여도 **터미널 입력은 필요**합니다. 출력된 `review.html`을 화면이 있는 PC에서 확인한 뒤 원래 터미널에서 승인합니다. `--timeout`은 촬영 세션 전체 제한 시간(초)입니다. 시간 초과 시 원본과 마지막 프레임을 남깁니다. USB 카메라 읽기는 별도 프로세스에서 실행하므로 드라이버 읽기가 멈춰도 메인 프로그램의 시간 제한과 종료 처리가 동작합니다.

## 이번 견적의 타깃

기본값은 [`calib/config/target.yaml`](calib/config/target.yaml)과 같습니다. 다른 제작 도면/실측값을 쓰려면 `--target <파일>`로 지정합니다.

| 항목 | 값 |
|---|---|
| 카메라 | 비방사형 Boson 320, 20320A050-9PAAX |
| 원본 해상도 | 320×256 |
| 렌즈 | 견적에 기재된 4.3 mm; 보정 시 K는 영상에서 추정 |
| 보드 외형 | 600×400×19 mm |
| 패턴 | 비대칭 원 27개, OpenCV `patternSize=(3,9)` |
| 원 지름 / 기본 간격 | 각각 60 mm |
| 원 중심 | `x=60+60i, y=50+60(2j+i%2)` mm, `i=0..8, j=0..2` |

원 지름과 중심 간격은 다른 개념입니다. 논리 행을 가로로 배치한 배열이며 단순 9×3 직사각 격자가 아닙니다. 원 좌표와 검출 순서는 함께 관리하고 결과에 실제 3D 점을 기록합니다.

견적서에서 **패턴은 제안 상태**이며 제작 도면·실물 검증이 남아 있어 `status: provisional`로 유지합니다. 실측 확인 후에만 `verified`로 변경합니다. 공칭 화각으로 추정한 초점거리를 보정 결과로 저장하지 않습니다. 비방사형 원시값을 섭씨 온도로 변환하지 않습니다.

## 저장 이미지로 보정

저장 원본은 단일 채널 8-bit 또는 16-bit PNG/TIFF를 사용합니다. 한 실행에 해상도와 인코딩이 다른 자료를 섞지 않습니다. 입력 폴더에는 보정 전 원본만 둡니다.

계산 후 보고서를 검토하고 바로 카메라 영상에 적용하려면:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 wizard --images .\my_images --source usb --device 0
```

```bash
bash run.sh wizard --images '/data/boson/*.png' --source ros --topic /boson/image_raw
```

`--output`은 모든 시도를 보관할 새 폴더입니다. 예: `--output calibration_data/review01`. 거절하면 `attempt_001` 결과를 보존하고 다음 입력을 `attempt_002`에서 처리합니다. 사진을 찍은 **동일한 카메라·렌즈·초점·출력 모드**를 적용 대상으로 선택해야 합니다. USB 장치 인덱스는 카메라 고유 시리얼 번호가 아니며 프로그램이 동일한 물리 장치를 자동 증명하지는 않습니다.

계산과 보고서 저장만 하는 기존 배치 명령도 유지합니다. 이 명령은 카메라를 열거나 적용을 시작하지 않습니다:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 intrinsic --images .\my_images --output .\calibration_data\offline01.yaml
```

```bash
bash run.sh intrinsic --images '/data/boson/*.png' --output calibration_data/offline01.yaml \
  --target calib/config/target.yaml --quality calib/config/quality.yaml
```

출력 경로는 기존 결과와 겹치지 않게 지정합니다. `--min-views`, `--max-rms`는 품질 YAML보다 우선합니다. 기존 체커보드는 명시적으로 선택할 수 있습니다:

```bash
bash run.sh intrinsic --images '/data/checker/*.png' --output calibration_data/checker01.yaml \
  --pattern checkerboard --corners-x 9 --corners-y 5 --square-m 0.05
```

## 결과와 품질 판정

라이브 촬영의 기본 경로는 `calibration_data/boson_<날짜시각>/`입니다.

- `images/*.png`: **원본 비트 깊이를 유지**한 촬영 이미지.
- `session.json`: 촬영 소스, 시각, SHA-256, 인코딩, 타깃 및 검출점.
- `boson_intrinsic.yaml`: `camera_matrix`, `distortion_coefficients`, 해상도, RMS, 프레임별 오차, 타깃과 품질 판정.
- `boson_intrinsic_debug/report.json`: 각 이미지의 채택/거부 사유와 관측점.
- `*_detected.png`, `coverage.png`: 검출과 화면 분포 확인.
- `sample_original.png`, `sample_undistorted.png`: 8-bit 표시용 보정 전후 비교.
- `sample_undistorted_native.png`: 원본 비트 깊이의 왜곡 보정 예제.
- `*_debug/review.html`: 이미지가 내장된 독립 HTML 보고서. K/D, RMS, 뷰별 오차 그래프, 검출 커버리지와 실제 재투영 잔차 히트맵, 원본/보정/검출 오버레이를 함께 보여줍니다.
- `*_debug/review_manifest.json`: 보고서와 보정 YAML의 SHA-256 연결. 검토 중 값이나 보고서가 바뀌면 적용하지 않습니다.
- `*_debug/decision_*.json`: 사용자 선택, 적용 대상, 승인한 결과의 해시와 적용 성공/실패 기록. 이전 선택을 다음 실행의 자동 승인으로 재사용하지 않습니다.

### 마지막 적용 선택과 실제 출력

보고서가 열리면 그래프의 높은 오차 뷰, 히트맵의 데이터 없는 영역, 검출점이 실제 원 중심에 놓이는지, 보정 후 영상이 자연스러운지 확인합니다. 히트맵의 회색 칸은 오차 0이 아니라 **관측 데이터 없음**입니다. RMS와 잔차는 보정에 사용한 영상의 모델 적합도이며 별도 검증 영상의 정확도는 아닙니다.

터미널의 마지막 질문에서:

| 입력 | 결과 |
|---|---|
| `yes` | 검토한 보정값으로 선택한 카메라의 실시간 영상 처리를 시작 |
| `no` | 적용 없이 새 원본 사진 입력/촬영 단계로 복귀 |
| `quit` 또는 입력 종료(EOF) | 적용 없이 종료 |
| Enter / 인식할 수 없는 답 | 승인하지 않고 다시 질문 |

**적용 범위는 호스트의 영상 처리입니다. Boson 펌웨어나 EEPROM에 K/D를 기록하지 않습니다.**

- **USB:** 원본(왼쪽)과 실시간 왜곡 보정(오른쪽)을 표시합니다. 계산은 Y16의 `uint16`을 유지하고 화면 표시만 8-bit로 정규화합니다. USB 적용은 화면이 필요합니다. 다른 프로그램의 UVC 출력이나 장치 설정 자체는 바뀌지 않습니다.
- **ROS:** 기본 `/boson/image_raw`를 받아 `/boson/image_rect`와 `/boson/camera_info_rect`를 발행합니다. 후속 노드는 이 두 출력 토픽을 사용합니다. 원본 시간·`frame_id`·비트 깊이를 유지하며, 이미 보정된 영상의 CameraInfo는 `D=0`, `R=I`, `P=[K|0]`입니다. 출력 토픽은 `--rectified-topic`, `--camera-info-topic`으로 바꿀 수 있고 입력과 서로 달라야 합니다.
- 실행 중 Q 또는 Ctrl+C로 중단합니다. `--frame-timeout 8`은 프레임이 끊기면 중단하는 시간(초)이며, `--apply-frames N`은 N개 처리 후 종료합니다. `0`은 사용자가 종료할 때까지 실행합니다. 프로그램 종료 후 보정 영상 처리는 중단됩니다.

수치 검사 실패나 `provisional` 타깃 상태는 보고서와 승인 직전에 표시합니다. 유효한 행렬이면 경고를 검토한 사용자가 적용 여부를 결정합니다. 구조가 잘못되었거나 비유한 파라미터·해상도 불일치 등으로 처리할 수 없는 결과는 적용하지 않습니다. 적용 중 장치 연결 실패나 인코딩 변경이 생기면 실패를 기록하고 새 입력 단계로 돌아갑니다.

저장한 결과를 다시 검토해 적용할 때도 승인을 다시 받습니다. 같은 실행에서 만들어진 YAML과 `*_debug` 폴더를 함께 보관하세요. 예:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 review --calibration .\calibration_data\offline01.yaml --source usb --device 0
```

```bash
bash run.sh review --calibration calibration_data/offline01.yaml --source ros --headless --no-browser
```

보고서가 없는 과거 버전의 YAML은 원본으로 `intrinsic` 또는 `wizard`를 다시 실행하여 보고서를 만듭니다. `Rectifier` 클래스의 `rectify(frame)`을 이용하면 별도 Python 영상 처리 코드에서도 같은 네이티브 보정을 사용할 수 있습니다.

기본 RMS 기준은 0.8 px입니다. 화면 coverage, 자세 다양성, 파라미터 유효성, 왜곡 함수의 단조성도 함께 확인합니다. 보정 전에 정규화한 homography의 Zhang 제약 특이값 비율(s4/s0 ≥ 0.01)을 검사해 정면·이동 위주 데이터의 퇴화를 거부합니다. 이 검사는 추정된 K의 기울기에 의존하지 않습니다. 품질 판정은 촬영 안내용 휴리스틱이며 장비 정확도 인증이 아닙니다. RMS는 학습에 사용한 관측점의 재투영 오차입니다.

종료 코드: `0` 정상 실행/수치 품질 통과, `2` 입력·장치·검출 부족 오류, `3` 결과가 생성됐지만 품질 재검토 필요, `130` 사용자 중단. `quality.deployment_ready`는 타깃도 verified이고 수치 검사도 통과한 경우에만 true입니다. 이 플래그도 실제 장비 반복 검증을 대체하지 않습니다.

## 연결 문제 해결

- **Windows에서 640×512 컬러 영상이 들어옴:** Boson 컬러화 출력일 수 있습니다. Y16 원본 협상과 USB 인덱스를 확인합니다. 이 영상은 자동으로 320×256으로 축소해 보정하지 않습니다.
- **320×258 프레임:** telemetry가 활성화된 경우에만 `--telemetry-rows 2`를 지정합니다. 마지막 두 행을 명시적으로 제외합니다.
- **ROS 프레임 없음:** `ros2 topic list`, `ros2 topic info /boson/image_raw -v`, `ROS_DOMAIN_ID`와 publisher 상태를 확인합니다. subscriber는 sensor-data QoS를 사용합니다.
- **원 검출 실패:** 열 대비, 초점, 거리, 원 전체가 영상 안에 있는지 확인합니다. Space에서 거부되면 `last_rejected.png`에 원본을 저장합니다. 명암 반전은 자동 시도합니다.
- **높은 RMS/coverage 부족:** 오버레이에서 오검출을 확인하고 더 넓은 화면 위치·다양한 기울기로 새로 촬영합니다. 장수만 늘리거나 임계값만 높여서 해결하지 않습니다.

[FLIR의 Y16 캡처 안내](https://flir.custhelp.com/app/answers/detail/a_id/3387/~/flir-oem---boson-video-and-image-capture-using-opencv-16-bit-y16), [OpenCV 보정 API](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html)를 기준으로 구현했습니다.

## 개발 및 ROS 패키지 빌드

일반 Python 설치는 **저장소 루트**에서 수행합니다.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
firesight-calib --help
```

ROS 워크스페이스에서는 기존 ament 패키지도 사용할 수 있습니다.

```bash
mkdir -p ~/firesight_ws/src
cd ~/firesight_ws/src
git clone https://github.com/HaechiAR/calib.git
cd ~/firesight_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 run firesight_calib firesight-calib wizard --source ros
```

외부 보정(`extrinsic`), 점 투영(`project`), 설정 검증(`verify-config`) 명령은 유지합니다. 이번 변경의 중심은 Boson 내부 보정입니다. 합성 self-check와 실제 장비 검증을 구분하며, 실제 Boson/보드의 검출률과 정확도는 장비에서 추가 확인해야 합니다.

현재 구현과 개발 기록은 모두 `main`에 포함합니다:

- [초기 설계 및 구현 범위](calib/docs/design.md)
- [Boson 타깃·촬영 설계](docs/superpowers/specs/2026-09-08-boson-intrinsic-design.md)
- [검토·승인·적용 설계](docs/superpowers/specs/2026-09-14-review-and-apply-design.md)
- [검토·적용 구현 및 검증 계획](docs/superpowers/plans/2026-09-14-review-and-apply.md)

GitHub Actions는 Windows/Ubuntu의 Python 3.10·3.13과 ROS 2 Humble에서 검사합니다. ROS 작업에서는 실제 DDS로 보정된 mono8/mono16 영상과 CameraInfo를 송수신하는 테스트를 필수 실행합니다. 이 테스트는 실제 Boson 하드웨어 검증과 구분됩니다.
