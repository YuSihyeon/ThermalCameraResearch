# 거리 굵기 실시간 실행과 Orin Nano 검증

센서는 미정이다. `run_realtime.py`는 USB/V4L2/CSI 카메라, 같은 호스트의 거리 어댑터, TEED, 거리 굵기, HUD 출력을 연결한다. `edge/orin/run_distance.py`도 같은 실행기의 진입점이다. 물리 센서의 SDK 어댑터와 실제 카메라 등록 보정은 장치 선정 후 필요하다. [어댑터 계약](DEVICE_ADAPTER.md)을 따른다.

## 실행 환경

PC 시험은 별도 `.venv-cuda`에 설치했다. CUDA에 맞는 PyTorch와 `opencv-contrib-python`(ximgproc 세선화)을 사용한다. Jetson에서는 설치한 JetPack과 호환되는 [NVIDIA PyTorch 배포판](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)을 사용한다. 보드의 GStreamer/카메라 지원 OpenCV를 일반 PC wheel로 덮어쓰지 않는다. CSI는 OpenCV GStreamer 빌드가 필요하다.

```bash
python3 -c "import torch,cv2; print(torch.__version__,torch.cuda.is_available()); print(cv2.__version__,hasattr(cv2.ximgproc,'thinning')); print(cv2.getBuildInformation())"
```

FP32가 기본이다. `--cuda-graph`는 고정 크기 추론의 실행 비용을 줄이며, 해당 API를 지원하는 PyTorch가 필요하다. 미지원 보드는 이 옵션을 빼고 CUDA eager로 측정한다. `--half`는 선택 사항이며 실제 입력에서 FP32와 품질 차이를 확인한 뒤 사용한다. 해상도는 8의 배수, 최소64다.

## 공개 영상으로 재현

저장소 루트에서:

```bash
python edge/teed/run_realtime.py \
  --source experiments/obstacle_edges/deliverables/media/smoke_heldout_original.mp4 \
  --loop --replay-fps 30 --target-fps 30 --duration 180 \
  --synthetic-depth --depth-port 0 --device cuda --cuda-graph \
  --record ../outputs/distance_width_realtime_2026-09-14/gpu_graph_30.mp4 \
  --metrics ../outputs/distance_width_realtime_2026-09-14/gpu_graph_30.json \
  --frame-csv ../outputs/distance_width_realtime_2026-09-14/gpu_graph_30.csv \
  --require-stable
```

원본은320×240·10fps·120프레임이다. 위 명령은 반복 재생을30fps로 가속하여 부하를 만든다. 서로 다른 공개 영상5400프레임이나 실제30fps 촬영 데이터라는 뜻이 아니다. 거리 맵은 `SYNTHETIC DEPTH`로 표시되며 실제 TCP 송수신 경로를 통과한다. 영상의 실제 거리를 추정한 결과가 아니다. 영상 저장의 `VideoWriter.write()` 호출과 주석 처리는 지연 측정에 포함되며 종료 시 인코더·디스크 flush는 제외한다. `--show`를 추가하면 창 표시 호출도 포함된다.

## 실제 보드 연결

보드에서 실제 보정 JSON과 센서 어댑터를 준비한 후, 첫 터미널에서:

```bash
python3 edge/orin/run_distance.py --source /dev/video0 \
  --capture-width 640 --capture-height 480 --target-fps 30 \
  --width 320 --height 240 --calibration actual_registration.json \
  --device cuda --cuda-graph --duration 180 --show \
  --metrics orin_distance.json --frame-csv orin_distance.csv \
  --verify-device
```

다른 터미널에서 [어댑터 전송 명령](DEVICE_ADAPTER.md)을 실행한다. 동일한 보정 ID와320×240 출력 해상도를 지정한다. USB 카메라는 `--source 0`, CSI는 `--source csi://0`도 가능하다. `--input-format y16`은 실제16bit 단일채널 캡처를 요구하고, `gray8`은 실제8bit 단일채널 입력용이다. 자동 모드는 실제 BGR8만 허용한다. 범용 GStreamer 경로는 지원하지만 파일 재생을 위장할 수 있으므로 `--verify-device`에는 직접 카메라 소스만 허용한다.

새 검증 실행기를 기존 `edge/orin/run_pipeline.py`의 색상 후보·OSD 프로파일·출력 화면 크기 옵션과 혼동하지 않는다. 이 실행기는320×240 출력 픽셀 기준 거리 굵기를 검증한다. 외부 뷰어가 확대하면 보이는 굵기도 확대된다. 실제 안경 해상도와 디스플레이 지연은 그 구성으로 별도 측정한다.

## 거리와 시간 계약

- float32 미터 단위의 카메라 Z 깊이. 포인트클라우드는 카메라 K·왜곡·센서→카메라 R,t로 투영한다. 이미지와 공통 시야여야 한다.
- 미측정 픽셀은 NaN이다. 빈 영역을 임의 거리로 채우지 않는다. 1m에서5px,8m에서1px이며 그 사이는 smoothstep 연속 폭이다.
- 작은 변화에는50ms 시정수의 인과적 EMA를 사용한다. 미래 프레임을 기다리지는 않지만 완만한 변화에 필터 응답 지연이 생긴다. 큰 폭 변화는 즉시 반영한다.
- 카메라 수신 완료와 센서 취득/SDK 수신 시각 모두 **같은 호스트 `time.perf_counter()`**를 사용한다. 센서 내부 tick, UTC, 다른 PC 시각을 그대로 보내지 않는다. 프로토콜 표기는 `host_perf_counter`다. Windows는 Python3.10 이상을 요구한다. [Python 시계 문서](https://docs.python.org/3.11/library/time.html#time.perf_counter).
- 영상 시각 이후의 거리값과100ms보다 오래된 거리값을 사용하지 않는다. 결측·지연·통신 끊김은1px와 상태 표시로 처리한다. 거리 입력을 기다리며 영상을 멈추지 않는다.
- 보정 ID 검사는 설정 혼용을 막는 수단이다. JSON 유효성 검사만으로 물리 보정의 정확성이 증명되지는 않는다.

## 안정 FPS 판정

예열 뒤 **180초 이상**을 측정하며 아래 조건을 모두 충족해야 한다.30fps의 프레임 예산은33.333ms다.

| 항목 | 기준 |
|---|---|
| 평균 출력 FPS | 목표의99% 이상 |
| 모든 완전한10초 구간 FPS | 목표의98% 이상 |
| 처리 P99 | 1프레임 예산 이하 |
| 처리 예산 초과 프레임 | 1% 이하 |
| 최신 슬롯 덮어쓰기 + 재생 스케줄 누락 | 예상 입력 수의1% 이하 |
| 호스트 프레임 수신→출력 완료 P99 | 2프레임 예산 이하 |
| 출력 간격 P99 / 최대 | 1.5 / 3프레임 예산 이하, 시작·종료 공백 포함 |
| 실행 오류 / 종료 | 오류 없음, 수집 스레드 정상 종료 |

`--require-stable`은 미달 시 exit3, 실행 오류는 exit2다. 짧은 시험은 기능이 정상이더라도 안정성 PASS가 될 수 없다. `--verify-device`는 실제 Orin Nano 호스트,CUDA,직접 카메라,30fps 이상,실측 거리,95% 이상의 유효 거리 프레임,검출선 거리 대응 비율과 통신 오류까지 확인한다. 대응 비율 기본은95%이며 `--min-depth-edge-coverage`로 명시적으로 조정하고 결과에 기준을 기록한다. 합성 입력으로 장치 검증을 통과할 수 없다.

지연은 GPU 작업 완료를 기다린 뒤 기록한다. `host_frame_age_ms`는 OpenCV `read()`가 프레임을 반환한 시점부터 출력 함수가 끝날 때까지다. 센서 노출 시간, 드라이버 내부 버퍼, 디스플레이 스캔아웃은 포함하지 않는다. MP4의 재생 FPS만으로 실시간 성능을 판단하지 말고 JSON과 프레임 CSV를 함께 확인한다.

`depth.buffer_drops`는8개 기록 버퍼에서 과거 데이터를 퇴출한 수이며 영상 누락 수가 아니다. `source_stats.overwritten`은 소비 전에 덮어쓴 영상 수다. 센서 SDK 내부 손실·장치 클럭 드리프트·물리 정렬 오차는 연결할 센서 어댑터와 실장 시험에서 추가 확인한다.
