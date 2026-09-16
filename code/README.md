# 코드 보존 범위

`original/`은 초기 실험 파일을 내용 변경 없이 복사했다. `selected/`는 후속 실험의 직접 근거가 되는 파일만 보존했다. 팀 저장소 전체, pretrained 모델, 외부 PiDiNet/TEED 전체 소스, 장치 adapter 전체, 모든 패키지와 데이터는 포함하지 않는다. 따라서 `selected`의 runner를 이 폴더에서 바로 실행할 수 있다고 보장하지 않는다.

| 파일 | 역할 |
|---|---|
| [flir_view_test.py](original/flir_view_test.py) | 첫 장치 OpenCV 표시, gray 정규화·INFERNO |
| [thermal_capture.py](original/thermal_capture.py) | `s` 키에서 gray PNG 저장 |
| [thermal_image_to_pointcloud.py](original/thermal_image_to_pointcloud.py) | stride 2, grayscale 색을 포함한 height field PLY |
| [thermal_image_to_pointcloud_backup.py](original/thermal_image_to_pointcloud_backup.py) | stride 4, XYZ만 있는 PLY |
| [firesight_radar_core.py](original/firesight_radar_core.py) | 난수 점군의 outlier/평면 분리 GUI |
| [contours.py](selected/obstacle_edges/contours.py) | 연결 경계 선택, 세선화, 표시 |
| [run_smoke.py](selected/obstacle_edges/run_smoke.py) | SmokeBasement 의사 라벨 adaptation·비교 실험 |
| [analyze.py](selected/boson_review/analyze.py) | 이미 합성된 HUD의 초록 stroke 감사 |
| [verify_video.py](selected/boson_review/verify_video.py) | 동일 확률맵의 상세도/출력 폭 비교 |
| [distance_width.py](selected/edge/teed/distance_width.py) | smoothstep 폭, EMA, 결측 정책, alpha raster |
| [depth_replay.py](selected/edge/teed/depth_replay.py) | offline 거리 재생 입력 계약 |
| [realtime_core.py](selected/edge/teed/realtime_core.py) | FP32 eager/CUDA Graph 모델 실행 |
| [stability.py](selected/edge/teed/stability.py) | 지속성 집계와 판정 |
| [benchmark.py](selected/experiments/distance_width/benchmark.py) | renderer/합성 거리 benchmark |
| [realtime_compare.py](selected/experiments/distance_width/realtime_compare.py) | CPU/eager/Graph 순차 계측 |

카메라 스크립트는 실행 즉시 `VideoCapture(0)`을 열고, pointcloud 스크립트는 현재 작업 폴더의 같은 이름 PLY를 덮어쓴다. 원본 보존용 폴더에서 실행하지 말고 별도 작업 폴더에 필요한 입력을 복사해 실행해야 한다. 이번 정리에서는 원본 코드를 실행하거나 수정하지 않았다.

PLY만 재현하려면 빈 작업 폴더에 `frame_0.png`와 해당 변환 스크립트를 복사하고 NumPy/OpenCV가 있는 Python으로 실행한다. `big`는 81,920점, backup은 20,480점이 되어야 한다. 이 실행은 카메라 없이 가능하지만 결과는 metric depth가 아니다. Open3D가 필요한 radar 스크립트는 합성 데이터 GUI이며 별도 기능이다.

카메라 표시·height field의 의존성은 OpenCV/NumPy, radar GUI는 Open3D/Pandas/NumPy다. 후속 경계 세선화에는 `cv2.ximgproc`가 필요해 OpenCV contrib 구성이 요구된다. 원 실험의 torch/CPU/CUDA 버전은 각 결과 JSON에 있으며, 그 버전 문자열을 이 아카이브 환경의 설치 완료 목록으로 해석하지 않는다.

구문이 깨진 `.save` 두 파일은 [history](../docs/evidence/history/)에 실행 대상과 분리했다. 수정하여 과거 성공본처럼 보이게 만들지 않았다. 모든 복사본 hash와 원본 경로는 로컬 전용 `source-map.json`에 보존했다.
