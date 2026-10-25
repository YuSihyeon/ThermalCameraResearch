# Firesight 캘리브레이션 시스템 설계

- 작성일: 2026-08-27
- 대상 저장소: `HaechiAR/firesight_sw`
- 대상 경로: `calib/`
- 실행 환경: Ubuntu 22.04 x86_64, ROS 2 Humble, Python 3.10
- 센서: FLIR Boson 320 (320×256), Intel RealSense D456, Livox Mid-360S
- 상태: 사용자 승인 설계

## 1. 목적

Firesight의 고정 센서 리그에 대해 다음 보정값을 측정하고 검증한다.

1. Boson 320 내부 파라미터 `K_boson`, `D_boson`
2. D456 공장 출고 내부·외부 파라미터의 유효성 및 provenance
3. `T_boson_from_d456_color`
4. `T_boson_from_livox`
5. D456 depth와 Livox point cloud를 Boson 열영상에 투영하는 검증 결과

시스템은 계산값만 출력하지 않는다. 원본 입력, 설정, 자동 검출점, 사람의 수정 기록,
수치 품질, 시각화, 승인 이력을 하나의 추적 가능한 결과로 보존한다.

## 2. 설계 원칙

Firesight 캘리브레이션 시스템은 다음 원칙을 따른다.

- 단일 시작점에서 현재 상태와 다음 작업을 안내한다.
- 좌표계, 해상도, 센서 프레임, 타임스탬프를 solve 경계에서 검증한다.
- 낮은 RMS만으로 결과를 승인하지 않는다.
- 입력과 출력의 provenance를 checksum으로 결합한다.
- 실행 결과를 덮어쓰지 않는 immutable run으로 저장한다.
- 수치 기준과 운영자 승인을 모두 통과한 결과만 `latest`로 승격한다.
- 정적 분석 결과뿐 아니라 투영 overlay와 3D 화면으로 최종 검증한다.

센서 구성과 작업 흐름은 Firesight 요구에 맞춰 설계한다.

- Boson 열영상 `mono8`/`mono16` 입력과 열 대비 전처리
- D456 factory calibration 스냅샷 및 신선도 검사
- Boson–D456–Livox 센서 관계
- 자동 검출 후 사람 확인·수정, 그리고 주변 데이터 기반 재정밀화
- D456 depth 및 Livox point cloud의 Boson 투영
- run별 정적 PNG와 독립 실행 가능한 HTML 리포트

## 3. 범위와 비범위

### 3.1 포함

- Boson pinhole/plumb-bob 내부 보정
- D456 color/depth factory intrinsic 및 depth-to-color extrinsic 검증
- Boson–D456 color 외부 보정
- Boson–Livox 외부 보정
- 설정 가능한 multimodal 보정판 모델
- ROS 2 live capture와 재현 가능한 offline re-solve
- 자동 검출, 직접 클릭 수정, sub-pixel/cluster 재정밀화
- immutable run, checksum, latest pointer, stale 판정
- PNG, CSV, YAML, NPZ, HTML, OpenCV, Open3D 진단
- D456 depth와 Livox point cloud의 Boson 투영

### 3.2 제외

- Boson과 D456의 stereo depth 계산
- 센서 드라이버 자체 구현
- 센서 펌웨어 변경
- 주행 중 온라인 self-calibration
- 자동 TF 배포 및 차량 운용 TF tree 변경
- targetless calibration
- fisheye 모델. v1은 임시 50° HFOV 렌즈 가정의 `plumb_bob`만 지원하며,
  실제 렌즈가 초광각으로 확인되어 물리 유효성 검사를 통과하지 못하면 결과 승격을
  차단하고 별도 설계 변경을 수행한다.

## 4. 물리 보정판

주문 대상은 Calib.io의 `Passive Thermal Checkerboard Target for LWIR camera
calibration`, 외형 600×400 mm이다.

현재 임시 기하값은 다음과 같다.

```yaml
target:
  status: provisional
  revision: firesight_thermal_board_provisional_v1
  outer_width_m: 0.600
  outer_height_m: 0.400
  checkerboard:
    squares_x: 10
    squares_y: 6
    square_length_m: 0.050
    internal_corners_x: 9
    internal_corners_y: 5
```

유효 패턴은 500×300 mm이고 바깥 여백은 각 방향 50 mm이다. 주문 규격이 확정되면
실측값으로 `target.yaml`만 갱신한다. 모든 run은 사용한 target 설정 전체와 SHA-256을
저장한다.

가시광 인식을 위해 여백 또는 탈착식 carrier frame에 AprilTag를 부착한다. Livox
인식을 위해 반사 마커 또는 구형 타깃을 체커보드 영역을 가리지 않는 위치에 고정한다.
각 tag corner와 LiDAR marker center의 보드 좌표를 `target.yaml`에 미터 단위로
기록한다. 이 좌표는 실측 또는 제작 도면으로 검증한 뒤에만
`target.status: verified`로 변경할 수 있다.

`provisional` target으로 캡처, 개발, solve, report 생성은 허용한다. 그러나
운용용 `ACCEPTED` 및 `latest` 승격은 금지한다. Boson intrinsic의 초점거리·왜곡은
square scale에 직접 의존하지 않지만, 보드 식별과 외부 보정 translation은 실제
기하값에 의존하므로 예외를 두지 않는다.

## 5. 좌표계와 변환 방향

Firesight 보정 결과의 기준은 `boson_optical_frame`이다. optical frame은 ROS/OpenCV
관례인 x 오른쪽, y 아래, z 전방을 사용한다.

주 결과는 다음 방향으로 저장한다.

```text
p_boson = T_boson_from_d456_color * p_d456_color
p_boson = T_boson_from_livox      * p_livox
```

저장 필드 이름은 각각 `T_boson_from_d456_color`와 `T_boson_from_livox`이며,
`source_frame`, `target_frame`, 4×4 행렬, translation, quaternion xyzw, RPY rad를
함께 기록한다. 변환의 역행렬은 필요할 때 계산하며 별도 진실 소스로 저장하지 않는다.

v1은 기존 차량 TF tree를 자동 변경하지 않는다. 검증된 결과에서
`static_transform_publisher` 예시를 생성할 수 있지만 운영자가 명시적으로 승인하기
전에는 실행하지 않는다.

## 6. 패키지 구조

하나의 `ament_python` 패키지로 시작한다.

```text
calib/
├── package.xml
├── setup.py
├── setup.cfg
├── config/
│   ├── sensors.yaml
│   ├── target.yaml
│   └── quality.yaml
├── launch/
├── firesight_calib/
│   ├── app/          # calib_home, 상태 머신, preflight
│   ├── ros/          # image/depth/cloud 입력, QoS, sync, TF snapshot
│   ├── target/       # checkerboard/tag/marker metric model
│   ├── detection/    # thermal, AprilTag, LiDAR 후보와 refinement
│   ├── interaction/  # OpenCV/Open3D point review와 edit log
│   ├── intrinsic/    # Boson intrinsic solve와 model diagnostics
│   ├── extrinsic/    # D456/Livox joint solve와 quality
│   ├── projection/   # distortion-aware projection, depth chain, z-buffer
│   ├── storage/      # session, run, manifest, checksum, latest
│   └── report/       # PNG/CSV/HTML 생성
├── test/
└── docs/
```

계산 코어는 ROS와 UI에 의존하지 않는다. ROS adapter는 입력을 NumPy 기반 내부
계약으로 변환하고, UI는 코어의 결과만 검토·수정한다. 따라서 저장된 sample을
사용한 offline 재계산이 live capture와 같은 코드를 사용한다.

## 7. 설정 계약

`sensors.yaml`은 다음 값을 명시적으로 요구한다.

- Boson image topic, optical frame, image encoding 허용 목록
- D456 color image/camera_info topic과 color optical frame
- D456 depth image/camera_info topic, depth optical frame, depth scale
- D456 depth-to-color transform를 읽을 TF frame pair
- Livox PointCloud2 topic, expected cloud frame, intensity/reflectivity field
- 센서별 QoS
- pairwise synchronization queue와 최대 timestamp 차이
- calibration data root
- 고정 장착물을 식별하는 `rig_id`와 `rig_revision`
- Boson lens/focus 상태를 식별하는 `boson_lens_revision`

토픽 이름과 frame ID는 자동 추측하지 않는다. 설정 누락 또는 live header 불일치는
preflight 실패다. 하드웨어 동기화는 없는 것으로 가정하고 pairwise
`ApproximateTimeSynchronizer`를 사용한다. 기본 최대 timestamp 차이는 0.10 s이며,
보드는 캡처 전 정지 상태를 유지한다. 실제 delta는 sample마다 저장한다.
센서 재장착, bracket 변경 또는 Boson focus 조정 시 revision을 올려야 하며, revision이
바뀌면 연결된 extrinsic과 final bundle은 `STALE`이다.

`quality.yaml`은 품질 threshold를, `target.yaml`은 보드·tag·LiDAR marker의 metric
geometry만 소유한다. CLI override는 run manifest에 반드시 기록한다.

## 8. 공통 캡처와 직접 수정

각 캡처는 solve보다 먼저 draft session에 저장한다.

```text
ROS messages
 → timestamp/frame/size/sensor gate
 → 원본 sample 원자 저장
 → 자동 검출
 → 운영자 확인·추가·이동·삭제
 → local refinement
 → 수정 이력 append
 → solve 입력 확정
```

자동 검출점, 운영자의 클릭 seed, 최종 refinement 결과를 모두 별도로 보존한다.
Boson과 D456의 클릭점은 주변 영상에서 sub-pixel refinement를 적용한다. Livox
클릭은 단일 point를 선택하지 않고 주변 ROI를 seed로 사용한다. ROI에서 intensity,
reflectivity, 거리 연속성과 기하 조건으로 cluster를 추출하고 plane, sphere 또는
marker center를 적합한다.

점 ID는 두 센서 화면에서 같은 색과 번호로 표시한다. 운영자는 undo, point 이동,
삭제, 재검출, capture 전체 제외를 수행할 수 있다. 자동점과 최종점의 차이가 큰
항목은 경고하고 승인 화면에서 최우선으로 보여준다.

## 9. Stage A — Boson 내부 보정

Boson 입력은 320×256의 `mono8` 또는 `mono16`을 지원한다. 원본 값은 보존하고
검출용 프레임만 8-bit로 정규화한다. global percentile normalization, polarity
반전, CLAHE 후보를 평가하고 가장 안정적으로 전체 checkerboard를 검출한 후보를
사용한다. 전처리 방법과 파라미터는 sample metadata에 기록한다.

고정밀 검출은 원본 해상도에서 `findChessboardCornersSB` 계열을 사용한다. 실시간
preview는 낮은 비용 경로를 사용하지만 capture 판정은 항상 고정밀 결과로 한다.

최소 15개 유효 view를 요구한다. 다음 다양성 gate를 함께 적용한다.

- 8×6 image grid coverage 80% 이상
- 8개 방향 모두에서 정규화 반경 0.90 이상 도달
- 방향별 서로 다른 view 2개 이상
- board tilt span 20° 이상
- 중복 pose, 낮은 열 대비, blur, 경계 밖 corner 거부

OpenCV pinhole/plumb-bob에서 `k3_fixed`와 `k3_free` 후보를 계산한다. view-wise
held-out 오차, 외곽 p95, fold 간 mapping spread, 전체 영상 Jacobian determinant와
fold 여부를 검사한다. 단순 모델이 통계적으로 열등하지 않으면 `k3_fixed`를
선택한다.

합격 기준은 RMS 0.8 px 이하이며, coverage와 물리 유효성 gate도 모두 통과해야 한다.

## 10. Stage B — D456 factory calibration 검증

D456의 intrinsic을 재측정하여 대체하지 않는다. 다음 정보를 한 snapshot으로
저장한다.

- device serial과 firmware
- color/depth 해상도와 encoding
- color/depth `CameraInfo`
- depth scale
- `T_d456_color_from_d456_depth`
- 관련 ROS frame ID와 TF authority

행렬 유한성, 양의 focal length, principal point 범위, distortion shape, 해상도 일치,
rotation orthonormality 및 determinant를 검사한다. snapshot의 checksum을 이후
Boson–D456 run에 결합한다. serial, 해상도, camera_info 또는 factory transform이
달라지면 기존 Boson–D456 결과는 `STALE`이다.

## 11. Stage C — Boson–D456 외부 보정

같은 정지 보드를 Boson에서는 thermal checkerboard로, D456 color에서는 AprilTag로
검출한다. tag와 thermal corner가 공유하는 `target.yaml` 보드 좌표가 두 관측을
연결한다.

각 view에서 두 카메라의 board pose를 초기 추정하고 다음 변환을 얻는다.

```text
T_boson_from_d456_color
  = T_boson_from_board * inverse(T_d456_color_from_board)
```

여러 거리와 기울기의 view를 모아 rig transform 하나와 view별 board pose를 함께
최적화한다. 최적화는 Boson thermal corner와 D456 tag corner의 distortion-aware
pixel residual을 최소화한다. 한 view의 결과를 최종값으로 사용하지 않는다. 최소
8개 accepted view를 요구하며, 보드가 Boson 영상 너비의 35–80%를 차지하는 범위에서
중앙·가장자리 위치와 정면·기울어진 자세를 모두 포함해야 한다.

held-out view에서는 D456 tag로 얻은 board pose로 thermal checkerboard corner를
Boson 영상에 투영해 실제 thermal corner와 비교한다. 중앙 pixel 오차 2.0 px 이하를
요구한다. 반복 solve 간 translation 차이는 10 mm 이하, rotation 차이는 0.3°
이하여야 한다.

## 12. Stage D — Boson–Livox 외부 보정

보드를 정지시킨 뒤 짧은 구간의 Livox frame을 누적한다. motion 또는 timestamp
조건을 위반한 frame은 누적하지 않는다. 보드 평면과 반사/구형 marker 후보를
자동 표시하고 운영자의 클릭으로 올바른 cluster를 확정한다.

Boson thermal corner와 Livox의 board plane·marker center를 `target.yaml` metric
geometry로 결합한다. view별 초기 transform을 구한 뒤 고정 rig transform과 view별
board pose를 공동 최적화한다. marker identity가 모호하거나 plane/marker가 충분한
기하 제약을 제공하지 않으면 solve를 차단한다. 최소 8개 accepted view를 요구하고,
보드 거리, 영상 내 위치, plane normal이 한 구간에 몰리지 않도록 diversity gate를
적용한다.

Livox board plane 적합 오차는 15 mm 이하를 요구한다. 반복 solve 간 translation
차이 10 mm 이하, rotation 차이 0.3° 이하를 함께 적용한다.

## 13. Stage E — 최종 투영 검증

D456 depth는 다음 체인으로 Boson에 투영한다.

```text
depth pixel + depth value
 → deproject(K_d456_depth, D_d456_depth, depth_scale)
 → T_d456_color_from_d456_depth
 → T_boson_from_d456_color
 → project(K_boson, D_boson)
```

Livox는 다음 체인을 사용한다.

```text
p_livox
 → T_boson_from_livox
 → project(K_boson, D_boson)
```

Boson 뒤쪽, 최소·최대 운용 거리 밖, 영상 경계 밖, 유효하지 않은 depth/point는
제외한다. 같은 pixel에 여러 점이 대응하면 z-buffer로 가장 가까운 값을 선택한다.
열영상 위에 depth 또는 point cloud를 거리색으로 표시하며 투영된 점의 원래 3D
좌표와 provenance를 조회할 수 있어야 한다.

## 14. 결과 상태와 승격

```text
DRAFT → SOLVED → CANDIDATE → ACCEPTED
                  └──────→ REJECTED

ACCEPTED → dependency 변경 → STALE
```

- `DRAFT`: 캡처 또는 직접 수정 중
- `SOLVED`: 계산됐으나 품질·사용자 검토 미완료
- `CANDIDATE`: 수치 gate 통과, 사용자 미승인
- `ACCEPTED`: 수치 gate 및 최종 overlay 사용자 승인 완료
- `REJECTED`: 수치 gate 실패
- `STALE`: target, sensor, intrinsic 또는 dependency checksum 변경

수치 합격만으로 자동 승격하지 않는다. 최종 projection overlay를 운영자가 승인해야
`ACCEPTED`가 된다. `target.status: provisional`이면 운영자 승인 여부와 무관하게
`ACCEPTED`가 될 수 없다.

## 15. 저장과 provenance

mutable data는 source tree 밖의 절대 경로에 저장한다. 기본 위치는
`~/.local/share/firesight_calib`이며 `FIRESIGHT_CALIB_DATA_DIR`로 변경할 수 있다.

```text
data_root/
├── sessions/<session_id>/
│   ├── session.yaml
│   ├── samples/<sample_id>/...
│   └── edits.yaml
├── runs/<run_id>/
│   ├── manifest.yaml
│   ├── config/
│   ├── input/
│   ├── results/
│   ├── debug/
│   └── report.html
└── latest/
    ├── boson_intrinsic.yaml
    ├── d456_factory.yaml
    ├── boson_d456.yaml
    ├── boson_livox.yaml
    └── firesight_bundle.yaml
```

run은 temp directory에 전체 파일을 기록하고 flush, fsync, SHA-256 manifest 검증 후
원자적으로 공개한다. `candidate`와 `rejected`도 보존하지만 `latest`를 바꾸지 않는다.
동일 `run_id` 중복, checksum 불일치, dependency stale은 기본 consumer에서 거부한다.

manifest에는 sensor identity, topics, frame IDs, timestamp delta, config snapshots,
target hash, dependency run IDs/checksums, 자동·수정·최종 points, software Git commit,
Python/OpenCV/ROS 버전, 품질 값, 승인자를 포함한다.

## 16. 운영 UI

단일 진입점은 다음과 같다.

```bash
ros2 run firesight_calib calib_home
```

홈 화면은 환경, 세 센서 토픽, target 상태, Boson intrinsic, D456 factory,
Boson–D456, Boson–Livox, 최종 bundle을 `OK/MISSING/STALE/FAILED`로 표시한다.
Enter는 아직 완료되지 않은 가장 이른 단계를 추천한다.

캡처 화면은 다음을 제공한다.

- 실시간 검출점과 point ID
- 열 대비, blur, timestamp delta, capture FPS
- coverage, tilt, 거리 및 다음 권장 보드 위치
- 자동점, 직접 수정점, refinement 결과의 색상 구분
- undo, point 이동/추가/삭제, capture 제외, re-detect
- solve 전 sample gallery

Open3D는 Livox point, board plane, marker 후보, 클릭 seed, fitted center를 서로 다른
색과 glyph로 표시한다.

## 17. 시각화와 리포트

각 run은 `report.html`과 다음 PNG/CSV를 생성한다.

### Boson intrinsic

- `coverage_heatmap.png`
- `per_view_errors.png`
- `error_histogram.png`
- `residual_vectors.png`
- `pose_distribution.png`
- `thermal_contrast.png`
- `radial_distortion.png`
- `distortion_grid.png`
- `jacobian_heatmap.png`
- `original_vs_undistorted.png`
- `model_comparison.png`

### Boson–D456

- 자동점·수정점·최종점 비교 overlay
- correspondence residual vector 및 heatmap
- view별 error와 held-out histogram
- 반복 translation/rotation 분산
- D456 depth-on-Boson overlay

### Boson–Livox

- board plane 및 marker fit 3D snapshot
- click seed와 fitted center 비교
- point-to-plane histogram
- view별 residual 및 거리별 residual
- Livox-on-Boson depth/intensity overlay

HTML 상단은 모든 합격 기준과 실제 값을 표로 비교하고 실패 이유를 표시한다. 각
그래프에서 관련 sample과 point ID를 찾을 수 있어야 한다. HTML은 별도 서버 없이
열 수 있는 self-contained report로 생성한다.

## 18. 오류 처리

다음 오류는 fail-closed로 처리하며 `latest`를 변경하지 않는다.

- 필수 토픽, camera_info 또는 TF 누락
- sensor serial, frame ID, encoding, 해상도 불일치
- timestamp delta 초과
- target config 누락, hash 불일치 또는 provisional 승격 시도
- 열 대비, blur, coverage, pose 다양성 부족
- 대응점 수 또는 공간 분포 부족
- 자동점과 직접 수정점의 identity 불일치
- LiDAR cluster fit 실패 또는 marker identity 모호성
- solve 실패, 비정상 rotation, negative-depth 또는 quality gate 실패
- dependency run checksum 불일치
- 파일 기록·checksum·atomic publication 실패

복구 가능한 오류는 해당 sample만 거절하고 세션을 유지한다. solve 또는 publication
오류는 draft session을 보존해 재시도할 수 있게 한다.

## 19. 테스트 전략

### 19.1 수학 단위 테스트

- 알려진 K/D/T와 합성 board pose로 intrinsic/extrinsic 복원
- noise, outlier, partial visibility에 대한 RANSAC 및 refinement
- D456 depth→color→Boson chain
- Livox→Boson projection 및 z-buffer
- transform 역방향 사용을 차단하는 frame 계약
- plumb-bob Jacobian과 full-frame fold 검사

### 19.2 메시지·계약 테스트

- `mono8`, `mono16`, D456 depth encoding, PointCloud2 field decode
- QoS 및 approximate sync
- frame, timestamp, image size, sensor identity gate
- target geometry와 point ID 일관성
- 자동점·수정점·최종점 audit trail

### 19.3 저장 장애 테스트

- YAML/NPZ/PNG 기록 중 실패
- temp run rename 실패
- checksum 변조
- duplicate run ID
- stale dependency
- rejected/candidate가 latest를 바꾸지 않음

### 19.4 Offline 회귀 테스트

실제 센서에서 얻은 익명화 sample 또는 rosbag2 fixture를 고정해 detector, solver,
quality, report 결과를 회귀 검사한다. golden 숫자는 허용 오차와 함께 관리한다.

### 19.5 실제 센서 검증

Ubuntu 22.04와 ROS 2 Humble에서 다음을 확인한다.

1. 세 센서 토픽 및 frame preflight
2. target warm-up과 thermal contrast
3. 각 stage capture/review/re-solve
4. 합격선과 반복 안정성
5. D456 depth-on-Boson overlay
6. Livox-on-Boson overlay
7. 프로세스 재시작 후 latest 재로딩과 checksum 검증

실제 센서 검증을 통과하지 않은 결과는 소프트웨어 테스트가 통과해도 운용 승인하지
않는다.

## 20. 합격 기준

```text
Boson intrinsic RMS                    <= 0.8 px
D456 → Boson held-out median            <= 2.0 px
Livox board plane residual              <= 15 mm
반복 extrinsic translation 차이         <= 10 mm
반복 extrinsic rotation 차이            <= 0.3 deg
```

위 기준 외에도 coverage, physical validity, provenance, dependency freshness와 최종
overlay 승인을 모두 만족해야 한다. 하나라도 실패하면 `ACCEPTED`가 아니다.

## 21. 구현 순서

1. package/config/core contracts와 immutable storage
2. Boson image decode, thermal preprocessing, target model
3. Boson intrinsic capture/solve/report
4. D456 factory snapshot 및 검증
5. 공통 point review/edit/refinement UI
6. Boson–D456 joint solve와 depth projection
7. Livox decode, marker/plane refinement, Boson–Livox solve
8. 통합 dashboard, final bundle, HTML report
9. offline fixture 및 실제 센서 검증

각 단계는 독립 테스트와 run artifact를 가진다. 다음 단계는 승인되거나 명시적으로
선택한 candidate dependency만 사용할 수 있으며, 운용 bundle은 accepted dependency만
참조한다.
