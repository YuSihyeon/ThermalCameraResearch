# FireSight 캘리브레이션 설계 요약

> 아래 내용은 v0.1의 초기 설계 기록입니다. v0.2에서 27원 패턴, USB/ROS 촬영과 내부 보정 품질 검사를 구현했습니다. v0.3은 HTML 보고서 검토 → 사용자 승인 → 실시간 영상 적용, 거절 시 새 입력 복귀를 추가합니다. 현재 실행 방법은 저장소 루트 README, 최신 설계는 `docs/superpowers/specs/2026-09-14-review-and-apply-design.md`를 기준으로 합니다. 아래의 후속 구현 목록은 초기 시점의 기록입니다.

## 목적

고정 센서 리그의 다음 보정값을 재현 가능하게 계산하고 검증한다.

- Boson 320 내부 파라미터 `K_boson`, `D_boson`
- D456 factory calibration 스냅샷
- `T_boson_from_d456_color`
- `T_boson_from_livox`
- D456 depth와 Livox point cloud의 Boson 열영상 투영 결과

## 실행 경계

캘리브레이션은 Ubuntu 22.04 x86_64 + ROS 2 Humble PC에서 수행한다. 결과 YAML은 추후 Jetson 또는 다른 FireSight 런타임이 읽는다. 이 저장소의 목적은 실시간 화재 인식이 아니라 센서 기하관계 측정이다.

## 단계

1. Boson 열 보정판 이미지로 내부 보정
2. D456 `CameraInfo`, depth scale, depth↔color transform 스냅샷 검증
3. 동일한 multimodal 보정판을 사용해 Boson↔D456 color 외부 보정
4. 보드 평면 및 반사/구형 마커로 Boson↔Livox 외부 보정
5. depth/point cloud를 Boson 영상에 투영해 최종 시각 검증

## 좌표계

기준은 `boson_optical_frame`이며 OpenCV/ROS optical convention(x 오른쪽, y 아래, z 전방)을 사용한다.

```text
p_boson = T_boson_from_d456_color × p_d456_color
p_boson = T_boson_from_livox × p_livox
```

## 물리 보정판

초기 가정은 600×400 mm 외형, 10×6 squares, 50 mm square다. 이는 `provisional`이며 제작 도면 또는 실측 전에는 운용용 결과를 승인할 수 없다. 가시광 AprilTag와 LiDAR 반사/구형 마커의 보드 좌표도 실측해 `target.yaml`에 기록해야 한다.

## 품질 기준

- Boson intrinsic RMS ≤ 0.8 px
- D456→Boson held-out median ≤ 2.0 px
- Livox board plane residual ≤ 15 mm
- 반복 extrinsic translation 차이 ≤ 10 mm
- 반복 extrinsic rotation 차이 ≤ 0.3°

낮은 RMS만으로 결과를 승인하지 않는다. 다양한 위치·거리·기울기, 전체 영상 coverage, 물리적으로 유효한 왜곡, 최종 overlay 검토가 함께 필요하다.

## 현재 구현과 후속 구현

현재 v0.1은 오프라인 코어를 제공한다.

- 열영상 checkerboard 검출 및 Boson 내부 보정
- 저장 대응점 기반 카메라 외부 보정
- 3D point의 Boson 투영
- YAML 설정과 결과
- 합성 self-check

다음 단계는 실제 장비와 보정판이 있어야 구현·검증할 수 있다.

1. ROS 2 토픽 preflight 및 동기화
2. 세 센서의 원본 sample 캡처
3. AprilTag와 Livox 마커 자동 검출
4. 운영자 point review/edit UI
5. immutable run/checksum/latest 승격
6. HTML 진단 리포트와 실제 센서 회귀 테스트
