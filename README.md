# 열화상 신호에서 농연 속 경계 표시까지

열화상 카메라의 출력을 받아 화면에 띄우는 것과, 그 화면에서 사람이 공간의 경계를 읽을 수 있게 만드는 것은 다른 문제다. 이 연구는 카메라 표시·저장과 밝기 기반 점군 표현을 탐색하는 데서 시작해, **농연 영상의 벽·문틀·장애물 경계를 배경을 덜 가리면서 표시하는 방법**으로 구체화되었다. 이후에는 주어진 거리값을 선 굵기로 표현하는 방법과, 그 처리가 느린 프레임까지 포함해 지속될 수 있는지를 따로 검증했다.

초기 실험의 정확한 날짜는 확정되지 않았고, 후속 보고서는 2026-09-06~14에 기록되었다. 남아 있는 자료가 보여 주는 진전은 밝기의 공간 시각화, 경계 선택과 표시 폭의 분리, 합성 거리의 연속 폭 변환, RTX PC에서의 지속 처리다. 실제 거리·절대온도 계측이나 Jetson·AR 안경에서의 현장 성능을 검증한 단계는 아니다.

<img src="docs/evidence/obstacle-edges/smoke-comparison.png" width="960" alt="공개 농연 열화상에 기존 TEED, 세선화, 구조 경계 처리, 제한적 adaptation과 PiDiNet을 적용한 비교">

*그림 1. SmokeBasement의 동일 장면에 대한 경계 처리 비교. 위 행은 왼쪽부터 입력, 기존 TEED, 얇은 선·밝은 배경이며, 아래 행은 구조 경계 TEED, fusion-head adaptation, PiDiNet이다. 위 가운데의 굵은 초록선과 아래의 선택된 경계는 화면 가림과 구조 유지가 함께 달라짐을 보여 준다. 이 그림만으로 필요한 경계의 검출 정확도를 매길 수는 없다. SmokeBasement — Jianzhu Huai (2025), CC BY 4.0의 선택·resize·처리 파생물이며 직접 Boson으로 촬영한 농연 실험은 아니다. [데이터 DOI](https://doi.org/10.5281/zenodo.15173055) · [출처와 변환 기록](docs/SOURCE_CREDITS.md)*

## 첫 질문은 입력의 의미와 저장 가능성이었다

초기 [표시 코드](code/original/flir_view_test.py)는 `VideoCapture(0)`으로 첫 카메라를 열고 BGR이면 gray로 바꾼 뒤 프레임별 최소·최대값 정규화와 INFERNO 컬러맵을 적용한다. OpenCV 하나로 연결·표시·저장 경로를 확인할 수 있는 구성이었다. [캡처 코드](code/original/thermal_capture.py)는 `s`를 누를 때 정규화·컬러맵 이전의 gray 배열을 PNG로 저장한다. 따라서 화면의 컬러와 저장된 PNG는 동일한 배열이 아니며, 센서·UVC 단계에서 AGC나 8bit 변환이 이미 적용되었는지도 이 코드만으로 확인되지 않는다.

보존된 `frame_0`~`frame_51`은 모두 **640×512, 단일 채널 uint8**인 52장이다. 수동 저장할 때 번호가 증가하므로 같은 간격의 연속 영상이라고 해석하지 않는다. timestamp·intrinsics·pose·온도 보정 정보는 없고 장치 serial·part number·radiometric variant도 확정되지 않았다. 파일명과 창 제목은 자료를 찾는 단서지만 장치의 출력 계약을 대신하지 못한다. [입력 조사](docs/01-inputs.md) · [52장 통계](docs/evidence/frame-inventory.csv)

<img src="docs/evidence/inputs/frame_0.png" width="640" alt="두 밝기 기반 PLY 파일의 계산 입력인 640×512 grayscale frame 0">

*그림 2. `frame_0.png`는 저장된 두 PLY와 입력 수식을 대조할 수 있어 대표 입력으로 사용했다. 픽셀은 8bit 표시값이며, 값 255가 존재한다는 사실은 센서의 포화 온도나 섭씨 상한을 뜻하지 않는다. 이 파일에는 깊이·절대온도·촬영 pose를 정하는 메타데이터가 없다.*

## 밝기를 높이로 바꾸면 무엇을 알 수 있는가

공간 표현을 탐색한 [PLY 변환](code/original/thermal_image_to_pointcloud.py)은 `X=x`, `Y=y`, `Z=float32(I)/255×3000`을 사용했다. 이미지 좌표를 평면에 놓고 밝기를 높이로 세우는 방식이다. `3000`은 임의 확대 계수이며 metric depth나 온도 단위가 아니다. 초점거리·주점·metric Z를 사용하는 카메라 back-projection도 없으므로, 결과는 **intensity height field**로 해석해야 한다.

이 표현은 2차원 배열을 점군으로 옮겨 저장하고 공간 뷰어에서 다루는 과정을 검사하는 데 의미가 있다. 동시에 표현의 한계도 분명하다. 서로 다른 거리에 있는 물체가 같은 밝기를 가지면 같은 Z가 되고, 하나의 평면에서도 밝기가 다르면 높낮이가 생긴다. 화면이 벽이나 바닥처럼 보인다는 인상만으로 실제 장면의 기하를 얻었다고 결론내릴 수 없다. 초기 탐색의 의도와 이 해석은 [처리 과정](docs/02-process.md)과 [분석 기록](docs/04-analysis.md)에 남아 있다.

<img src="docs/evidence/pointcloud/floor-view.png" width="850" alt="밝기 기반 PLY를 CloudCompare에서 열어 관찰한 화면">

*그림 3. CloudCompare에서 밝기 기반 PLY를 열고 관찰한 기록. 화면의 공간 좌표와 scale은 변환 코드의 임의 단위이며 실제 바닥의 거리나 경사를 측정한 값이 아니다. 과거 화면 속 파일과 현재 보존 파일의 bit 단위 동일성은 당시 hash가 없어 확정하지 않았고, 아래 계산 검증은 현재 남아 있는 PLY와 `frame_0.png` 사이에서 수행했다.*

[backup 변환](code/original/thermal_image_to_pointcloud_backup.py)의 stride 4는 **20,480점**, stride 2는 **81,920점**을 만들었다. 간격을 절반으로 줄여 두 축의 표본 수가 각각 두 배가 된 결과다. 전체 vertex를 대조했을 때 두 파일의 X/Y와 `I/255×3000` 계산의 최대 절대 오차는 **0**, 조밀한 파일의 RGB도 모두 입력 gray와 일치했다. 이는 저장·변환 구현의 일치이며 점 수를 늘려 거리 복원 정확도가 높아졌다는 결과는 아니다. [전체 검산](docs/evidence/local-audit.json)

초기에 함께 탐색한 [radar 평면 분리 코드](code/original/firesight_radar_core.py)는 별도의 합성 실험이다. 실제 실행부는 바닥 1,000점·벽 800점·장애물 300점을 난수로 만들고, outlier 제거와 RANSAC 두 평면 분리를 거쳐 색을 표시한다. thermal PLY를 읽거나 radar 장치에 연결하는 경로는 없다. 가장 큰 평면을 바닥이라 이름 붙이는 규칙도 중력 방향과 센서 좌표계를 확인하지 않으므로 실측 공간 분할에는 별도 검증이 필요하다. 두 탐색이 나란히 있었다는 사실을 thermal–radar 융합의 실행 증거로 합치지는 않는다.

## 굵은 윤곽선이 장면을 가리는 문제

후속 연구는 공간 자체를 복원하기보다 농연 영상에서 구조를 읽기 쉽게 만드는 문제로 좁혀졌다. 이미 합성된 Boson HUD에는 초록선이 넓게 덮인 부분이 있었고, 그 표시를 먼저 조사했다. 여기서는 원래 열화상 입력이 없으므로 새 검출 모델을 적용해 성능을 비교할 수 없다. 대신 [감사 코드](code/selected/boson_review/analyze.py)는 HSV H=40~85와 S/V threshold로 화면에 그려진 초록 stroke를 근사 추출하고 세선화했다.

[<img src="public-media/boson-test.jpg" width="640" alt="어두운 배경 위에 이미 초록 윤곽이 합성된 Boson HUD의 한 시점">](public-media/boson-test.mp4)

*영상 1. [Boson HUD 보존 영상](public-media/boson-test.mp4), 10초·320×256·9fps·90프레임. 직접 제공된 표시 결과이며, 센서 원시 입력이나 온도 맵은 아니다. 감사는 이미 그려진 stroke의 면적과 형태를 대상으로 하므로 가려진 열 배경이나 누락된 경계를 복원하지 못한다.*

90프레임의 평균 초록선 점유율은 **18.45798%**, 세선화한 stroke는 **4.03887%**였다. 약 78.1%의 면적 감소는 표시된 선을 얇게 할 여지가 있음을 보여 준다. 다만 색 추출 threshold를 바꾸면 원래 초록 면적도 17.01515%~18.71516%로 달라져 압축색과 추출 규칙의 영향이 남았다. [감사 JSON](docs/evidence/boson-review/evidence/audit.json)

검출과 표시를 실제로 나누어 시험하기 위해서는 HUD가 합성되기 전 입력이 필요했다. 핵심 비교에는 공개 **SmokeBasement** 열화상을 사용했다. 실험 기록상 run3 중앙 24초에서 학습 32장, run5 중앙 12초에서 검증 120장을 선정했다. 서로 다른 run이지만 같은 지하실이므로 새로운 현장에 대한 일반화 시험은 아니다. 원 PNG는 640×512 uint8 3채널이며 모델 입력은 320×240으로 바꿨다. 120장 모두 다른 frame이고 시간 재표본화의 최대 오차는 검증 40.82ms, 학습 39.40ms로 기록된다.

[<img src="public-media/smoke-original.jpg" width="640" alt="모델 비교와 표시 실험에 공통으로 사용한 SmokeBasement 농연 열화상 입력">](public-media/smoke-original.mp4)

*영상 2. [농연 입력 구간](public-media/smoke-original.mp4), 12초·320×240·10fps. SmokeBasement의 선택·재표본화·resize·재인코딩 파생물로, 직접 보유한 카메라에서 수집한 농연 영상이 아니다. 이 구간을 공통 입력으로 사용해 모델과 후처리의 차이를 비교했다. 데이터셋에 LiDAR가 포함되어 있어도 뒤의 거리 굵기 실험이 그 LiDAR 실측값을 사용한 것은 아니다.*

원 기록은 전체 27GB ZIP 대신 필요한 152개 PNG를 HTTP Range로 추출하고 CRC/SHA-256을 확인했다고 적는다. 2026-09-16 보존 검토는 남아 있는 파일과 기록의 대조 범위이며 전체 데이터의 재다운로드·152장 재추출 검증은 포함하지 않는다. Fire360/IFSI Video 8의 보조 학습과 Video 4의 625프레임 보조 검증도 있었지만 농연 강도가 확인된 핵심 시험과 합산하지 않았다. [입력·선정 근거](docs/01-inputs.md) · [원 실험 보고서](docs/evidence/obstacle-edges/REPORT.md)

## 모델을 바꾸기 전에 경계 선택과 표시를 비교하다

2026-09-06 실험은 더 큰 모델을 곧바로 채택하기보다 기존 TEED의 출력에서 무엇이 화면 가림을 만드는지를 나누어 보았다. 당시 기준선은 CLAHE 2.0, threshold .75, 3px dilation, 배경 밝기 .24였다. 같은 확률맵을 세선화하고 배경을 .55로 밝힌 조건은 선의 겹침과 배경 소실을 줄이려는 비교였다. 이 조건은 선 두께와 배경을 함께 바꾸므로 두께만의 순수 ablation은 아니다.

구조 경계 조건은 CLAHE를 빼고 bilateral(5,25,3), hysteresis .60/.85, skeletonization을 적용했다. [hysteresis 구현](code/selected/obstacle_edges/contours.py)은 low 이상인 8-connectivity 성분 중 high 이상 응답을 포함하는 성분을 남긴다. 따라서 강한 경계에 연결된 약한 응답은 보존하면서 단독의 약한 응답은 제외하는 선택이다. 그림 1과 아래 영상은 이 선택이 표시 형태에 미친 효과를 보여 준다.

[<img src="public-media/smoke-structural.jpg" width="640" alt="SmokeBasement 입력에서 강한 구조와 연결된 경계를 선택하고 세선화한 TEED 출력">](public-media/smoke-structural.mp4)

*영상 3. [TEED 구조 경계 출력](public-media/smoke-structural.mp4), 12초·320×240·10fps, SmokeBasement 처리 파생물. threshold와 연결 성분 선택 이후 남은 경계를 가는 선으로 표시했다. 점유율 감소는 배경을 덮는 면적의 변화이며, 얇은 장애물의 누락 여부나 boundary recall은 이 영상과 면적 지표만으로 확인되지 않는다.*

추가 학습이 필요한지도 제한적으로 시험했다. TEED feature extractor를 고정하고 `block_cat`의 **480개 fusion 파라미터**만 학습했다. seed 20260906, 32장×8 epochs=256 optimizer step, Adam lr 1e-4이며 320×240/160×120의 TEED 자체 응답을 의사 라벨로 사용했다. 다중 scale에서 남는 응답을 강화하려는 시험이지만, 사람이 부여한 장애물 정답이나 회피 의미를 새로 학습한 것은 아니다. 비교 모델 PiDiNet은 공식 checkpoint·RGB 정규화와 hysteresis .20/.40을 사용했다. 서로 다른 threshold는 같은 precision/recall로 교정한 값이 아니다. [학습·평가 코드](code/selected/obstacle_edges/run_smoke.py)

| 방법 | 평균 엣지 점유율 | 전체 처리 중앙값 | 전체 처리 P95 |
|---|---:|---:|---:|
| 기존 TEED | 12.27% | 68.23ms | 89.37ms |
| 얇은 선·밝은 배경 | 2.36% | 69.15ms | 85.61ms |
| 구조 경계 TEED | 1.51% | 73.54ms | 104.55ms |
| 의사 라벨 adaptation | 1.45% | 69.49ms | 93.68ms |
| PiDiNet | 1.94% | 443.93ms | 556.90ms |

조건은 SmokeBasement run5의 120프레임, 320×240, CPU 4 threads다. 시간은 첫 10개를 제외한 warm 110개에서 집계하며 전처리·추론·후처리·렌더를 포함하고 읽기·인코딩·저장은 제외한다. 점유율은 출력 mask가 차지한 픽셀 비율의 프레임 평균이다. **12.27%→1.51%는 약 87.7%의 표시 면적 감소**이며 정답 boundary가 없어 검출 정확도 개선률은 계산하지 않았다. [원 결과 JSON](docs/evidence/obstacle-edges/results_summary.json)

학습의 weighted soft BCE는 **0.7557140→0.7550460**, 변경 tensor는 4개였다. 학습 실행은 기록되었지만 loss 감소와 시각 차이가 작아 기본 checkpoint를 교체할 근거는 부족하다는 결론이 남았다. PiDiNet은 일부 경계가 연속적으로 보였으나 개구부 주변을 닫힌 형태로 연결하는 부분도 관찰되었고, 이 CPU 조건의 처리 비용이 컸다. 따라서 “추가 학습 또는 모델 교체가 지금 필요한가”라는 질문에 대해 기존 TEED를 유지하며 경계 선택과 표시를 먼저 다루는 방향이 선택되었다. 이것은 TEED가 모든 장비와 현장에서 더 정확하다는 일반적인 모델 순위가 아니다. [학습 hash·비교 결과](docs/03-results.md) · [당시 해석](docs/04-analysis.md)

## 같은 경계라도 표시 폭이 달라지면 읽는 화면이 달라진다

2026-09-08 비교는 검출 상세도와 선 굵기를 더 명확히 나누었다. 같은 TEED 확률맵에서 detailed=(.55,.75), balanced=(.60,.85), sparse=(.70,.90)의 경계를 선택하고, 최종 640×480 화면의 1px/2px 표시를 비교했다. Balanced 1px와 2px는 같은 경계를 그리므로 2px에서 눈에 더 잘 띄는 부분이 생겨도 새 장애물을 검출했다는 뜻은 아니다. [표시 비교 코드](code/selected/boson_review/verify_video.py)

[<img src="public-media/smoke-balanced-2px.jpg" width="760" alt="동일 확률맵에서 선택한 Balanced 경계를 최종 640×480 화면에 2px로 표시한 결과">](public-media/smoke-balanced-2px.mp4)

*영상 4. [Balanced 2px 표시](public-media/smoke-balanced-2px.mp4), 12초·640×480·10fps, SmokeBasement 처리 파생물. 경계 선택과 렌더 폭을 분리한 비교에서 모니터상 가독성을 검토한 후보이다. 런타임 기본값은 1px로 유지되었고 실제 안경에서의 시야각·대비·사용자 반응을 평가한 결과는 아니다.*

평균 초록선 점유율은 legacy **11.4979%**, detailed 1px **1.2701%**, balanced 1px **1.1776%**, balanced 2px **2.6322%**, sparse 1px **1.0272%**였다. Balanced 2px는 1px보다 눈에 띄면서 기존 굵은 표시보다 배경을 덜 가린다는 정성 판단으로 후속 장비 시연 후보가 되었다. 120프레임 모두 경계가 완전히 비지는 않았지만 이는 개별 호스·단차·가는 장애물의 누락이 없다는 보장은 아니다. [표시 판단 원문](docs/evidence/boson-review/RECOMMENDATION.md)

선이 적어지면 점유율이나 시간 residual도 낮아질 수 있으므로 그 자체를 더 좋은 인식이라고 해석하기는 어렵다. flow로 정렬한 residual에는 장면 변화·가림·새 구조·flow 오차가 함께 들어간다. 또한 9월 6일의 12.27%와 이 비교의 11.4979%는 출력 해상도·표시 조건이 달라 하나의 연속 개선 곡선으로 합치지 않는다. 후속 채택에는 같은 경계를 놓고 폭만 바꾸는 표시 평가와, 같은 출력 폭에서 정답 boundary를 대조하는 검출 평가가 각각 필요하다.

## 거리 구간의 갑작스러운 변화를 연속적인 굵기로 바꾸다

거리를 선 굵기로 전달하려는 초기 시연은 화면 원근을 보고 먼 방향을 임의로 지정했다. 같은 Balanced 경계를 고정 2px와 가정 거리의 1/2/3px로 비교한 자료다. 여기서 시험한 것은 가까운 선을 굵게 보여 주는 표현 방식이며, 열화상에서 실제 거리를 복원하는 모델은 들어 있지 않다.

[<img src="public-media/distance-preview.jpg" width="900" alt="같은 Balanced 경계를 고정 2px와 임의 원근에 따른 1·2·3px로 비교한 화면">](public-media/distance-preview.mp4)

*영상 5. [가정 거리별 선 굵기 비교](public-media/distance-preview.mp4), 12초·1280×736·10fps. 왼쪽은 고정 2px, 오른쪽은 임의 원근에 따라 먼 곳 1px·중간 2px·가까운 곳 3px를 부여한 표시다. 화면의 거리 분포와 작은 맵은 합성값이다. SmokeBasement 열화상에 표시를 더한 파생물이며 실제 radar·LiDAR 거리 또는 아래 후속 모듈의 실측 입력으로 취급하지 않는다.*

2026-09-14에는 표현을 명시적인 입력 계약으로 바꿨다. [거리 폭 모듈](code/selected/edge/teed/distance_width.py)은 영상과 정렬된 meter 거리 맵 또는 scalar를 받으며, 기본값은 **1m→5px, 8m→1px**다. `t=(clamp(d,1,8)-1)/7`, `w=5-4t²(3-2t)`의 smoothstep으로 거리가 변할 때 폭이 연속적으로 변하도록 했다. 시험의 `d`는 여전히 합성값이며 센서가 그 값을 얼마나 정확히 측정하는지는 별개다.

연속 함수만으로는 작은 거리 노이즈가 그대로 폭의 흔들림이 될 수 있다. 그래서 작은 변화에는 `α=1-exp(-dt/0.05)`인 50ms EMA를 적용하고, 직전 폭과 차이가 0.75px보다 크면 새 목표를 즉시 반영했다. 작은 노이즈를 줄이면서 큰 변화에 오래 지연되지 않게 하려는 구성이다. 필터는 화면 좌표별 history를 사용하며 optical flow나 물체 tracking을 수행하지 않는다. 따라서 경계 검출 자체의 깜빡임이나 움직이는 물체의 대응까지 해결한 것은 아니다.

Guo-Hall 중심선의 시작 픽셀에 폭을 부여하고 사각 dilation·alpha blending으로 펼친다. 이는 선을 펼친 도착 위치의 배경 거리 때문에 폭이 갑자기 바뀌는 문제를 줄이려는 선택이다. 사각 커널의 대각선·교차점에서 보이는 폭은 수평·수직 alpha 합으로 정의한 폭과 다를 수 있다. NaN/Inf/0, 거리 없음, 100ms 초과 stale, 미래 timestamp는 기본 1px로 돌아가며 history를 지운다. **이 1px는 결측 표시 정책이며 먼 물체를 측정했다는 판정이 아니다.** [거리 replay 계약](code/selected/edge/teed/depth_replay.py)

| 합성 통제 입력 | 구간식 | 연속 무필터 | 연속+50ms EMA |
|---|---:|---:|---:|
| 4.5m, 거리 noise σ=.18m의 폭 표준편차 | 0px | .1560px | .0864px |
| 구간 경계 3.28m의 폭 표준편차 | .9963px | .1367px | .0755px |
| 1→8m 완만한 변화의 최대 폭 변화 | 2px/frame | .0251px/frame | .0251px/frame |

구간 중앙의 정지 입력에서는 구간식이 오히려 폭을 전혀 흔들지 않을 수 있다. 문제가 되는 지점은 구간 경계에서 값이 왕복하며 폭이 뛰는 경우다. 연속 함수는 그 jump를 줄였고 EMA는 남은 작은 흔들림을 줄였다. 30fps의 이론적 저주파 지연은 **35.2ms**이며, 큰 8m→1m step은 첫 frame에 5px로 반영되었다. 이 표는 주어진 거리 입력에 대한 renderer의 응답을 검증하며 거리 오차를 개선한 수치는 아니다. [통제 시험 summary](docs/evidence/distance-width/summary.json)

[<img src="public-media/realtime-comparison.jpg" width="720" alt="같은 공개 열화상과 TEED 경계에 고정 폭, 합성 거리 구간식, 연속 폭과 EMA를 적용한 네 화면">](public-media/realtime-comparison.mp4)

*영상 6. [고정·구간·연속 폭 비교](public-media/realtime-comparison.mp4), 12초·640×556·10fps. 위 왼쪽은 공개 열화상, 위 오른쪽은 같은 TEED mask의 고정 3px 중심선, 아래 왼쪽은 합성 거리의 1/3/5px 구간식, 아래 오른쪽은 연속 폭과 50ms EMA다. SmokeBasement 처리 파생물이며, 이 영상의 인코딩 FPS는 다음 180초 benchmark의 처리량이나 센서 지연을 나타내지 않는다.*

## 평균 FPS 대신 지속 처리의 실패 조건을 확인하다

표시가 부드러워도 간헐적으로 늦은 frame이 나오면 시간상의 품질은 달라진다. 초기 CPU benchmark에서 연속+EMA의 TEED 포함 P95는 **43.583ms**, 33.333ms 예산을 넘은 비율은 **23.3%**였다. 평균 약 36.6fps만 보면 30fps를 넘지만, 개별 frame의 마감 시간까지 만족한 것은 아니다. 조건은 320×240, OpenCV/PyTorch 각 2 threads, 모델 예열 5회·후처리 예열 30회, 방식별 600회 측정이다. [프레임·후처리 원자료](docs/evidence/distance-width/benchmark_raw.csv)

후속 실행은 최신 카메라 frame 1개와 거리 최근 8개만 보관하고 유효한 과거 거리만 선택했다. 고정 버퍼와 FP32 CUDA Graph로 model dispatch 비용을 줄이는 경로를 CPU·CUDA eager와 비교했다. [실행부](code/selected/edge/teed/realtime_core.py)와 [지속성 판정](code/selected/edge/teed/stability.py)은 평균 속도 외에 예산 초과·누락·출력 간격·정상 종료를 함께 검사한다.

원 기록에는 낮은 정밀도 clock으로 timestamp가 중복되는 실패, 화면 경계의 세선화 문제, 작은 연결 성분 소실, 종료 직전 멈춤의 집계 누락, 640×480 calibration을 320×240 입력에 적용하는 불일치가 남아 있다. 최종 보고서는 `perf_counter`로 시각 기준을 통일하고 관련 회귀 검증을 추가했다고 기록한다. 이 수정의 근거는 당시 보고서와 선별 소스이며 새 환경의 재실행 검증은 남아 있지 않다. [처리 과정과 시행착오](docs/02-process.md)

| 처리 경로 | 측정 시간 / 프레임 | 평균 FPS | 처리 P99 | 호스트 수신→출력 P99 | 예산 초과 | 판정 |
|---|---|---:|---:|---:|---:|---|
| CPU FP32 | 180.003초 / 5,400 | 29.999 | 33.452ms | 34.988ms | 1.056% | FAIL |
| CUDA FP32 | 180.008초 / 5,400 | 29.999 | 11.319ms | 12.021ms | 0% | PASS |
| CUDA Graph FP32 | 180.021초 / 5,401 | 30.002 | 10.981ms | 11.601ms | 0% | PASS |

환경은 Windows·RTX 5060 Ti, torch 2.11.0+cu128, FP32, 320×240, 각 2 CPU threads, 예열 30회다. 공개 120프레임/10fps 영상을 30fps로 가속 반복하고 합성 거리를 TCP로 전달했다. 시간에는 GPU 완료 대기와 `VideoWriter.write`가 포함되지만 종료 flush·창 표시·센서 노출/driver buffer·물리 display scanout은 제외한다. **측정된 호스트 수신→출력 지연은 센서 노출→안경 표시 지연이 아니며, 이 PC 결과로 Jetson 성능을 추정하지 않는다.** [환경 JSON](docs/evidence/realtime/environment.json) · [프레임 집계 CSV](docs/evidence/realtime/comparison.csv)

당시 PASS는 180초 이상, 평균≥29.7fps, 각 10초 구간≥29.4fps, 처리 P99≤33.333ms, 예산 초과·누락≤1%, 수신→출력 P99≤66.667ms, 출력 간격 P99≤50ms·최대≤100ms, 오류 없음·정상 종료를 모두 요구했다. CPU는 평균 FPS가 30에 가까워도 처리 P99와 예산 초과율 기준을 넘었다. 평균이 같은 세 경로를 같은 성공으로 분류할 수 없다는 결과다. [전체 조건과 판정](docs/evidence/realtime/realtime_results.json)

Graph와 eager FP32의 120프레임 확률 최대 절대차는 `4.76837158e-7`, 이진 경계 불일치 0%, IoU 1이었다. 이 시험에서는 FP16으로 표현 정밀도를 낮추어 속도를 얻은 것이 아니다. 다만 초기 CPU benchmark와는 runner·PyTorch 환경이 달라 숫자를 이어 단순한 개선 배수를 계산하지 않는다. [출력 동등성](docs/evidence/realtime/quality.json)

## 남은 질문은 표시 품질을 실제 입력과 사용에 연결하는 일이다

저장된 결과를 종합하면, 굵은 선이 배경을 가리는 문제에는 경계 선택과 세선화가 유효했고, 합성 거리의 구간 경계 문제에는 연속 함수와 EMA가 효과를 보였다. PC의 반복 처리 경로도 GPU 조건에서 지속성 기준을 충족했다. 각 결론이 적용되는 대상은 표시 면적, 주어진 거리의 폭 응답, 정의된 호스트 처리 구간이다. 장애물 회피 정확도·절대온도·실제 거리·새 농연 현장의 일반화는 현재 자료가 답하지 못한다.

다음 실험에서는 먼저 장치 part number·pixel encoding·AGC/FFC·timestamp·보정 정보를 입력과 함께 저장하고 HUD 합성 전·후 영상을 분리해야 한다. 거리 센서를 연결할 때는 meter 단위뿐 아니라 좌표계·camera-to-sensor 변환·등록 해상도·시각 기준·유효 품질이 필요하다. 알려진 거리 표적, 가림 경계, 결측, 늦은 sample과 time reset을 각각 시험해야 매끈한 선이 잘못 대응된 거리를 표현하는 상황을 구분할 수 있다.

검출 평가에는 호스·단차·개구부·빈 장면을 포함한 boundary 정답과 현장·세션별 분리가 필요하다. 표시 평가에서는 같은 경계를 놓고 1px/2px와 대비를 실제 안경에서 비교하여 필요한 경계 누락, 통행 판단·반응시간, 시야 가림을 함께 측정해야 한다. 목표 Jetson·카메라·거리 센서·안경을 연결한 180초 이상 시험에서는 현재의 지속성 기준과 함께 열·전원·drop·재연결을 기록하고, 노출→표시 지연을 별도로 계측해야 한다. 이 항목들은 [보존 분석에서 제안한 후속 연구](docs/04-analysis.md)이며 이미 완료된 장비·사용자 검증으로 제시하지 않는다.

## 연구 자료와 복원의 경계

이 저장소는 초기 자체 입력·코드와 FireSight 관련 실험의 직접 근거를 선별한 기록이다. TEED·PiDiNet의 핵심 모델과 pretrained weight는 외부 기반이며, 팀 앱 전체를 개인이 개발한 성과로 재구성하지 않는다. 개인별 코드 작성 분량을 단정할 자료도 여기에는 없다. 초기 원본과 후속 코드 발췌의 역할은 [code/README.md](code/README.md)에, 입력·방법·수치·해석의 상세 기록은 [입력](docs/01-inputs.md), [처리 과정](docs/02-process.md), [결과](docs/03-results.md), [분석](docs/04-analysis.md)에 남아 있다. 변경 전 서술은 [RESEARCH_REPORT.md](RESEARCH_REPORT.md)에 바이트 그대로 보존했다.

공개 영상 여섯 개 중 Boson HUD를 제외한 다섯 개는 **SmokeBasement — Jianzhu Huai (2025), CC BY 4.0**의 선택·재표본화·resize·인코딩·경계 처리 또는 합성 거리 표시 파생물이다. 원 데이터의 LiDAR를 실측 거리로 사용하지 않았으며 Fire360/IFSI에 같은 라이선스를 적용하지 않는다. [SOURCE_CREDITS](docs/SOURCE_CREDITS.md)에 출처와 변환을, [MEDIA.md](MEDIA.md)에 영상별 메타데이터와 보존 MP4 배포 경로를 기록했다. 장치·모니터를 휴대폰으로 촬영한 원본은 주변 사람·작업 화면이 포함될 수 있어 개인 보존본에 두었고 [선정 근거](docs/evidence/publication-review.json)를 남겼다.

전체 52장 입력·PLY·휴대폰 원본·FireSight 전체 작업 상태와 이전 아카이브의 위치는 [DATA_AND_RESTORE.md](DATA_AND_RESTORE.md)에 있다. 조사한 Windows 체크아웃에서는 일부 선택 PNG·source manifest·적응 checkpoint·PiDiNet checkpoint·IFSI 원 영상이 확인되지 않았다. 남아 있는 재인코딩 영상은 원 PNG와 정확한 weight를 대체하지 못한다. 새 다운로드나 재학습의 hash가 다르면 원 실험 복원과 구분되는 새 실험으로 기록해야 한다.

카메라 없이 재현 가능한 최소 범위는 별도 작업 폴더의 `frame_0.png`와 원본 변환 스크립트로 PLY 계산을 확인하는 것이다. stride 4는 20,480점, stride 2는 81,920점이 기준이다. 후속 경계 실험에는 OpenCV contrib의 `ximgproc`, 정확한 TEED/PiDiNet 소스·weight·환경이 필요하며 선별 runner만으로 전체 FireSight 앱이 실행되지는 않는다.

[VALIDATION.md](VALIDATION.md)는 2026-09-16의 52장 통계, PLY 전체 검산, 영상 디코드, 구문·출처·해시 대조 범위를 기록한다. 모델 재학습·카메라 재촬영·180초 benchmark 재실행·새 PC 전체 복원을 완료한 기록과는 구분한다. 로컬 보존본의 파일 검증, USB 전송, 외부 사본 검증도 각각 별도 단계다.
