# Fire360 윤곽 추적 안정화

이 문서는 기존 TEED 확률 시간 안정화 뒤에 추가한 윤곽 추적과 출력 선 렌더링을 설명한다. Fire360의 같은 TEED 결과로 비교하며, 온도 추정 정책과 TEED 가중치는 바꾸지 않는다.

## 처리 순서

원본 영상 → TEED 확률 → 움직임을 보정한 확률 누적 → 공간 필터/중심선 → 윤곽 트랙 연결 → 표시 여부와 좌표 보정 → 출력 해상도에서 안티앨리어싱 선 합성.

- 같은 윤곽에 ID를 유지하고 이전 좌표를 영상 움직임에 맞춰 이동시킨다.
- 떨어져 있는 약한 짧은 조각(길이 18 모델 픽셀 이하)은 최근 4프레임 중 3회 관측해야 표시한다. 현재 TEED 확률이 강한 새 경계는 바로 표시한다. 연결된 구조선의 약한 연결부는 따로 지연시키지 않는다.
- 확률 단계의 시간 누적을 이용하고, 윤곽 단계의 추가 표시 유지는 기본 0프레임으로 둔다. 두 단계가 각각 선을 유지하면 실제 영상에서 잔여 조각이 늘어났다. API에서 `ContourConfig(max_missing_frames=2)`를 지정하면 현재 확률의 지지가 있는 독립 윤곽만 최대 2프레임 추가 유지한다. 표시 전인 약한 트랙은 확인 기간 동안 기록만 유지하며 선을 그리지 않는다.
- 이전 좌표를 전후방 광류/밝기 일치로 검증하고, 현재 관측에서 최대 1 모델 픽셀 이내로 보정한다. 공통 분기점은 현재 연결 위치를 공유한다. 사람 윤곽을 직선으로 바꾸지 않는다.
- 약한 독립 중복 윤곽만 정리하고 강한 근접 평행선과 연결된 구조선은 보존한다. 재구성한 출력에도 공간 필터의 최소 길이 12/최소 범위 8 모델 픽셀을 다시 적용한다. 소수점 좌표를 출력 크기에 직접 그리며 기본 선 폭은 출력 기준 1.5픽셀이다.

## 실행

```sh
python edge/orin/run_pipeline.py --source <video> --profiles edge/teed/profiles/fire360_video_profiles.json --pace --contour-edges
```

CLI 기본은 기존 확률 시간 안정화다. 윤곽 추적은 `--contour-edges`로 켜는 실험 옵션이다. 실제 Fire360에서 전체 점멸 개선은 작고 CPU 비용이 증가해 기본 실행에는 자동 적용하지 않았다. `--no-contour-edges`는 직전 확률 시간 안정화 화면과 비교한다. 여기에 `--no-temporal-edges`를 더하면 공간 필터만 남는다. `--legacy-edges`는 필터 이전 기준 화면이다. 선 폭은 `--contour-line-width 1.5`로 조절한다.

기존 API 호출의 결과를 보존하기 위해 `HUDConfig.contour_edges` 기본값은 `False`다. API에서는 명시적으로 켠다. `HUDRenderer.reset_temporal()`은 확률과 윤곽 기록을 함께 초기화한다. 영상 반복 재생, 프레임/시간 불연속, 크기 또는 제외 영역 변경에서 이전 기록이 이어지지 않는다.

`HUDFrame.edge_mask`는 표시 윤곽의 모델 해상도 중심선으로, 실제 화면의 선 폭/안티앨리어싱 알파와는 다르다. 안정성 지표는 중심선으로 측정하고, 실제 화질은 합성 영상으로 확인한다.

## 검증 결과

합성/통합/기존 공간·시간 처리 검사 111개 통과. 변경 범위 Ruff E/F/I 검사, basedpyright 기본 모드, Python 구문 검사 통과. 동일 Fire360 600프레임 비교 결과는 아래에 기록한다.

측정 정의:
- 전체 경계 변화율: 원본 영상의 움직임으로 이전 마스크를 정렬한 뒤, 가로·세로 ±1 모델 픽셀 안에서 짝이 없는 경계의 비율. 낮을수록 좋지만 선 위치의 소수점 떨림을 모두 측정하지는 않는다.
- 작은 윤곽 수: 연결 성분의 바운딩박스 대각선이 20 모델 픽셀 이하인 개수. 실제 불필요 윤곽 정답이 아니며 작은 물체 경계도 포함한다.
- 작은 윤곽 수 변동: 인접 프레임의 개수 차 절댓값 평균. 개별 객체의 점멸 빈도와 같지 않다.
- 기존 경계 근방 보존: 기존 중심선 중 새 중심선의 가로·세로 ±2 모델 픽셀 안에 남은 비율. 장애물 탐지 정확도나 회피 성공률이 아니다.
- HUD 시간: 현재 Windows CPU에서 측정한 후처리·합성 시간. TEED 추론과 파일 인코딩은 제외한다.

검증용 합성 정지 경계의 1픽셀 교대 떨림은 좌표 표준편차 0.500→0.109 모델 픽셀로 줄었다. 이는 통제된 합성 검사 결과이며 Fire360 전체의 개선율로 해석하지 않는다.

| 구간 | 프레임 | 작은 윤곽 수 이전→새 방식 | 개수 변동 이전→새 방식 | 전체 경계 변화율 감소 | 기존 경계 근방 보존 | CPU HUD 평균 이전→새 방식 |
|---|---:|---:|---:|---:|---:|---:|
| fire360_ifsi4 | 240 | 6.67 → 6.25 | 2.14 → 2.02 | 0.26% | 99.81% | 70.2 → 307.2 ms |
| fire360_ifsi8 | 240 | 5.90 → 5.59 | 2.08 → 2.00 | 0.30% | 99.88% | 65.7 → 247.9 ms |
| fire360_visible | 120 | 6.88 → 6.73 | 2.40 → 2.32 | 1.90% | 99.85% | 52.8 → 180.9 ms |

작은 윤곽 개수는 약 2.1–6.3%, 그 개수 변동은 약 3.5–5.5% 감소했다. 전체 경계 변화율은 열화상 두 구간에서 약 0.3%, 가시광 구간에서 약 1.9%만 개선됐다. 출력 선의 계단 현상은 줄었지만, 사용자가 원하는 수준으로 동적 장면의 불필요 윤곽과 점멸이 충분히 줄었다고 판단하지 않는다. 옷 주름/화염 내부처럼 복잡한 곳의 경계는 여전히 남는다. 기본 실행을 바꾸지 않고 실험 옵션으로 유지한다.

총 10개 영상 파일의 전 프레임을 디코딩해 프레임 수·해상도·재생률을 확인했다. 실제 인코딩된 영상에서 연속 구간 프레임과 확대 화면을 추출해 정합·선 연결·잔상을 점검했다. 브라우저에서 전체 영상을 재생해 확인한 것은 아니다.

로컬 생성 근거:
- 열화상 IFSI4 전후 비교: `experiments/teed_ui/contour_stability/video_final_graph/fire360_ifsi4.webm`
- 열화상 IFSI8 전후 비교: `experiments/teed_ui/contour_stability/video_final_graph/fire360_ifsi8.webm`
- IFSI8 동일 영역 확대 비교: `experiments/teed_ui/contour_stability/video_final_graph/fire360_ifsi8_detail.webm`
- 가시광 화염 장면 전후 비교: `experiments/teed_ui/contour_stability/video_final_graph/fire360_visible.webm`
- 전체 영상 디코딩 결과: `experiments/teed_ui/contour_stability/video_final_graph/video_decode_qa.json`
- 회귀 검사와 검증 기록: `experiments/teed_ui/contour_stability/regressions.json`, `experiments/teed_ui/contour_stability/review.md`, `experiments/teed_ui/contour_stability/validation_manifest.json`

위 영상과 JSON은 로컬 검증 산출물이며 GitHub 배포 대상이 아니다.


재현:
```sh
python experiments/teed_ui/evaluate_contour_stability.py --output video_final_graph --render
python experiments/teed_ui/qa_contour_videos.py --root video_final_graph
```

비교 기준은 직전 확률 시간 안정화 결과다. 모든 프레임의 기준 픽셀 수를 이전 저장 결과와 대조하고, 영상 생성 시 코드 SHA-256을 함께 기록한다. 영상은 원본 29.97fps의 동일 시점으로 재생한다.

## 해석 한계

Fire360 영상에는 현재 입력으로 사용할 수 있는 보정된 섭씨 온도맵과 거리맵이 없다. 이 변경은 화면의 윤곽 안정화를 다루며 절대온도나 근거리 장애물을 추정하지 않는다. 경계 보존 비율과 점멸 지표는 장애물 회피 정답률이 아니다. 사람/실제 장애물의 정답 라벨, SmokeBasement의 과제 성능, Orin 실기 지연은 별도 검증이 필요하다. 비교 영상은 원본 재생 시간으로 저장되므로 처리 가능한 실시간 FPS를 나타내지 않는다.
