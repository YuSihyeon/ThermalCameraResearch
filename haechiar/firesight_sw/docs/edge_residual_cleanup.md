# 잔여 픽셀 개선 결과

기존 최소 길이 필터는 연결 성분 전체의 그래프 길이를 계산하므로 긴 선에 붙은 짧은 가지와 작은 고리가 남을 수 있다. 이번 변경은 그 두 경우를 공간 필터로 처리한다. 프레임마다 깜빡이는 선과 연결된 큰 질감 엣지는 해결하지 않는다.

## 적용

- `edge/orin/edge_filter.py`: thinning 이후 짧은 가지를 한 번 제거하고, 기존 최소 길이 검사 이후 작은 범위의 연결 성분을 제거한다.
- `edge/orin/run_pipeline.py`: `--min-branch-length 6`, `--min-edge-extent 8` 기본값과 metrics를 연결했다. 두 값을 모두 0으로 하면 기존 12 px 길이 필터 동작을 재현한다.
- 모든 기준은 모델 해상도 320×256의 픽셀 단위다. 거리 또는 실제 물체 크기를 의미하지 않는다.
- 가지는 끝점에서 분기점까지 6 px 미만일 때만 제거하며 분기점은 유지한다. 독립된 선을 끝에서 반복해서 깎지 않는다.
- 작은 덩어리는 바운딩 박스 대각선이 8 px 미만인 연결 성분이다. 픽셀 개수가 많더라도 작은 고리처럼 좁게 뭉친 경우를 걸러낸다.
- TEED 가중치, 확률/gradient 임계값, 온도·화염 로직은 이번 변경에서 바꾸지 않았다.

## 실제 연속 영상

왼쪽은 기존 최소 길이 12 px만 적용한 상태, 오른쪽은 같은 입력과 같은 TEED 출력에 새 필터 두 개를 추가한 상태다. 영상은 원본 장면 시간으로 재생되며 실시간 추론 성능을 뜻하지 않는다.

| 영상 | 검사 프레임 | 기존 표시 엣지 픽셀 합 | 개선 후 | 추가 감소 |
|---|---:|---:|---:|---:|
| SmokeBasement run5_left, 시작 후 6–12초 | 75 | 182,278 | 178,527 | 2.06% |
| Fire360 IFSI Video 4, 원본 360–539프레임 | 180 | 614,748 | 599,399 | 2.50% |

로컬 생성 영상: `experiments/teed_ui/residual_edge_results/fire360_cleanup.webm`, `experiments/teed_ui/residual_edge_results/fire360_cleanup.mp4`, `experiments/teed_ui/residual_edge_results/smokebasement_cleanup.webm`, `experiments/teed_ui/residual_edge_results/smokebasement_cleanup.mp4`. 이 비교 영상은 GitHub 배포 대상이 아니다.

원본/개선 합계는 1 px 중심선에서 측정했다. 가지 제거 카운트는 기존 길이 필터 이전 단계의 수치이므로 최종 픽셀 감소량과 같지 않을 수 있다. 두 영상의 모든 프레임에서 표시 픽셀 수가 감소했다. 이 수치는 노이즈 정답률이나 장애물 회피 성능이 아니다.

## 검증

- `experiments/teed_ui/check_edge_cleanup.py`: 직선·대각선·모서리·긴 가지·임계값 경계·영상 경계 보존과 작은 고리·짧은 가지 제거 등 19개 검사 통과.
- 새 필터를 끈 경우 저장한 이전 코드와 무작위 입력 20개에서 mask/alpha가 동일함을 확인.
- 실제 CLI로 Fire360 3프레임 실행, 새 설정과 표시 픽셀 수가 비교 영상과 일치.
- Ruff E/F/I(E501 제외), basedpyright basic(수정한 런타임 파일), Python 문법 검사 통과.
- WebM/MP4의 전체 프레임을 다시 디코딩하여 개수·1440×540 해상도·프레임 변화·재생률을 검사했다. 실제 인코딩 프레임에서 좌우 정렬·문구·주요 윤곽을 확인했다. 브라우저 재생 여부는 확인하지 않았다.
- 독립 코드 리뷰에서 차단 이슈 없음. Orin 실측과 실제 장애물 정답 라벨에 대한 recall 평가는 수행하지 않았다.

근거: `experiments/teed_ui/residual_edge_results/manifest.json`, `video_validation.json`, `code_validation.json`, `cli_metrics.json` 및 프레임별 JSON.

## 남는 문제와 다음 개선

이번 결과는 잔가지 정리 수준이며 전체 노이즈 제거를 의미하지 않는다. 작은 실제 물체도 지워질 수 있고 연결된 큰 무늬는 길이/범위 기준을 통과한다. 최소 길이를 무조건 크게 올리기보다, 다음 단계로 카메라 움직임을 보정한 시간 신뢰도를 평가해야 한다. 약한 엣지는 2–3프레임 지지를 요구하고 강한 신규 엣지는 빠르게 표시하는 방식을 후보로 두되, 잔상·깜빡임·새 장애물 표시 지연을 함께 측정해야 한다. 이번 변경에는 시간 필터를 넣지 않았다.
