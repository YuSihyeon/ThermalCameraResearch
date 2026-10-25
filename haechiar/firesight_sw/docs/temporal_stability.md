# Fire360 TEED 시간 안정화

첨부 화면 기록의 목표는 시점이 움직여도 긴 윤곽이 이어지고, 독립 점과 깜빡임·잔상이 적은 표시다. Fire360에 움직임 보정 후 TEED 확률을 누적하는 시간 필터를 추가했다. 모델 가중치와 기존 공간 필터·열/화염 판단은 그대로다.

## 적용 방식

1. 320×256 모델 입력에서 양방향 Farneback optical flow를 구한다.
2. 이전 확률을 현재 좌표로 이동한다. 왕복 오차 1.5 px, 영상 밝기 오차 35 이하이고, 이전·현재 모두 유효한 영역에서만 과거 값을 사용한다.
3. 현재 확률이 0.25 이상인 위치에서 이전 0.6 / 현재 0.4로 누적한다. 타임스탬프가 있으면 `0.6 ** (프레임 간격 × 30)`으로 가중치를 조정한다.
4. 현재 확률이 표시 임계값(기본 0.75) 이상인 강한 경계는 시간 누적 없이 즉시 통과한다. 이후 기존 gradient·길이·가지·범위 필터는 동일하게 적용된다. 현재 근거가 없어진 위치에는 과거 선을 유지하지 않는다.
5. 정합이 나쁘거나, 입력 크기·OSD 영역이 바뀌거나, 프레임 번호/시간이 끊기거나, 간격이 250 ms를 넘으면 초기화한다. 영상 루프는 입력 계층에서 명시적으로 초기화한다.

정합 실패는 현재 프레임으로 복귀한다. 표시 임계값 아래의 약한 경계에는 누적으로 인한 지연이 있을 수 있고, 강한 경계 즉시 통과가 모든 실제 장애물의 즉시 검출을 보장하지는 않는다.

- 구현: `edge/orin/temporal_edges.py`
- 연결: `edge/orin/hud.py`, `edge/orin/run_pipeline.py`, `edge/orin/replay.py`
- 실행법: [Orin README](../edge/orin/README.md#fire360-영상-시간-안정화)
- API 근거: [OpenCV Farneback](https://docs.opencv.org/4.x/de/d9e/classcv_1_1FarnebackOpticalFlow.html), [역방향 remap 좌표](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)

## 실제 영상 검증

동일 TEED 출력과 동일 공간 필터로 시간 안정화만 켜고 끈 결과를 비교한다. 입력과 결과는 29.970fps의 원본 시간 간격을 유지한다. 저장 영상의 FPS는 모델 추론 속도가 아니다.

| 입력 | 원본 프레임 | 길이 |
|---|---:|---:|
| Fire360 IFSI Video 4 | 300–539 | 240프레임 / 8.008초 |
| Fire360 IFSI Video 8 | 90–329 | 240프레임 / 8.008초 |
| Fire360 가시광 02815 (2).MTS | 0–119 | 120프레임 / 4.004초 |

가시광 파일은 더 뒤의 디코딩에서 실패가 발생해 유효한 첫 120프레임만 검증했다. 전체 파일을 검증했다는 의미는 아니다.

| 구간 | 엣지 변화량 감소 | 기존 경계 근접 유지율 | 표시 픽셀 변화 |
|---|---:|---:|---:|
| IFSI 4 | 19.4% | 97.6% | +5.6% |
| IFSI 8 | 16.5% | 98.1% | +4.9% |
| 가시광 | 25.3% | 96.6% | +15.9% |

- IFSI 4 비교 영상: 로컬 생성 `experiments/teed_ui/temporal_stability/final/fire360_ifsi4.webm`
- IFSI 8 비교 영상: 로컬 생성 `experiments/teed_ui/temporal_stability/final/fire360_ifsi8.webm`
- 가시광 비교 영상: 로컬 생성 `experiments/teed_ui/temporal_stability/final/fire360_visible.webm`
- IFSI 4 개선 결과 단독 영상: 로컬 생성 `experiments/teed_ui/temporal_stability/final/fire360_ifsi4_stabilized.webm`

왼쪽 기존 / 오른쪽 개선. 각 비교 영상의 MP4 사본도 같은 폴더에 있다. 안정화로 끊기던 선이 유지되므로 표시 픽셀 수는 증가했다. 잔여 점이 모두 제거됐다는 뜻은 아니다.


`엣지 변화량`은 현재/이전 마스크를 동일 optical flow로 맞춘 후 1 모델 픽셀 안에서 대응하지 않는 픽셀의 비율이다. 실제 물체 등장·사라짐과 optical flow 오차도 포함하므로 깜빡임만의 정답 지표가 아니다. `기존 경계 근접 유지율`은 기존 마스크 픽셀이 개선 마스크 2 모델 픽셀 안에 있는 비율이며 실제 장애물 recall이 아니다. 픽셀 증가는 끊긴 선 유지 효과와 불필요한 선 유지 가능성을 모두 포함한다.

## 재현과 검사

```bash
# 작업 폴더 firesight_sw_ui, Windows 공용 venv 기준
../.venv/Scripts/python.exe experiments/teed_ui/check_temporal_edges.py
../.venv/Scripts/python.exe experiments/teed_ui/check_edge_cleanup.py
../.venv/Scripts/python.exe experiments/teed_ui/check_temporal_integration.py
../.venv/Scripts/python.exe experiments/teed_ui/evaluate_temporal_edges.py --output final --render
../.venv/Scripts/python.exe experiments/teed_ui/qa_temporal_videos.py
```

평가는 `cache_temporal_inputs.py`로 생성한 실제 입력/확률 캐시를 사용한다. 캐시의 입력 경로·원본 시각·체크포인트는 각 결과 JSON의 `source`에, 렌더 코드 SHA-256은 `code_sha256`에 기록된다.

- 시간 동작 39개, 기존 공간 필터 19개, 실제 실행기/루프 연결 7개 계약 통과.
- 변경한 production Python 4개와 회귀 스크립트 2개 파일 basic 타입 검사: 오류/경고 없음. 문법 검사 통과.
- 신규 시간 안정화와 HUD/CLI/실험 코드의 선택 Ruff 검사 통과.
- `replay.py`에는 기존 I001/F401/F541 3건이 남아 있다. HEAD 원본에서도 같은 진단임을 확인했고 이번 기능과 관계없는 정리는 하지 않았다.
- 320×256 합성 입력에서 시간 필터만 CPU 1스레드 약 26 ms/update. 전체 추론 지연이나 Orin 성능 측정이 아니다.
- 저장된 WebM/MP4 9개를 끝까지 디코딩하여 프레임 수·크기·FPS와 프레임 변화를 확인했다. 비교 프레임과 motion strip에서 문틀·사람 윤곽이 현재 장면을 따라 움직이며 뚜렷하게 길게 남는 이중 윤곽이 없는지 검토했다. 작은 선 변화는 남아 있다. 브라우저 재생 검증은 포함하지 않는다.
- 첫 4프레임의 실제 CLI 출력 엣지 수가 평가 결과와 일치하며, 영상 생성 코드 해시가 현재 코드와 일치한다.
- 상세 실행 근거: 로컬 생성 기록 `experiments/teed_ui/temporal_stability/verification.json`.

## 첨부 영상과의 차이 및 한계

첨부 영상은 7.47초·60fps 녹화지만 HUD 내용 변화는 초당 약 23–25회로 관측되었다. 시점 회전에 맞춰 윤곽이 움직이며 전체를 고정한 영상은 아니다. 기록 영상만으로 내부 알고리즘·실제 장치 지연은 판단할 수 없다. 첨부 영상 분석은 로컬 생성 기록 `experiments/teed_ui/temporal_stability/reference_audit/findings.md`에 보존했다. 이 영상·JSON·분석 기록은 GitHub 배포 대상이 아니다.

Fire360은 센서·장면·경계 대비가 다르므로 첨부 영상과 같은 안정성에 도달했다고 단정하지 않는다. 반복적인 작은 선 변화는 남을 수 있다. 화면 녹화를 느리게 만들거나 이전 화면을 고정해서 안정화로 보이게 하지 않았다. 실제 장애물 회피/누락률, 소방 현장 안전성, Orin 실시간 성능은 별도 입력과 장비 검증이 필요하다.
