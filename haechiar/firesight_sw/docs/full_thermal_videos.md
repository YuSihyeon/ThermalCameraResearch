# Fire360 열화상 전체 영상 적용

범위는 현재 로컬에 확보된 열화상 표시 영상 두 파일의 처음부터 끝까지다. IFSI Video 4 (2).mp4는 625프레임/20.854초, IFSI Video 8 (2).mp4는 661프레임/22.055초이며 원본 재생률은 모두 29.970fps다. 가시광 MTS·GoPro 영상은 대상에서 제외한다.

기존에 검증한 확률 안정화와 윤곽 추적 설정을 그대로 사용한다. 각 영상의 0번 프레임에서 상태를 새로 시작하고 모든 프레임을 연속 재추론한다. 8초 결과를 반복하거나 연결하지 않는다. 비교 영상 좌측과 우측은 동일 프레임의 TEED 추론 한 번을 공유한다.

출력은 적용 영상 720×406(아래 1픽셀 검정 패딩)과 전후 비교 영상 1440×540이다. 원본 전체 프레임과 재생률을 유지하며 출력은 무음이다. 원본 파일은 수정하지 않는다. 두 원본 모두 시작과 끝에 검정 화면이 있으며 이 구간도 포함한다. 열화상 자체를 새로 측온하지 않으며 이전 결과의 안정화·성능 한계는 그대로 적용된다.

## 재현

```sh
python experiments/teed_ui/render_full_thermal_videos.py
python experiments/teed_ui/qa_full_thermal_videos.py
```

## 결과

총 1,286프레임을 처리했고 전체 적용본 2개와 비교본 2개를 생성했다.

| 열화상 원본 | 프레임 | 전체 길이 | 결과 |
| --- | ---: | ---: | --- |
| IFSI Video 4 | 625 | 20.854초 | 로컬 생성: `experiments/teed_ui/contour_stability/thermal_full/fire360_ifsi4_full_applied.webm`, `fire360_ifsi4_full_comparison.webm` |
| IFSI Video 8 | 661 | 22.055초 | 로컬 생성: `experiments/teed_ui/contour_stability/thermal_full/fire360_ifsi8_full_applied.webm`, `fire360_ifsi8_full_comparison.webm` |

검증 결과는 로컬 생성 기록 `experiments/teed_ui/contour_stability/thermal_full/qa.json`에 있다. 원본 2개와 출력 4개를 끝까지 디코딩했으며 프레임 수, 매 프레임 해상도, 재생률, 0번부터 마지막까지의 순차 기록, 원본 및 렌더 코드 해시가 모두 일치했다. 시작·25%·50%·75%·끝의 디코딩 화면과 큰 비교 화면도 확인했다. 렌더·검증 스크립트의 Ruff와 Basedpyright 검사는 통과했다.

검증 범위와 근거는 로컬 생성 기록 `experiments/teed_ui/contour_stability/thermal_full/verification.json`에 함께 보존했다. 이번 검증은 전체 길이 출력의 무결성에 대한 것으로, 새로운 안정화 성능 개선이나 장애물 회피 정확도·실기 실시간 성능을 의미하지 않는다. 브라우저 재생은 검증하지 않았으며 영상 디코딩과 추출 화면을 확인했다. 위 WebM과 JSON은 GitHub 배포 대상이 아니다.
