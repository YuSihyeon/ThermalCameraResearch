# 온도맵 없는 영상의 가시광 불꽃 후보 표시

사용자가 지목한 02815 장면에서 이전 화면은 온도맵이 없다는 이유로 빨간 표시를 만들지 않았다. 이번 추가 기능은 가시광 영상에 보이는 불꽃 후보를 별도로 표시한다. 섭씨 온도 추정이나 발화점 판정은 하지 않는다.

## 동작

- 기본은 꺼짐. **일반 가시광 영상임을 확인한 입력**에서 `--visible-flame-candidates`로 켠다. 열화상 팔레트·색상화 영상에 적용하지 않는다. BGR 파일 형식 자체가 가시광 센서의 증거는 아니다.
- TEED와 독립적으로 따뜻한 색 범위, 밝은 중심부, 프레임 간 변화, 연속 확인, 최소 면적을 결합한다.
- 감지된 영역을 빨간 윤곽과 옅은 반투명 채움으로 표시한다. `FLAME?`는 **불꽃 후보**라는 뜻이다.
- 온도맵이 없으면 `TEMP N/A`가 유지되며 고온 등온선의 픽셀 수는 0이다. 후보 픽셀 수는 별도 지표다.
- 첫 프레임에는 후보가 없다. 현재 색·밝기 근거가 사라지면 표시도 제거한다. 변화 이력은 최대 3프레임만 유지해 짧은 정지 구간을 허용하고 오래 정지하면 만료된다. 서로 다른 영상·seek 이후 HUD 객체를 재생성하거나 detector를 reset해야 한다.
- OSD는 후보 연결성분을 구하기 전에 제거한다.
- `--loop`, 명시적인 gray8/y16, 온도/거리 sidecar와 함께 쓰면 오류를 낸다. `--no-fire-highlight`는 온도 표시와 불꽃 후보를 모두 끈다.

## 실행

저장소 루트에서 기존 환경을 사용한다.

~~~bash
python edge/orin/run_pipeline.py --source "../data/fire360/raw/02815 (2).MTS" --visible-flame-candidates --device cpu --display-width 960 --display-height 540 --background-scale 0.42 --headless --record --record-path visible_flame.mp4 --metrics visible_flame.json
~~~

재현 가능한 실제 영상 비교:

~~~bash
python experiments/teed_ui/evaluate_visible_flame.py
~~~

결과 폴더: `experiments/teed_ui/visible_flame_results`.
`frame_066_comparison.png`는 원본 / 이전 출력 / 새 출력,
`frame_066_updated.png`는 새 화면,
`visible_flame_comparison.mp4`는 연속 재생 결과다.

## 해석 한계

이번 기능은 새로 학습된 화재 AI 모델이 아니라 개발용 영상 규칙이다. 움직이는 주황색 물체, 조명, 반사광, 카메라 움직임에 의한 오검출이 가능하다. 정지한 불꽃, 무색 불꽃, 연기에 가려진 불꽃은 놓칠 수 있다. 특정 실제 장면의 표시 성공은 검출 정확도나 안전 성능을 입증하지 않는다. 라벨이 있는 여러 화재/비화재 장면 평가가 별도로 필요하다.

색과 시간적 변화를 함께 쓰는 접근은 [가시광 화염 검출 연구](https://repository.bilkent.edu.tr/bitstreams/7d0e49ea-aab5-47dd-b701-eeae3dc5335b/download)에서도 사용된다. 이 구현은 해당 논문의 wavelet 기반 깜빡임 분류기를 재현한 것이 아니며 프레임 간 차이만 확인한다.

## 이번 검증 결과

실제 02815 연속 120프레임을 재생했다. 사용자가 지목한 유형의 오른쪽 아래 불꽃이 있는 66번 프레임에서 빨간 후보 윤곽을 확인했다. 모델 좌표 후보 픽셀은 5,693개이며 온도 기반 고온 픽셀은 0이다. 22개 경계조건과 5개 CLI 검사, 녹화 파일 디코딩, lint/typecheck/compile 검사를 통과했다. 원본 코드 해시는 replay_metrics.json과 final_validation.json에 기록했다.

수정 화면, 전후 비교, 비교 영상은 로컬 생성 산출물 `experiments/teed_ui/visible_flame_results/frame_066_updated.png`, `experiments/teed_ui/visible_flame_results/frame_066_comparison.png`, `experiments/teed_ui/visible_flame_results/visible_flame_comparison.mp4`에 보존했다. 이 PNG와 MP4는 GitHub 배포 대상이 아니다.
