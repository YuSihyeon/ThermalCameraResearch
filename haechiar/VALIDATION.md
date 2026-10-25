# 코드 이관 검증 — 2026-10-25

검증 대상은 `firesight_sw/main@339af10`과 `calib/YSH@659dca3`의 코드 복사본입니다.
실제 장비 없이 Windows에서 확인했습니다.

## 파일 보존

| 항목 | 결과 |
|---|---|
| 가져온 원본 파일 | 212개 SHA-256 일치: 소프트웨어 139개, 보정 73개 |
| 가져온 Python 소스 | 126개 구문 분석 통과 |
| 기존 이미지 | 33개 SHA-256 변경 없음 |
| 기존 영상 | MP4 6개 + WebM 6개 SHA-256 변경 없음 |
| 기존 PLY | 2개 SHA-256 변경 없음 |
| README의 이미지·영상 임베드 | 9개 줄의 내용·경로 변경 없음 |
| 기존 파일 전체 | 129개 중 README, UPSTREAM_STATUS, code/README만 수정; 나머지 126개 동일 |

기존 자료 비교 기준은 이 저장소의 `d1ee7fb0402bcb8881029dbf9c6e983d527603d5`입니다.
가져온 코드의 원본 경로·커밋·해시는 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)에 있습니다.
커밋하는 Git blob 212개와 실행 파일 모드도 원본과 일치합니다. 복사 후 알고리즘이나 테스트
기준을 수정하지 않았습니다. 원본 보존 때문에 일부 파일 끝의 기존 빈 줄은 유지했습니다.

## 실행 검사

최종 검사 환경: Windows, Python **3.13.17**, NumPy **2.5.4**, OpenCV **4.14.0**
(`opencv-python` 4.14.0.94), PyTorch **2.14.1+cpu**, pytest **9.1.1**, PyYAML **6.0.3**.

| 검사 | 결과 |
|---|---|
| 보정 패키지 editable 설치 | 성공 |
| 보정 전체 테스트 `python -m pytest -q -rs` | **100 passed, 10 skipped** |
| 독립 윤곽선 검사 | **10개 스크립트 통과** |
| 실제 TEED 체크포인트 로딩 | 알려진 SHA-256 일치, 36개 tensor 로딩 |
| 기존 `public-media/smoke-original.mp4` CPU 실행 | 윤곽 추적을 켠 10프레임 처리, JSON 출력 성공 |
| RGB-D + 열화상 demo | 원본 36 samples → 12 frames, dropped/unmatched 0 |
| 열화상 AI demo | 원본 12 samples → 12 frames, dropped/unmatched 0 |
| 레이더 + 열화상 demo | 원본 24 samples → 12 frames, dropped/unmatched 0 |
| 세 demo의 `verify-session` | 원본 배열·메타데이터 검증 성공 |

건너뛴 보정 테스트는 Bash가 없는 환경의 Linux 실행기 검사 7개와 ROS 2 `rclpy`가 필요한
통합 검사 3개입니다. 통과 수에서 제외했습니다. 세 demo는 모두 `provenance=synthetic`,
`hardware_verified=false`이며 AI 모델 미지정 경계는 Canny 기준선입니다.

윤곽선 검사는 경계 정리, 시간 안정화, 윤곽 추적·행동·렌더링, 움직임 분리·확률 누적,
열화상 팔레트와 HUD 통합을 포함합니다. 실행 파일 목록은 [verify.py](verify.py)에 있습니다.

## 환경과 검증 한계

Windows Python 3.12.14의 첫 검사에서는 98 passed, 2 failed, 10 skipped였습니다.
실패는 자동 촬영 테스트의 23/24 프레임 뒤 타임아웃과 짧은 간격 샘플의 timestamp 중복입니다.
해당 환경의 `monotonic` 시계 해상도는 15.625 ms였습니다. Python 3.13의 고해상도 시계 환경으로
바꾸어 원본 테스트를 다시 실행한 최종 결과가 위의 100 passed입니다. Windows에서 보정 검사에는
Python 3.13 이상을 권장합니다. 첫 3.13 실행에는 NumPy DLL 로딩 오류가 있었으며, 보안 설정 변경
없이 같은 설치 상태에서 재실행한 검사와 CLI 실행은 완료됐습니다.

실제 Boson·RGB-D·레이더 연결, Windows raw 녹화 부하 시험, Jetson/CUDA 성능,
외부 깊이 모델 전체 추론·학습은 실행하지 않았습니다. 원본 Fire360 전체 영상이나
HaechiAR Git 이력이 필요한 비교 스크립트도 이번 검사에서 제외했습니다.
10프레임 CPU 실행은 설치·입출력 확인이며 성능 benchmark로 해석하지 않습니다.

재현 명령과 모델·장비별 준비는 [실행 안내](README.md)를 참고합니다.
