# 실험 기반 검증 기록 — 2026-09-24

검증 환경: Windows, Python 3.13.7, NumPy 2.5.3, OpenCV 4.14.0, pytest 9.1.1.
장치가 연결되지 않은 개발 환경에서 실행했다.

| 검사 | 결과 |
|---|---|
| 전체 `python -m pytest -q` | 107 passed, 3 skipped |
| 신규 실험 테스트 | 41 passed |
| 기존 회귀 테스트 | 66 passed; ROS 2 런타임이 필요한 3개는 로컬에서 skip |
| RGB-D+열화상 CLI demo | 원본 18 samples → 6 bundles, 탈락/잔여 0 |
| 열화상 단독 CLI demo | 원본 6 samples → 6 frames, 탈락/잔여 0 |
| 레이더+열화상 CLI demo | 원본 12 samples → 6 bundles, 탈락/잔여 0 |
| RGB-D `verify-session` | 18개 원본 배열/메타데이터 검증 |
| radar YAML 보정 로딩 후 별도 `run` | 저장 세션 6 bundles 재처리 |
| wheel 빌드/별도 디렉터리 설치 | 성공; 설치본에서 3개 YAML 프로필 로딩 확인 |
| `git diff --check` | 통과 |

데모는 전부 synthetic이다. 실제 SDK 대신 frame 객체를 넣어 검증한 Boson/RealSense
어댑터 테스트도 실제 USB/D456 연결 검증으로 간주하지 않는다. 기존 CI는 Linux/Windows
Python 3.10/3.13과 ROS Humble에서 전체 테스트를 실행하도록 이미 구성되어 있다.

독립 코드 검토에서 발견된 항목과 수정:

- 최종 JSONL 인덱스 SHA-256 및 세션/샘플 provenance·clock·count 일관성 검사.
- 필수 센서 스트림이 하나도 기록되지 않으면 수집을 interrupted로 종료.
- RGB에 depth intrinsic이 붙지 않도록 metadata 분리. 기록된 SDK intrinsic과
  재생 시 source_intrinsic이 다르면 융합을 거부.

SHA-256은 파일/메타데이터의 비의도적 변경 검출용이며 서명이나 물리적 출처 증명이 아니다.
세션과 체크섬을 모두 의도적으로 다시 작성한 자료의 진위를 판단하지는 않는다.

재현 절차는 [실험 안내](experiments.md)의 장비 없는 시작 명령을 따른다.
실제 장비 연결, 타깃 실측, radar firmware decoder, Jetson 성능, AI 학습은 아직 검증하지 않았다.
