# 원본 데이터와 복원 자료

**2026-09-16 최종 보존 상태:** 조사한 Windows 연구 원본과 WSL 전체 export, 연구별 직접 추출본, conda·Unity·Unreal 환경 archive의 로컬 내용 검증을 마쳤다. USB 전송·외부 사본 검증과 초기화 후 전체 실행은 아직 수행하지 않았다. 실제 복사 목록·해시·확인하지 못한 자료는 개인 보존 묶음의 `PRESERVATION_STATUS.md`와 `_control/manifests/`를 기준으로 확인한다.

이 저장소는 열화상 연구의 설명·선별 코드·공개 허용 결과를 담는다. 전체 직접 촬영 입력, 휴대폰 원본, FireSight 팀 코드 및 실행 환경은 별도의 개인 보존 묶음에 둔다. 이 저장소를 clone하는 것만으로 전체 연구가 복원되지는 않는다.

아래는 저장소가 개인 보존 묶음의 `06-ThermalCameraResearch/repository/`에 있을 때의 상대경로다. 공개 Git에는 private 원본을 포함하지 않는다. 실제 복사·해시 검증의 최종 상태는 개인 보존 관리 기록을 확인한다.

| 개인 보존 위치 | 역할 |
|---|---|
| `../originals/thermal_original/` | 직접 캡처 52 PNG, PLY 2개, 원래 코드/백업, 스크린샷, 휴대폰 원본 |
| `../originals/thermal_empty/` | 현재 비어 있던 과거 작업 폴더의 위치 기록 |
| `../originals/download-originals/` | 제공받은 Boson·distance 영상 등의 원 파일명 자료 |
| `../originals/firesight_private/` | 전체 private 팀 연구 작업 폴더, Git/worktree, calibration 입력, ignored 출력/환경 |
| `../originals/previous-thermal-archive/` | 이전 로컬 아카이브의 전체 media, 52장, private 검토 기록 및 provenance |
| `../../_shared/wsl/` | 다른 Linux 작업 폴더를 확인할 수 있는 전체 WSL exports; manifest와 복원 검증 필요 |

## 입력과 결과를 구분하기

직접 캡처 `frame_0.png`–`frame_51.png`는 기존 검토에서 640×512 uint8 grayscale이었다. 온도 calibration이나 16-bit radiometric RAW, 센서 timestamp는 확인되지 않았다. PLY는 영상 강도를 높이로 표현한 heightfield이므로 metric depth 또는 °C 실측값으로 취급하지 않는다.

휴대폰 원본 영상과 공개 preview는 보존 역할이 다르다. 원본에는 주변 사람·작업 화면이 포함될 수 있어 개인 백업에서만 다룬다. distance 결과의 `ASSUMED` depth 배열·합성 시각화는 실제 거리 측정 데이터가 아니다. calibration의 검토용 데이터도 현장 정확도 검증과 구별한다.

## 현재 복원 한계

직접 촬영 52장과 PLY·원본 phone 영상은 Windows 원본 폴더에 존재했다. 그러나 FireSight 문서가 가리키는 SmokeBasement 선택 PNG 152장과 source manifest, Fire360 IFSI 4/8 원본 MP4, PiDiNet 비교 checkpoint, Smoke/IFSI 적응 TEED checkpoint는 조사한 Windows 체크아웃에서 확인하지 못했다. 표준 TEED checkpoint와 선별 결과 영상은 남아 있다.

`smoke_heldout_original.mp4`라는 이름의 영상도 리사이즈/재인코딩본이다. raw PNG와 정확한 학습 가중치를 대체할 수 없다. 다른 WSL 작업 폴더 등은 별도 확인이 필요하며, 이 문서는 모든 장치에서 원본이 사라졌다고 단정하지 않는다. 재다운로드/재학습 결과는 원래 SHA가 맞지 않으면 새 실험으로 기록한다. depth-model wrapper가 있다는 사실도 해당 모델 checkpoint 확보 또는 실행 성공을 의미하지 않는다.

## 복원 순서

1. 개인 보존 묶음의 source/copy 해시와 manifest를 확인하고 원본과 분리된 작업 복제를 만든다.
2. 직접 입력·PLY·영상이 열리는지 확인한다. 당시 metadata·코드의 stride/scale·전처리 계약을 유지한다.
3. FireSight의 여러 checkout과 공통 Git object store를 함께 복원한다. worktree 지시 파일의 절대경로는 작업 복제에서 수정한다. 다른 이력을 가진 checkout을 같은 remote라는 이유로 합치거나 삭제하지 않는다.
4. Windows venv의 Python 본체·DLL과 package/version을 맞춘다. CUDA 환경과 문서 생성용 외부 Node 의존성은 새 머신에서 확인한다. venv 디렉터리 복사만으로 실행을 보장하지 않는다.
5. synthetic/recorded replay부터 확인하고, 실제 입력·정확한 모델 가중치가 있는 경우에 과거 실험과 비교한다. 새 출력 경로를 써서 과거 결과를 덮어쓰지 않는다.

상위 Git에는 HEAD가 없어도 많은 dangling 파일 객체가 남아 있었다. 복구 가능성을 유지하려면 원래 `.git`를 포함한 전체 private 보관본을 유지하고 `git gc`, `git prune`, `git clean`을 실행하지 않는다. forensic 복구는 별도 복제에서 한다.

전체 FireSight 팀 코드를 공개 저장소에 올리는 절차는 포함하지 않는다. 이 문서는 USB 보존 완료, 모든 누락 원본의 회수, WSL 복원 또는 모든 실험의 재현 성공을 뜻하지 않는다.


FireSight의 로컬 Node 의존성 junction은 링크 대상을 같은 상대 위치의 일반 디렉터리로 풀어 7,796개 파일/335,197,562바이트를 검증 복사했다. 원래 링크 대상 정보와 공유 Node 런타임도 개인 보존본에 남겼다.
