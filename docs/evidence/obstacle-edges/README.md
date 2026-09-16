# FireSight 장애물 윤곽 비교 실험

사진·측정표·학습 설명과 영상 링크를 담은 Word 실험 보고서 (원본 프로젝트 참조: `deliverables/FireSight_thermal_experiment_report.docx`; 선별본에는 미포함)와
동봉 농연 영상 7개 (원본 프로젝트 참조: `deliverables/media`; 선별본에는 미포함)를 제공한다. 문서를 내려받을 때 `media/`와 `images/` 폴더도 함께 유지한다.
출처와 변경 내역은 SOURCE_CREDITS.txt (원본 프로젝트 참조: `deliverables/SOURCE_CREDITS.txt`; 선별본에는 미포함)에 기록했다.

파일별 이름·내용·처리 방식은 [결과물 파일 안내](ARTIFACT_GUIDE.md)를 참고한다.

## 농연 조건의 핵심 실행

최종 핵심 샘플은 **SmokeBasement**의 실제 열화상 연속 PNG다. 공개 수집 설명에
소방 훈련용 연기로 채운 지하실이라는 조건이 명시되어 있다. 현장 사진도 함께
확인했다. IFSI 비교는 보조 열화상 실험이다.

```powershell
./.venv/Scripts/python.exe -u experiments/obstacle_edges/download_smoke_clips.py
./.venv/Scripts/python.exe -u experiments/obstacle_edges/run_smoke.py
```

27GB ZIP의 중앙 디렉터리와 필요한 파일 범위만 다운로드하고 ZIP CRC와 SHA256를
검증한다. `data/smoke_manifest.json`에 원래 시각과 선택된 파일명을 모두 기록한다.
run3 중앙 24초에서 32장으로 학습하고, run5 중앙 12초를 10fps로 최근접 시각
재표본화하여 120프레임을 검증한다. 서로 다른 주행이지만 같은 지하실이므로
독립 현장 일반화 시험은 아니다. 구간별 연기 농도나 가시거리는 측정되어 있지 않다.

`outputs/smoke/smoke_heldout_{original,baseline,thin_only,structural,adapted,pidinet}.mp4`
가 농연 데이터 결과다. 영상 안에 새 UI/텍스트를 넣지 않으며 이 원본에는 IFSI의
숫자/로고도 없다. 가중치, `training.json`, `metrics.json`, boolean 마스크 및
대표 프레임 비교 이미지도 같은 폴더에 저장된다. 실제 제공 PNG는 uint8, 640×512,
3채널이다. 별도의 16-bit 센서 원본이나 온도값으로 해석하지 않는다.

출처: https://zenodo.org/records/15173055 (DOI 10.5281/zenodo.15173055).
현장 사진은 촬영 환경의 참고 자료이며 처리 영상과 동기화된 RGB 정답이 아니다.

실행 결과와 채택 판단은 [REPORT.md](REPORT.md), 모델 조사 및 실제 라벨 학습
권고는 [MODEL_RESEARCH.md](MODEL_RESEARCH.md)에 기록한다.

영상에 새 패널·설명·박스·물체명을 넣지 않는다. 원본에 이미 포함된 열화상
카메라 숫자와 기관 마크는 원본 배경에 남으며, 해당 영역의 엣지는 기존 저장소
프로파일과 동일하게 제외한다. 실제 센서 Y16 입력이 아닌 8-bit 표시 영상이다.

## Windows CPU 재현

저장소 루트, Python 3.12 기준:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r experiments/obstacle_edges/requirements.txt
powershell -File experiments/obstacle_edges/download_data.ps1
git clone https://github.com/hellozhuo/pidinet.git experiments/obstacle_edges/third_party/pidinet
git -C experiments/obstacle_edges/third_party/pidinet checkout d21aa881ed9c628571636fad39acfe1fad517ebd
./.venv/Scripts/python.exe -m unittest discover -s experiments/obstacle_edges -p 'test_*.py' -v
./.venv/Scripts/python.exe -u experiments/obstacle_edges/run_experiment.py --suite thermal
```

`--suite rgb`는 초기 보조 RGB 실험을 수행한다. `--reuse-checkpoint`는 해당 suite의
`outputs/<suite>/training.json`에 기록된 SHA256와 일치하는 가중치만 재사용한다.
동일 suite 재실행은 그 suite의 결과를 덮어쓰므로 보존할 실험은 먼저 별도 복사한다.

두 thermal/smoke 실험 완료 후 추가 속도 비교와 최종 보고서 생성:

```powershell
./.venv/Scripts/python.exe experiments/obstacle_edges/benchmark_conversion.py
./.venv/Scripts/python.exe experiments/obstacle_edges/build_report.py
```

`build_report.py`는 모든 농연/IFSI 출력 영상을 다시 끝까지 디코드하고
`artifact_verification.json`, `results_summary.json`, `REPORT.md`를 생성한다.

## 출력

`outputs/thermal/`:

- `heldout_original.mp4`: IFSI Video 4 원본 리사이즈.
- `heldout_baseline.mp4`: 현재 CLAHE 2.0, threshold .75, 3px dilation, 배경 .24.
- `heldout_thin_only.mp4`: 같은 입력/모델/threshold, skeletonization, 배경 .55.
  이름은 축약명이며 배경도 바뀌므로 '선 두께만'의 순수 영상 ablation은 아니다.
  엣지 마스크로 비교하면 배경 밝기의 효과를 분리할 수 있다.
- `heldout_structural.mp4`: bilateral(5,25,3), CLAHE 없음, hysteresis .60/.85,
  skeletonization, 배경 .55.
- `heldout_adapted.mp4`: thermal fusion-head 적응 후 동일 structural 후처리.
- `heldout_pidinet.mp4`: 공식 PiDiNet table5, ImageNet RGB 정규화, hysteresis
  .20/.40, skeletonization, 배경 .55. 모델별 threshold는 독립적인 시작 설정이며
  정답 라벨로 calibration한 동일 operating point가 아니다.
- `training_*.mp4`: IFSI Video 8의 같은 출력. 학습 영상이므로 일반화 증거로
  사용하지 않는다. 학습에서는 앞 4초를 제외하지만 출력은 전체 타임라인이다.
- `*_masks.npz`: 각 방식의 손실 없는 boolean 엣지 마스크, [T,240,320].
- `*_frame_*.png`: 위 행 원본/기존/thin, 아래 행 structural/adapted/PiDiNet.
- `teed_fusion_adapted.pth`, `training.json`: 실제 학습 가중치와 학습 이력.
- `metrics.json`: 입력·가중치 hash, 환경, 출력 디코드 검증, 픽셀 점유율,
  warm CPU 추론/후처리 시간, flow 정합 후 선 불일치 진단.

원본 fps와 프레임 수를 유지하고 H.264로 저장한다. 영상은 무음이며 320×240은
기존 실행 기본값과 비교하기 위한 크기다. 와이드 원본을 4:3으로 리사이즈하는
기존 동작을 유지했으므로 모양 비율에 왜곡이 있다. 실장 시 센서 비율에 맞춘다.

원본 데이터/외부 코드/전체 실행 출력/체크포인트는 Git에 넣지 않는다. 보고서와 작은 요약 지표,
`deliverables/`의 Word 문서·대표 사진·출처 표시가 있는 농연 검토 영상 7개는 Git에 포함한다.
공개 데이터의 접근 및 사용 조건은 원 출처를 따른다.
