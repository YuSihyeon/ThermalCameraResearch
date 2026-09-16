# 결과물 파일 안내

이 문서는 2026-09-06 실행으로 저장된 결과물의 **파일명, 내용, 처리 방식, 비교 목적**을 설명한다. 아래 파일명은 클릭하여 열 수 있다. 전체 실행 출력·가중치·원본 데이터는 로컬에 보관되므로 Git 저장소만 내려받으면 일부 링크 대상이 없을 수 있다.

GitHub에서도 받을 수 있는 Word 보고서 (원본 프로젝트 참조: `deliverables/FireSight_thermal_experiment_report.docx`; 선별본에는 미포함),
농연 영상 7개 (원본 프로젝트 참조: `deliverables/media`; 선별본에는 미포함), 대표 사진 (원본 프로젝트 참조: `deliverables/images`; 선별본에는 미포함)은 `deliverables/`에 별도로 포함했다.
영상의 파일명과 처리 방식은 아래 설명과 동일하다. 출처 표시 (원본 프로젝트 참조: `deliverables/SOURCE_CREDITS.txt`; 선별본에는 미포함)와
파일별 SHA256 (원본 프로젝트 참조: `deliverables/MEDIA_MANIFEST.json`; 선별본에는 미포함)도 함께 제공한다.

## 먼저 볼 영상

1. 농연 원본 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_original.mp4`; 선별본에는 미포함)에서 열화상에 실제로 남아 있는 구조를 확인한다.
2. 기존 처리 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_baseline.mp4`; 선별본에는 미포함)와 구조 경계 개선 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_structural.mp4`; 선별본에는 미포함)을 비교한다.
3. 파인튜닝 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_adapted.mp4`; 선별본에는 미포함)으로 학습의 추가 효과를, PiDiNet (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_pidinet.mp4`; 선별본에는 미포함)으로 대체 모델의 경계를 비교한다.
4. 순수 엣지 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_edges_only.mp4`; 선별본에는 미포함)에서는 배경 밝기의 영향을 제외하고 남은 선을 확인한다.

선이 적다는 것만으로 장애물 검출이 좋아졌다고 판단할 수 없다. 기둥 외곽·바닥 접점·통로 경계가 유지되는지, 실제 경계가 지워지거나 잘못 연결되는지를 함께 본다. 정답 라벨 기반 회피 정확도는 미검증이다.

## 이름에 들어간 처리 방식

모든 출력 영상은 320×240 무음 H.264다. 새 패널·물체명·박스·설명 자막은 넣지 않았다. `original`도 원본 파일 그대로가 아니라 비교 크기로 리사이즈하여 다시 저장한 영상이다. 임계값은 모델의 엣지 출력값에 적용하며 장애물 존재 확률로 해석하지 않는다.

| 접미사 | 이름과 처리 방식 | 비교할 내용 |
|---|---|---|
| `original` | 입력 표시 영상 리사이즈. 엣지 추론·오버레이 없음. | 열화상/가시광 원본에 보이는 구조의 기준. |
| `baseline` | 기존 TEED. CLAHE 2.0으로 국소 대비 강화 → 임계값 0.75 → 3픽셀 팽창 → 배경 밝기 계수 0.24 위 초록 선. | 기존 굵은 선과 복잡한 경계의 기준. |
| `thin_only` | 같은 TEED 입력·임계값 → 선을 가늘게 만드는 skeletonization → 배경 계수 0.55. | 선 팽창 제거와 밝은 배경의 효과. 배경도 바꾸므로 두께만의 순수 비교는 아님. |
| `structural` | bilateral(5,25,3)로 미세 변동 완화, CLAHE 없음 → 기존 TEED → 낮은/높은 임계값 0.60/0.85의 hysteresis → skeletonization → 배경 0.55. | 강한 선과 이어지는 약한 선을 살리고 독립적인 약한 선을 제거. 현재 권장 비교안. |
| `adapted` | 해당 학습 데이터로 TEED fusion head 480개 파라미터를 실제 미세조정한 가중치 → `structural`과 동일 후처리. | 후처리만 바꾼 결과에 대한 학습의 추가 효과. 전체 모델 재학습이나 사람이 붙인 장애물 정답 학습은 아님. |
| `pidinet` | 공식 PiDiNet table5 사전학습 가중치, RGB ImageNet 정규화 → hysteresis 0.20/0.40 → skeletonization → 배경 0.55. | 대체 모델의 경계 연결과 처리 속도. TEED와 임계값이 달라 정확도 순위로 단정할 수 없음. |
| `edges_only` | `structural`의 저장된 마스크를 검은 배경 위 초록 선으로 출력. | 배경 영상 없이 경계만 확인. 새로운 모델이나 추가 학습 결과가 아님. |

`adapted` 학습은 다중 해상도 TEED 출력으로 만든 의사 라벨을 사용하고 fusion head만 8 epoch, 256 step 갱신했다. 농연용과 IFSI용 가중치는 별개다. 이번 출력에는 depth 추론이나 시간 평균을 적용하지 않았다.

## 이미지와 수치 파일 읽는 법

- 비교 PNG는 **위 행 왼쪽부터 원본 / baseline / thin_only, 아래 행 왼쪽부터 structural / adapted / PiDiNet**인 2행×3열 이미지다. 영상에는 이 비교 패널을 넣지 않았다.
- `frame_0060` 같은 끝 번호는 해당 출력 시퀀스의 0부터 시작하는 프레임 번호다. 입력 파일명에도 `frame_...`이 들어간 단일 이미지 실험은 마지막 번호 `0000`이 그 단일 입력의 첫 프레임을 뜻한다.
- `*_masks.npz`에는 `baseline`, `thin_only`, `structural`, `adapted`, `pidinet`별 boolean 배열 `[프레임 수, 240, 320]`이 있다. True는 표시할 엣지 픽셀이다. 동영상 압축이나 배경 밝기 영향을 받지 않는 비교용이며 정답 마스크가 아니다.
- `metrics.json`은 실행 환경, 입력/가중치 식별 정보, 방식별 엣지 점유율·처리 시간 등의 기록이다. 점유율은 화면에서 선이 차지하는 비율이고 정확도가 아니다. optical flow 정합 후 불일치는 깜박임 진단값이며 선 두께와 flow 오류의 영향도 받는다. 초기 RGB 기록은 최신 진단 항목과 다를 수 있다.
- `training.json`은 학습 설정·loss·시간·가중치 hash 등의 이력이다. loss 감소가 장애물 회피 개선을 입증하지 않는다.
- `teed_fusion_adapted.pth`는 학습 결과 TEED state dictionary다. 저장 파일에는 모델 전체 가중치가 들어 있지만 실제 갱신 범위는 fusion head다. 동영상 플레이어로 여는 파일이 아니다.

## 농연 열화상 핵심 결과

SmokeBasement run5 중앙 12초, 10fps, 120프레임. 학습은 같은 지하실의 다른 주행 run3에서 추출한 32장. 실제 소방 훈련 연기 환경의 열화상이며 구간별 연기 농도는 미측정이다.

| 파일 | 내용·처리 방식 |
|---|---|
| metrics.json (원본 프로젝트 참조: `outputs/smoke/metrics.json`; 선별본에는 미포함) | 이 폴더 실험의 실행 환경 및 방식별 정량 측정 기록. 최종 비교에는 smoke/thermal 기록 사용. |
| smoke_heldout_adapted.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_adapted.mp4`; 선별본에는 미포함) | 해당 데이터의 fusion-head 파인튜닝 TEED + structural 후처리. 학습의 추가 효과 비교. |
| smoke_heldout_baseline.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_baseline.mp4`; 선별본에는 미포함) | 기존 TEED 처리. CLAHE 2.0, 임계값 0.75, 3픽셀 팽창, 배경 0.24. |
| smoke_heldout_edges_only.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_edges_only.mp4`; 선별본에는 미포함) | structural 마스크만 검은 배경에 초록 선으로 렌더링. |
| smoke_heldout_frame_0000.png (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| smoke_heldout_frame_0060.png (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_frame_0060.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| smoke_heldout_frame_0119.png (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_frame_0119.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| smoke_heldout_masks.npz (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_masks.npz`; 선별본에는 미포함) | 파일명에 해당하는 입력의 5개 처리 방식별 무손실 boolean 엣지 마스크. 위 배열 설명 참조. |
| smoke_heldout_original.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_original.mp4`; 선별본에는 미포함) | 원본 비교 영상. 엣지 없이 리사이즈·재인코딩. |
| smoke_heldout_pidinet.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_pidinet.mp4`; 선별본에는 미포함) | 공식 PiDiNet table5 + hysteresis 0.20/0.40 + skeletonization, 배경 0.55. 대체 모델 비교. |
| smoke_heldout_structural.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_structural.mp4`; 선별본에는 미포함) | 구조 경계 개선. bilateral + TEED + hysteresis 0.60/0.85 + skeletonization, 배경 0.55. |
| smoke_heldout_thin_only.mp4 (원본 프로젝트 참조: `outputs/smoke/smoke_heldout_thin_only.mp4`; 선별본에는 미포함) | TEED 얇은 선과 밝은 배경 비교. 기존 입력/임계값, skeletonization, 배경 0.55. |
| teed_fusion_adapted.pth (원본 프로젝트 참조: `outputs/smoke/teed_fusion_adapted.pth`; 선별본에는 미포함) | 이 폴더 데이터로 fusion head만 적응시킨 TEED 가중치. 다른 폴더의 동명 가중치와 구분해서 사용. |
| training.json (원본 프로젝트 참조: `outputs/smoke/training.json`; 선별본에는 미포함) | 이 폴더 실험에 사용한 의사 라벨 학습 이력. 설정·loss·학습 시간·체크포인트 식별 정보. |

## IFSI 열화상 보조 결과

`heldout_*`는 학습에 쓰지 않은 IFSI Video 4 전체 625프레임(약 20.85초), `training_*`는 IFSI Video 8 전체 661프레임(약 22.06초). 학습에는 앞 4초를 제외한 32장을 사용했다. 두 영상은 약 29.97fps다. training 영상은 일반화 근거로 쓰지 않는다. 원본 카메라 숫자/로고는 배경에 남으며 기존 OSD 영역은 엣지에서 제외했다. 이 데이터의 구간별 농연 강도는 확인하지 못했다.

| 파일 | 내용·처리 방식 |
|---|---|
| clip_02815_frame_000068_frame_0000.png (원본 프로젝트 참조: `outputs/thermal/clip_02815_frame_000068_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. 입력은 가시광 RGB 02815의 68번 프레임으로, thermal 폴더에 있어도 열화상이 아니다. |
| clip_02815_frame_000068_masks.npz (원본 프로젝트 참조: `outputs/thermal/clip_02815_frame_000068_masks.npz`; 선별본에는 미포함) | 파일명에 해당하는 입력의 5개 처리 방식별 무손실 boolean 엣지 마스크. 위 배열 설명 참조. 입력은 가시광 RGB 02815의 68번 프레임으로, thermal 폴더에 있어도 열화상이 아니다. |
| conversion_benchmark.json (원본 프로젝트 참조: `outputs/thermal/conversion_benchmark.json`; 선별본에는 미포함) | IFSI Video 4 프레임 0/150/450에서 PiDiNet 공식 CNN 변환 전후 출력 오차와 CPU 추론 시간 측정. 저장 영상은 변환 전 구현 사용. |
| heldout_adapted.mp4 (원본 프로젝트 참조: `outputs/thermal/heldout_adapted.mp4`; 선별본에는 미포함) | 해당 데이터의 fusion-head 파인튜닝 TEED + structural 후처리. 학습의 추가 효과 비교. IFSI Video 4 검증 영상 결과. |
| heldout_baseline.mp4 (원본 프로젝트 참조: `outputs/thermal/heldout_baseline.mp4`; 선별본에는 미포함) | 기존 TEED 처리. CLAHE 2.0, 임계값 0.75, 3픽셀 팽창, 배경 0.24. IFSI Video 4 검증 영상 결과. |
| heldout_frame_0000.png (원본 프로젝트 참조: `outputs/thermal/heldout_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. IFSI Video 4 검증 영상 결과. |
| heldout_frame_0312.png (원본 프로젝트 참조: `outputs/thermal/heldout_frame_0312.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. IFSI Video 4 검증 영상 결과. |
| heldout_frame_0624.png (원본 프로젝트 참조: `outputs/thermal/heldout_frame_0624.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. IFSI Video 4 검증 영상 결과. |
| heldout_masks.npz (원본 프로젝트 참조: `outputs/thermal/heldout_masks.npz`; 선별본에는 미포함) | 파일명에 해당하는 입력의 5개 처리 방식별 무손실 boolean 엣지 마스크. 위 배열 설명 참조. IFSI Video 4 검증 영상 결과. |
| heldout_original.mp4 (원본 프로젝트 참조: `outputs/thermal/heldout_original.mp4`; 선별본에는 미포함) | 원본 비교 영상. 엣지 없이 리사이즈·재인코딩. IFSI Video 4 검증 영상 결과. |
| heldout_pidinet.mp4 (원본 프로젝트 참조: `outputs/thermal/heldout_pidinet.mp4`; 선별본에는 미포함) | 공식 PiDiNet table5 + hysteresis 0.20/0.40 + skeletonization, 배경 0.55. 대체 모델 비교. IFSI Video 4 검증 영상 결과. |
| heldout_structural.mp4 (원본 프로젝트 참조: `outputs/thermal/heldout_structural.mp4`; 선별본에는 미포함) | 구조 경계 개선. bilateral + TEED + hysteresis 0.60/0.85 + skeletonization, 배경 0.55. IFSI Video 4 검증 영상 결과. |
| heldout_thin_only.mp4 (원본 프로젝트 참조: `outputs/thermal/heldout_thin_only.mp4`; 선별본에는 미포함) | TEED 얇은 선과 밝은 배경 비교. 기존 입력/임계값, skeletonization, 배경 0.55. IFSI Video 4 검증 영상 결과. |
| ifsi_video_4_frame_000454_frame_0000.png (원본 프로젝트 참조: `outputs/thermal/ifsi_video_4_frame_000454_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. 입력은 IFSI Video 4의 454번 프레임을 따로 추출한 보조 점검이다. |
| ifsi_video_4_frame_000454_masks.npz (원본 프로젝트 참조: `outputs/thermal/ifsi_video_4_frame_000454_masks.npz`; 선별본에는 미포함) | 파일명에 해당하는 입력의 5개 처리 방식별 무손실 boolean 엣지 마스크. 위 배열 설명 참조. 입력은 IFSI Video 4의 454번 프레임을 따로 추출한 보조 점검이다. |
| metrics.json (원본 프로젝트 참조: `outputs/thermal/metrics.json`; 선별본에는 미포함) | 이 폴더 실험의 실행 환경 및 방식별 정량 측정 기록. 최종 비교에는 smoke/thermal 기록 사용. |
| preview310_frame_0000.png (원본 프로젝트 참조: `outputs/thermal/preview310_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. 단일 입력 preview310의 보조 점검 결과. 정확한 원본 대응 이력이 최종 측정 기록에 없어 최종 판단 근거에서 제외한다. |
| preview310_masks.npz (원본 프로젝트 참조: `outputs/thermal/preview310_masks.npz`; 선별본에는 미포함) | 파일명에 해당하는 입력의 5개 처리 방식별 무손실 boolean 엣지 마스크. 위 배열 설명 참조. 단일 입력 preview310의 보조 점검 결과. 정확한 원본 대응 이력이 최종 측정 기록에 없어 최종 판단 근거에서 제외한다. |
| teed_fusion_adapted.pth (원본 프로젝트 참조: `outputs/thermal/teed_fusion_adapted.pth`; 선별본에는 미포함) | 이 폴더 데이터로 fusion head만 적응시킨 TEED 가중치. 다른 폴더의 동명 가중치와 구분해서 사용. |
| training.json (원본 프로젝트 참조: `outputs/thermal/training.json`; 선별본에는 미포함) | 이 폴더 실험에 사용한 의사 라벨 학습 이력. 설정·loss·학습 시간·체크포인트 식별 정보. |
| training_adapted.mp4 (원본 프로젝트 참조: `outputs/thermal/training_adapted.mp4`; 선별본에는 미포함) | 해당 데이터의 fusion-head 파인튜닝 TEED + structural 후처리. 학습의 추가 효과 비교. IFSI Video 8 학습 영상 결과. |
| training_baseline.mp4 (원본 프로젝트 참조: `outputs/thermal/training_baseline.mp4`; 선별본에는 미포함) | 기존 TEED 처리. CLAHE 2.0, 임계값 0.75, 3픽셀 팽창, 배경 0.24. IFSI Video 8 학습 영상 결과. |
| training_frame_0000.png (원본 프로젝트 참조: `outputs/thermal/training_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. IFSI Video 8 학습 영상 결과. |
| training_frame_0330.png (원본 프로젝트 참조: `outputs/thermal/training_frame_0330.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. IFSI Video 8 학습 영상 결과. |
| training_frame_0660.png (원본 프로젝트 참조: `outputs/thermal/training_frame_0660.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. IFSI Video 8 학습 영상 결과. |
| training_masks.npz (원본 프로젝트 참조: `outputs/thermal/training_masks.npz`; 선별본에는 미포함) | 파일명에 해당하는 입력의 5개 처리 방식별 무손실 boolean 엣지 마스크. 위 배열 설명 참조. IFSI Video 8 학습 영상 결과. |
| training_original.mp4 (원본 프로젝트 참조: `outputs/thermal/training_original.mp4`; 선별본에는 미포함) | 원본 비교 영상. 엣지 없이 리사이즈·재인코딩. IFSI Video 8 학습 영상 결과. |
| training_pidinet.mp4 (원본 프로젝트 참조: `outputs/thermal/training_pidinet.mp4`; 선별본에는 미포함) | 공식 PiDiNet table5 + hysteresis 0.20/0.40 + skeletonization, 배경 0.55. 대체 모델 비교. IFSI Video 8 학습 영상 결과. |
| training_structural.mp4 (원본 프로젝트 참조: `outputs/thermal/training_structural.mp4`; 선별본에는 미포함) | 구조 경계 개선. bilateral + TEED + hysteresis 0.60/0.85 + skeletonization, 배경 0.55. IFSI Video 8 학습 영상 결과. |
| training_thin_only.mp4 (원본 프로젝트 참조: `outputs/thermal/training_thin_only.mp4`; 선별본에는 미포함) | TEED 얇은 선과 밝은 배경 비교. 기존 입력/임계값, skeletonization, 배경 0.55. IFSI Video 8 학습 영상 결과. |

## 초기 RGB 및 단일 프레임 보조 결과

`outputs/` 바로 아래 자료는 초기 준비 실험이다. `sample_3_training_*`는 RGB sample_3 학습 영상, `sample_4_heldout_*`는 RGB sample_4 검증 영상이다. 실제 농연 열화상 성능의 핵심 근거로 사용하지 않는다. 현재 재현 스크립트의 `--suite rgb` 출력 위치는 `outputs/rgb/`이며 기존 초기 결과의 위치와 다르다.

| 파일 | 내용·처리 방식 |
|---|---|
| clip_02815_frame_000068_frame_0000.png (원본 프로젝트 참조: `outputs/clip_02815_frame_000068_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. 입력은 가시광 RGB 02815의 68번 프레임으로, thermal 폴더에 있어도 열화상이 아니다. |
| ifsi_video_4_frame_000454_frame_0000.png (원본 프로젝트 참조: `outputs/ifsi_video_4_frame_000454_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. 입력은 IFSI Video 4의 454번 프레임을 따로 추출한 보조 점검이다. |
| metrics.json (원본 프로젝트 참조: `outputs/metrics.json`; 선별본에는 미포함) | 이 폴더 실험의 실행 환경 및 방식별 정량 측정 기록. 최종 비교에는 smoke/thermal 기록 사용. |
| sample_3_training_adapted.mp4 (원본 프로젝트 참조: `outputs/sample_3_training_adapted.mp4`; 선별본에는 미포함) | 해당 데이터의 fusion-head 파인튜닝 TEED + structural 후처리. 학습의 추가 효과 비교. |
| sample_3_training_baseline.mp4 (원본 프로젝트 참조: `outputs/sample_3_training_baseline.mp4`; 선별본에는 미포함) | 기존 TEED 처리. CLAHE 2.0, 임계값 0.75, 3픽셀 팽창, 배경 0.24. |
| sample_3_training_frame_0000.png (원본 프로젝트 참조: `outputs/sample_3_training_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| sample_3_training_frame_0122.png (원본 프로젝트 참조: `outputs/sample_3_training_frame_0122.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| sample_3_training_frame_0244.png (원본 프로젝트 참조: `outputs/sample_3_training_frame_0244.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| sample_3_training_original.mp4 (원본 프로젝트 참조: `outputs/sample_3_training_original.mp4`; 선별본에는 미포함) | 원본 비교 영상. 엣지 없이 리사이즈·재인코딩. |
| sample_3_training_pidinet.mp4 (원본 프로젝트 참조: `outputs/sample_3_training_pidinet.mp4`; 선별본에는 미포함) | 공식 PiDiNet table5 + hysteresis 0.20/0.40 + skeletonization, 배경 0.55. 대체 모델 비교. |
| sample_3_training_structural.mp4 (원본 프로젝트 참조: `outputs/sample_3_training_structural.mp4`; 선별본에는 미포함) | 구조 경계 개선. bilateral + TEED + hysteresis 0.60/0.85 + skeletonization, 배경 0.55. |
| sample_3_training_thin_only.mp4 (원본 프로젝트 참조: `outputs/sample_3_training_thin_only.mp4`; 선별본에는 미포함) | TEED 얇은 선과 밝은 배경 비교. 기존 입력/임계값, skeletonization, 배경 0.55. |
| sample_4_heldout_adapted.mp4 (원본 프로젝트 참조: `outputs/sample_4_heldout_adapted.mp4`; 선별본에는 미포함) | 해당 데이터의 fusion-head 파인튜닝 TEED + structural 후처리. 학습의 추가 효과 비교. |
| sample_4_heldout_baseline.mp4 (원본 프로젝트 참조: `outputs/sample_4_heldout_baseline.mp4`; 선별본에는 미포함) | 기존 TEED 처리. CLAHE 2.0, 임계값 0.75, 3픽셀 팽창, 배경 0.24. |
| sample_4_heldout_frame_0000.png (원본 프로젝트 참조: `outputs/sample_4_heldout_frame_0000.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| sample_4_heldout_frame_0153.png (원본 프로젝트 참조: `outputs/sample_4_heldout_frame_0153.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| sample_4_heldout_frame_0305.png (원본 프로젝트 참조: `outputs/sample_4_heldout_frame_0305.png`; 선별본에는 미포함) | 해당 입력의 대표 프레임 6방식 비교 이미지. 위 2행×3열 순서로 읽음. |
| sample_4_heldout_original.mp4 (원본 프로젝트 참조: `outputs/sample_4_heldout_original.mp4`; 선별본에는 미포함) | 원본 비교 영상. 엣지 없이 리사이즈·재인코딩. |
| sample_4_heldout_pidinet.mp4 (원본 프로젝트 참조: `outputs/sample_4_heldout_pidinet.mp4`; 선별본에는 미포함) | 공식 PiDiNet table5 + hysteresis 0.20/0.40 + skeletonization, 배경 0.55. 대체 모델 비교. |
| sample_4_heldout_structural.mp4 (원본 프로젝트 참조: `outputs/sample_4_heldout_structural.mp4`; 선별본에는 미포함) | 구조 경계 개선. bilateral + TEED + hysteresis 0.60/0.85 + skeletonization, 배경 0.55. |
| sample_4_heldout_thin_only.mp4 (원본 프로젝트 참조: `outputs/sample_4_heldout_thin_only.mp4`; 선별본에는 미포함) | TEED 얇은 선과 밝은 배경 비교. 기존 입력/임계값, skeletonization, 배경 0.55. |
| training.json (원본 프로젝트 참조: `outputs/training.json`; 선별본에는 미포함) | 이 폴더 실험에 사용한 의사 라벨 학습 이력. 설정·loss·학습 시간·체크포인트 식별 정보. |

## 보고서·요약·추적 자료

| 파일 | 내용과 용도 |
|---|---|
| [REPORT.md](REPORT.md) | 최종 판단, 농연/IFSI 수치, 실제 파인튜닝 조건과 한계. 결과 해석의 본문. |
| [MODEL_RESEARCH.md](MODEL_RESEARCH.md) | TEED·PiDiNet·LED-Net 및 thermal/depth 대안 조사, 출처, 실제 실행 여부, 다음 학습 제안. |
| [results_summary.json](results_summary.json) | 최종 보고서에 사용한 핵심 측정값과 학습 이력을 기계 판독용으로 요약. |
| artifact_verification.json (원본 프로젝트 참조: `artifact_verification.json`; 선별본에는 미포함) | 농연 7개 + IFSI 12개 영상의 전체 디코딩·프레임 수·fps 검증. 초기 RGB 영상까지 검증한 목록은 아님. |
| [review_frames/smoke_comparison.png](smoke-comparison.png) | 농연 출력 60번 프레임의 6칸 비교 이미지 사본. Git에 포함한 대표 이미지. |
| review_frames/ifsi_comparison.png (원본 프로젝트 참조: `review_frames/ifsi_comparison.png`; 선별본에는 미포함) | IFSI 검증 출력 312번 프레임의 6칸 비교 이미지 사본. Git에 포함한 대표 이미지. |
| checkpoints/teed_fusion_adapted.pth (원본 프로젝트 참조: `checkpoints/teed_fusion_adapted.pth`; 선별본에는 미포함) | 초기 RGB 준비 실험의 적응 가중치. 농연/IFSI 최종 가중치는 각 outputs 하위 폴더에 있음. |
| data/smoke_manifest.json (원본 프로젝트 참조: `data/smoke_manifest.json`; 선별본에는 미포함) | 농연 학습/검증 입력의 원본 파일명·타임스탬프·hash·선택 구간. 출처 및 재표본화 추적용. |
| [README.md](README.md) | 환경 설치와 실행·재현 명령. |
| [PLAN.md](PLAN.md) | 실험 목적, 비교 설계와 검증 계획. 실제 실행 결과는 REPORT.md 참조. |

`data/`의 원본 영상·PNG·현장 사진과 다운로드 캐시는 처리 결과가 아닌 입력/수집 자료다. 현장 사진은 열화상과 동기화된 RGB 전후 비교 영상이 아니다. `third_party/`는 외부 모델 코드이며 `data/viewer.pid`는 로컬 재생 서버 관리용 프로세스 번호다.
