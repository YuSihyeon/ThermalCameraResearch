# 대체 모델 검토 (2026-09-06)

## 결론의 범위

TEED보다 정확도·속도·용량·훈련시간·열화상 일반화가 **모두** 좋은 모델은
확인하지 못했다. 다른 데이터셋/장비의 수치를 직접 순위로 합치지 않았다.
아래 논문 성능은 저자의 벤치마크이며 FireSight 현장 장애물 인지 점수가 아니다.

| 모델 | 확인한 근거 | 이번 판단 / 실행 |
|---|---|---|
| TEED (ICCVW 2023) | 약 58K 파라미터, BIPED 학습의 경량 엣지 모델 | 이미 매우 작다. 후처리 개선과 작업에 맞는 경계 라벨 학습이 우선. 실제 실행 |
| PiDiNet (ICCV 2021, 후속 TPAMI) | 공식 table5 모델 BSDS ODS .807, OIS .823; 공개 체크포인트/추론 코드 | 최신 모델은 아니다. 다른 학습 데이터와 표현 방식의 비교군. 실제 실행 후 실측으로 판단. 논문의 2080 Ti FPS를 이 PC/Nano 수치로 쓰지 않음 |
| LED-Net (PRL 2025) | 50K 파라미터, UDED ODS .839/OIS .855/AP .830, 얇은 경계용 loss | TEED보다 작고 최근인 유력 후보. 이번 조사에서 저자 공식 실행 코드와 사전학습 가중치를 확보하지 못해 실행/우월성 검증 못함. 동명 저조도 LEDNet과 혼동 금지 |
| BIOED | 저자 GitHub에 모델 코드 공개 | 확인한 저장소에 사전학습 체크포인트와 충분한 재현 안내가 없어 미실행 |
| MonoTher-Depth (2025) | thermal metric depth, RGB prior의 confidence-aware distillation, MS2/Vivid 평가와 공개 가중치 | 열화상 깊이 교사 후보. TEED 대체 엣지 모델은 아니며 크기·속도 우월 근거 없음. 이번 CPU 엣지 실험에서는 미실행 |
| AnyThermal (ICRA 2026) | thermal 표현 학습, frozen DINOv2+MiDaS depth head, MS2 depth checkpoint | 최신 열화상 교사 후보. 공식 환경은 CUDA/PyTorch3D 의존성이 있음. TEED보다 경량이라는 근거 없음. 미실행 |
| Video Depth Anything (CVPR 2025) | Small 28.4M, 긴 영상 시간 일관성. 공개 A100 FP16 수치는 32프레임 518 입력 조건 | TEED보다 파라미터가 수백 배 많음. 열화상 검증 없이 장애물 거리로 사용하면 안 됨. 미실행 |
| Depth Anything 3 (2025) | single/multi-view geometry와 streaming 코드, 모델별 공개 라이선스 | 최신 범용 기하 후보지만 열화상 적합성과 Jetson 비용을 따로 확인해야 함. 이번 목적에서 전면 교체 근거 없음. 미실행 |

## 원문

- TEED 공식 코드: https://github.com/xavysp/TEED
- TEED 논문: https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Soria_Tiny_and_Efficient_Model_for_the_Edge_Detection_Generalization_ICCVW_2023_paper.pdf
- PiDiNet 공식 코드/성능표: https://github.com/hellozhuo/pidinet
- LED-Net 논문: https://www.sciencedirect.com/science/article/pii/S016786552400312X
- BIOED 저자 코드: https://github.com/flypeople8/bioed
- MonoTher-Depth 공식 코드: https://github.com/ZuoJiaxing/monother_depth
- AnyThermal 공식 코드: https://github.com/castacks/AnyThermal
- Video Depth Anything 공식 코드: https://github.com/DepthAnything/Video-Depth-Anything
- Depth Anything 3 공식 코드: https://github.com/ByteDance-Seed/Depth-Anything-3
- Fire360 데이터/논문: https://fire360bench.github.io/ ; https://arxiv.org/abs/2506.02167

## 권장 실제 파인튜닝 설계

추가 확보 데이터: [SmokeBasement](https://zenodo.org/records/15173055)는 소방 훈련용
연기로 채운 건설현장 지하실에서 열화상과 LiDAR를 수집한 자료다. 이번 핵심 실험은
이 데이터의 실제 thermal PNG로 수행한다. [FIReStereo](https://firestereo.github.io/)도
연기 조건의 stereo thermal/depth 자료를 제공하지만 실외 비중이 높아 별도 후속
검증 후보로 남겼다. 실제 FireSight 센서/착용자 시점 검증은 별도로 필요하다.

코드상 주의점: 현재 TEED 마지막 smish 출력에 sigmoid를 취하면 바탕값이 약
0.438까지밖에 내려가지 않았다. 따라서 출력은 교정된 장애물 확률이 아니며,
일반적인 0.5 threshold를 무조건 적용하지 않는다. supervised 학습을 도입할 때는
기존 TEED 손실/출력 구조를 존중하거나 선형 logit head를 별도로 학습하고,
head 변경과 threshold calibration을 함께 검증해야 한다.

범용 '모든 명암 경계'를 그대로 재학습하면 잔가지, 벽 무늬, 열 변화도 계속
정답이 된다. 학습 타깃을 **회피 판단에 필요한 기하학적 경계**로 바꿔야 한다.

1. Boson 실제 입력을 수집하고 원본 Y16/표시용 8-bit, 시간, 노출/AGC 모드,
   연기 상태를 함께 보존한다. RGB 공개 영상의 성능을 thermal 성능으로 바꾸어
   말하지 않는다. 동일 현장/촬영 세션의 인접 프레임을 train/test에 섞지 않는다.
2. 시작 규모 제안: 20–30개 독립 클립에서 500–1,000장을 골라 라벨링한다.
   이는 실험 설계 제안이며 이번에 확보/학습한 데이터 수가 아니다.
   라벨은 사람/가구/장비 이름 대신 장애물 외곽, 기둥/벽 경계, 문 개구부,
   바닥 단차, 늘어진 호스/케이블로 한다. 연기로 실제 보이지 않는 영역은
   background 음성이 아니라 ignore로 둔다. OSD도 ignore 처리한다.
3. 바닥 질감·벽 무늬·반사·연기 경계는 hard negative로 넣되, 짧은 선이라는
   이유만으로 제거하지 않는다. 작은 돌출물과 호스가 사라질 수 있기 때문이다.
4. TEED backbone 동결 후 fusion/head부터 학습(lr 1e-4 시작), 이후 검증이 좋아질
   때만 마지막 feature block을 lr 1e-5로 푼다. 기존 범용 경계 데이터도 일부
   섞어 기본 외곽선 능력의 망각을 제한한다. epoch 수는 검증 성능으로 결정한다.
5. 손실 시작안: class-balanced boundary BCE + 0.5*boundary Dice +
   0.1*temporal consistency. 가까운 얇은 장애물의 false negative를 별도 가중한다.
   temporal 항은 flow 정합 신뢰 영역에서만 쓰고 신규 출현/가려짐/장면 전환은
   제외한다. 계수는 고정 정답이 아니라 검증 세트에서 조정할 하이퍼파라미터다.
6. depth는 우선 offline teacher로만 사용: thermal 모델의 depth discontinuity 중
   신뢰도 높은 경계만 보조 라벨로 제안하고 사람이 교정한다. 학습 후 배포는
   TEED 단독으로 유지하면 depth 추론 비용을 피할 수 있다. 소방 장면에 맞는
   깊이 검증이 먼저이며, 단안 상대 깊이를 미터 거리로 해석하지 않는다.
7. 테스트는 boundary recall/precision (거리 허용 오차 명시), 얇은 장애물 누락,
   문/통행 공간 경계 보존, motion-aligned temporal stability, 사용자 회피 판단
   정확도/반응시간을 함께 본다. 화면 점유율 감소만으로 모델을 채택하지 않는다.
8. 최종 속도/메모리/전력은 실제 Nano 또는 Orin 중 목표 보드를 확정한 뒤 측정.
   CPU 시간, 논문 GPU FPS, TensorRT FPS는 별개다. INT8 전환 시 낮은 대비의
   장애물 recall을 다시 검사한다.
