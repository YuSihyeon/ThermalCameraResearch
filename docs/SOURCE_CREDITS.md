# 출처와 변환 기록

## 공개 데이터

SmokeBasement - Stereo thermal camera and Hesai lidar dataset. Jianzhu Huai, Zenodo, 2025. DOI [10.5281/zenodo.15173055](https://doi.org/10.5281/zenodo.15173055). [공식 레코드](https://zenodo.org/records/15173055)의 라이선스는 CC BY 4.0이다. 데이터 자체의 저작권과 라이선스를 유지한다. 로컬 원 실험 문서는 Dezhong Chen의 수집 기여도 표기한다. 2026-09-16에 확인한 API 메타데이터에는 creator로 Jianzhu Huai가 표시된다. [확인 기록](evidence/external-source-check.json)

이 아카이브에 포함된 SmokeBasement 기반 결과는 선택 구간의 resize, 시간 재표본화, 영상 encoding, TEED/비교 모델 처리, HUD overlay, 또는 합성 거리 폭 표시를 거친 파생물이다. 원 저자가 이 처리나 해석을 보증하는 것은 아니다. 합성 거리는 데이터셋에서 제공한 LiDAR의 측정값이 아니다.

Fire360/IFSI는 원 실험의 별도 보조 입력이다. 그 자료에 SmokeBasement의 CC BY 4.0을 적용하지 않는다. 이번 선별본에는 수치 기록과 처리 설명을 남겼고 Fire360 원본 동영상 전체는 복제하지 않았다.

## 장비·온도 해석

[FLIR OEM Boson Radiometry 공식 안내](https://flir.custhelp.com/app/answers/detail/a_id/4148/~/flir-oem---boson-radiometry-%28absolute-temperature-measurement%29)를 2026-09-16에 확인했다. 일반형과 radiometric variant가 구분된다는 점을 자료 해석에 사용했다. 실제 사용 장치의 part number를 확인한 기록은 현재 아카이브에 없으므로 제품 variant를 단정하지 않는다.

## 자체 자료와 코드

초기 카메라 PNG/PLY, 장비와 모니터를 촬영한 영상, Boson HUD 파일, 기존 자체 실험의 보고서/수치/선별 코드를 원본에서 복사했다. 원본 전체의 법적 저작권이나 팀 프로젝트의 라이선스를 이 문서가 새로 부여하지 않는다. 특정 오픈소스 라이선스를 임의로 붙이지 않았으며 외부 모델 전체 소스/가중치는 포함하지 않았다.

`docs/evidence`의 기존 보고서는 선별 보존본이다. 로컬 절대경로를 분리하고, 보존된 대상의 상대 링크를 새 폴더 구조에 맞춰 조정하거나 선별 제외 경로를 일반 텍스트로 남길 수 있다. 수치와 결론은 원 자료를 기준으로 유지한다. 무가공 복사와 가공된 복사, 원본/보존본 hash를 로컬 전용 `source-map.json`에서 구분한다.

영상 미리보기 JPG는 로컬 동영상을 특정 frame에서 디코드한 뒤 보기 좋게 축소하거나 나란히 배치한 파생 자료다. 원본이 아니며 픽셀 정량 분석에는 원본 PNG/PLY/CSV를 사용한다. `media`는 로컬 원본 보존 공간으로 Git에서 제외한다. 현장 촬영 영상은 주변 작업 화면이 포함되어 있어 코드·정량 결과와 분리해 관리한다.
