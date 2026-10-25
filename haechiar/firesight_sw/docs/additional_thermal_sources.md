# 추가 열화상 영상 확인

2026-10-08에 Fire360 공개 Box 폴더를 브라우저에서 직접 열어 indoor 목록 7페이지를 확인했다. IFSI Video 5·6·7·9·10을 추가로 확보했고, 영상의 회색조 열화상 장면과 우측 온도 색상 눈금을 직접 확인했다. 기존 IFSI4·8을 포함하면 현재 확보한 열화상 표시 영상은 7개다.

## 새로 확보한 원본

| 영상 | 전체 길이 | 프레임 수 | 공개 원본 |
| --- | ---: | ---: | --- |
| IFSI5 | 55.689초 | 1,669 | [Box](https://uofi.app.box.com/v/fire360dataset/file/1864227340588) |
| IFSI6 | 71.838초 | 2,153 | [Box](https://uofi.app.box.com/v/fire360dataset/file/1864227400588) |
| IFSI7 | 116.383초 | 3,488 | [Box](https://uofi.app.box.com/v/fire360dataset/file/1864227371788) |
| IFSI9 | 53.720초 | 1,610 | [Box](https://uofi.app.box.com/v/fire360dataset/file/1864227328588) |
| IFSI10 | 62.329초 | 1,868 | [Box](https://uofi.app.box.com/v/fire360dataset/file/1864227285388) |

모두 3840×2160, 29.97003fps MP4이며 합계 10,788프레임, 359.960초다. 파일은 저장소 루트 기준 `../data/fire360/raw/IFSI Video N (2).mp4`에 보관했다. SHA-256, 출처 URL, 프레임 수와 해상도는 로컬 생성 기록 `experiments/teed_ui/additional_thermal_sources/downloaded_thermal_sources.json`에 저장했다. 이 JSON과 렌더 결과물은 GitHub 배포 대상이 아니다.

[공개 indoor 폴더](https://uofi.app.box.com/v/fire360dataset/folder/321513258241)에는 IFSI4–10 각각 접미사 없는 파일, `(1)`, `(2)` 변형이 있다. 표시된 파일 크기는 같지만 `(2)`만 다운로드했으므로 변형 사이의 바이너리 동일성은 단정하지 않는다. 이번 조사로 전체 공개 폴더의 모든 열화상 영상을 빠짐없이 분류했다고 주장하지 않는다.

## 적용 조건

새 5개 영상에도 앞서 사용한 TEED 320×256 / 임계값 0.75 / CLAHE 2, 확률 안정화 및 윤곽 추적을 동일하게 적용한다. 모델 가중치나 알고리즘은 변경하지 않는다. 각 파일의 0번 프레임부터 마지막 프레임까지 처리하며, 전체 길이를 유지한 720×406 무음 WebM으로 저장한다. 실제 HUD 높이 405px에 인코딩을 위한 검은색 1행만 덧붙인다.

영상에 내장된 우측 온도 눈금과 좌측 하단 로고의 위치가 IFSI4와 같아 IFSI4 OSD 제외 영역을 명시적으로 재사용한다. 온도 숫자를 읽거나 영상 밝기를 섭씨로 환산하지 않는다. 출력의 `TEMP N/A`는 실제 온도맵이 없다는 뜻이다.

전체 렌더와 QA 결과는 `experiments/teed_ui/contour_stability/thermal_additional/`에 저장한다. 출력 전체 디코딩으로 프레임 수·fps·해상도를 확인하고 시작·25%·50%·75%·끝의 장면을 시각 점검한다.

## 다른 로컬 데이터와 속도 측정

SmokeBasement의 짧은 MP4는 미리보기다. 로컬에는 별도의 전체 원본 아카이브와 프레임 시퀀스도 있으므로, 미리보기 길이를 데이터셋 전체 길이로 취급하지 않는다. 이번 추가 적용 대상은 위 Fire360 열화상 5개다.

실시간 가능 여부는 별도 순차 CPU 벤치마크 및 운영 CLI로 확인했다. 최종 측정과 제한은 [열화상 실시간 처리 실측](thermal_realtime_results.md)을 따른다. 병렬 영상 렌더의 처리 속도는 실시간 성능 비교에 사용하지 않는다.
