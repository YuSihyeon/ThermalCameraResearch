# 정리본 검증 기록

검증일: 2026-09-16. 이 아카이브는 기존 실험 자료를 정리하고 수치·파일·출처를 다시 대조한 기록이다. 새 카메라 촬영, 모델 재학습, 실측 거리·온도 검증, Jetson 실장 시험을 수행했다고 주장하지 않는다.

- 입력 52장의 해상도·자료형을 확인하고, 두 PLY의 모든 점을 입력 픽셀과 대조했다. `z = intensity / 255 × 3000`과 일치하므로 밝기 기반 height field임을 확인했다. [원자료 감사](docs/evidence/local-audit.json)
- 입력·코드·출력을 [pipeline-map.json](docs/evidence/pipeline-map.json)으로 연결했다. 초기 코드 5개는 구문 검사를 통과했으며, 깨진 `.save` 사본 두 개는 역사 자료로 보존하고 실행 대상에서 분리했다.
- 로컬 영상 7개 전체를 디코드했고 공개 후보 6개와 로컬 전용 현장 촬영 1개를 구분했다. [선정과 검토 범위](docs/evidence/publication-review.json)
- 공개 미리보기는 다시 전체 프레임을 디코드하고, Markdown 상대 링크·JSON 구문·Python 구문·알려진 credential 형식을 검사했다. [검증 결과](docs/evidence/archive-validation.json)
- 공개 파일의 크기와 SHA-256은 [manifest](docs/evidence/archive-manifest.json)에 기록한다. 검증 JSON·manifest 자체·이 문서는 자기참조 hash를 피하기 위해 목록에서 제외한다. 원본 절대경로와 복사 이력은 로컬 전용 `source-map.json`에 있다.

영상의 대표 프레임 검토는 모든 프레임을 사람이 전수 검토했다는 뜻이 아니다. 공개 데이터의 출처와 변환은 [SOURCE_CREDITS](docs/SOURCE_CREDITS.md)에 명시했다. 영상 재생 페이지는 VP9 WebM과 H.264 MP4를 함께 제공하며, 브라우저에서 재생이 제한되면 MP4 파일로 확인할 수 있다.
