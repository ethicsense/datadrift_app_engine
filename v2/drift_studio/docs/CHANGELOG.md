# CHANGELOG

모든 주요 변경사항은 이 파일에 기록합니다. (간략)

## v1.0 (2026-01-06)

### Added

- 서버(UI 포함) + CLI에서 **Plan/Runtime 기반 EDA → Drift → Report** 실행
- 리포트(HTML, PDF 옵션), Docker compose로 `api + web` 기본 구동

### Notes

- workspace/git/dvc 기능은 제거 방향(일부 명칭만 호환용 유지)
- 모달리티 플러그인(wrapper) 및 Docker “선택 설치” 빌드 구성은 추가 정리 필요
- 모달리티별 분석 지원(요약)
  - vision: EDA/Drift 지원(플러그인 우선, 일부 내장 fallback)
  - text: EDA 일부 지원(플러그인/내장), Drift는 플러그인 필요
  - audio: EDA/Drift는 플러그인 필요
  - timeseries: EDA/Drift 최소 내장 지원(플러그인 우선)


