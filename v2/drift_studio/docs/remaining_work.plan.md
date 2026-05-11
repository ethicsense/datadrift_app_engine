# drift_studio 남은 작업 To-Do (2026-01-06)

## 범위

- 이 문서는 현재 워크스페이스에서 “남은 작업/리스크”만 정리합니다.
- 플랜 원문(`.cursor/plans/*.plan.md`)은 수정하지 않습니다.

## 현황 요약(현재 메인스트림)

- **실행 엔진**: v1 `ddoc`를 `packages/ddoc`로 vendoring하여 같은 레포에서 설치/수정 가능
- **실행 방식**: drift_studio 런타임이 `PythonExecutor`로 ddoc 플러그인(pluggy entrypoints)을 직접 호출
- **비동기/진행률/WS**: 단순화를 위해 제거됨(필요 시 재도입)
- **모달리티 선택 설치**: Docker build-arg `DDOC_PLUGINS`로 설치 플러그인 선택 가능

---

## To-Do: 5) 모달리티 플러그인 실행/배포(정리)

### 5.1 패키징/의존성 전략(현 상태)

- **vendored 구조**
  - `packages/ddoc/` (ddoc)
  - `packages/ddoc/plugins/ddoc-plugin-{vision,text,audio,timeseries}/` (플러그인)
- **설치 방법(예시)**
  - Python(로컬): `pip install -e packages/ddoc -e packages/ddoc/plugins/ddoc-plugin-<modality>`
  - Docker: `DDOC_PLUGINS="text,audio,timeseries"` 기본, 필요 시 `vision,yolo,vis` 추가

### 5.2 런타임 동작(현 상태)

- `RuntimeRunner` → `PythonExecutor` → `ddoc.core.analysis_facade` → pluggy provider(`ddoc_text`, `ddoc_audio`, ...)
- 플러그인이 없거나 결과가 없으면 **명확히 실패(ValueError/HTTP 400)** 합니다.
- 내장 EDA/Drift fallback은 제거되어 **플러그인 필수**입니다.

---

## To-Do: 8) Docker 고객사별 배포 최적화(정리)

### 8.1 API 이미지에서 모달리티 선택 설치 지원(현 상태)

- `apps/api/Dockerfile`에 `ARG DDOC_PLUGINS`로 분기 설치 지원
- `docker-compose.yml`에서 `build.args.DDOC_PLUGINS`로 기본값 지정

### 8.2 런타임 플러그인 탐지/상태 노출(선택)

- 목표: 현재 어떤 모달리티 플러그인이 활성인지 API/UI에서 즉시 확인
- 예: `driftstudio_runtime.registry.detect_installed_modalities()`를 `/health/plugins` 같은 엔드포인트로 노출


