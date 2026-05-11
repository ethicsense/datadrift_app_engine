# One-Command Deployment Guide

이 문서는 아래 목표를 가장 간단하게 달성하기 위한 배포 가이드입니다.

- 저장소 `git clone`
- 원천 데이터 `data/` 삽입
- `docker compose` 한 줄로 분석 + 시각화 API + 웹 앱 실행

## 1) 사전 조건

- Docker Desktop (또는 Docker Engine + Compose v2)
- 프로젝트 루트에 `data/` 디렉터리 존재
- `data/` 하위에 최소 1개 이상의 `ranking_summary.json` 존재
- (선택) 로컬 파이썬 실행 시 표준 가상환경: `silhouette_venv`

```bash
python -m venv silhouette_venv
source silhouette_venv/bin/activate
pip install -r requirements.txt
```

## 2) 실행 방법 (E2E 한 줄)

```bash
docker compose up --build
```

최초 실행 시:
- `qdrant` 시작
- `analytics-pipeline`가 데이터 preflight 확인 후 분석 산출물 생성
- 분석 완료 성공 시 `visualization-api` 시작
- API 헬스체크 통과 시 `visualization-web` 시작

## 3) 접속 주소

- Web: `http://localhost:38173`
- API Health: `http://localhost:38000/health`
- Qdrant: `http://localhost:38633`

## 4) 포트 커스터마이즈

`.env.example`를 `.env`로 복사 후 수정합니다.

```bash
cp .env.example .env
```

지원 변수:
- `SILHOUETTE_API_PORT` (기본 `38000`)
- `SILHOUETTE_WEB_PORT` (기본 `38173`)
- `SILHOUETTE_QDRANT_PORT` (기본 `38633`)

참고:
- 웹은 Nginx에서 `/api/*`를 내부 `visualization-api:8000`으로 프록시합니다.
- 기본 빌드에서는 프론트가 별도 API 포트를 직접 호출하지 않아도 됩니다.

## 5) 운영 시 자주 쓰는 명령

```bash
# 백그라운드 실행
docker compose up --build -d

# 로그 확인
docker compose logs -f analytics-pipeline visualization-api visualization-web

# 종료
docker compose down
```

## 6) 실패 시 점검 포인트

- `analytics-pipeline` 실패:
  - `data/` 경로와 `ranking_summary.json` 존재 여부 확인
  - 로그: `docker compose logs analytics-pipeline`
  - Qdrant가 순간적으로 불안정하면 임베딩 업서트만 실패(`status=error`)하고 파이프라인 산출물 저장은 계속 진행됩니다.
- API가 안 뜸:
  - 파이프라인 완료 여부 확인 (`service_completed_successfully` 의존)
  - `output/analytics` 산출물 생성 여부 확인
- Web가 안 뜸:
  - API 헬스체크 상태 확인 (`/health`)
