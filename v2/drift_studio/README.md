# Drift Studio v2

Drift Studio는 **ZIP(+ `ddoc.yaml`) 규약 기반 데이터셋**에 대해 **EDA → Drift → Report(HTML/PDF)** 파이프라인을 제공하는 시스템입니다.

- **UI**: `apps/web` (Vite/React)
- **API**: `apps/api` (FastAPI)
- **Runtime/Spec**: `packages/driftstudio_runtime`, `packages/driftstudio_spec`
- **Analysis Engine(Plugin 기반)**: `packages/ddoc` (+ `packages/ddoc/plugins/*`)

## 빠른 시작(Docker)

```bash
docker compose up --build
```

- Web: `http://localhost:3000`
- API: `http://localhost:18765`

### 플러그인 선택 설치(모달리티)

API 이미지는 `docker-compose.yml`의 build-arg `DDOC_PLUGINS`로 모달리티 플러그인을 선택 설치합니다.

```yaml
DDOC_PLUGINS: "text,timeseries,audio-wave,audio-midi,vision-image,vision-video"
```

## 로컬 개발 실행

### Python 의존성(네이티브) 한 번에 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[native-runtime,native-plugins-default]" \
  -e packages/driftstudio_spec \
  -e packages/driftstudio_runtime \
  -e packages/driftstudio_reports \
  -e packages/ddoc \
  -e packages/ddoc/plugins/ddoc-plugin-text \
  -e packages/ddoc/plugins/ddoc-plugin-timeseries \
  -e packages/ddoc/plugins/ddoc-plugin-audio-wave \
  -e packages/ddoc/plugins/ddoc-plugin-audio-midi \
  -e apps/api
```

> 참고: WeasyPrint PDF 생성을 위해 OS 라이브러리(cairo/pango/gdk-pixbuf)도 필요합니다.

### API

```bash
cd apps/api
uvicorn app.main:app --reload --port 18765
```

### Web

```bash
cd apps/web
npm install
npm run dev -- --port 3000
```

## 데이터셋 입력 규약(필독)

- 업로드 입력은 **`.zip`**이며, ZIP 내부(압축 해제 후 루트)에 **`ddoc.yaml`이 필수**입니다.
- `ddoc.yaml`이 없거나 스키마/값이 규약과 다르면 **처리되지 않습니다(= fallback 없음)**.

상세 스펙/템플릿은 `packages/driftstudio_spec/README.md`를 참고하세요.

### 예시(`ddoc.yaml`, text)

```yaml
modality: text
data:
  csv: data.csv
  columns: content
```

## 분석 실행 흐름(개요)

1. 사용자가 ZIP 업로드
2. API가 ZIP을 압축 해제하고 `ddoc.yaml`로 데이터셋을 검증
3. Runtime이 플러그인 기반 분석(EDA/Drift)을 실행하고 산출물/캐시를 저장
4. Web이 산출물을 시각화 카드로 렌더링

## 프로젝트 구조

- `apps/api/`: FastAPI 서버(업로드/검증/런타임 실행/리포트/임베딩 API)
- `apps/web/`: UI(데이터셋/EDA/Drift/리포트/임베딩 시각화)
- `packages/ddoc/`: 플러그인 기반 분석 엔진(최소 CLI 포함)
- `packages/driftstudio_runtime/`: plan 실행기/zip 처리/모달리티 추론·검증
- `packages/driftstudio_reports/`: 리포트 렌더(HTML/PDF)
- `packages/driftstudio_spec/`: `ddoc.yaml`/artifact 스펙 정의

## 트러블슈팅

- **텍스트 분석 중 `Quantization is not supported for ArchType::neon...`**
  - 대개 컨테이너 내부의 네이티브 라이브러리(모델 로더/최적화 코드)에서 출력되는 로그입니다.
  - 기능 오류가 아니라면 무시 가능하며, 필요 시 어떤 라이브러리에서 발생하는지 컨테이너 내부에서 문자열 검색으로 추적할 수 있습니다.
