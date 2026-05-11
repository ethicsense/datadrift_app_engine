# Silhouette Analytics Guide

`silhouette`는 패션 랭킹 스냅샷 데이터를 분석하고 시각화하는 프로젝트입니다.  
OCR과 크롤링은 이 저장소의 책임 범위에서 제외되며, 이 프로젝트는 이미 수집된 스냅샷 데이터를 입력으로 받아 분석 산출물과 대시보드를 생성합니다.

## 한 줄 배포 (권장)

`git clone` 후 원천 데이터를 `data/`에 넣고, 프로젝트 루트에서 아래 한 줄만 실행하세요.

```bash
docker compose up --build
```

실행 순서:
- `qdrant` 시작
- `analytics-pipeline` 데이터 preflight + 분석 수행
- 분석 성공 시 `visualization-api` 시작
- API 헬스체크 통과 시 `visualization-web` 시작

자세한 운영 가이드는 `docs/ONE_COMMAND_DEPLOYMENT.md`를 참고하세요.

## 환경 설정

- Python 3.10+ 권장
- 이 프로젝트의 표준 가상환경 경로는 `silhouette_venv`입니다.
- 프로젝트 루트에서 가상환경 생성:

```bash
python -m venv silhouette_venv
source silhouette_venv/bin/activate
pip install -r requirements.txt
```

- 임베딩 가속이 필요하면 `torch`, `open_clip_torch`가 동작하는 GPU 환경을 사용하세요.

## 입력 데이터 계약

기본 원칙은 `data/` 전체를 재귀 탐색해, `source_dataset/YYYY-MM-DD/HH-MM/ranking_summary.json` 구조를 만족하는 입력을 모두 canonical 시간축 저장소로 통합하는 것입니다.

채널 변화 대응을 위한 공통 계약은 `docs/data-contracts/core-canonical-v1.md`를 기준으로 운영합니다.
파이프라인은 `raw 보존 + core 정규화 + extension 격리` 원칙을 따르며, 공통 분석/대시보드는 Core 필드 계약을 우선 사용합니다.

`--data-dir` 아래에는 다음과 같은 source dataset들이 섞여 있어도 됩니다.

```text
data_dir/
├── musinsa_feb/
│   └── 2026-02-20/
│       └── 09-00/
│           ├── ranking_summary.json
│           └── products/
│               └── <product_id>/
├── musinsa_march/
│   └── 2026-03-05/
│       └── 09-00/
│           ├── ranking_summary.json
│           └── products/
│               └── <product_id>/
└── other_source/.../YYYY-MM-DD/HH-MM/ranking_summary.json
```

필수 입력:
- `ranking_summary.json`

선택 입력:
- `product_info.csv`
- `tags.csv`
- `ocr_data.json`
- `size_table.csv`
- `detail_images/*`

실데이터에서는 `detail_XX.jpg` 패턴이 일반적이며, `main_image.jpg`는 없어도 됩니다.
이미지는 분석 코어에는 필수가 아니며, 멀티모달 임베딩을 사용할 때만 활용됩니다.
`ocr_data.json`은 제품 폴더 단위 OCR 결과이며, 내부 텍스트는 제품별 대표본만 `dim_products`에 보존하고 스냅샷 팩트에는 경량 메타데이터만 남깁니다.

## 분석 데이터셋 생성

프로젝트 루트에서 실행:

```bash
source silhouette_venv/bin/activate
python -m analytics.pipeline.build_dataset --data-dir data --output-dir output/analytics
```

기간 또는 source를 제한하는 예시:

```bash
source silhouette_venv/bin/activate
python -m analytics.pipeline.build_dataset \
  --data-dir data \
  --output-dir output/analytics \
  --start-date 2026-02-20 \
  --end-date 2026-02-26 \
  --include-source musinsa_feb \
  --include-source musinsa_march
```

주요 옵션:
- `--latest-snapshot-limit N`
- `--include-source musinsa_feb`
- `--exclude-source musinsa_legacy`
- `--disable-multimodal`
- `--disable-embeddings`
- `--disable-qdrant-upsert`
- `--metrics-refresh-only` (기존 `output/analytics/fact_snapshots.parquet` 기반으로 지표/분석 테이블만 빠르게 재계산)
- `--qdrant-url http://localhost:38633`
- `--quota-product-packshot 1`
- `--quota-detail-closeup 1`
- `--quota-model-wearing 1`

빠른 재분석(임베딩/이미지 처리 없이 지표만 갱신) 예시:

```bash
source silhouette_venv/bin/activate
python -m analytics.pipeline.build_dataset --output-dir output/analytics --metrics-refresh-only
```

Docker Compose 예시:

```bash
docker compose up --build
```

## 시각화 엔진 실행

가상환경 활성화 후:

```bash
uvicorn analytics.api.main:app --reload --host 127.0.0.1 --port 8000
```

프론트엔드는 별도 셸에서 실행:

```bash
cd apps/visualization-web
npm install
npm run dev
```

로컬 개발 시 `VITE_API_BASE_URL`을 비워 두면(기본), Vite가 `/api`·`/health`를 위 uvicorn 주소로 프록시합니다. API를 다른 호스트·포트에서 띄우면 `SILHOUETTE_VITE_PROXY_TARGET`(예: `http://127.0.0.1:9000`)을 설정하세요.

`apps/visualization-web/vite.config.js`가 `vite.config.ts` 옆에 있으면 Vite가 JS 설정을 우선해 프록시가 빠진 채로 뜰 수 있습니다. 해당 파일은 사용하지 않으며(빌드 시 선언만 `.tmp/`로 emit), 저장소에 두지 마세요.

기본 접속 주소(로컬):
- Web(Vite): `http://localhost:4173` (브라우저는 이 오리진만 사용하면 됨)
- API(직접 헬스 확인): `http://127.0.0.1:8000/health`

Docker Compose로 띄울 때:
- API: `http://localhost:38000`
- Web: `http://localhost:38173`

Docker Compose 예시:

```bash
docker compose up --build
```

## 주요 산출물

핵심 분석 산출물:
- `output/analytics/fact_snapshots.parquet`
- `output/analytics/fact_snapshots.csv`
- `output/analytics/product_latest.parquet`
- `output/analytics/dim_products.parquet`
- `output/analytics/product_snapshot_coverage.parquet`
- `output/analytics/kpi_summary.json`
- `output/analytics/analysis_tag_performance.parquet`
- `output/analytics/analysis_brand_index.parquet`
- `output/analytics/analysis_trends.parquet`
- `output/analytics/analysis_product_profile.parquet`
- `output/analytics/analysis_embedding_projection.parquet`
- `output/analytics/analysis_rank_trajectories.parquet`
- `output/analytics/analysis_rank_race.parquet`
- `output/analytics/analytics.duckdb`

선택적 멀티모달 산출물:
- `output/analytics/image_manifest.parquet`
- `output/analytics/image_segments.parquet`
- `output/analytics/image_embeddings.parquet`
- `output/analytics/text_features.parquet`

산출물 역할:
- `fact_snapshots`: 모든 source dataset을 `crawl_datetime` 기준으로 통합한 canonical 시간축 관측치
- `dim_products`: 제품별 대표 상세 정보와 OCR 집계, 대표 이미지 메타데이터
- `product_snapshot_coverage`: 스냅샷별 상세 수집 성공 여부와 OCR/이미지 커버리지
- `raw_snapshot_products`: provenance 및 신규 필드 관측용 경량 raw sidecar
- `schema_raw.json`, `schema_normalized.json`, `schema_diff.json`: canonical 저장소와 raw sidecar의 스키마 관측 자산

Canonical 관측 컬럼에는 최소한 다음 출처 메타가 포함됩니다.
- `platform`
- `source_dataset`
- `schema_version`
- `source_snapshot_id`
- `crawl_datetime`

## 지표 정의

- `rank_velocity`: 직전 스냅샷 대비 순위 개선 폭
- `rank_acceleration`: 순위 개선 속도의 변화량
- `rank_energy`: `rank_filled=51` 보정 후 `score=51-rank_filled`, `rank_energy=score*(1+2*(score/50)^2)`로 계산한 연속 비선형 순위 에너지
- `energy_velocity`: 직전 시점과 현재 시점에 모두 순위권에 관측된 상품에 한해 `rank_energy(t) - rank_energy(t-1)`로 계산
- `energy_acceleration`: 연속 관측 구간의 `energy_velocity(t) - energy_velocity(t-1)`
- `entry_score`: 첫 관측 또는 순위권 재진입 시점의 `rank_energy(t)`; `momentum_score`에는 섞지 않음
- `exit_score`: 순위권 탈락 첫 시점의 `rank_energy(t-1)`; 이후 순위권 밖 체류에는 반복 감점을 부여하지 않음
- `cumulative_rank_energy`: 선택 기간 동안 상품이 순위권에 관측된 모든 시점의 `rank_energy` 합
- `sustained_rank_energy`: `cumulative_rank_energy * (0.5 + presence_ratio)`로, 단발 고순위보다 반복적으로 순위권에 머문 상품을 우선하기 위한 제품 단위 점수
- `momentum_score`: z-score 표준화 없이 연속 관측 구간의 `energy_velocity`를 그대로 사용
- `momentum_event_state`: `first_seen`, `chart_in_spike`, `chart_out_drop`, `out_of_chart`, `breakout`, `sustained_growth`, `cooling`, `reversal`, `steady` 상태 태그
- `stability_score`: 최근 6개 관측치 순위 표준편차 기반 안정성 점수
- `discount_efficiency`: 할인율 1%당 순위 변화 효율
- `discount_prev`: 직전 스냅샷의 할인율
- `discount_velocity`: 이번 스냅샷 할인율 - 직전 스냅샷 할인율

## 중복 및 누락 처리

- 파이프라인은 `ranking_summary.json`을 스냅샷 기준 truth source로 사용합니다.
- 동일 스냅샷 안의 `(snapshot_id, product_id)` 중복은 방어적으로 제거합니다.
- 같은 제품이 여러 스냅샷에 반복 등장하는 것은 정상으로 간주하고, 순위 연속성 계산에 그대로 사용합니다.
- OCR 본문과 상세 페이지 유래 정적 자산은 제품 단위 대표본만 `dim_products`에 보존해 중복 저장을 줄입니다.
- 제품 폴더가 없거나 `ocr_data.json`, `tags.csv`, `product_info.csv`, `detail_images`가 빠진 경우도 허용하며, 누락 현황은 `kpi_summary.json`의 `extra` 메타데이터에 기록합니다.

## 대시보드 해석 메모

- 상단 KPI는 항상 전체 파이프라인 결과 기준입니다.
- 필터 적용 후에는 현재 보기 기준 건수가 별도로 표시됩니다.
- 웹 필터는 `dataset` 선택이 아니라 `source_dataset`, `platform`, `schema_version`, 기간 중심으로 canonical 결과를 분리 조회합니다.
- 할인율 수준과 순위 변화는 참고용 분포이며, 해석은 `discount_velocity`와 `rank_velocity`를 함께 보는 쪽이 더 적절합니다.

