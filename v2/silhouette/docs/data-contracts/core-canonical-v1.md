# Core Canonical Contract v1

Date: 2026-04-30

## 목적

여러 채널(소스 데이터셋)의 포맷 차이와 변경을 흡수하기 위해, `silhouette` 파이프라인은 공통 Core 필드와 채널별 Extension 필드를 분리한다.

핵심 원칙:
- 원본(`raw`)은 불변(immutable)으로 보존한다.
- 공통 분석/시각화는 Core Canonical 필드만 의존한다.
- 채널 특화 속성은 Extension으로 격리해 보존한다.

## 레이어 모델

- Raw layer: 채널 원본 payload + 원본 경로 + 채널 메타데이터
- Core layer: 채널 공통 분석 필드 (하위 호환 유지 대상)
- Extension layer: 채널별 가변 필드 (JSON/object)

## Core Canonical v1 필드

아래 필드는 `fact_snapshots` 및 API 공통 계약의 기반 필드다.

필수:
- `snapshot_id`: 스냅샷 식별자
- `crawl_datetime`: 스냅샷 수집 시각(ISO8601)
- `product_id`: 상품 식별자
- `name`: 상품명
- `source_dataset`: 소스 데이터셋 식별자
- `platform`: 플랫폼 식별자
- `schema_version`: 추정/명시 스키마 버전

권장(있으면 제공):
- `brand`
- `rank`
- `price`
- `discount_pct`
- `category_label`

## Extension 정책

- Core에 포함되지 않은 필드는 `extension` 객체로 저장한다.
- `extension`에는 채널별 원본 키를 최대한 보존한다.
- Key 충돌 방지를 위해 채널 prefix 또는 namespace를 권장한다.

## 품질/유효성 필드

Core 정규화 결과에는 아래 품질 메타데이터를 포함할 수 있다.
- `missing_required_fields`: 누락된 Core 필수 필드 목록
- `quality_score`: 누락률/타입 일치율 기반 점수(0~1)
- `adapter_version`: 레코드를 생성한 어댑터 버전

## 호환성 정책

- additive 변경(필드 추가): 허용
- breaking 변경(필드 삭제/의미 변경): 금지, 신규 버전으로 승격
- `schema_version`이 변경되면 채널 프로파일 리포트에서 diff를 제공한다.

## 운영 규칙

- 신규 채널은 어댑터를 통해 Core 매핑을 제공해야 한다.
- Core 필수 필드 누락률이 임계치를 넘으면 `warn` 또는 `fail` 정책으로 처리한다.
- 원본 재처리가 가능하도록 raw 보존 데이터는 덮어쓰지 않는다.
