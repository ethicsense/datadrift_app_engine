# 랭킹 소스 페이지 노트

작성일: 2026-05-13

## 기준 URL

랭킹 웹 URL과 JSON API 베이스는 코드의 `silhouette_outliner.endpoints` 및 `silhouette_outliner.config`에 정의되어 있습니다. 파라미터 조합 예시는 다음과 같습니다.

- `gf`: 성별 필터 (예: `A`)
- `storeCode`: 스토어 코드 (기본은 내장 소스와 동일)
- `sectionId`: 상위 랭킹 목록 구분 (기본 예시 `199`는 전체 섹션)
- `contentsId`: 콘텐츠 식별자 (빈 값 가능)
- `categoryCode`: 카테고리 코드
- `ageBand`: 연령 필터 (예: `AGE_BAND_ALL`)
- `subPan`: 하위판 (예: `product`)
- `period` (선택): 랭킹 **집계 기간**. 웹 랭킹 URL 쿼리와 클라이언트 API `.../pans/ranking?...` 모두에서 동일 키로 관측됨 (2026-05-14 기준).

## 필터 바(DOM)와 `period`

랭킹 상단에서 **성별 · 연령 · 시간(기간) · 품절 포함** 등을 묶어 그리는 바는, 스타일드 컴포넌트 기준 클래스 이름에 `UIQueryUpdatedat__Wrap`이 포함된 래퍼 안에 들어 있습니다. 배포마다 해시 서픽스(`-sc-xxxxx-0` 등)가 바뀔 수 있으므로 **전체 문자열 고정 매칭보다 접두·역할 기반**으로 찾는 편이 안전합니다.

기간(시간) 축은 위 UI로 바꾸거나, **웹 URL에 `period=…`를 붙여도** 동일하게 동작합니다.

예시 (상의, 전체 섹션):

- 주간: `https://www.musinsa.com/main/musinsa/ranking?gf=A&storeCode=musinsa&sectionId=199&contentsId=&categoryCode=001000&ageBand=AGE_BAND_ALL&subPan=product&period=WEEKLY`

Network에서 확인한 첫 판 API는 예를 들어 다음과 같이 **`period`가 쿼리에 포함**됩니다.

- `GET .../api/home/web/v5/pans/ranking?...&subPan=product&period=WEEKLY`

추가 페이지네이션 호출(`.../pans/ranking/sections/199?...`)에도 같은 `period`와 함께 `eventPeriod=BASIC_REALTIME` 등이 붙는 경우가 있습니다. 수집 MVP는 **첫 `pans/ranking` JSON**을 우선 사용합니다.

### `period` 값 (무신사, 2026-05-14 Playwright로 1차 호출 검증)

| 값 | 설명에 가깝게 쓰인 말 |
|----|---------------------|
| `REALTIME` | 실시간(기본 동작에 가깝함; URL에 생략 시에도 내부 페이로드에서 REALTIME이 보일 수 있음) |
| `DAILY` | 일간 |
| `WEEKLY` | 주간 |
| `MONTHLY` | 월간 |

앱 설정에서는 `ranking_windows[].query_params`에 예를 들어 `{ "period": "WEEKLY" }`처럼 넣으면 `build_ranking_api_url` 결과에 그대로 합쳐집니다.

```json
"ranking_windows": [
  { "id": "1d", "label": "일간", "query_params": { "period": "DAILY" } },
  { "id": "1w", "label": "주간", "query_params": { "period": "WEEKLY" } },
  { "id": "1m", "label": "월간", "query_params": { "period": "MONTHLY" } }
]
```

## 정적 HTML 관찰

공개 HTML fetch 기준으로는 상단 메뉴, 커스텀판 배너, 성별 필터 텍스트가 보일 수 있습니다. 상품 랭킹 목록은 정적 HTML에 안정적으로 포함되지 않을 수 있으므로 Playwright 기반 discovery가 필요합니다.

## API 후보

2026-05-13 discovery에서, 내장 클라이언트 API 경로(`/api/home/web/v5/pans/ranking`)가 상품 랭킹 데이터를 반환하는 것을 확인했습니다.

응답의 상품 카드에는 `id`, `image.rank`, `image.url`, `info.brandName`, `info.productName`, `info.finalPrice`, `info.discountRatio`가 포함됩니다.

## 구현 판단

MVP는 다음 순서로 수집합니다.

1. 내장 랭킹 JSON API를 직접 호출합니다.
2. API 응답 JSON 안에서 상품명, 브랜드, 가격, 순위 중 일부를 가진 레코드를 재귀적으로 찾습니다.
3. 직접 API 호출이 실패하면 Playwright로 페이지를 로드하고 네트워크 JSON 응답을 후보로 봅니다.
4. 상품형 JSON이 발견되면 해당 응답을 원천 데이터로 저장합니다.
5. 상품형 JSON이 없으면 렌더링된 DOM에서 `/products/` 링크를 찾아 상품 후보를 구성합니다.

## 랭킹 기간(일·주·월 등) — 설정 절차

작성일: 2026-05-14

무신사는 위 표의 **`period` 쿼리**로 기간을 구분합니다. 그 외 성별(`gf`), 연령(`ageBand`), 품절 포함 등은 같은 필터 바에서 바뀌며, 필요하면 `sections[].params` 또는 동일 키를 `ranking_windows[].query_params`에 두어 API에 합칩니다.

1. 브라우저 **Network**에서 `pans/ranking` 요청 URL을 확인합니다.
2. UI에서 기간을 바꿀 때마다 **`period` 값**이 어떻게 바뀌는지 기록합니다.
3. `ranking_windows`에 `query_params`로 반영합니다. 저장소에는 `configs/realtime.json`(실시간)과 `configs/daily-weekly-monthly.json`(일·주·월) 프리셋이 있으며, 실행 시 `silhouette-outliner collect --config configs/…`로 지정하면 됩니다.

**병합 우선순위**: `build_ranking_url` / `build_ranking_api_url`에서 기본 파라미터에 `section.params`를 합친 뒤, **`ranking_window.query_params`가 마지막에 덮어씁니다.** 동일 키가 `section.params`와 충돌하면 윈도우 쪽이 우선입니다.

## 리스크

- 실제 `categoryCode` 매핑은 소스 사이트 필터 요청과 다를 수 있습니다.
- DOM fallback은 브랜드명/가격 추정 정확도가 API 수집보다 낮습니다.
- 소스 사이트가 봇 차단, 지역/세션별 응답 차이, 실험군 UI를 적용할 경우 수집 결과가 달라질 수 있습니다.
- `used`, `부티크`, `급상승`은 `sectionId` 또는 추가 파라미터 매핑이 필요할 수 있습니다.
