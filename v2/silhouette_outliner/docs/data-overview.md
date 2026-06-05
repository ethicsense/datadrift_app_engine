# 수집 데이터 개요

작성일: 2026-05-13

## 원천 수집 단위

MVP는 `sectionId × categoryCode` 조합을 하나의 수집 단위로 봅니다.

- `sectionId`: 전체, 급상승, 부티크, used 같은 상위 랭킹 목록 구분 ID
- `categoryCode`: 카테고리 구분 코드
- `gf`: 성별 필터
- `ageBand`: 연령 필터
- `subPan`: 랭킹 하위판, MVP 기본값은 `product`
- `storeCode`: 스토어 코드 (기본값은 앱 내장 랭킹 소스와 동일)

## categoryCode 규칙

`categoryCode`는 문자열로 보관하며, 분석 단계에서는 6자리로 정규화합니다.

- 앞 세자리: 상위 분류
- 뒤 세자리: 세부 분류
- 뒤 세자리가 `000`: 해당 상위 분류 전체
- 예: `001000`은 `001` 상위 분류 전체, `001001`은 `001`의 첫 번째 세부 분류
- 소스 URL에 `000`처럼 3자리 코드가 들어오면 내부 분석에서는 `000000`으로 해석합니다.

## 기본 카테고리

초기 MVP는 2026-05-13 discovery 결과에서 확인한 다음 seed mapping을 사용합니다. 실제 운영 전에는 `silhouette-outliner discover` 결과와 소스 사이트 필터 요청을 다시 확인해 보정해야 합니다.

- `000`: 전체
- `001000`: 상의
- `003000`: 바지
- `002000`: 아우터
- `103000`: 신발
- `100000`: 원피스/스커트
- `017000`: 스포츠/레저
- `106000`: 키즈
- `101000`: 악세서리

## 정규화 필드

`normalized.json`의 `items` 배열은 다음 필드를 갖습니다.

- `rank`: 수집 조합 안에서의 순위
- `brand`: 브랜드명
- `product`: 상품명
- `price`: 판매가
- `original_price`: 정가 또는 정상가
- `discount_rate`: 할인율
- `product_id`: 상품 ID
- `product_url`: 상품 URL
- `image_url`: 이미지 URL
- `section_label`, `section_id`: 랭킹 구분
- `category_label`, `category_code`: 카테고리
- `category_major_code`, `category_minor_code`: 정규화된 카테고리 코드 분해
- `category_parent_label`: 상위 카테고리 라벨
- `source`: `network-json`, `dom`, `none`
- `collected_at`: 수집 시각

## 분석 결과

`analysis.json`은 리포트 생성을 위해 다음 영역을 포함합니다.

- `kpis`: 수집 조합, 성공/실패, 상품 수, 브랜드 수
- `coverage`: 수집 조합별 상태
- `quality`: 누락/중복/실패 URL
- `top_items`: 전체 TOP 상품
- `category_summary`: 카테고리별 상품 수, 브랜드 수, 가격, 대표 브랜드
- `section_summary`: 랭킹 구분별 요약
- `brand_summary`: 브랜드별 상품 수, 최고 순위, 카테고리 분포
- `price_buckets`: 가격대 분포
- `discount_buckets`: 할인율 분포
- `insights`: 리포트 상단 핵심 문장
