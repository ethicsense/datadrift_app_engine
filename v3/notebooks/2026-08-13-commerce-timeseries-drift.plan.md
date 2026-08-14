# 커머스 시계열 데이터 드리프트 탐색 실험

- 날짜: 2026-08-13
- 위치: `notebooks/01_commerce_timeseries_drift.ipynb`

## 목표

커머스 공개 데이터셋의 구성(grain, 엔티티, 지표)을 펼쳐 보고, 데이터 드리프트를 어떤 층에서 측정할 수 있는지 재현 가능한 분석/시각화로 정리한다.

## 데이터

| 데이터셋 | 사용 | 이유 |
|---|---|---|
| UCI Online Retail II | 주 실험 | 거래 원장, 2년, CC BY 4.0, 인증 없이 다운로드 |
| Olist Brazilian E-Commerce | 주 실험 | 관계형 주문/배송/카테고리/결제, GitHub 미러 |
| Tableau Sample Superstore | 보조 | 카테고리·지역·세그먼트 믹스 시각화 |
| M5 / Favorita / Instacart | 카탈로그만 | Kaggle 인증 필요 |

## 분석 축

1. 스키마/grain/엔티티 프로파일
2. 커머스 KPI 시계열 (GMV, orders, AOV, cancel, SLA)
3. 구성(mix) 시계열 (국가, 카테고리, 결제수단)
4. 드리프트: PSI, KS, Wasserstein, TV/JS, SKU churn, volume-mix-price 분해, rolling score
5. 제품 인사이트: 커머스 도메인 킷이 추적해야 할 시계열 슬롯
