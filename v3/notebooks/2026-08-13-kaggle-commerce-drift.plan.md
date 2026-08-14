# Kaggle 커머스 시계열 드리프트 실험

- 날짜: 2026-08-13 (M5 공식 전환: 2026-08-14)
- 노트북: `notebooks/02_kaggle_commerce_timeseries_drift.ipynb`
- 인증: `~/.kaggle/access_token` + competition Rules 수락

## 다운로드 결과

| 데이터 | 결과 | 비고 |
|---|---|---|
| Olist | OK (`olistbr/brazilian-ecommerce`) | `data/olist_kaggle/` |
| Favorita | OK (competition) | `data/favorita/` |
| Instacart | OK (dataset mirror) | `data/instacart/` |
| M5 official | OK (`kagglehub.competition_download`) | Rules 수락 후 공식 CSV |

## M5 다운로드

```python
import kagglehub
path = kagglehub.competition_download("m5-forecasting-accuracy")
```

`lib/kaggle_data.download_m5()` 가 kagglehub → CLI → parquet 미러 순으로 시도하고,
공식 CSV가 있으면 그걸로 패널을 만든다.

## 패널 캐시

- `data/m5/panels/` (`source.txt=official`)
- `data/favorita/panels/`
- `data/instacart/panels/`
