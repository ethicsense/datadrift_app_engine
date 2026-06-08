# 수집용 설정 프리셋

날짜: 2026-05-14

| 파일 | 용도 |
|------|------|
| `realtime.json` | `period=REALTIME` 한 번만 수집 (실시간 랭킹). |
| `realtime-multag.json` | 실시간 + 성별 3 × 연령 7; 리포트 **연령별 실시간 랭킹** TOP 30 표·가격×나이 히트맵(실시간 기준). |
| `daily-weekly-monthly.json` | `DAILY` / `WEEKLY` / `MONTHLY` 세 번 수집 후 교차 분석. |
| `periodic-multag.json` | 일·주·월 + 성별 3 × 연령 7(히트맵·주간) + 성별 전체 × 연령 7(연령별 랭킹 표·**실시간** `age_rankings_window`). |

실행 예:

```bash
silhouette-outliner collect --config configs/realtime.json --out runs
silhouette-outliner collect --config configs/realtime-multag.json --out runs
silhouette-outliner collect --config configs/daily-weekly-monthly.json --out runs
```

`period` 값은 무신사 기준이며, 상세는 [docs/ranking-source-notes.md](../docs/ranking-source-notes.md)를 참고하세요. 카테고리·섹션·`limit` 등은 각 JSON을 복사해 수정하면 됩니다.

각 윈도우 엔트리에는 `days` 필드를 명시할 수 있습니다 (예: `1`, `7`, `30`). 생략 시 `id`/`label`에서 자동 추정합니다. 이 값은 모멘텀 누적 가중(`sqrt(days)`)과 라인 차트 x축 간격(`log10(1+days)`)에 사용되며, 모멘텀 방향(`momentum_span`) 자체에는 곱해지지 않습니다. 자세한 정의는 루트 README의 "모멘텀 정의" 절을 참고하세요.

| `age-gender-heatmap.json` | 성별 3종 × 연령 7종 × 주간 랭킹 (가격×나이 히트맵용, 상의 1카테고리). |
