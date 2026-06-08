# Silhouette Ranking Outliner

패션 랭킹 페이지를 일회성으로 수집하고, 정규화된 데이터와 정적 HTML 리포트를 생성하는 MVP입니다.

## 범위

- 기본 랭킹 구분: `sectionId=199`
- 기본 성별/연령: `gf=A`, `ageBand=AGE_BAND_ALL`
- 기본 하위판: `subPan=product`
- 기본 카테고리: 전체, 상의, 바지, 아우터, 신발, 원피스/스커트, 스포츠/레저, 키즈, 악세서리
- 수집 방식: Playwright로 페이지를 열고 네트워크 JSON을 우선 수집합니다. JSON을 찾지 못하면 렌더링된 DOM의 상품 링크를 fallback으로 수집합니다.

## 설치

프로젝트 루트에 `.venv`를 만든 뒤 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui]"
playwright install chromium
```

## GUI 실행

데스크톱 GUI는 PySide6(Qt) 기반이며 Ubuntu, Windows, macOS에서 동일하게 실행할 수 있습니다. 앱 자체는 오프라인에서도 실행되지만, **분석 실행 시점에는 네트워크 연결이 필요**합니다. 연결이 없으면 GUI에서 안내 팝업이 표시됩니다.

```bash
source .venv/bin/activate
python -m silhouette_outliner.gui
```

또는:

```bash
silhouette-outliner-gui
```

편의 스크립트:

```bash
# macOS / Linux
./scripts/run_gui.sh

# Windows
scripts\run_gui.bat
```

GUI 기능:

- 플랫폼 선택: 무신사(동작), 28CM/유튜브/인스타그램/네이버뉴스(준비 중)
- 분석 실행, 보고서 열기, 보고서 폴더 열기
- 실행 로그 패널 및 진행률 표시
- 네트워크 오류·분석 실패 등 사용자 확인 사항 alert 팝업

GUI의 **분석 실행**은 CLI에서 `configs/periodic-multag.json`을 쓸 때와 동일한 전체 프리셋(일·주·월 기간, 성별·연령 신호, 모멘텀·기간별 TOP 등)을 사용합니다.

## 데모 설치 패키지 (Windows / macOS)

OS별 설치형 데모 패키지는 **GitHub Actions**로 빌드합니다. 로컬 Windows PC 없이도 CI에서 Windows zip을 받을 수 있습니다.

- 워크플로: [`.github/workflows/silhouette-outliner-demo.yml`](../../.github/workflows/silhouette-outliner-demo.yml) (저장소 루트)
- 상세: [packaging/README.md](packaging/README.md)

**Actions에서 수동 실행:** GitHub → Actions → *Silhouette Outliner Demo Package* → Run workflow → Artifacts 다운로드

**태그 실행(Artifacts 생성):** `git tag demo-v0.1.0 && git push origin demo-v0.1.0`

## 실행

```bash
source .venv/bin/activate
silhouette-outliner collect
```

결과는 `runs/{timestamp}/` 아래에 저장됩니다.

- `config.json`: 실행 설정
- `raw/`: 수집 원본 JSON/HTML
- `normalized.json`: 정규화 상품 데이터
- `analysis.json`: 리포트용 분석 결과
- `report.html`: 정적 리포트

## 페이지 구조 탐색

실제 소스 사이트의 API/DOM 구조를 먼저 확인하려면 다음을 실행합니다.

```bash
source .venv/bin/activate
silhouette-outliner discover
```

`runs/{timestamp}_discovery/discovery.json`에 상품형 JSON 응답 후보와 DOM 상품 링크 수가 기록됩니다.

## 설정 변경

기본 설정을 출력한 뒤 필요한 카테고리나 섹션을 수정할 수 있습니다.

```bash
silhouette-outliner sample-config > config.local.json
silhouette-outliner collect --config config.local.json
```

미리 나둔 프리셋은 `configs/`에서 `--config`로 골라 쓰면 됩니다.

```bash
silhouette-outliner collect --config configs/realtime.json --out runs
silhouette-outliner collect --config configs/daily-weekly-monthly.json --out runs
```

`configs/README.md`에 각 파일 용도가 정리되어 있습니다.

기본 `collect` 실행 시 상의 상품 랭킹과 함께, BCave 포트폴리오(커버낫·와키윌리·팔렛·리)에 대한 **무신사 브랜드 탭** 순위가 자동 수집되어 동일 `report.html`의 **BCave 포트폴리오** 섹션에 표시됩니다. 수집 대상 4판은:

- 전체 상의 (`sectionId=1054`, `categoryCode=001000`, `subPan=brand`)
- 영캐주얼 (`sectionId=1056`, `categoryCode=` 전체)
- 여성캐주얼 (`sectionId=1063`, `categoryCode=` 전체)
- 스트릿캐주얼 (`sectionId=1066`, `categoryCode=` 전체)

스타일 레인 매핑: 커버낫·리 → 영캐주얼, 와키윌리 → 스트릿캐주얼, 팔렛 → 여성캐주얼. 끄려면 설정에 `"track_bcave_portfolio": false`를 넣습니다.

카테고리 코드는 `앞 세자리 = 상위 분류`, `뒤 세자리 = 세부 분류`, `xxx000 = 상위 전체` 규칙으로 해석합니다.

## 일·주·월 등 랭킹 기간(다중 스냅샷)

`ranking_windows` 배열을 넣으면 `section × category × 기간`만큼 수집이 반복됩니다. 각 항목의 `query_params`는 랭킹 API URL에 그대로 병합되며(섹션 `params`보다 우선), 무신사 `period` 값은 [docs/ranking-source-notes.md](docs/ranking-source-notes.md)와 `configs/daily-weekly-monthly.json`을 참고하세요.

각 윈도우 엔트리에는 선택적으로 `days`(일수)를 지정할 수 있습니다. 미지정 시 `id`(`1d`/`1w`/`1m` 등)와 `label`(`일간` 등)에서 자동 추정합니다. 이 값은 모멘텀 가중과 라인 차트 x축 간격에 사용됩니다.

수집 후 `normalized.json`의 각 상품에 `ranking_window_id` / `ranking_window_label` / `ranking_window_days`가 붙고, `analysis.json`에는 `windows`(기간별 단면, TOP 10 포함)와 `cross_window`(모멘텀 대시보드)가 추가됩니다.

### 모멘텀 정의

`cross_window`는 동일 상품을 `product_id`(없으면 URL · 브랜드+상품명) 기준으로 묶어 `1m → 1w → 1d` 흐름으로 계산합니다.

- **`rank_energy(w) = (limit(w) + 1 − rank) / limit(w)`** — 윈도우 안에서의 정규화된 점수 (1위가 1.0).
- **`momentum_span = rank_energy(현재) − rank_energy(과거)`** — 월·일 양 끝 윈도우 순위가 **모두 있을 때만** 계산합니다. `+`면 현재 펄스가 강함, `−`면 과거 누적이 더 좋았음.
- **이벤트 클러스터** — 진입(최신만), 이탈(과거만), 중간 단독(주간만), 부분 관측 등은 `momentum_span`과 분리해 별도 집계합니다. `event_strength`는 클러스터 내부 정렬용이며 유지 상품의 스팬과 직접 비교하지 않습니다.
- **`sustained_rank_energy = Σ sqrt(days(w)) · rank_energy(w)`** — 누적 검증 강도. 일간:주간:월간 ≈ 1 : 2.65 : 5.48. 선형(1:7:30)은 월간 비중이 과대해지므로 sqrt로 감쇠합니다.
- **인접 윈도우 속도** `v_wm = rank_energy(1w) − rank_energy(1m)`, `v_dw = rank_energy(1d) − rank_energy(1w)`.
- **패턴 라벨**: `steady_climb`, `breakout`/`entry_breakout`, `stable_top`, `fading`, `classic_drop`, `mid_blip`, `transient_dw`, `transient_wm`, `mixed`. 위 라벨에 따라 인사이트 카드 4종이 자동 채워집니다.

라인 차트의 x축은 등간격이 아니라 `log10(1 + days)` 간격을 씁니다. 월(30일)→주(7일)와 주(7일)→일(1일)의 실제 시간 격차 차이를 시각적으로도 반영하기 위함입니다.

### 이미지 정책

리포트의 TOP 10 표·인사이트 카드 썸네일은 무신사 CDN URL을 **다운로드 없이 직접 핫링크**합니다.

- `<img loading="lazy" decoding="async" referrerpolicy="no-referrer">` 속성으로 트래픽 영향을 최소화합니다.
- CSS 호버 확대(2.4~2.6배)는 JS 없이 동작하지만, 이미지 자체는 외부 호스트에서 받기 때문에 **오프라인에서 리포트를 열면 이미지가 보이지 않습니다.** 카드 자리에는 브랜드 첫 글자 placeholder가 표시됩니다.
- 사양상 의도된 동작입니다. 정적 캡처가 필요하다면 별도의 이미지 캐싱 단계를 추가해야 합니다.
