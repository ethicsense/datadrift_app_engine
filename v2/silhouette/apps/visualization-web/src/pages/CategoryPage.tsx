import { useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { apiGet } from "../lib/api";
import { describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import type { ExplainabilityFact, WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

type CategoryOverviewResponse = {
  summaryRows: Record<string, unknown>[];
  marketMap: Record<string, unknown>[];
  leaderRows: Record<string, unknown>[];
};

type CategoryRelationshipsResponse = {
  priceHeatmap: Record<string, unknown>[];
  materialHeatmap: Record<string, unknown>[];
  colorHeatmap: Record<string, unknown>[];
};

type CategoryTimeseriesResponse = {
  shareSeries: Record<string, unknown>[];
  rankSeries: Record<string, unknown>[];
  momentumSeries: Record<string, unknown>[];
};

type CategoryQualityResponse = {
  statusRows: Record<string, unknown>[];
  sourceRows: Record<string, unknown>[];
  issueRows: Record<string, unknown>[];
};

type CategoryExamplesResponse = {
  rows: Record<string, unknown>[];
};

const PRICE_BAND_ORDER = ["~3만", "3-7만", "7-12만", "12-20만", "20-50만", "50만+", "미분류"];

function parseSourceLabel(sourceDataset: string): string {
  const lower = sourceDataset.toLowerCase();
  if (lower.includes("musinsa")) {
    return "무신사";
  }
  const bits = sourceDataset.split("_").filter(Boolean);
  if (bits.length >= 2) {
    return bits[1];
  }
  return sourceDataset;
}

export function CategoryPage() {
  const { filters } = useDashboardFilters();
  const [categoryLevel, setCategoryLevel] = useState<"l1" | "l2" | "l3">("l3");
  const [qualityMode, setQualityMode] = useState<"success_only" | "success_partial">("success_only");
  const [includeFallback, setIncludeFallback] = useState(false);
  const [relationshipAxis, setRelationshipAxis] = useState<"price" | "material" | "color">("price");
  const [timeseriesMetric, setTimeseriesMetric] = useState<"share" | "rank" | "momentum">("rank");
  const [leaderMetric, setLeaderMetric] = useState<"shareOfCatalog" | "avgRank" | "avgMomentumScore">("avgRank");
  const [selectedCategoryLabel, setSelectedCategoryLabel] = useState<string | null>(null);
  const handleCategorySelection = (row: Record<string, unknown> | null) => {
    const next = String(row?.categoryLabel ?? "").trim();
    if (!next) {
      setSelectedCategoryLabel(null);
      return;
    }
    setSelectedCategoryLabel((current) => (current === next ? null : next));
  };
  const queryArgs = useMemo(
    () => ({
      level: categoryLevel,
      quality_mode: qualityMode,
      include_fallback: includeFallback ? "true" : "false",
    }),
    [categoryLevel, includeFallback, qualityMode],
  );

  const overviewQuery = useQuery({
    queryKey: ["category-overview", filters, queryArgs],
    queryFn: () => apiGet<CategoryOverviewResponse>("/api/category/overview", filters, queryArgs),
  });
  const relationshipsQuery = useQuery({
    queryKey: ["category-relationships", filters, queryArgs],
    queryFn: () => apiGet<CategoryRelationshipsResponse>("/api/category/relationships", filters, queryArgs),
  });
  const timeseriesQuery = useQuery({
    queryKey: ["category-timeseries", filters, queryArgs],
    queryFn: () => apiGet<CategoryTimeseriesResponse>("/api/category/timeseries", filters, queryArgs),
  });
  const qualityQuery = useQuery({
    queryKey: ["category-quality", filters, categoryLevel],
    queryFn: () => apiGet<CategoryQualityResponse>("/api/category/quality", filters, { level: categoryLevel }),
  });
  const examplesQuery = useQuery({
    queryKey: ["category-examples", filters, queryArgs],
    queryFn: () => apiGet<CategoryExamplesResponse>("/api/category/examples", filters, queryArgs),
  });
  const leaderMetricLabel = leaderMetric === "shareOfCatalog"
    ? "점유율"
    : leaderMetric === "avgRank"
      ? "평균 순위"
      : "평균 모멘텀";
  const leaderRows = useMemo(() => {
    const sourceRows = overviewQuery.data?.leaderRows ?? [];
    const toNumber = (value: unknown) => {
      if (typeof value === "number") {
        return Number.isFinite(value) ? value : null;
      }
      if (typeof value === "string") {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      }
      return null;
    };
    const sorted = [...sourceRows].sort((left, right) => {
      const leftValue = toNumber(left[leaderMetric]);
      const rightValue = toNumber(right[leaderMetric]);
      if (leftValue === null && rightValue === null) {
        return String(left.categoryLabel ?? "").localeCompare(String(right.categoryLabel ?? ""), "ko");
      }
      if (leftValue === null) {
        return 1;
      }
      if (rightValue === null) {
        return -1;
      }
      if (leaderMetric === "avgRank") {
        return leftValue - rightValue;
      }
      return rightValue - leftValue;
    });
    return sorted.map((row, index) => ({
      기준순위: index + 1,
      ...row,
    }));
  }, [leaderMetric, overviewQuery.data?.leaderRows]);

  if (
    overviewQuery.isLoading
    || relationshipsQuery.isLoading
    || timeseriesQuery.isLoading
    || qualityQuery.isLoading
    || examplesQuery.isLoading
  ) {
    return <div className="loading-state">카테고리 데이터를 불러오는 중입니다.</div>;
  }

  const relationshipRows =
    relationshipAxis === "price"
      ? relationshipsQuery.data?.priceHeatmap ?? []
      : relationshipAxis === "material"
        ? relationshipsQuery.data?.materialHeatmap ?? []
        : relationshipsQuery.data?.colorHeatmap ?? [];

  const relationshipSpec =
    relationshipAxis === "price"
      ? {
          x: "priceBand",
          y: "categoryLabel",
          value: "count",
          xLabel: "판매가 가격대",
          yLabel: "카테고리",
          xDomain: PRICE_BAND_ORDER,
        }
      : relationshipAxis === "material"
        ? {
            x: "materialValue",
            y: "categoryLabel",
            value: "count",
            xLabel: "소재",
            yLabel: "카테고리",
          }
        : {
            x: "colorValue",
            y: "categoryLabel",
            value: "count",
            xLabel: "색상",
            yLabel: "카테고리",
          };

  const timeseriesRows =
    timeseriesMetric === "share"
      ? (() => {
          const rows = timeseriesQuery.data?.shareSeries ?? [];
          if (!rows.length) {
            return rows;
          }
          const latestByCategory = new Map<string, { dateValue: number; shareValue: number }>();
          rows.forEach((row) => {
            const category = String(row.categoryLabel ?? "").trim();
            if (!category) {
              return;
            }
            const snapshotRaw = String(row.snapshotDate ?? "");
            const dateValue = Number.isNaN(Date.parse(snapshotRaw)) ? -Infinity : Date.parse(snapshotRaw);
            const shareRaw = row.value;
            const shareValue = typeof shareRaw === "number" ? shareRaw : Number(shareRaw);
            if (!Number.isFinite(shareValue)) {
              return;
            }
            const prev = latestByCategory.get(category);
            if (!prev || dateValue >= prev.dateValue) {
              latestByCategory.set(category, { dateValue, shareValue });
            }
          });
          const topCategories = Array.from(latestByCategory.entries())
            .sort((left, right) => right[1].shareValue - left[1].shareValue)
            .slice(0, 8)
            .map(([category]) => category);
          const focusCategories = new Set(topCategories);
          if (selectedCategoryLabel) {
            focusCategories.add(selectedCategoryLabel);
          }
          return rows.filter((row) => focusCategories.has(String(row.categoryLabel ?? "").trim()));
        })()
      : timeseriesMetric === "rank"
        ? (() => {
            const rows = timeseriesQuery.data?.rankSeries ?? [];
            if (!rows.length) {
              return rows;
            }
            // Rank line chart readability: keep core categories by latest rank
            // and always include currently selected category.
            const latestByCategory = new Map<string, { dateValue: number; rankValue: number }>();
            rows.forEach((row) => {
              const category = String(row.categoryLabel ?? "").trim();
              if (!category) {
                return;
              }
              const snapshotRaw = String(row.snapshotDate ?? "");
              const dateValue = Number.isNaN(Date.parse(snapshotRaw)) ? -Infinity : Date.parse(snapshotRaw);
              const rankRaw = row.value;
              const rankValue = typeof rankRaw === "number" ? rankRaw : Number(rankRaw);
              if (!Number.isFinite(rankValue)) {
                return;
              }
              const prev = latestByCategory.get(category);
              if (!prev || dateValue >= prev.dateValue) {
                latestByCategory.set(category, { dateValue, rankValue });
              }
            });
            const focusedCategories = Array.from(latestByCategory.entries())
              .sort((left, right) => left[1].rankValue - right[1].rankValue)
              .slice(0, 8)
              .map(([category]) => category);
            const focusSet = new Set(focusedCategories);
            if (selectedCategoryLabel) {
              focusSet.add(selectedCategoryLabel);
            }
            return rows.filter((row) => focusSet.has(String(row.categoryLabel ?? "").trim()));
          })()
        : timeseriesQuery.data?.momentumSeries ?? [];

  const timeseriesSpec =
    timeseriesMetric === "share"
      ? {
          x: "snapshotDate",
          y: "value",
          seriesBy: "categoryLabel",
          xLabel: "스냅샷 날짜",
          yLabel: "점유율",
          yFormat: "percent" as const,
          lineSmooth: false,
          lineShowSymbol: false,
          lineSampling: "lttb" as const,
        }
      : timeseriesMetric === "rank"
        ? {
            x: "snapshotDate",
            y: "value",
            seriesBy: "categoryLabel",
            xLabel: "스냅샷 날짜",
            yLabel: "평균 순위",
            yFormat: "number" as const,
            yAxisInverse: true,
            lineSmooth: false,
            lineShowSymbol: false,
            lineSampling: "lttb" as const,
          }
        : {
            x: "snapshotDate",
            y: "value",
            seriesBy: "categoryLabel",
            xLabel: "스냅샷 날짜",
            yLabel: "평균 모멘텀",
            yFormat: "number" as const,
          };

  const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
  const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
  const targetSummary = targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";
  const sharedContext: ExplainabilityFact[] = [
    { label: "카테고리 레벨", value: categoryLevel.toUpperCase() },
    { label: "품질 모드", value: qualityMode === "success_only" ? "성공만" : "성공+부분 성공" },
    { label: "선택 카테고리", value: selectedCategoryLabel ?? "전체" },
    { label: "보완 분류", value: includeFallback ? "포함" : "제외", tone: includeFallback ? "warning" : "accent" },
  ];

  const widgets: WidgetConfig[] = [
    {
      id: "category-market-map",
      type: "chart",
      title: "품목 점유율과 평균 순위",
      description: "카테고리별 점유율과 평균 순위를 한 화면에 겹쳐 봅니다.",
      section: "result",
      takeaway: "점유율이 큰데 순위가 밀리는 품목, 점유율은 작아도 순위가 앞선 품목을 빠르게 분리할 수 있습니다.",
      chartKind: "scatter",
      rows: overviewQuery.data?.marketMap ?? [],
      spec: {
        x: "shareOfCatalog",
        y: "avgRank",
        seriesBy: "categoryLabel",
        scatterSizeBy: "productCount",
        scatterSizeRange: [10, 56],
        scatterSizeExponent: 1.8,
        xLabel: "카탈로그 점유율",
        yLabel: "평균 순위",
        xFormat: "percent",
        yFormat: "number",
        yAxisInverse: true,
        palette: "categorical",
        tooltipFields: [
          { key: "categoryLabel", label: "카테고리", format: "string" },
          { key: "productCount", label: "상품 수", format: "integer", fallback: "-" },
          { key: "shareOfCatalog", label: "시장 점유율", format: "percent", fallback: "-" },
          { key: "avgRank", label: "평균 순위", format: "number", fallback: "-" },
          { key: "avgMomentumScore", label: "평균 모멘텀", format: "number", fallback: "-" },
        ],
        onSelectDatum: handleCategorySelection,
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "점 1개는 카테고리 1개이며, 가로축은 점유율, 세로축은 평균 순위입니다." },
        ],
        drilldown: [
          { text: "점을 클릭하면 아래 리더보드와 대표 사례 표에서 같은 카테고리가 강조됩니다. 같은 점을 다시 누르면 선택이 해제됩니다." },
        ],
      },
    },
    {
      id: "category-timeseries",
      type: "chart",
      title: timeseriesMetric === "share" ? "품목 점유율 추이" : timeseriesMetric === "rank" ? "품목 평균 순위 추이" : "품목 평균 모멘텀 추이",
      description: timeseriesMetric === "share"
        ? "최신 시점 점유율 상위 카테고리를 중심으로 시간 흐름을 보여줍니다."
        : "카테고리별 점유율과 성과가 시간에 따라 어떻게 달라지는지 보여줍니다.",
      section: "result",
      takeaway: timeseriesMetric === "share"
        ? "선이 너무 많아 생기는 가독성 저하를 줄이기 위해 상위 카테고리 중심으로 표시하며, 선택한 카테고리는 항상 함께 보여줍니다."
        : "정적인 분포만 보면 보이지 않는 성장 카테고리와 약화 카테고리를 시계열에서 확인할 수 있습니다.",
      chartKind: "line",
      rows: timeseriesRows,
      spec: {
        ...timeseriesSpec,
        palette: "categorical",
        onSelectDatum: handleCategorySelection,
      },
    },
    {
      id: "category-leaders",
      type: "table",
      title: `${leaderMetricLabel} 상위 카테고리`,
      description: `${leaderMetricLabel} 기준으로 정렬한 순위를 보여줍니다. 점유율/모멘텀은 내림차순, 평균 순위는 오름차순으로 계산합니다.`,
      section: "result",
      takeaway: "기준을 바꿔 보면 시장 지배 카테고리와 성과 개선 카테고리가 어떻게 다른지 빠르게 비교할 수 있습니다.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "기준순위 계산식: rank_i = sort(metric_i). 점유율·평균 모멘텀은 값이 클수록 상위, 평균 순위는 값이 작을수록 상위입니다." },
          { text: "동점이면 카테고리명을 가나다순으로 정렬해 순위를 고정합니다." },
        ],
      },
      rows: leaderRows,
      highlightKey: "categoryLabel",
      highlightValue: selectedCategoryLabel,
    },
    {
      id: "category-relationships",
      type: "chart",
      title: relationshipAxis === "price" ? "품목-가격대 분포" : relationshipAxis === "material" ? "품목-소재 분포" : "품목-색상 분포",
      description: "품목과 선택 속성이 어떻게 묶여 나타나는지 히트맵으로 확인합니다.",
      section: "interpretation",
      takeaway: "카테고리 자체보다 카테고리와 속성의 조합이 시장 차이를 더 선명하게 보여주는 경우가 많습니다.",
      chartKind: "heatmap",
      rows: relationshipRows,
      spec: {
        ...relationshipSpec,
        valueFormat: "integer",
        palette: "sequential",
        tooltipFields: [
          { key: "categoryLabel", label: "카테고리", format: "string" },
          { key: relationshipAxis === "price" ? "priceBand" : relationshipAxis === "material" ? "materialValue" : "colorValue", label: relationshipAxis === "price" ? "가격대" : relationshipAxis === "material" ? "소재" : "색상", format: "string" },
          { key: "count", label: "상품 수", format: "integer" },
        ],
        onSelectDatum: handleCategorySelection,
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "셀 1개는 카테고리 1개와 선택한 관계 축 1개의 조합입니다." },
        ],
        drilldown: [
          { text: "셀을 클릭하면 같은 카테고리가 사례 표에서 강조되어 실제 상품 수준으로 내려가서 검증할 수 있습니다." },
        ],
      },
    },
    {
      id: "category-quality-status",
      type: "chart",
      title: "품질 상태 분포",
      description: "카테고리 인입의 성공/부분 성공/실패/스킵 비중을 확인합니다.",
      section: "summary",
      takeaway: "시장 차트보다 먼저 이 상태 분포를 보면 카테고리 집계의 신뢰 구간을 가늠할 수 있습니다.",
      chartKind: "bar",
      rows: qualityQuery.data?.statusRows ?? [],
      spec: {
        x: "status",
        y: "productCount",
        yLabel: "상품 수",
        yFormat: "integer",
        palette: "semantic",
      },
    },
    {
      id: "category-quality-source",
      type: "chart",
      title: "분류 소스 분포",
      description: "원천 분류와 보완 분류 사용 비중을 분리해 확인합니다.",
      section: "summary",
      takeaway: "보완 분류 비율이 높을수록 카테고리 해석은 더 보수적으로 해야 합니다.",
      chartKind: "bar",
      rows: qualityQuery.data?.sourceRows ?? [],
      spec: {
        x: "source",
        y: "productCount",
        yLabel: "상품 수",
        yFormat: "integer",
        palette: "categorical",
      },
    },
    {
      id: "category-summary",
      type: "table",
      title: "집계 기준",
      description: "현재 품질 모드와 보완 분류 포함 여부를 기준으로 집계 모수를 확인합니다.",
      section: "summary",
      takeaway: "카테고리 해석은 항상 집계 모수와 품질 기준을 먼저 확인한 뒤 시작해야 합니다.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "행 1개는 현재 집계 기준의 카테고리 모수 또는 수집 상태 1개입니다." },
        ],
        caveats: [
          { text: "품질 모드와 보완 분류 여부가 바뀌면 같은 시장 분포도도 다른 모집단을 반영합니다.", tone: "warning" },
        ],
      },
      rows: overviewQuery.data?.summaryRows ?? [],
    },
    {
      id: "category-quality-issues",
      type: "table",
      title: "품질 이슈 사례",
      description: "실패, 스킵, 부분 성공 사례를 표본으로 보여줍니다.",
      section: "examples",
      takeaway: "잘못 분류된 카테고리를 시장 신호로 오해하지 않으려면 실제 이슈 사례를 같이 보는 습관이 중요합니다.",
      rows: qualityQuery.data?.issueRows ?? [],
    },
    {
      id: "category-examples",
      type: "table",
      title: "대표 상품 사례",
      description: "현재 필터 기준으로 분석에 포함된 실제 상품 사례를 확인합니다.",
      section: "examples",
      takeaway: "추상적인 카테고리 집계가 실제 상품 수준에서 어떤 모습인지 연결해 보는 마지막 검증 단계입니다.",
      rows: examplesQuery.data?.rows ?? [],
      highlightKey: "categoryLabel",
      highlightValue: selectedCategoryLabel,
    },
  ];

  return (
    <PageContainer
      title="카테고리"
      description="품목별 점유율과 순위가 어떻게 연결되는지 먼저 보고, 속성·시간 흐름·품질 기준까지 이어서 해석합니다."
    >
      <section className="overview-story-hero">
        <small>CATEGORY</small>
        <h2>품목과 순위는 어떻게 연결되나</h2>
        <p>
          {targetSummary} 데이터에서 품목별 점유율과 평균 순위를 같이 놓고, <strong>많이 보이는 품목이 실제로도 상위권인지</strong>부터 확인합니다.
        </p>
        <p>
          이후 속성 관계와 시계열까지 이어 보면, 왜 특정 품목이 올라오거나 밀리는지 해석 근거를 더 분명히 잡을 수 있습니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>핵심 관계</span>
          <strong>품목과 순위의 거리</strong>
          <small>점유율·평균 순위·리더보드로 품목별 위치를 먼저 잡습니다.</small>
        </article>
        <article className="overview-story-card">
          <span>원인 단서</span>
          <strong>속성과 가격대 결합</strong>
          <small>품목이 어떤 가격대·소재·색상 조합에서 강한지 비교합니다.</small>
        </article>
        <article className="overview-story-card">
          <span>검증 흐름</span>
          <strong>시간과 품질 체크</strong>
          <small>시계열·품질 이슈·대표 사례를 함께 보며 해석이 맞는지 점검합니다.</small>
        </article>
      </section>

      <section className="category-page-toolbar" aria-label="필터 요약">
        <p className="category-page-toolbar__scope">{describeDashboardFilterScope(filters)}</p>
        <p className="category-page-toolbar__selection">
          선택 카테고리: <strong>{selectedCategoryLabel ?? "전체"}</strong>
        </p>
        <div className="legend-filter__controls">
          <label>
            카테고리 레벨
            <select value={categoryLevel} onChange={(event) => setCategoryLevel(event.target.value as typeof categoryLevel)}>
              <option value="l1">L1</option>
              <option value="l2">L2</option>
              <option value="l3">L3</option>
            </select>
          </label>
          <label>
            품질 모드
            <select value={qualityMode} onChange={(event) => setQualityMode(event.target.value as typeof qualityMode)}>
              <option value="success_only">성공만</option>
              <option value="success_partial">성공+부분 성공</option>
            </select>
          </label>
          <label>
            관계 축
            <select value={relationshipAxis} onChange={(event) => setRelationshipAxis(event.target.value as typeof relationshipAxis)}>
              <option value="price">가격대</option>
              <option value="material">소재</option>
              <option value="color">색상</option>
            </select>
          </label>
          <label>
            시계열 지표
            <select value={timeseriesMetric} onChange={(event) => setTimeseriesMetric(event.target.value as typeof timeseriesMetric)}>
              <option value="share">점유율</option>
              <option value="rank">평균 순위</option>
              <option value="momentum">평균 모멘텀</option>
            </select>
          </label>
          <label>
            상위 기준
            <select value={leaderMetric} onChange={(event) => setLeaderMetric(event.target.value as typeof leaderMetric)}>
              <option value="shareOfCatalog">점유율</option>
              <option value="avgRank">평균 순위</option>
              <option value="avgMomentumScore">평균 모멘텀</option>
            </select>
          </label>
          <label>
            보완 분류 포함
            <input type="checkbox" checked={includeFallback} onChange={(event) => setIncludeFallback(event.target.checked)} />
          </label>
          <button
            type="button"
            className="ghost-button"
            onClick={() => setSelectedCategoryLabel(null)}
            disabled={!selectedCategoryLabel}
          >
            선택 초기화
          </button>
        </div>
      </section>
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
