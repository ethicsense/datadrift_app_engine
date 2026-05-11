import { useEffect, useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { apiGet } from "../lib/api";
import { pickString } from "../lib/coreAccessors";
import { describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type {
  DiscountDrilldownResponse,
  DiscountEffectsResponse,
  PriceDistributionResponse,
  PriceTimeseriesResponse,
  WidgetConfig,
} from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

const PRICE_BAND_ORDER = ["~3만", "3-7만", "7-12만", "12-20만", "20-50만", "50만+"];

function comparePriceBands(left: string, right: string) {
  const leftIndex = PRICE_BAND_ORDER.indexOf(left);
  const rightIndex = PRICE_BAND_ORDER.indexOf(right);
  const normalizedLeft = leftIndex >= 0 ? leftIndex : Number.MAX_SAFE_INTEGER;
  const normalizedRight = rightIndex >= 0 ? rightIndex : Number.MAX_SAFE_INTEGER;
  if (normalizedLeft !== normalizedRight) {
    return normalizedLeft - normalizedRight;
  }
  return left.localeCompare(right, "ko");
}

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

function toNumber(value: unknown) {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function PricePage() {
  const { filters } = useDashboardFilters();
  const [selectedPriceBand, setSelectedPriceBand] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [availableCategories, setAvailableCategories] = useState<string[]>([]);
  const [distributionSort, setDistributionSort] = useState<"priceBand" | "recordCount" | "avgRank" | "avgDiscountPct">("priceBand");
  const [performanceSort, setPerformanceSort] = useState<"avgRank" | "priceBand" | "avgDiscountPct" | "avgMomentumScore">("avgRank");
  const [topRankedAggregation, setTopRankedAggregation] = useState<"productCount" | "brandCount">("productCount");
  const [selectedDiscountProductId, setSelectedDiscountProductId] = useState<string | null>(null);
  const distributionQuery = useQuery({
    queryKey: ["price-distribution", filters, selectedCategory],
    queryFn: () =>
      apiGet<PriceDistributionResponse>("/api/price/distribution", filters, {
        category: selectedCategory ?? undefined,
      }),
  });
  const timeseriesQuery = useQuery({
    queryKey: ["price-timeseries", filters, selectedCategory],
    queryFn: () =>
      apiGet<PriceTimeseriesResponse>("/api/price/timeseries", filters, {
        category: selectedCategory ?? undefined,
      }),
  });
  const discountEffectsQuery = useQuery({
    queryKey: ["price-discount-effects", filters],
    queryFn: () => apiGet<DiscountEffectsResponse>("/api/price/discount-effects", filters),
  });
  const drilldownQuery = useQuery({
    queryKey: ["price-discount-drilldown", filters, selectedDiscountProductId],
    queryFn: () =>
      apiGet<DiscountDrilldownResponse>("/api/price/discount-drilldown", filters, {
        product_id: selectedDiscountProductId ?? "",
      }),
    enabled: Boolean(selectedDiscountProductId),
  });
  useEffect(() => {
    const nextCategories = (distributionQuery.data?.priceBandCategoryHeatmap ?? [])
      .map((row) => pickString(row, "nameItem", "categoryLabel", "category_label") ?? "")
      .filter((name) => name && name.toLowerCase() !== "unknown");
    if (!nextCategories.length) {
      return;
    }
    setAvailableCategories((prev) =>
      Array.from(new Set([...prev, ...nextCategories])).sort((left, right) => left.localeCompare(right, "ko")),
    );
  }, [distributionQuery.data?.priceBandCategoryHeatmap]);
  const categoryOptions = useMemo(
    () => [...availableCategories].sort((left, right) => left.localeCompare(right, "ko")),
    [availableCategories],
  );
  const priceBandLegendItems = useMemo(
    () =>
      [...(distributionQuery.data?.priceBandDistribution ?? [])]
        .map((row) => String(row.priceBand ?? ""))
        .filter((value) => value && value !== "미분류")
        .sort(comparePriceBands),
    [distributionQuery.data?.priceBandDistribution],
  );
  const sortedPriceBandDistribution = useMemo(() => {
    const rows = [...(distributionQuery.data?.priceBandDistribution ?? [])];
    rows.sort((left, right) => {
      if (distributionSort === "recordCount") {
        return (toNumber(right.recordCount) ?? -Infinity) - (toNumber(left.recordCount) ?? -Infinity);
      }
      if (distributionSort === "avgRank") {
        return (toNumber(left.avgRank) ?? Number.MAX_SAFE_INTEGER) - (toNumber(right.avgRank) ?? Number.MAX_SAFE_INTEGER);
      }
      if (distributionSort === "avgDiscountPct") {
        return (toNumber(right.avgDiscountPct) ?? -Infinity) - (toNumber(left.avgDiscountPct) ?? -Infinity);
      }
      return comparePriceBands(String(left.priceBand ?? ""), String(right.priceBand ?? ""));
    });
    return rows;
  }, [distributionQuery.data?.priceBandDistribution, distributionSort]);
  const sortedPriceBandPerformance = useMemo(() => {
    const rows = [...(distributionQuery.data?.priceBandPerformance ?? [])];
    rows.sort((left, right) => {
      if (performanceSort === "priceBand") {
        return comparePriceBands(String(left.priceBand ?? ""), String(right.priceBand ?? ""));
      }
      if (performanceSort === "avgDiscountPct") {
        return (toNumber(right.avgDiscountPct) ?? -Infinity) - (toNumber(left.avgDiscountPct) ?? -Infinity);
      }
      if (performanceSort === "avgMomentumScore") {
        return (toNumber(right.avgMomentumScore) ?? -Infinity) - (toNumber(left.avgMomentumScore) ?? -Infinity);
      }
      return (toNumber(left.avgRank) ?? Number.MAX_SAFE_INTEGER) - (toNumber(right.avgRank) ?? Number.MAX_SAFE_INTEGER);
    });
    return rows;
  }, [distributionQuery.data?.priceBandPerformance, performanceSort]);
  const topRankedPriceBandDistribution = useMemo(() => {
    const rows = timeseriesQuery.data?.topRankedProducts ?? [];
    const grouped = new Map<string, { productCount: number; brands: Set<string> }>();
    rows.forEach((row) => {
      const band = String(row.originalPriceBand ?? "미분류");
      const current = grouped.get(band) ?? { productCount: 0, brands: new Set<string>() };
      current.productCount += 1;
      const brand = String(row.brand ?? "");
      if (brand) {
        current.brands.add(brand);
      }
      grouped.set(band, current);
    });
    return Array.from(grouped.entries())
      .map(([priceBand, value]) => ({
        priceBand,
        productCount: value.productCount,
        brandCount: value.brands.size,
      }))
      .sort((left, right) => comparePriceBands(left.priceBand, right.priceBand));
  }, [timeseriesQuery.data?.topRankedProducts]);
  const filteredCategoryHeatmapRows = useMemo(() => {
    const rows = (distributionQuery.data?.priceBandCategoryHeatmap ?? []).filter(
      (row) => String(row.nameItem ?? "").trim().toLowerCase() !== "unknown",
    );
    const categoryScopedRows = selectedCategory
      ? rows.filter((row) => String(row.nameItem ?? "") === selectedCategory)
      : rows;
    const scopedRows = selectedPriceBand
      ? categoryScopedRows.filter((row) => String(row.priceBand ?? "") === selectedPriceBand)
      : categoryScopedRows;
    const categoryOrder = Array.from(
      scopedRows.reduce((map, row) => {
        const key = String(row.nameItem ?? "unknown");
        map.set(key, (map.get(key) ?? 0) + (toNumber(row.count) ?? 0));
        return map;
      }, new Map<string, number>()).entries(),
    )
      .sort((left, right) => right[1] - left[1])
      .map(([key]) => key);
    return [...scopedRows].sort((left, right) => {
      const categoryDiff = categoryOrder.indexOf(String(left.nameItem ?? "unknown")) - categoryOrder.indexOf(String(right.nameItem ?? "unknown"));
      if (categoryDiff !== 0) {
        return categoryDiff;
      }
      return comparePriceBands(String(left.priceBand ?? ""), String(right.priceBand ?? ""));
    });
  }, [distributionQuery.data?.priceBandCategoryHeatmap, selectedCategory, selectedPriceBand]);
  const categoryHeatmapYDomain = useMemo(
    () => Array.from(new Set(filteredCategoryHeatmapRows.map((row) => String(row.nameItem ?? "unknown")))),
    [filteredCategoryHeatmapRows],
  );
  const priceBandProjectionRows = useMemo(
    () =>
      sortedPriceBandPerformance.map((row) => ({
        ...row,
        name: `${String(row.priceBand ?? "미분류")} 가격대`,
        brand: "가격대 집계",
        rank: row.avgRank,
        price: row.recordCount,
        discountPct: row.avgDiscountPct,
      })),
    [sortedPriceBandPerformance],
  );
  const discountEffectsSummary = discountEffectsQuery.data?.summary;
  const discountEffectsEvents = discountEffectsQuery.data?.events ?? [];
  const discountEventStudyCurves = discountEffectsQuery.data?.eventStudyCurves ?? [];
  const discountEffectsScatter = discountEffectsQuery.data?.effectScatter ?? [];
  const topDiscountEvents = useMemo(
    () =>
      discountEffectsEvents
        .filter((row) => !row.lowConfidence)
        .slice(0, 12)
        .map((row, index) => {
          const discountDelta = toNumber(row.discountDelta);
          const abnormal = toNumber(row.abnormalRankDelta);
          const rankDelta = toNumber(row.rankDelta);
          return {
            순위: index + 1,
            productId: String(row.productId ?? ""),
            상품명: String(row.name ?? "-"),
            브랜드: String(row.brand ?? "-"),
            가격대: String(row.originalPriceBand ?? "-"),
            이벤트일: String(row.eventDate ?? "-"),
            이벤트유형: String(row.eventType ?? "-"),
            "할인 변화(%p)": discountDelta !== null ? Number(discountDelta.toFixed(1)) : "-",
            "순위 개선": rankDelta !== null ? Number(rankDelta.toFixed(1)) : "-",
            "초과 효과": abnormal !== null ? Number(abnormal.toFixed(1)) : "-",
            "비교군 수": toNumber(row.controlSampleSize) ?? 0,
          } as Record<string, unknown>;
        }),
    [discountEffectsEvents],
  );
  const drilldownProduct = drilldownQuery.data?.product ?? null;
  const drilldownTimeline = drilldownQuery.data?.timeline ?? [];
  const drilldownEvents = drilldownQuery.data?.events ?? [];
  const drilldownEventMarkLines = useMemo(() => {
    if (!drilldownEvents.length) {
      return undefined;
    }
    return drilldownEvents
      .map((event) => {
        const timestamp = Date.parse(String(event.eventDate ?? ""));
        const delta = toNumber(event.discountDelta);
        if (!Number.isFinite(timestamp) || delta === null) {
          return null;
        }
        return {
          axis: "x" as const,
          value: timestamp,
          label: `${delta > 0 ? "▲" : "▼"} ${formatNumber(Math.abs(delta), "number")}p`,
          tone: (delta > 0 ? "rising" : "falling") as "rising" | "falling",
          labelPosition: "end" as const,
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
  }, [drilldownEvents]);
  const selectedEventInScatter = useMemo(
    () => discountEffectsEvents.find((event) => String(event.productId ?? "") === selectedDiscountProductId) ?? null,
    [discountEffectsEvents, selectedDiscountProductId],
  );

  if (distributionQuery.isLoading || timeseriesQuery.isLoading) {
    return <div className="loading-state">가격 데이터를 불러오는 중입니다.</div>;
  }

  const highlightSeries = selectedPriceBand ? [selectedPriceBand] : undefined;
  const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
  const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
  const targetSummary = targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";
  const sharedContext = [
    { label: "선택 가격대", value: selectedPriceBand ?? "전체" },
    { label: "선택 카테고리", value: selectedCategory ?? "전체" },
    { label: "가격대 수", value: `${formatNumber(priceBandLegendItems.length, "integer")}개` },
    { label: "상위권 사례", value: `${formatNumber((timeseriesQuery.data?.topRankedProducts ?? []).length, "integer")}행` },
  ];

  const widgets: WidgetConfig[] = [
    {
      id: "price-summary-rows",
      type: "table",
      title: "이 화면에서 쓰는 숫자 요약",
      description: "지금 필터 안에서, 판매가·추정 정가·할인이 어떻게 잡혀 있는지 한눈에 정리합니다.",
      section: "summary",
      takeaway: "표에 나온 가격은 ‘할인까지 반영한 추정 정가’가 기준이라, 매장가만 보고 판단하면 어긋날 수 있어요.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "각 줄은 이 탭에서 자주 보게 될 요약 숫자 하나입니다." },
        ],
        caveats: [
          { text: "가격대 막대·순위 비교의 기본축은 판매가가 아니라 할인을 반영한 추정 정가 구간입니다.", tone: "warning" },
        ],
      },
      rows: distributionQuery.data?.summaryRows ?? [],
    },
    {
      id: "price-band-distribution",
      type: "chart",
      title: "어느 가격 구간에 상품이 가장 많이 모이나",
      description: "추정 정가 기준으로, 기록이 많이 쌓인 가격 구간을 막대로 보여줍니다.",
      section: "summary",
      takeaway: "많이 쌓인 구간은 ‘그 가격대에서 경쟁이 붙는 상품이 많다’는 뜻에 가깝습니다.",
      chartKind: "bar",
      rows: sortedPriceBandDistribution,
      spec: {
        x: "priceBand",
        y: "recordCount",
        yLabel: "관측 건수",
        yFormat: "integer",
        palette: "categorical",
        highlightSeries,
        xDomain: PRICE_BAND_ORDER,
        onSelectDatum: (row) => setSelectedPriceBand(String(row.priceBand ?? "") || null),
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "막대 하나는 추정 정가 구간 하나이고, 높이는 그 구간에 해당하는 기록 수입니다." },
        ],
        drilldown: [
          { text: "막대를 누르면 아래 차트에서 같은 가격 구간이 함께 강조됩니다." },
        ],
      },
    },
    {
      id: "discount-band-distribution",
      type: "chart",
      title: "할인은 대체로 어느 정도인가",
      description: "할인 폭이 어느 구간에 많이 몰려 있는지 비교합니다.",
      section: "input",
      takeaway: "중심이 어디인지, 꼬리가 길게 늘어지는지 보면 할인 운영 성격이 보입니다.",
      chartKind: "bar",
      rows: distributionQuery.data?.discountBandDistribution ?? [],
      spec: { x: "discountBand", y: "recordCount", yLabel: "관측 건수", yFormat: "integer", palette: "semantic" },
    },
    {
      id: "price-band-avg-rank",
      type: "chart",
      title: "가격 구간마다 순위는 어떤가",
      description: "구간별로 평균 순위가 얼마나 앞당겨져 있는지 비교합니다.",
      section: "interpretation",
      takeaway: "숫자가 작을수록(그래프에서는 아래쪽일수록) 그 가격 구간에서 상대적으로 잘 나가는 편입니다.",
      chartKind: "bar",
      rows: sortedPriceBandPerformance,
      spec: {
        x: "priceBand",
        y: "avgRank",
        yLabel: "평균 순위",
        yFormat: "number",
        yAxisInverse: true,
        palette: "categorical",
        highlightSeries,
        xDomain: PRICE_BAND_ORDER,
        onSelectDatum: (row) => setSelectedPriceBand(String(row.priceBand ?? "") || null),
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "막대 하나는 가격 구간 하나입니다." },
        ],
        interpretationRules: [
          { text: "순위는 작을수록 좋습니다. 그래프 세로축이 뒤집혀 있으니 헷갈리면 이 문구를 다시 보세요.", tone: "warning" },
        ],
      },
    },
    {
      id: "price-band-avg-momentum",
      type: "chart",
      title: "가격 구간마다 순위가 오르는 힘은",
      description: "순위가 얼마나 빨리 좋아지는지(모멘텀)를 구간별로 비교합니다.",
      section: "interpretation",
      takeaway: "순위만 보면 놓칠 수 있는 ‘반등 중인 가격대’를 짚는 데 도움이 됩니다.",
      chartKind: "bar",
      rows: sortedPriceBandPerformance,
      spec: {
        x: "priceBand",
        y: "avgMomentumScore",
        yLabel: "평균 모멘텀",
        yFormat: "number",
        palette: "categorical",
        highlightSeries,
        xDomain: PRICE_BAND_ORDER,
        onSelectDatum: (row) => setSelectedPriceBand(String(row.priceBand ?? "") || null),
      },
    },
    {
      id: "top-ranked-price-band-distribution",
      type: "chart",
      title: "잘 나가는 상품은 어느 가격대에 몰리나",
      description:
        topRankedAggregation === "productCount"
          ? "순위가 앞선 상품들이 어느 가격 구간에 많이 모여 있는지, 상품 수로 봅니다."
          : "순위가 앞선 상품들이 어느 가격 구간에 많이 모여 있는지, 브랜드 수로 봅니다.",
      section: "interpretation",
      takeaway: "여기서 두드러지는 구간은 ‘지금 인기가 실린 가격대 후보’로 삼아볼 만합니다.",
      chartKind: "bar",
      rows: topRankedPriceBandDistribution,
      spec: {
        x: "priceBand",
        y: topRankedAggregation,
        yLabel: topRankedAggregation === "productCount" ? "상위권 상품 수" : "상위권 브랜드 수",
        yFormat: "integer",
        palette: "categorical",
        highlightSeries,
        xDomain: PRICE_BAND_ORDER,
        onSelectDatum: (row) => setSelectedPriceBand(String(row.priceBand ?? "") || null),
      },
    },
    {
      id: "price-band-category-heatmap",
      type: "chart",
      title: "가격 구간마다 어떤 종류가 많이 보이나",
      description: "가격 구간과 상품 유형이 어떻게 겹치는지 색으로 보여줍니다.",
      section: "summary",
      takeaway: "같은 가격대라도 안에 들어 있는 상품 종류가 다르면, 해석이 달라질 수 있습니다.",
      chartKind: "heatmap",
      rows: filteredCategoryHeatmapRows,
      spec: {
        x: "priceBand",
        y: "nameItem",
        value: "count",
        valueFormat: "integer",
        xLabel: "정가 가격대",
        yLabel: "상품 카테고리",
        palette: "sequential",
        xDomain: selectedPriceBand ? [selectedPriceBand] : PRICE_BAND_ORDER,
        yDomain: categoryHeatmapYDomain,
        tooltipFields: [
          { key: "priceBand", label: "가격대", format: "string" },
          { key: "nameItem", label: "상품 카테고리", format: "string" },
          { key: "count", label: "상품 수", format: "integer" },
        ],
        onSelectDatum: (row) => setSelectedPriceBand(String(row.priceBand ?? "") || null),
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "칸 하나는 가격 구간 하나와 상품 유형 하나의 조합이고, 진할수록 그 조합이 자주 보였다는 뜻입니다." },
        ],
        drilldown: [
          { text: "칸을 누르면 그 가격 구간이 유지된 채로 아래 표와 다른 차트를 같이 볼 수 있습니다." },
        ],
      },
    },
    {
      id: "discount-effect-scatter",
      type: "chart",
      title: "할인 강도 vs 초과 순위 효과",
      description:
        "이벤트별로 할인 변화량(가로)과 같은 가격대의 비할인 상품 대비 ‘초과 순위 효과’(세로)를 점으로 표시합니다. 점을 누르면 아래 드릴다운에서 해당 상품의 시계열을 볼 수 있어요.",
      section: "interpretation",
      takeaway:
        "오른쪽 위에 있을수록 ‘세게 할인했고 동종 비할인 상품보다 순위가 더 좋아진’ 케이스입니다. 0 근처에 몰려 있다면 할인이 순위로 잘 이어지지 않았다는 신호.",
      chartKind: "scatter",
      rows: discountEffectsScatter,
      spec: {
        x: "discountDelta",
        y: "abnormalRankDelta",
        seriesBy: "eventType",
        xLabel: "할인 변화량(%p)",
        yLabel: "초과 순위 효과 (대비 비할인군)",
        xFormat: "number",
        yFormat: "number",
        palette: "semantic",
        markLines: [
          { axis: "x", value: 0, tone: "neutral", labelPosition: "end" },
          { axis: "y", value: 0, tone: "neutral", labelPosition: "end" },
        ],
        quadrantHints: [
          { x: "right", y: "top", text: "할인↑ + 초과 효과↑" },
          { x: "left", y: "top", text: "할인↓ + 초과 효과↑" },
          { x: "right", y: "bottom", text: "할인↑ + 효과 미미" },
          { x: "left", y: "bottom", text: "할인↓ + 순위 하락" },
        ],
        tooltipFields: [
          { key: "name", label: "상품", format: "string" },
          { key: "brand", label: "브랜드", format: "string" },
          { key: "eventDate", label: "이벤트일", format: "string" },
          { key: "originalPriceBand", label: "가격대", format: "string" },
          { key: "rankDelta", label: "원 순위 변화", format: "number" },
          { key: "controlSampleSize", label: "비교군 수", format: "integer" },
        ],
        onSelectDatum: (row) => {
          const productId = String(row.productId ?? "");
          if (!productId) {
            return;
          }
          setSelectedDiscountProductId((prev) => (prev === productId ? null : productId));
        },
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "가로축은 ‘할인이 얼마나 움직였나’, 세로축은 ‘비할인군 대비 순위가 얼마나 더 좋아졌나’입니다." },
        ],
        caveats: [
          {
            text: "비교군 수가 적은 점은 효과 추정이 흔들릴 수 있습니다. 툴팁에서 비교군 수를 확인하세요.",
            tone: "warning",
          },
        ],
      },
    },
    {
      id: "discount-top-events",
      type: "table",
      title: "할인 정책이 가장 잘 먹힌 이벤트 Top 12",
      description: discountEffectsSummary
        ? `총 ${formatNumber(discountEffectsSummary.eventCount, "integer")}개 이벤트 중 ${formatNumber(
            discountEffectsSummary.confidentEventCount,
            "integer",
          )}개가 비교군 신뢰도 충족(개선 ${formatNumber(
            discountEffectsSummary.improvedCount,
            "integer",
          )} · 하락 ${formatNumber(discountEffectsSummary.worsenedCount, "integer")} · 중립 ${formatNumber(
            discountEffectsSummary.neutralCount,
            "integer",
          )}). 개선율 ${
            discountEffectsSummary.improvementRate !== null
              ? `${formatNumber(discountEffectsSummary.improvementRate * 100, "number")}%`
              : "-"
          }.`
        : "할인 이벤트별 효과 순위표입니다.",
      section: "interpretation",
      takeaway:
        "‘초과 효과’가 큰 상위 케이스부터 살펴보면 어떤 가격대·브랜드의 할인 정책이 실제로 작동했는지 단서를 얻을 수 있어요. 행을 클릭하면 아래에서 해당 상품의 할인·순위 시계열을 봅니다.",
      rows: topDiscountEvents,
      highlightKey: "productId",
      highlightValue: selectedDiscountProductId,
      onRowSelect: (row) => {
        const productId = String(row.productId ?? "");
        if (!productId) {
          return;
        }
        setSelectedDiscountProductId((prev) => (prev === productId ? null : productId));
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "‘초과 효과’는 같은 가격대의 비할인 상품 대비 얼마나 더 순위가 좋아졌는지를 의미합니다." },
          { text: "양수가 클수록 할인이 순위 개선으로 이어졌을 가능성이 높습니다." },
        ],
        caveats: [
          {
            text: "외부 변수(광고, 시즌, 경쟁사 할인)는 통제되지 않았습니다. 인과보다는 후보 신호로 활용하세요.",
            tone: "warning",
          },
        ],
      },
    },
    ...(selectedDiscountProductId
      ? ([
          {
            id: "discount-drilldown-discount",
            type: "chart" as const,
            title: drilldownProduct
              ? `${String(drilldownProduct.brand ?? "-")} · ${String(drilldownProduct.name ?? selectedDiscountProductId)} · 할인율 추이`
              : "선택 상품의 할인율 추이",
            description: drilldownProduct
              ? `관측 ${formatNumber(toNumber(drilldownProduct.observationCount) ?? 0, "integer")}회 / 기간 ${String(
                  drilldownProduct.firstObservedAt ?? "",
                )} ~ ${String(drilldownProduct.lastObservedAt ?? "")} / 추정 정가대 ${String(
                  drilldownProduct.originalPriceBand ?? "-",
                )}`
              : "선택한 제품의 할인율 시계열입니다.",
            section: "result" as const,
            takeaway:
              "녹색(▲)/분홍(▼) 점선은 이 상품의 할인 변동 시점입니다. 같은 시점이 아래 순위 차트에도 표시되니, 변동 전후 순위가 어떻게 움직였는지 같이 비교하세요.",
            chartKind: "line" as const,
            rows: drilldownTimeline,
            spec: {
              x: "snapshotDate",
              y: "discountPct",
              xLabel: "스냅샷 날짜",
              yLabel: "할인율(%)",
              yFormat: "number",
              palette: "categorical",
              lineSmooth: false,
              lineShowSymbol: true,
              lineSymbolSize: 5,
              lineDiscreteHoldRatio: 0.85,
              timeAxisLabel: "day",
              markLines: drilldownEventMarkLines,
            },
            explainability: {
              context: sharedContext,
              readingGuide: [
                { text: "가로축은 날짜, 세로축은 할인율(%)입니다." },
                { text: "수평 구간이 길수록 같은 할인이 유지된 기간이 길었다는 뜻이에요." },
              ],
            },
          },
          {
            id: "discount-drilldown-rank",
            type: "chart" as const,
            title: drilldownProduct
              ? `${String(drilldownProduct.brand ?? "-")} · ${String(drilldownProduct.name ?? selectedDiscountProductId)} · 순위 추이`
              : "선택 상품의 순위 추이",
            description:
              "위 할인율 차트와 같은 점선이 표시됩니다. 점선 직후 선이 위로 올라가면 순위가 개선됐다는 뜻입니다.",
            section: "result" as const,
            takeaway: drilldownProduct
              ? `평균 순위 ${formatNumber(toNumber(drilldownProduct.avgRank) ?? 0, "number")} · 최고 ${formatNumber(
                  toNumber(drilldownProduct.bestRank) ?? 0,
                  "number",
                )} · 최저 ${formatNumber(toNumber(drilldownProduct.worstRank) ?? 0, "number")}`
              : "선택 상품의 시계열 순위입니다.",
            chartKind: "line" as const,
            rows: drilldownTimeline,
            spec: {
              x: "snapshotDate",
              y: "rank",
              xLabel: "스냅샷 날짜",
              yLabel: "순위",
              yFormat: "number",
              yAxisInverse: true,
              palette: "categorical",
              lineSmooth: true,
              lineShowSymbol: true,
              lineSymbolSize: 5,
              timeAxisLabel: "day",
              markLines: drilldownEventMarkLines,
            },
            explainability: {
              context: [
                ...sharedContext,
                {
                  label: "관측 횟수",
                  value:
                    drilldownProduct?.observationCount !== undefined
                      ? `${formatNumber(toNumber(drilldownProduct.observationCount) ?? 0, "integer")}회`
                      : "-",
                },
                {
                  label: "이번 산점 이벤트",
                  value: selectedEventInScatter
                    ? `${String(selectedEventInScatter.eventDate ?? "-")} (${String(selectedEventInScatter.eventType ?? "-")})`
                    : "-",
                },
              ],
              readingGuide: [
                { text: "세로축이 뒤집혀 있어 위로 올라갈수록 순위가 좋아진 것입니다." },
              ],
            },
          },
        ] as WidgetConfig[])
      : []),
  ];

  return (
    <PageContainer
      title="가격"
      description="가격 구간별로 얼마나 몰리는지, 할인과 순위까지 겹쳐서 ‘어느 대가 적당한지’ 짚어봅니다."
    >
      <section className="overview-story-hero">
        <small>PRICE</small>
        <h2>어느 가격대가 적절할까</h2>
        <p>
          {targetSummary} 데이터에서 상품이 <strong>어느 가격 구간에 많이 모이는지</strong> 보고, 그 구간이{" "}
          <strong>순위(인기)</strong>와 <strong>할인</strong>과 어떻게 겹치는지 함께 봅니다.
        </p>
        <p>
          여기서 말하는 가격은 판매가만이 아니라 할인까지 반영한 <strong>추정 정가</strong> 기준이에요. 아래에서 구간을
          골라보면 차트가 한꺼번에 맞춰집니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>분포·구성</span>
          <strong>가격·할인·상품 유형</strong>
          <small>
            가격 구간 막대, 할인 구간, 가격대×상품 유형 히트맵으로 어디에 얼마나 쌓였는지와 무엇이 겹쳐 있는지 봅니다.
          </small>
        </article>
        <article className="overview-story-card">
          <span>성과·신호</span>
          <strong>구간별 순위와 변화량</strong>
          <small>
            평균 순위·모멘텀 막대, 할인과 순위 한 화면, 상위권이 모이는 가격대까지 묶어 ‘잘 나가는 대’를 좁힙니다.
          </small>
        </article>
        <article className="overview-story-card">
          <span>흐름·기준</span>
          <strong>시간축 변화</strong>
          <small>할인·순위 시계열과 맨 위 숫자 요약으로 변화와 전제를 같이 확인합니다.</small>
        </article>
      </section>

      <section className="price-page-toolbar" aria-label="필터 요약">
        <p className="price-page-toolbar__scope">{describeDashboardFilterScope(filters)}</p>
        <p className="price-page-toolbar__selection">
          선택 가격 구간: <strong>{selectedPriceBand ?? "전체"}</strong>
          {" · "}
          상품 유형 필터: <strong>{selectedCategory ?? "전체"}</strong>
        </p>
      </section>

      <section className="legend-filter">
        <strong>관측 기준 및 정렬</strong>
        <div className="legend-filter__controls">
          <label>
            카테고리 관측
            <select
              value={selectedCategory ?? ""}
              onChange={(event) => setSelectedCategory(event.target.value || null)}
            >
              <option value="">전체 카테고리</option>
              {categoryOptions.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </label>
          <label>
            가격대 분포 정렬
            <select value={distributionSort} onChange={(event) => setDistributionSort(event.target.value as typeof distributionSort)}>
              <option value="priceBand">가격대 순</option>
              <option value="recordCount">관측 수 순</option>
              <option value="avgRank">평균 순위 순</option>
              <option value="avgDiscountPct">평균 할인율 순</option>
            </select>
          </label>
          <label>
            성과 차트 정렬
            <select value={performanceSort} onChange={(event) => setPerformanceSort(event.target.value as typeof performanceSort)}>
              <option value="avgRank">평균 순위 순</option>
              <option value="priceBand">가격대 순</option>
              <option value="avgDiscountPct">평균 할인율 순</option>
              <option value="avgMomentumScore">평균 모멘텀 순</option>
            </select>
          </label>
          <label>
            상위권 집계 기준
            <select
              value={topRankedAggregation}
              onChange={(event) => setTopRankedAggregation(event.target.value as typeof topRankedAggregation)}
            >
              <option value="productCount">상품 수</option>
              <option value="brandCount">브랜드 수</option>
            </select>
          </label>
        </div>
        <div className="legend-filter__chips">
          <button
            type="button"
            className={`legend-filter__chip${selectedPriceBand === null ? " is-active" : ""}`}
            onClick={() => setSelectedPriceBand(null)}
          >
            전체 보기
          </button>
          {priceBandLegendItems.map((priceBand) => (
            <button
              key={priceBand}
              type="button"
              className={`legend-filter__chip${selectedPriceBand === priceBand ? " is-active" : ""}`}
              onClick={() => setSelectedPriceBand((current) => (current === priceBand ? null : priceBand))}
            >
              {priceBand}
            </button>
          ))}
        </div>
      </section>
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
