import { useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router-dom";

import { PageContainer } from "../components/PageContainer";
import { apiGet } from "../lib/api";
import { pickNumber, pickString } from "../lib/coreAccessors";
import { describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type {
  MomentumDistributionResponse,
  MomentumInputsResponse,
  RankTrajectoryPoint,
  RankTrajectoriesResponse,
  WidgetConfig,
} from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

const PRICE_BAND_ORDER = ["~3만", "3-7만", "7-12만", "12-20만", "20-50만", "50만+", "미분류"];
const MOMENTUM_TOPK_OPTIONS = [5, 10, 20, 30];
const MOMENTUM_EVENT_LEGEND = [
  { key: "첫 관측", label: "첫 관측", group: "이벤트", description: "선택 기간 안에서 처음 관측된 상품" },
  { key: "순위권 진입", label: "순위권 진입", group: "이벤트", description: "전 시점 순위권 밖에서 새로 진입한 급등 후보" },
  { key: "순위권 탈락", label: "순위권 탈락", group: "이벤트", description: "직전 시점에는 있었지만 현재 순위 정보가 사라진 상품" },
  { key: "순위권 밖", label: "순위권 밖", group: "상태", description: "이미 탈락한 뒤 현재도 순위권에 보이지 않는 상품" },
  { key: "가속 상승", label: "가속 상승", group: "상승", description: "에너지 속도와 가속도가 동시에 양수인 breakout 후보" },
  { key: "지속 상승", label: "지속 상승", group: "상승", description: "최근 상승 비율이 높은 안정적 상승 흐름" },
  { key: "상승 둔화", label: "상승 둔화", group: "둔화", description: "아직 오르지만 가속도가 꺾이는 구간" },
  { key: "하락 전환", label: "하락 전환", group: "하락", description: "에너지 속도와 가속도가 모두 음수인 경고 구간" },
  { key: "정체", label: "정체", group: "중립", description: "움직임이 약하거나 방향성이 뚜렷하지 않은 상태" },
];

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

type MomentumExampleRow = {
  productId: string;
  brand?: string | null;
  name?: string | null;
  rank?: number | null;
  standardScore?: number | null;
  rankEnergy?: number | null;
  rankVelocity?: number | null;
  rankAcceleration?: number | null;
  energyVelocity?: number | null;
  energyAcceleration?: number | null;
  entryScore?: number | null;
  actionScore?: number | null;
  observationCount?: number | null;
  presenceRatio?: number | null;
  cumulativeRankEnergy?: number | null;
  avgRankEnergy?: number | null;
  bestRank?: number | null;
  bestRankEnergy?: number | null;
  sustainedRankEnergy?: number | null;
  momentumScore?: number | null;
  persistence?: number | null;
  eventLabel?: string | null;
  discountPct?: number | null;
  price?: number | null;
  priceBand?: string | null;
};

type MomentumSearchType = "all" | "name" | "productId" | "brand" | "rank";

export function MomentumPage() {
  const { filters } = useDashboardFilters();
  const location = useLocation();
  const [selectedSellingPriceBand, setSelectedSellingPriceBand] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(10);
  const [searchType, setSearchType] = useState<MomentumSearchType>("all");
  const [exampleSearch, setExampleSearch] = useState("");

  const inputsQuery = useQuery({
    queryKey: ["momentum-inputs", filters],
    queryFn: () => apiGet<MomentumInputsResponse>("/api/momentum/inputs", filters),
  });
  const distributionQuery = useQuery({
    queryKey: ["momentum-distribution", filters],
    queryFn: () => apiGet<MomentumDistributionResponse>("/api/momentum/distribution", filters),
  });
  const trajectoriesQuery = useQuery({
    queryKey: ["momentum-rank-trajectories", filters],
    queryFn: () => apiGet<RankTrajectoriesResponse>("/api/animation/rank-trajectories", filters, { entity_type: "product" }),
  });

  const sellingPriceBandItems = useMemo(
    () =>
      [...(distributionQuery.data?.priceBandMomentum ?? [])]
        .map((row) => String(row.priceBand ?? "미분류"))
        .sort(comparePriceBands),
    [distributionQuery.data?.priceBandMomentum],
  );

  const filteredTrajectorySeries = useMemo(() => {
    const rows = trajectoriesQuery.data?.series ?? [];
    if (!selectedSellingPriceBand) {
      return rows;
    }
    return rows.filter((row) => String(row.priceBand ?? "미분류") === selectedSellingPriceBand);
  }, [selectedSellingPriceBand, trajectoriesQuery.data?.series]);

  const filteredTopMomentumRows = useMemo(() => {
    const rawRows = distributionQuery.data?.topMomentum ?? [];
    const rows: MomentumExampleRow[] = rawRows.map((row) => ({
      productId: pickString(row, "productId", "product_id") ?? "",
      brand: pickString(row, "brand") ?? null,
      name: pickString(row, "name") ?? null,
      rank: pickNumber(row, "rank"),
      standardScore: pickNumber(row, "standardScore", "score"),
      rankEnergy: pickNumber(row, "rankEnergy", "rank_energy"),
      rankVelocity: pickNumber(row, "rankVelocity", "rank_velocity"),
      rankAcceleration: pickNumber(row, "rankAcceleration", "rank_acceleration"),
      energyVelocity: pickNumber(row, "energyVelocity", "energy_velocity"),
      energyAcceleration: pickNumber(row, "energyAcceleration", "energy_acceleration"),
      entryScore: pickNumber(row, "entryScore", "entry_score"),
      actionScore: pickNumber(row, "actionScore", "action_score"),
      observationCount: pickNumber(row, "observationCount", "observation_count"),
      presenceRatio: pickNumber(row, "presenceRatio", "presence_ratio"),
      cumulativeRankEnergy: pickNumber(row, "cumulativeRankEnergy", "cumulative_rank_energy"),
      avgRankEnergy: pickNumber(row, "avgRankEnergy", "avg_rank_energy"),
      bestRank: pickNumber(row, "bestRank", "best_rank"),
      bestRankEnergy: pickNumber(row, "bestRankEnergy", "best_rank_energy"),
      sustainedRankEnergy: pickNumber(row, "sustainedRankEnergy", "sustained_rank_energy"),
      momentumScore: pickNumber(row, "momentumScore", "momentum_score"),
      persistence: pickNumber(row, "persistence", "consistencyScore", "consistency_score"),
      eventLabel: pickString(row, "eventLabel", "momentum_event_label") ?? null,
      discountPct: pickNumber(row, "discountPct", "discount_pct"),
      price: pickNumber(row, "price"),
      priceBand: pickString(row, "priceBand", "price_band"),
    })).filter((row) => row.productId !== "");
    if (!selectedSellingPriceBand) {
      return rows;
    }
    return rows.filter((row) => String(row.priceBand ?? "미분류") === selectedSellingPriceBand);
  }, [distributionQuery.data?.topMomentum, selectedSellingPriceBand]);

  const sortedTopMomentumRows = useMemo(() => {
    const rows = [...filteredTopMomentumRows];
    rows.sort((left, right) => {
      const momentumDiff = (right.sustainedRankEnergy ?? right.actionScore ?? -Infinity) - (left.sustainedRankEnergy ?? left.actionScore ?? -Infinity);
      if (momentumDiff !== 0) {
        return momentumDiff;
      }
      return (right.observationCount ?? -Infinity) - (left.observationCount ?? -Infinity);
    });
    return rows;
  }, [filteredTopMomentumRows]);

  const topMomentumRows = useMemo(
    () => sortedTopMomentumRows.slice(0, topK),
    [sortedTopMomentumRows, topK],
  );

  const normalizedSearch = exampleSearch.trim().toLowerCase();
  const tableRows = useMemo(() => {
    const baseRows = normalizedSearch ? sortedTopMomentumRows : topMomentumRows;
    const searchedRows = normalizedSearch
      ? baseRows.filter((row) => {
        if (searchType === "name") {
          return String(row.name ?? "").toLowerCase().includes(normalizedSearch);
        }
        if (searchType === "productId") {
          return String(row.productId ?? "").toLowerCase().includes(normalizedSearch);
        }
        if (searchType === "brand") {
          return String(row.brand ?? "").toLowerCase().includes(normalizedSearch);
        }
        if (searchType === "rank") {
          return String(row.rank ?? "").toLowerCase().includes(normalizedSearch);
        }
        return [row.productId, row.name, row.brand, row.priceBand, String(row.rank ?? "")]
          .filter(Boolean)
          .join(" ")
          .toLowerCase()
          .includes(normalizedSearch);
      })
      : baseRows;
    return searchedRows.slice(0, 60);
  }, [sortedTopMomentumRows, topMomentumRows, normalizedSearch, searchType]);
  const thumbnailSearchPrefix = useMemo(() => {
    const params = new URLSearchParams(location.search);
    params.set("thumbnailMode", "point");
    params.delete("thumbnailStartMonth");
    params.delete("thumbnailStartDay");
    params.delete("thumbnailStartSnapshot");
    params.delete("thumbnailEndMonth");
    params.delete("thumbnailEndDay");
    params.delete("thumbnailEndSnapshot");
    return params;
  }, [location.search]);

  const rankTrajectoryWidget = useMemo(() => {
    const series = filteredTrajectorySeries;
    const hasData = series.length > 0;
    const rows: Record<string, unknown>[] = hasData
      ? series.map((s) => ({
          crawlDatetime: s.crawlDatetime ?? s.snapshotId,
          rank: s.rank ?? 0,
          entityId: s.entityId,
          entityLabel: s.entityLabel,
          name: s.entityLabel,
          brand: s.brand ?? "-",
          rankEnergy: s.rankEnergy ?? null,
          energyVelocity: s.energyVelocity ?? s.momentumScore ?? null,
          energyAcceleration: s.energyAcceleration ?? null,
          persistence: s.persistence ?? null,
          eventLabel: s.eventLabel ?? "-",
          priceBand: s.priceBand ?? "미분류",
          estimatedOriginalPriceBand: s.estimatedOriginalPriceBand ?? "미분류",
        }))
      : [];
    const byEntity = new Map<string, RankTrajectoryPoint[]>();
    series.forEach((point) => {
      const list = byEntity.get(point.entityId) ?? [];
      list.push(point);
      byEntity.set(point.entityId, list);
    });
    const byEntityWithMeta = Array.from(byEntity.entries()).map(([entityId, points]) => {
      const sorted = [...points].sort((a, b) => (b.crawlDatetime ?? "").localeCompare(a.crawlDatetime ?? ""));
      const latest = sorted[0];
      const momentum = latest?.momentumScore ?? -Infinity;
      return {
        entityId,
        entryCount: points.length,
        rank: latest?.rank ?? 9999,
        label: latest?.entityLabel ?? entityId,
        brand: latest?.brand ?? null,
        momentumScore: momentum === -Infinity ? null : momentum,
        rankEnergy: latest?.rankEnergy ?? null,
        energyVelocity: latest?.energyVelocity ?? latest?.momentumScore ?? null,
        energyAcceleration: latest?.energyAcceleration ?? null,
        eventLabel: latest?.eventLabel ?? null,
        priceBand: latest?.priceBand ?? "미분류",
        estimatedOriginalPriceBand: latest?.estimatedOriginalPriceBand ?? "미분류",
      };
    });
    byEntityWithMeta.sort((a, b) => {
      if (b.entryCount !== a.entryCount) return b.entryCount - a.entryCount;
      if (a.rank !== b.rank) return a.rank - b.rank;
      const ma = a.momentumScore ?? -Infinity;
      const mb = b.momentumScore ?? -Infinity;
      return mb - ma;
    });
    const defaultSeriesIds = byEntityWithMeta.slice(0, 3).map((entity) => entity.entityId);
    const availableSeries = byEntityWithMeta.map((entity) => ({
      id: entity.entityId,
      label: entity.label,
      brand: entity.brand ?? undefined,
      latestRank: entity.rank,
      latestMomentum: entity.momentumScore,
      priceBand: entity.priceBand,
      estimatedOriginalPriceBand: entity.estimatedOriginalPriceBand,
    }));
    return {
      type: "rankTrajectories" as const,
      payload: {
        rows,
        baseSpec: {
          x: "crawlDatetime",
          y: "rank",
          xLabel: "시점",
          yLabel: "순위",
          yFormat: "integer" as const,
          seriesBy: "entityId",
          yAxisInverse: true,
          tooltipFields: [
            { key: "energyVelocity", label: "에너지 속도", format: "number" as const },
            { key: "energyAcceleration", label: "에너지 가속도", format: "number" as const },
            { key: "eventLabel", label: "움직임 상태", format: "string" as const },
          ],
          availableSeries,
        },
        defaultSeriesIds,
        availableSeries,
        series,
      },
    };
  }, [filteredTrajectorySeries]);

  if (inputsQuery.isLoading || distributionQuery.isLoading || trajectoriesQuery.isLoading) {
    return <div className="loading-state">모멘텀 데이터를 불러오는 중입니다.</div>;
  }

  const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
  const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
  const targetSummary = targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";
  const sharedContext = [
    { label: "판매가 기준 가격대", value: selectedSellingPriceBand ?? "전체" },
    { label: "궤적 포인트", value: `${formatNumber(filteredTrajectorySeries.length, "integer")}개` },
    { label: "대표 사례", value: `${formatNumber(sortedTopMomentumRows.length, "integer")}행` },
  ];

  const momentumAnalysisWidgets: WidgetConfig[] = [
    {
      id: "product-rank-trajectories",
      title: "시간에 따른 순위 변화",
      description:
        selectedSellingPriceBand
          ? `판매가 ${selectedSellingPriceBand} 구간만 골라, 상품마다 순위가 어떻게 움직였는지 선으로 봅니다. 기본 3개 상품이 켜져 있고 번호로 더 넣거나 뺄 수 있어요.`
          : "상품마다 순위가 시간에 따라 어떻게 움직였는지 선으로 봅니다. 기본 3개가 켜져 있고, 목록을 누르면 그 상품 선이 강조돼요.",
      section: "examples" as const,
      takeaway: "순위가 좋아지면 선은 아래로, 나빠지면 위로 갑니다. 가격 구간 칩을 고르면 그 안에서만 비교돼요.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "선 하나는 상품 하나입니다. 아래로 갈수록 순위가 좋아진 거예요." },
        ],
        drilldown: [
          { text: "가격 구간 칩을 좁힌 뒤, 아래 사례 표와 겹쳐 보면 이해가 빨라집니다." },
        ],
      },
      ...rankTrajectoryWidget,
    },
    {
      id: "price-band-momentum",
      type: "chart",
      title: "판매가 구간마다 지속 순위 에너지",
      description: "같은 판매가 구간 안에서 순위권에 얼마나 자주, 얼마나 높게 머물렀는지 누적 에너지로 봅니다.",
      section: "interpretation",
      takeaway: "막대를 누르면 순위 그래프와 사례 표가 같은 구간으로 맞춰져요.",
      chartKind: "bar",
      rows: distributionQuery.data?.priceBandMomentum ?? [],
      spec: {
        x: "priceBand",
        y: "avgMomentum",
        yLabel: "평균 지속 순위 에너지",
        yFormat: "number",
        palette: "categorical",
        highlightSeries: selectedSellingPriceBand ? [selectedSellingPriceBand] : undefined,
        xDomain: PRICE_BAND_ORDER,
        onSelectDatum: (row) => setSelectedSellingPriceBand(String(row.priceBand ?? "") || null),
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "막대 하나는 판매가 구간 하나입니다." },
        ],
        drilldown: [
          { text: "막대를 누르면 같은 구간이 순위 그래프·상위 사례에서 함께 강조됩니다." },
        ],
      },
    },
    {
      id: "energy-velocity-acceleration-map",
      type: "chart",
      title: "속도와 가속도로 보는 액션 후보",
      description: "연속 관측 상품의 에너지 속도와 가속도를 함께 놓고, 외부 원인을 추적할 우선순위를 봅니다.",
      section: "result",
      takeaway: "오른쪽 위는 지금 올라가면서 더 빨라지는 breakout 후보입니다. 오른쪽 아래는 오르지만 힘이 빠지는 구간이에요.",
      chartKind: "scatter",
      rows: sortedTopMomentumRows
        .filter((row) => row.energyVelocity != null && row.energyAcceleration != null)
        .map((row) => ({
          ...row,
          eventLabel: row.eventLabel ?? "정체",
        })),
      spec: {
        x: "energyVelocity",
        y: "energyAcceleration",
        color: "eventLabel",
        xLabel: "에너지 속도",
        yLabel: "에너지 가속도",
        xFormat: "number",
        yFormat: "number",
        palette: "semantic",
        markLines: [
          { axis: "x", value: 0, label: "속도 0", tone: "neutral" },
          { axis: "y", value: 0, label: "가속도 0", tone: "neutral" },
        ],
        quadrantHints: [
          { x: "right", y: "top", text: "상승 가속: 외부 노출 추적" },
          { x: "right", y: "bottom", text: "상승 둔화: 지속성 확인" },
          { x: "left", y: "bottom", text: "하락 가속: 리스크 점검" },
          { x: "left", y: "top", text: "하락 완화: 반등 후보" },
        ],
        customLegendTitle: "움직임 상태",
        customLegendItems: MOMENTUM_EVENT_LEGEND,
        tooltipFields: [
          { key: "rankEnergy", label: "순위 에너지", format: "number" },
          { key: "momentumScore", label: "모멘텀 점수", format: "number" },
          { key: "persistence", label: "지속성", format: "number" },
        ],
      },
    },
    {
      id: "momentum-band-distribution",
      type: "chart",
      title: "에너지 속도가 어디에 몰리나",
      description: "첫 관측·재진입·탈락 이벤트를 제외한 연속 관측 모멘텀 점수가 어떻게 퍼져 있는지 봅니다.",
      section: "result",
      takeaway: "오른쪽 꼬리는 순위권 안에서 성장 에너지가 커진 상품군, 왼쪽 꼬리는 빠르게 식는 상품군입니다.",
      chartKind: "bar",
      rows: distributionQuery.data?.momentumBandDistribution ?? [],
      spec: { x: "momentumBand", y: "count", yLabel: "상품 수", yFormat: "integer", palette: "categorical" },
    },
    {
      id: "momentum-event-state-distribution",
      type: "chart",
      title: "움직임 상태별 상품 수",
      description: "첫 관측, 순위권 진입·탈락, 가속 상승처럼 바로 액션으로 이어지는 상태로 모멘텀을 묶어 봅니다.",
      section: "interpretation",
      takeaway: "순위권 진입은 별도 진입 강도로, 순위권 탈락은 첫 탈락 이벤트로만 잡아 반복 감점하지 않습니다.",
      chartKind: "bar",
      rows: distributionQuery.data?.eventStateDistribution ?? [],
      spec: {
        x: "eventLabel",
        y: "count",
        yLabel: "상품 수",
        yFormat: "integer",
        palette: "categorical",
        xDomain: MOMENTUM_EVENT_LEGEND.map((item) => item.label),
      },
    },
    {
      id: "brand-momentum",
      type: "chart",
      title: "브랜드마다 지속 순위 에너지",
      description: "브랜드 단위로 순위권 체류 빈도와 순위 에너지를 함께 반영해, 어디가 전반적으로 강한지 봅니다.",
      section: "interpretation",
      takeaway: "단품 하나보다는 ‘그 브랜드 전체가 밀고 있는지’를 보는 데 맞춰져 있어요.",
      chartKind: "bar",
      rows: distributionQuery.data?.brandMomentum ?? [],
      spec: {
        x: "brand",
        y: "avgMomentum",
        yLabel: "평균 지속 순위 에너지",
        yFormat: "number",
        palette: "brand",
      },
    },
  ];

  const momentumInputWidgets: WidgetConfig[] = [
    {
      id: "energy-velocity-distribution",
      type: "chart",
      title: "에너지 속도가 얼마나 큰가",
      description: "첫 관측·재진입을 제외하고, 연속 관측된 상품의 순위 에너지가 직전 시점 대비 얼마나 움직였는지 봅니다.",
      section: "input",
      takeaway: "양수 꼬리가 길수록 순위권 안에서 실제 성장 에너지가 커진 상품이 많다는 뜻에 가깝습니다.",
      chartKind: "bar",
      rows: inputsQuery.data?.energyVelocityDistribution ?? inputsQuery.data?.rankVelocityDistribution ?? [],
      spec: { x: "band", y: "count", yLabel: "관측 건수", yFormat: "integer", palette: "categorical" },
    },
    {
      id: "energy-acceleration-distribution",
      type: "chart",
      title: "에너지 상승이 더 빨라지나, 둔해지나",
      description: "연속 관측 구간에서 에너지 속도가 이번에 더 붙는지(가속) 느슨해지는지 분포로 봅니다.",
      section: "input",
      takeaway: "오른쪽 꼬리는 외부 마케팅 요인을 역추적할 우선 후보입니다.",
      chartKind: "bar",
      rows: inputsQuery.data?.energyAccelerationDistribution ?? inputsQuery.data?.rankAccelerationDistribution ?? [],
      spec: { x: "band", y: "count", yLabel: "관측 건수", yFormat: "integer", palette: "categorical" },
    },
    {
      id: "stability-distribution",
      type: "chart",
      title: "오름이 잠깐인지, 좀 이어지나",
      description: "최근 에너지 속도가 양수였던 비율로, 상승이 우연 한 번인지 조금 더 버티는 패턴인지 가늠합니다.",
      section: "input",
      takeaway: "점수만 크고 안정성이 낮으면, 짧게 튀었다가 사라질 수 있는 움직임일 수 있어요.",
      chartKind: "bar",
      rows: inputsQuery.data?.stabilityDistribution ?? [],
      spec: { x: "band", y: "count", yLabel: "관측 건수", yFormat: "integer", palette: "categorical" },
    },
  ];

  return (
    <PageContainer
      title="모멘텀"
      description="순위가 얼마나 빨리 움직이는지, 가격 구간·브랜드별로 어디가 뜨거운지, 상품 궤적으로 어떤 제품이 올라오고 꺾이는지 봅니다."
    >
      <section className="overview-story-hero">
        <small>TRACTION</small>
        <h2>어떤 제품이 올라오고 꺾이나</h2>
        <p>
          {targetSummary} 데이터에서 <strong>순위가 아니라 순위의 움직임</strong>을 봅니다. 지금 잘 보이는 상품만이 아니라,
          갑자기 치고 올라오거나 기세가 꺾이는 흐름을 절대 에너지 기준으로 짚는 탭이에요.
        </p>
        <p>
          아래는 <strong>판매가 구간</strong>을 기준으로 모아 둔 값이에요. 위쪽 칩으로 구간을 고르면 궤적·표·막대가 같이
          맞춰집니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>측정·밑그림</span>
          <strong>에너지 속도와 가속</strong>
          <small>Rank Energy·속도·가속·지속성 막대와 계산식으로, 점수에 무엇이 들어가는지 먼저 짚습니다.</small>
        </article>
        <article className="overview-story-card">
          <span>묶음·비교</span>
          <strong>가격대와 브랜드</strong>
          <small>판매가 구간·에너지 속도 분포·브랜드 막대 순으로, 세그먼트가 어떻게 갈리는지 비교합니다.</small>
        </article>
        <article className="overview-story-card">
          <span>이름·흐름</span>
          <strong>상품 궤적</strong>
          <small>순위 선 그래프와 상위 사례 표로, 실제로 어떤 상품이 움직였는지 확인합니다.</small>
        </article>
      </section>

      <section className="momentum-page-toolbar" aria-label="필터 요약">
        <p className="momentum-page-toolbar__scope">{describeDashboardFilterScope(filters)}</p>
        <p className="momentum-page-toolbar__selection">
          선택 판매가 구간: <strong>{selectedSellingPriceBand ?? "전체"}</strong>
        </p>
        <div className="legend-filter__chips">
          <button
            type="button"
            className={`legend-filter__chip${selectedSellingPriceBand === null ? " is-active" : ""}`}
            onClick={() => setSelectedSellingPriceBand(null)}
          >
            전체 판매가 가격대
          </button>
          {sellingPriceBandItems.map((priceBand) => (
            <button
              key={priceBand}
              type="button"
              className={`legend-filter__chip${selectedSellingPriceBand === priceBand ? " is-active" : ""}`}
              onClick={() => setSelectedSellingPriceBand((current) => (current === priceBand ? null : priceBand))}
            >
              {priceBand}
            </button>
          ))}
        </div>
      </section>

      <WidgetRenderer widgets={momentumAnalysisWidgets} />

      <section className="section-card momentum-formula-card">
        <header className="section-card__header">
          <span className="section-card__eyebrow">계산식</span>
          <h2>모멘텀을 계산하는 법</h2>
          <p>z-score 표준화 없이, 연속 관측 모멘텀과 순위권 진입·탈락 이벤트를 분리해 해석합니다.</p>
        </header>
        <div className="momentum-formula-card__equation">
          <code>momentum_score = rank_energy(t) - rank_energy(t-1) only if observed(t) and observed(t-1); entry_score = rank_energy(t); exit_score = rank_energy(t-1)</code>
        </div>
        <ul className="section-card__note-list">
          <li>
            <strong>연속 관측만 모멘텀</strong>
            <span>전 시점과 현재 시점에 모두 순위권에 있는 상품만 energy_velocity와 momentum_score를 계산합니다.</span>
          </li>
          <li>
            <strong>진입은 별도 이벤트</strong>
            <span>첫 관측과 순위권 재진입은 momentum_score에 섞지 않고 entry_score와 이벤트 상태로 분리합니다.</span>
          </li>
          <li>
            <strong>제품 단위는 누적 에너지</strong>
            <span>상위 사례는 단일 시점 rank_energy가 아니라 누적 rank_energy와 관측 횟수를 반영한 sustained_rank_energy로 정렬합니다.</span>
          </li>
          <li>
            <strong>탈락은 1회만 기록</strong>
            <span>순위 정보가 사라지는 첫 시점은 chart_out_drop으로 잡고, 이후 순위권 밖 체류는 out_of_chart 상태로만 봅니다.</span>
          </li>
          <li>
            <strong>지속성 보조 판단</strong>
            <span>persistence는 최근 연속 관측 구간 중 에너지 속도가 양수인 비율로, 단발 급등과 꾸준한 상승을 구분합니다.</span>
          </li>
        </ul>
      </section>
      <section className="section-card momentum-example-card">
        <header className="section-card__header">
          <span className="section-card__eyebrow">사례</span>
          <h2>상품의 모멘텀 검색</h2>
          <p>검색으로 이름·번호·브랜드를 찾고, 필요하면 섬네일 탭에서 이미지까지 이어서 보세요.</p>
        </header>
        <div className="momentum-example-card__toolbar">
          <label>
            상위 개수
            <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {MOMENTUM_TOPK_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  상위 {option}개
                </option>
              ))}
            </select>
          </label>
          <label className="momentum-example-card__search-inline">
            검색 타입
            <select value={searchType} onChange={(event) => setSearchType(event.target.value as MomentumSearchType)}>
              <option value="all">전체</option>
              <option value="name">제품명</option>
              <option value="productId">상품 번호</option>
              <option value="brand">브랜드</option>
              <option value="rank">순위</option>
            </select>
          </label>
          <label className="momentum-example-card__search-inline">
            검색어
            <input
              type="search"
              value={exampleSearch}
              placeholder={
                searchType === "name"
                  ? "제품명으로 검색"
                  : searchType === "productId"
                    ? "상품 번호로 검색"
                    : searchType === "brand"
                      ? "브랜드로 검색"
                      : searchType === "rank"
                        ? "순위(숫자)로 검색"
                        : "제품명, 상품 번호, 브랜드, 순위 검색"
              }
              onChange={(event) => setExampleSearch(event.target.value)}
            />
          </label>
          <span>
            검색 결과 {formatNumber(tableRows.length, "integer")}건
            {normalizedSearch ? ` (전체 ${formatNumber(sortedTopMomentumRows.length, "integer")}건 중)` : ""}
          </span>
        </div>
        {tableRows.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>순번</th>
                  <th>상품</th>
                  <th>브랜드</th>
                  <th>판매가 가격대</th>
                  <th>순위</th>
                  <th>sustained_rank_energy</th>
                  <th>관측 수</th>
                  <th>누적 rank_energy</th>
                  <th>energy_velocity</th>
                  <th>energy_acceleration</th>
                  <th>entry_score</th>
                  <th>상태</th>
                  <th>상세</th>
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row, index) => {
                  const rankOrder = normalizedSearch
                    ? sortedTopMomentumRows.findIndex((candidate) => candidate.productId === row.productId) + 1
                    : index + 1;
                  const linkParams = new URLSearchParams(thumbnailSearchPrefix);
                  linkParams.set("thumbnailProductId", row.productId);
                  linkParams.set("thumbnailOpen", "1");
                  return (
                    <tr key={row.productId}>
                      <td>{formatNumber(rankOrder, "integer")}</td>
                      <td>
                        <strong>{row.name ?? "-"}</strong>
                        <div className="momentum-example-card__subtext">{row.productId}</div>
                      </td>
                      <td>{row.brand ?? "-"}</td>
                      <td>{row.priceBand ?? "미분류"}</td>
                      <td>{formatNumber(row.rank, "integer")}</td>
                      <td>{formatNumber(row.sustainedRankEnergy ?? row.actionScore, "number")}</td>
                      <td>{formatNumber(row.observationCount, "integer")}</td>
                      <td>{formatNumber(row.cumulativeRankEnergy, "number")}</td>
                      <td>{formatNumber(row.energyVelocity ?? row.momentumScore, "number")}</td>
                      <td>{formatNumber(row.energyAcceleration, "number")}</td>
                      <td>{formatNumber(row.entryScore, "number")}</td>
                      <td>{row.eventLabel ?? "-"}</td>
                      <td>
                        <Link
                          to={`/thumbnails?${linkParams.toString()}`}
                          className="momentum-example-card__link"
                          aria-label="썸네일 상세 보기"
                          title="썸네일 상세 보기"
                        >
                          ↗
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">검색 조건에 맞는 사례가 없습니다.</div>
        )}
      </section>
      <WidgetRenderer widgets={momentumInputWidgets} />
    </PageContainer>
  );
}
