import { memo, useMemo } from "react";

import ReactECharts from "echarts-for-react";

import { SectionCard } from "../SectionCard";
import type { ExplainabilityFact } from "../../types";
import { formatNumber } from "../../lib/formatters";

type BinRow = {
  label?: string | null;
  low?: number | null;
  high?: number | null;
  productCount?: number | null;
  cumulativeCount?: number | null;
  cumulativePercent?: number | null;
};

type Props = {
  bins: Record<string, unknown>[];
  statsRows: Record<string, unknown>[];
  sharedContext?: ExplainabilityFact[];
};

const chartTextColor = "#d4d4d8";
const chartAxisColor = "#71717a";
const chartSplitLineColor = "rgba(255,255,255,0.12)";
const chartTooltipBackground = "rgba(12,12,14,0.96)";
const chartAxisLabelFontSize = 14;
const chartTooltipFontSize = 14;
const chartBaseFontSize = 15;

const barColor = "#8fb3ff";
const barHighlightColor = "#f8b86e";
const lineColor = "#4ade80";

function toNumberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function formatMaybeInt(value: number | null): string {
  if (value == null) return "–";
  if (Number.isInteger(value)) return formatNumber(value);
  return formatNumber(Math.round(value));
}

function ReviewsPerProductDistributionPanelComponent({ bins, statsRows, sharedContext }: Props) {
  const normalized: BinRow[] = useMemo(
    () =>
      bins.map((row) => ({
        label: typeof row.label === "string" ? row.label : null,
        low: toNumberOrNull(row.low),
        high: toNumberOrNull(row.high),
        productCount: toNumberOrNull(row.productCount),
        cumulativeCount: toNumberOrNull(row.cumulativeCount),
        cumulativePercent: toNumberOrNull(row.cumulativePercent),
      })),
    [bins],
  );

  const hasData = normalized.some((row) => (row.productCount ?? 0) > 0);

  const statByMetric = useMemo(() => {
    const map = new Map<string, { value: number | null; unit: string }>();
    statsRows.forEach((row) => {
      const metric = String((row as Record<string, unknown>).metric ?? "");
      const value = toNumberOrNull((row as Record<string, unknown>).value);
      const unit = String((row as Record<string, unknown>).unit ?? "");
      map.set(metric, { value, unit });
    });
    return map;
  }, [statsRows]);

  const sourceLabel = useMemo(() => {
    for (const key of Array.from(statByMetric.keys())) {
      if (key.startsWith("전체 리뷰 수")) return "전체 리뷰 수";
      if (key.startsWith("수집 리뷰 수")) return "수집 리뷰 수";
    }
    return "전체 리뷰 수";
  }, [statByMetric]);

  const summaryChips = useMemo(
    () => [
      { label: "0건 상품", metric: `${sourceLabel} 0건 상품`, suffix: "개" },
      { label: "300건 초과", metric: `${sourceLabel} 300건 초과 상품`, suffix: "개" },
      { label: "P25", metric: "상품당 리뷰 P25", suffix: "건" },
      { label: "중앙값(P50)", metric: "상품당 리뷰 중앙값", suffix: "건", emphasize: true },
      { label: "P75", metric: "상품당 리뷰 P75", suffix: "건" },
      { label: "P90", metric: "상품당 리뷰 P90", suffix: "건" },
      { label: "P99", metric: "상품당 리뷰 P99", suffix: "건", emphasize: true },
      { label: "최대", metric: "상품당 리뷰 최대", suffix: "건", emphasize: true },
      { label: "평균", metric: "상품당 리뷰 평균", suffix: "건" },
    ],
    [sourceLabel],
  );

  const option = useMemo(() => {
    const categories = normalized.map((row) => row.label ?? "");
    const productCounts = normalized.map((row) => row.productCount ?? 0);
    const cumulativePercents = normalized.map((row) => row.cumulativePercent ?? 0);
    const maxBarValue = productCounts.reduce((acc, v) => (v > acc ? v : acc), 0);

    // 300 구간 강조 (하이라이트)
    const capBinIndex = normalized.findIndex(
      (row) => row.low === 300 && row.high === 500,
    );

    const barDataWithStyle = productCounts.map((value, idx) => ({
      value,
      itemStyle:
        idx === capBinIndex
          ? { color: barHighlightColor, borderColor: "#f59e0b", borderWidth: 1 }
          : { color: barColor },
    }));

    return {
      backgroundColor: "transparent",
      grid: { left: 64, right: 72, top: 48, bottom: 72 },
      legend: {
        top: 8,
        left: "center",
        textStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
        data: ["상품 수", "누적 비율"],
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: chartTooltipBackground,
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: chartTextColor, fontSize: chartTooltipFontSize },
        formatter: (
          params: Array<{ dataIndex: number; seriesName: string; marker: string; value: number }>,
        ) => {
          if (!params.length) return "";
          const idx = params[0].dataIndex;
          const row = normalized[idx];
          const rangeLabel =
            row?.high == null
              ? `${formatNumber(row?.low ?? 0)}건 이상`
              : `${formatNumber(row?.low ?? 0)} ~ ${formatNumber((row?.high ?? 1) - 1)}건`;
          const pc = row?.productCount ?? 0;
          const cum = row?.cumulativePercent ?? 0;
          const cumCnt = row?.cumulativeCount ?? 0;
          return [
            `<strong>${row?.label ?? ""} (${rangeLabel})</strong>`,
            `${params[0].marker} 상품 수: ${formatNumber(pc)}개`,
            `누적: ${formatNumber(cumCnt)}개 (${cum.toFixed(1)}%)`,
          ].join("<br/>");
        },
      },
      xAxis: {
        type: "category",
        data: categories,
        name: "상품당 리뷰 수 구간(로그)",
        nameGap: 36,
        nameLocation: "middle",
        nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
        axisLabel: {
          color: chartTextColor,
          fontSize: chartAxisLabelFontSize,
          rotate: 30,
          interval: 0,
        },
        axisLine: { lineStyle: { color: chartAxisColor } },
        axisTick: { alignWithLabel: true },
      },
      yAxis: [
        {
          type: "value",
          name: "상품 수",
          min: 0,
          max: Math.max(maxBarValue, 10),
          nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
          axisLabel: {
            color: chartTextColor,
            fontSize: chartAxisLabelFontSize,
            formatter: (value: number) => formatNumber(value),
          },
          axisLine: { lineStyle: { color: chartAxisColor } },
          splitLine: { lineStyle: { color: chartSplitLineColor } },
        },
        {
          type: "value",
          name: "누적 비율(%)",
          min: 0,
          max: 100,
          position: "right",
          nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
          axisLabel: {
            color: chartTextColor,
            fontSize: chartAxisLabelFontSize,
            formatter: (value: number) => `${value}%`,
          },
          axisLine: { lineStyle: { color: chartAxisColor } },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "상품 수",
          type: "bar",
          yAxisIndex: 0,
          barCategoryGap: "15%",
          data: barDataWithStyle,
          label: {
            show: true,
            position: "top",
            color: chartTextColor,
            fontSize: 12,
            formatter: (p: { value: number }) => (p.value > 0 ? formatNumber(p.value) : ""),
          },
        },
        {
          name: "누적 비율",
          type: "line",
          yAxisIndex: 1,
          smooth: true,
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { color: lineColor, width: 2 },
          itemStyle: { color: lineColor },
          data: cumulativePercents,
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#fbbf24", type: "dashed", opacity: 0.6 },
            label: {
              color: "#fbbf24",
              fontSize: 11,
              formatter: (p: { name?: string }) => p.name ?? "",
            },
            data: [
              { yAxis: 50, name: "P50 (50%)" },
              { yAxis: 90, name: "P90 (90%)" },
              { yAxis: 99, name: "P99 (99%)" },
            ],
          },
        },
      ],
    };
  }, [normalized]);

  return (
    <SectionCard
      title="상품당 전체 리뷰 수 분포 (로그 구간)"
      description="가로축은 상품별 `meta.json`의 `total_reviews_reported`(없으면 다른 소스 폴백)를 로그 스케일로 묶은 구간, 왼쪽 세로축은 해당 구간의 상품 수, 오른쪽 세로축은 누적 비율(CDF)입니다. 분포가 극단적 롱테일이라 선형축에서는 보이지 않는 저·고 구간이 함께 드러납니다."
      section="summary"
      takeaway="대부분 어느 구간에 몰려 있는지(막대)와 상위 꼬리가 어디까지 뻗는지(막대의 오른쪽 끝 + 누적 라인 기울기)를 한 번에 확인합니다."
      explainability={{
        context: sharedContext,
        readingGuide: [
          {
            text: "막대: 각 로그 구간에 속한 상품 수. 파란색은 일반 구간, 주황색은 수집 캡 300이 걸리는 [300–499) 구간을 강조한 것.",
          },
          {
            text: "녹색 라인: 왼쪽부터 누적된 상품의 비율(%). 50/90/99% 선을 넘는 지점의 x축 구간을 보면 P50·P90·P99가 어느 구간에 위치하는지 직관적으로 보입니다.",
          },
          {
            text: "툴팁에서는 구간의 실제 범위([low, high)), 해당 구간 상품 수, 좌측 누적 상품 수 및 누적 비율을 모두 제공합니다.",
          },
        ],
        caveats: [
          {
            text: "0건 구간은 리뷰가 아예 집계되지 않은 상품(미수집·비공개 포함)을 한 곳에 모은 것이라 본 분포의 일반 경향과 분리해 해석해야 합니다.",
            tone: "warning",
          },
          {
            text: "요약 칩(P25/P50/P75/P90/P99/최대 등)은 실제 리뷰 수의 백분위 값이며, 표시는 로그 구간으로 묶여 있지만 칩 숫자 자체는 원본 값 기준입니다.",
          },
        ],
      }}
    >
      <div className="review-dist-chips" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
        {summaryChips.map((chip) => {
          const entry = statByMetric.get(chip.metric);
          const value = entry?.value ?? null;
          const isCountChip = chip.suffix === "개";
          const displayValue = isCountChip
            ? formatMaybeInt(value)
            : value == null
            ? "–"
            : Number.isInteger(value)
            ? formatNumber(value)
            : formatNumber(Number(value.toFixed(1)));
          return (
            <div
              key={chip.metric}
              style={{
                padding: "6px 12px",
                borderRadius: 999,
                border: "1px solid rgba(255,255,255,0.12)",
                background: chip.emphasize ? "rgba(143,179,255,0.14)" : "rgba(255,255,255,0.04)",
                color: chartTextColor,
                fontSize: 13,
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span style={{ color: chartAxisColor }}>{chip.label}</span>
              <strong style={{ color: "#e4e4e7" }}>{displayValue}</strong>
              <span style={{ color: chartAxisColor }}>{chip.suffix}</span>
            </div>
          );
        })}
      </div>
      {hasData ? (
        <ReactECharts option={option} notMerge lazyUpdate style={{ height: 460 }} />
      ) : (
        <div className="empty-state">리뷰 분포를 계산할 데이터가 없습니다.</div>
      )}
    </SectionCard>
  );
}

export const ReviewsPerProductDistributionPanel = memo(ReviewsPerProductDistributionPanelComponent);
