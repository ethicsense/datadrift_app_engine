import { useEffect, useMemo, useState } from "react";

import ReactECharts from "echarts-for-react";

import { productImageApiUrl } from "../../lib/api";
import { computeEqualScaleScatterExtents } from "../../lib/chartScatterBounds";
import { escapeHtml } from "../../lib/escapeHtml";
import { formatDimensionValue, formatNumber } from "../../lib/formatters";
import type { ChartSeriesOption, ChartSpec } from "../../types";
import { ScatterSquareChart } from "./ScatterSquareChart";

type EChartPanelProps = {
  rows: Record<string, unknown>[];
  chartKind: "bar" | "line" | "scatter" | "heatmap" | "bump";
  spec: ChartSpec;
};

function groupBySeries(
  rows: Record<string, unknown>[],
  spec: ChartSpec,
  chartKind: "bar" | "line" | "scatter" | "heatmap" | "bump",
) {
  const groupKey =
    spec.seriesBy ??
    ((chartKind === "scatter" || chartKind === "line" || chartKind === "bump") ? spec.color : undefined);
  if (!groupKey) {
    return [{ name: spec.y, rows }];
  }
  const groups = new Map<string, Record<string, unknown>[]>();
  rows.forEach((row) => {
    const key = String(row[groupKey as keyof typeof row] ?? "기타");
    const current = groups.get(key) ?? [];
    current.push(row);
    groups.set(key, current);
  });
  return Array.from(groups.entries()).map(([name, groupRows]) => ({ name, rows: groupRows }));
}

function formatSeriesLabel(label: string, spec: ChartSpec) {
  const key = spec.seriesBy ?? spec.color ?? spec.x;
  return formatDimensionValue(label, key);
}

const categoricalPalette = [
  "#8fb3ff",
  "#6ee7b7",
  "#f6c177",
  "#f48fb1",
  "#b39ddb",
  "#ff8a80",
  "#67e8f9",
  "#cbd5e1",
  "#f9a826",
  "#7dd3a7",
  "#d8b4fe",
  "#93c5fd",
];

const semanticColorMap: Record<string, string> = {
  "할인 증가": "#4ade80",
  "유지(±5%p)": "#a1a1aa",
  "유지(±5%포인트)": "#a1a1aa",
  "할인 감소": "#fb7185",
  rising: "#4ade80",
  stable: "#a1a1aa",
  falling: "#fb7185",
  fast_riser: "#22c55e",
  fast_faller: "#ef4444",
  rank_up_fast: "#1d4ed8",
  rank_up: "#38bdf8",
  rank_unchanged: "#64748b",
  rank_down: "#f59e0b",
  rank_down_fast: "#b91c1c",
  no_prior_rank: "#8b5cf6",
  "큰 폭 순위 상승": "#1d4ed8",
  "순위 상승": "#38bdf8",
  "순위 유지": "#64748b",
  "순위 하락": "#f59e0b",
  "큰 폭 순위 하락": "#b91c1c",
  "비교 기준 없음": "#8b5cf6",
};

const chartTextColor = "#d4d4d8";
const chartAxisColor = "#71717a";
const chartSplitLineColor = "rgba(255,255,255,0.12)";
const chartTooltipBackground = "rgba(12,12,14,0.96)";
const chartBaseFontSize = 16;
const chartAxisLabelFontSize = 15;
const chartLegendFontSize = 15;
const chartTooltipFontSize = 15;
const chartAuxiliaryFontSize = 14;
const oneDayMs = 24 * 60 * 60 * 1000;

function isDateLike(value: unknown) {
  return typeof value === "string" && !Number.isNaN(Date.parse(value));
}

function toNumeric(value: unknown) {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function flattenSmallLineChanges(
  rows: Record<string, unknown>[],
  yKey: string,
  threshold?: number,
) {
  if (!threshold || threshold <= 0 || rows.length < 2) {
    return rows;
  }
  const flattened: Record<string, unknown>[] = [];
  rows.forEach((row) => {
    const currentValue = toNumeric(row[yKey as keyof typeof row]);
    const previous = flattened.at(-1);
    if (!previous || currentValue === null) {
      flattened.push(row);
      return;
    }
    const previousValue = toNumeric(previous[yKey as keyof typeof previous]);
    if (previousValue === null || Math.abs(currentValue - previousValue) > threshold) {
      flattened.push(row);
      return;
    }
    flattened.push({
      ...row,
      [yKey]: previousValue,
    });
  });
  return flattened;
}

function applyDiscreteHoldCurve(
  rows: Record<string, unknown>[],
  xKey: string,
  yKey: string,
  holdRatio?: number,
) {
  if (!holdRatio || holdRatio <= 0 || rows.length < 2) {
    return rows;
  }
  const clampedRatio = Math.min(0.98, Math.max(0, holdRatio));
  const expanded: Record<string, unknown>[] = [];
  for (let index = 0; index < rows.length; index += 1) {
    const current = rows[index];
    if (index === 0) {
      expanded.push(current);
      continue;
    }
    const previous = rows[index - 1];
    const previousTime = Date.parse(String(previous[xKey as keyof typeof previous] ?? ""));
    const currentTime = Date.parse(String(current[xKey as keyof typeof current] ?? ""));
    const previousValue = toNumeric(previous[yKey as keyof typeof previous]);
    if (
      Number.isFinite(previousTime) &&
      Number.isFinite(currentTime) &&
      currentTime > previousTime &&
      previousValue !== null
    ) {
      const holdTime = previousTime + (currentTime - previousTime) * clampedRatio;
      expanded.push({
        ...current,
        [xKey]: new Date(holdTime).toISOString(),
        [yKey]: previousValue,
      });
    }
    expanded.push(current);
  }
  return expanded;
}

function applyMovingAverage(
  rows: Record<string, unknown>[],
  yKey: string,
  windowSize?: number,
) {
  if (!windowSize || windowSize <= 1 || rows.length < 3) {
    return rows;
  }
  const normalizedWindow = Math.max(2, Math.floor(windowSize));
  const smoothed: Record<string, unknown>[] = [];
  const values = rows.map((row) => toNumeric(row[yKey as keyof typeof row]));
  rows.forEach((row, index) => {
    const start = Math.max(0, index - normalizedWindow + 1);
    let sum = 0;
    let count = 0;
    for (let cursor = start; cursor <= index; cursor += 1) {
      const value = values[cursor];
      if (value !== null) {
        sum += value;
        count += 1;
      }
    }
    if (count === 0) {
      smoothed.push(row);
      return;
    }
    smoothed.push({
      ...row,
      [yKey]: sum / count,
    });
  });
  return smoothed;
}

function formatDateLabel(value: string | number) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
}

function formatDateDayLabel(value: string | number) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function getPalette(spec: ChartSpec, groups?: string[]) {
  if (spec.palette === "semantic" && groups?.length) {
    return groups.map((group, index) => semanticColorMap[group] ?? categoricalPalette[index % categoricalPalette.length]);
  }
  return categoricalPalette;
}

function resolveCategoryColor(label: string, index: number, spec: ChartSpec) {
  if (spec.palette === "semantic") {
    return semanticColorMap[label] ?? categoricalPalette[index % categoricalPalette.length];
  }
  return categoricalPalette[index % categoricalPalette.length];
}

function parseHexColor(hex: string) {
  const normalized = hex.replace("#", "");
  const expanded = normalized.length === 3
    ? normalized.split("").map((char) => char + char).join("")
    : normalized;
  const value = Number.parseInt(expanded, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function mixColor(
  left: { r: number; g: number; b: number },
  right: { r: number; g: number; b: number },
  ratio: number,
) {
  const clamped = Math.max(0, Math.min(1, ratio));
  return {
    r: Math.round(left.r + (right.r - left.r) * clamped),
    g: Math.round(left.g + (right.g - left.g) * clamped),
    b: Math.round(left.b + (right.b - left.b) * clamped),
  };
}

function rgbToCss(color: { r: number; g: number; b: number }) {
  return `rgb(${color.r}, ${color.g}, ${color.b})`;
}

function hslToRgb(h: number, s: number, l: number) {
  const hue = ((h % 360) + 360) % 360;
  const sat = Math.max(0, Math.min(100, s)) / 100;
  const light = Math.max(0, Math.min(100, l)) / 100;
  const chroma = (1 - Math.abs(2 * light - 1)) * sat;
  const segment = hue / 60;
  const x = chroma * (1 - Math.abs((segment % 2) - 1));
  let red = 0;
  let green = 0;
  let blue = 0;

  if (segment >= 0 && segment < 1) {
    red = chroma;
    green = x;
  } else if (segment < 2) {
    red = x;
    green = chroma;
  } else if (segment < 3) {
    green = chroma;
    blue = x;
  } else if (segment < 4) {
    green = x;
    blue = chroma;
  } else if (segment < 5) {
    red = x;
    blue = chroma;
  } else {
    red = chroma;
    blue = x;
  }

  const match = light - chroma / 2;
  return {
    r: Math.round((red + match) * 255),
    g: Math.round((green + match) * 255),
    b: Math.round((blue + match) * 255),
  };
}

function hierarchyGroupKey(label: string) {
  const parts = label.split(">").map((part) => part.trim()).filter(Boolean);
  if (parts.length >= 3) {
    return `${parts[0]} > ${parts[1]}`;
  }
  if (parts.length === 2) {
    return parts[0];
  }
  return "기타";
}

function hierarchyBaseHue(parentKey: string, fallbackIndex: number) {
  const explicit: Record<string, number> = {
    "패션 > 의류": 214,
    패션: 328,
    기타: 146,
  };
  if (parentKey in explicit) {
    return explicit[parentKey];
  }
  const fallback = [214, 228, 198, 246, 206];
  return fallback[fallbackIndex % fallback.length];
}

function buildHierarchicalColorMap(groupNames: string[]) {
  const parentMap = new Map<string, string[]>();
  groupNames.forEach((name) => {
    const parentKey = hierarchyGroupKey(name);
    const current = parentMap.get(parentKey) ?? [];
    current.push(name);
    parentMap.set(parentKey, current);
  });

  const colorMap = new Map<string, string>();
  const childVariants = [
    { hueOffset: 0, saturation: 82, lightness: 68 },
    { hueOffset: -12, saturation: 90, lightness: 54 },
    { hueOffset: 14, saturation: 76, lightness: 78 },
    { hueOffset: -20, saturation: 94, lightness: 42 },
    { hueOffset: 22, saturation: 70, lightness: 60 },
    { hueOffset: -6, saturation: 64, lightness: 84 },
    { hueOffset: 28, saturation: 88, lightness: 48 },
    { hueOffset: -26, saturation: 72, lightness: 72 },
  ];

  Array.from(parentMap.entries()).forEach(([parentKey, names], parentIndex) => {
    const fallbackBase = parseHexColor(categoricalPalette[parentIndex % categoricalPalette.length]);
    const baseHue = hierarchyBaseHue(parentKey, parentIndex);
    const sortedNames = [...names].sort((left, right) => left.localeCompare(right, "ko"));
    sortedNames.forEach((name, childIndex) => {
      const variant = childVariants[childIndex % childVariants.length];
      const color = hslToRgb(baseHue + variant.hueOffset, variant.saturation, variant.lightness);
      colorMap.set(name, rgbToCss(color));
    });
    const parentColor = sortedNames.length
      ? hslToRgb(baseHue, 68, 64)
      : fallbackBase;
    colorMap.set(parentKey, rgbToCss(parentColor));
  });

  return colorMap;
}

function buildLegendColorMap(groupNames: string[], spec: ChartSpec) {
  if (spec.palette === "categorical" && groupNames.some((name) => name.includes(">"))) {
    return buildHierarchicalColorMap(groupNames);
  }
  return new Map(groupNames.map((name, index) => [name, resolveCategoryColor(name, index, spec)]));
}

function renderCustomLegendPanel(spec: ChartSpec, groupNames: string[]) {
  const items = spec.customLegendItems ?? [];
  if (!items.length) {
    return null;
  }
  const grouped = new Map<string, typeof items>();
  items.forEach((item) => {
    const key = item.group?.trim() || "기타";
    const current = grouped.get(key) ?? [];
    current.push(item);
    grouped.set(key, current);
  });
  const colorMap = buildLegendColorMap(groupNames, spec);
  return (
    <div className="chart-custom-legend">
      <div className="chart-custom-legend__header">
        <strong>{spec.customLegendTitle ?? "범례"}</strong>
        <span>{items.length}개 범주</span>
      </div>
      <div className="chart-custom-legend__groups">
        {Array.from(grouped.entries()).map(([groupLabel, groupItems]) => (
          <section key={groupLabel} className="chart-custom-legend__group">
            <h4>{groupLabel}</h4>
            <div className="chart-custom-legend__items">
              {groupItems.map((item) => (
                <div
                  key={item.key}
                  className="chart-custom-legend__item"
                  title={item.description ?? item.key}
                >
                  <span
                    className="chart-custom-legend__swatch"
                    style={{ backgroundColor: colorMap.get(item.key) ?? "#a1a1aa" }}
                  />
                  <span className="chart-custom-legend__label">{item.label}</span>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function buildMarkLines(spec: ChartSpec) {
  if (!spec.markLines?.length) {
    return undefined;
  }
  const toneColorMap: Record<"rising" | "falling" | "neutral", string> = {
    rising: "#4ade80",
    falling: "#fb7185",
    neutral: "#d4d4d8",
  };
  const toneLineColorMap: Record<"rising" | "falling" | "neutral", string> = {
    rising: "rgba(74,222,128,0.55)",
    falling: "rgba(251,113,133,0.55)",
    neutral: "rgba(255,255,255,0.24)",
  };
  return spec.markLines.map((line) => {
    const tone = line.tone ?? "neutral";
    return {
      [line.axis === "x" ? "xAxis" : "yAxis"]: line.value,
      label: {
        formatter: line.label ?? String(line.value),
        color: toneColorMap[tone],
        fontSize: chartAuxiliaryFontSize,
        fontWeight: 600,
        position: line.labelPosition ?? "end",
        distance: 6,
        backgroundColor: "rgba(12,12,14,0.85)",
        padding: [3, 6],
        borderRadius: 4,
        borderColor: toneLineColorMap[tone],
        borderWidth: 1,
      },
      lineStyle: {
        color: toneLineColorMap[tone],
        type: "dashed",
        width: 1.2,
      },
    };
  });
}

function buildQuadrantHints(spec: ChartSpec) {
  if (!spec.quadrantHints) {
    return undefined;
  }
  const hints = Array.isArray(spec.quadrantHints)
    ? spec.quadrantHints
    : [
        { x: "left" as const, y: "top" as const, text: "할인 감소 + 순위 상승" },
        { x: "right" as const, y: "top" as const, text: "할인 증가 + 순위 상승" },
        { x: "left" as const, y: "bottom" as const, text: "할인 감소 + 순위 하락" },
        { x: "right" as const, y: "bottom" as const, text: "할인 증가 + 순위 하락" },
      ];
  return hints.map((hint) => ({
    type: "text",
    [hint.x]: hint.x === "left" ? "18%" : "16%",
    [hint.y]: hint.y === "top" ? "16%" : "14%",
    style: {
      text: hint.text,
      fill: "rgba(244,244,245,0.62)",
      fontSize: chartAuxiliaryFontSize,
    },
  }));
}

function formatTooltipFieldValue(raw: Record<string, unknown>, key: string, spec: ChartSpec) {
  const field = spec.tooltipFields?.find((candidate) => candidate.key === key);
  const rawValue = raw[key];
  if ((rawValue === null || rawValue === undefined || rawValue === "") && field?.fallback) {
    return field.fallback;
  }
  return formatNumber(rawValue, field?.format ?? "auto", key);
}

function buildTooltipFieldLines(raw: Record<string, unknown>, spec: ChartSpec) {
  if (!spec.tooltipFields?.length) {
    return [];
  }
  return spec.tooltipFields.map((field) => `${field.label}: ${formatTooltipFieldValue(raw, field.key, spec)}`);
}

function buildScatterAnnotationSeries(spec: ChartSpec) {
  if (!spec.scatterAnnotations?.length) {
    return [];
  }
  return [
    {
      name: "annotation",
      type: "scatter",
      silent: false,
      tooltip: { show: false },
      symbol: "circle",
      z: 8,
      data: spec.scatterAnnotations.map((annotation) => ({
        value: [annotation.x, annotation.y],
        meta: {
          ...annotation,
          clusterId: annotation.key,
          __scatterAnnotation: true,
        },
        symbolSize: annotation.tone === "accent" ? 18 : 14,
        itemStyle: annotation.tone === "accent"
          ? {
              color: "rgba(245, 158, 11, 0.22)",
              borderColor: "rgba(251, 191, 36, 0.92)",
              borderWidth: 1.6,
            }
          : {
              color: "rgba(148, 163, 184, 0.18)",
              borderColor: "rgba(203, 213, 225, 0.78)",
              borderWidth: 1.2,
            },
        label: {
          show: true,
          position: "top",
          distance: 10,
          formatter: () => (
            annotation.subLabel
              ? `{title|${annotation.label}}\n{sub|${annotation.subLabel}}`
              : `{title|${annotation.label}}`
          ),
          rich: {
            title: {
              color: "#f4f4f5",
              fontSize: chartAuxiliaryFontSize,
              fontWeight: 700,
              lineHeight: 18,
              backgroundColor: annotation.tone === "accent" ? "rgba(120, 53, 15, 0.9)" : "rgba(24, 24, 27, 0.82)",
              borderColor: annotation.tone === "accent" ? "rgba(251, 191, 36, 0.65)" : "rgba(255,255,255,0.12)",
              borderWidth: 1,
              borderRadius: 999,
              padding: [5, 10, 5, 10],
            },
            sub: {
              color: "#a1a1aa",
              fontSize: chartAuxiliaryFontSize,
              lineHeight: 16,
              padding: [4, 2, 0, 2],
            },
          },
        },
      })),
    },
  ];
}

function buildScatterRegionSeries(spec: ChartSpec) {
  if (!spec.scatterRegions?.length) {
    return [];
  }
  return spec.scatterRegions
    .filter((region) => region.points.length >= 3)
    .map((region) => {
      const closed = [...region.points, region.points[0]];
      const accent = region.tone === "accent";
      const active = spec.scatterActiveRegionKey === region.key;
      const hasActiveRegion = Boolean(spec.scatterActiveRegionKey);
      return {
        name: region.label ?? region.key,
        type: "line",
        silent: true,
        tooltip: { show: false },
        showSymbol: false,
        z: 1,
        smooth: 0.42,
        lineStyle: accent
          ? {
              color: active
                ? "rgba(250, 204, 21, 0.82)"
                : hasActiveRegion
                ? "rgba(250, 204, 21, 0.14)"
                : "rgba(250, 204, 21, 0.34)",
              width: active ? 2.2 : 1.6,
              opacity: 1,
            }
          : {
              color: active
                ? "rgba(226, 232, 240, 0.5)"
                : hasActiveRegion
                ? "rgba(148, 163, 184, 0.08)"
                : "rgba(148, 163, 184, 0.18)",
              width: active ? 1.9 : 1.2,
              opacity: 1,
            },
        areaStyle: accent
          ? {
              color: active
                ? "rgba(250, 204, 21, 0.12)"
                : hasActiveRegion
                ? "rgba(250, 204, 21, 0.015)"
                : "rgba(250, 204, 21, 0.05)",
            }
          : {
              color: active
                ? "rgba(148, 163, 184, 0.09)"
                : hasActiveRegion
                ? "rgba(148, 163, 184, 0.01)"
                : "rgba(148, 163, 184, 0.028)",
            },
        emphasis: { disabled: true },
        data: closed,
      };
    });
}

function resolveEventRow(params: { data?: unknown; value?: unknown }) {
  if (params.data && typeof params.data === "object" && "meta" in params.data) {
    return (params.data as { meta?: Record<string, unknown> }).meta ?? null;
  }
  if (Array.isArray(params.value) && params.value.length >= 3 && typeof params.value[2] === "object" && params.value[2] !== null) {
    return params.value[2] as Record<string, unknown>;
  }
  return null;
}

function buildChartEvents(spec: ChartSpec) {
  if (!spec.onSelectDatum && !spec.onHoverDatum) {
    return undefined;
  }
  return {
    click: (params: unknown) => {
      const resolved = (params ?? {}) as { data?: unknown; value?: unknown };
      const row = resolveEventRow(resolved);
      if (row) {
        spec.onSelectDatum?.(row);
      }
    },
    mouseover: (params: unknown) => {
      const resolved = (params ?? {}) as { data?: unknown; value?: unknown };
      const row = resolveEventRow(resolved);
      spec.onHoverDatum?.(row);
    },
    mouseout: () => {
      spec.onHoverDatum?.(null);
    },
    globalout: () => {
      spec.onHoverDatum?.(null);
    },
  };
}

function buildSampledIndexSet(length: number, maxVisiblePointCount: number) {
  if (length <= maxVisiblePointCount) {
    return new Set(Array.from({ length }, (_, index) => index));
  }
  const keep = new Set<number>([0, length - 1]);
  const segments = Math.max(maxVisiblePointCount - 1, 1);
  for (let step = 1; step < segments; step += 1) {
    const index = Math.round((step * (length - 1)) / segments);
    keep.add(index);
  }
  return keep;
}

function buildBumpOption(
  rows: Record<string, unknown>[],
  spec: ChartSpec,
  groups: Array<{ name: string; rows: Record<string, unknown>[] }>,
) {
  const highlightSeries = new Set((spec.highlightSeries ?? []).map(String));
  const visibleSeriesIds = new Set((spec.loadedSeriesIds ?? groups.map((group) => group.name)).map(String));
  const maxVisiblePointCount = spec.maxVisiblePointCount ?? 12;
  const fadeNonHighlighted = spec.fadeNonHighlighted ?? true;
  const visibleGroups = groups.filter((group) => visibleSeriesIds.has(group.name));
  const timeValues = rows
    .map((row) => Date.parse(String(row[spec.x as keyof typeof row] ?? "")))
    .filter((value) => Number.isFinite(value));
  const rankValues = rows
    .map((row) => toNumeric(row[spec.y as keyof typeof row]))
    .filter((value): value is number => value !== null);
  const xMin = timeValues.length ? Math.min(...timeValues) : undefined;
  const xMax = timeValues.length ? Math.max(...timeValues) : undefined;
  const yMin = rankValues.length ? Math.max(0, Math.floor(Math.min(...rankValues) - 1)) : 0;
  const yMax = rankValues.length ? Math.ceil(Math.max(...rankValues) + 1) : 1;

  const preparedGroups = visibleGroups.map((group, index) => {
    const sortedRows = [...group.rows].sort((left, right) => {
      const leftTime = Date.parse(String(left[spec.x as keyof typeof left] ?? ""));
      const rightTime = Date.parse(String(right[spec.x as keyof typeof right] ?? ""));
      return leftTime - rightTime;
    });
    const sparse = sortedRows.length <= 2;
    const highlighted = highlightSeries.has(group.name);
    const sampledIndexes = buildSampledIndexSet(sortedRows.length, maxVisiblePointCount);
    return {
      name: group.name,
      rows: sortedRows,
      sparse,
      highlighted,
      sampledIndexes,
      color: categoricalPalette[index % categoricalPalette.length],
    };
  });

  const series = preparedGroups.map((group) => {
    const shouldFade = fadeNonHighlighted && !group.highlighted && !group.sparse;
    const lineOpacity = shouldFade ? 0.18 : 0.9;
    const pointOpacity = shouldFade ? 0.32 : 1;

    return {
      name: group.name,
      type: "line",
      smooth: false,
      connectNulls: false,
      showSymbol: true,
      showAllSymbol: false,
      animationDurationUpdate: 600,
      universalTransition: true,
      color: group.color,
      lineStyle: {
        width: group.sparse ? 0 : group.highlighted ? 2.8 : 1.8,
        opacity: lineOpacity,
      },
      itemStyle: {
        color: group.color,
        opacity: pointOpacity,
        borderColor: "#f4f4f5",
        borderWidth: group.sparse ? 2 : 1.2,
      },
      emphasis: {
        focus: "series",
        lineStyle: {
          opacity: 1,
          width: group.sparse ? 0 : 3.2,
        },
        itemStyle: {
          opacity: 1,
          borderColor: "#ffffff",
          borderWidth: 2,
        },
      },
      endLabel:
        group.highlighted || group.sparse
          ? {
              show: true,
              formatter: group.name,
              color: group.color,
              fontSize: chartAuxiliaryFontSize,
            }
          : undefined,
      labelLayout: {
        moveOverlap: "shiftY",
      },
      data: group.rows.map((row, rowIndex) => {
        const xValue = String(row[spec.x as keyof typeof row] ?? "");
        const yValue = toNumeric(row[spec.y as keyof typeof row]) ?? 0;
        const symbolVisible = group.sparse || group.sampledIndexes.has(rowIndex);
        return {
          value: [xValue, yValue],
          symbolSize: symbolVisible ? (group.sparse ? 12 : group.highlighted ? 10 : 7) : 0,
          itemStyle: {
            opacity: symbolVisible ? pointOpacity : 0,
          },
          meta: row,
        };
      }),
    };
  });

  return {
    backgroundColor: "transparent",
    grid: { left: 64, right: 84, top: 40, bottom: 64, containLabel: true },
    tooltip: {
      trigger: "item",
      backgroundColor: chartTooltipBackground,
      borderColor: "rgba(255,255,255,0.08)",
      textStyle: { color: chartTextColor, fontSize: chartTooltipFontSize },
      formatter: (params: { seriesName: string; data: { meta?: Record<string, unknown>; value: [string, number] } }) => {
        const raw = params.data?.meta ?? {};
        const name = raw.name ?? params.seriesName;
        const brand = raw.brand ?? "-";
        const rank = raw.rank ?? params.data?.value?.[1] ?? "-";
        return [
          `<strong>${String(name)}</strong>`,
          `상품 ID: ${params.seriesName}`,
          `브랜드: ${String(brand)}`,
          `시점: ${formatDateLabel(params.data?.value?.[0] ?? "")}`,
          `순위: ${formatNumber(rank, "integer")}`,
        ].join("<br/>");
      },
    },
    legend: { show: false },
    xAxis: {
      type: "time",
      name: spec.xLabel ?? spec.x,
      min: xMin,
      max: xMax,
      interval: oneDayMs,
      minInterval: oneDayMs,
      nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
      axisLabel: {
        show: true,
        color: chartTextColor,
        fontSize: chartAxisLabelFontSize,
        formatter: (value: string | number) => formatDateDayLabel(value),
        interval: 0,
      },
      axisTick: {
        show: true,
        alignWithLabel: true,
        lineStyle: { color: "rgba(255,255,255,0.18)" },
      },
      axisLine: { lineStyle: { color: chartAxisColor } },
      splitLine: {
        show: true,
        lineStyle: { color: chartSplitLineColor },
      },
    },
    yAxis: {
      type: "value",
      name: spec.yLabel ?? spec.y,
      inverse: spec.yAxisInverse ?? true,
      nameLocation: "middle",
      nameGap: 48,
      nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
      axisLabel: {
        color: chartTextColor,
        fontSize: chartAxisLabelFontSize,
        formatter: (value: number) => formatNumber(value, spec.yFormat ?? "integer", spec.y),
      },
      axisLine: { lineStyle: { color: chartAxisColor } },
      splitLine: { lineStyle: { color: chartSplitLineColor } },
      min: yMin,
      max: yMax,
    },
    series,
  };
}

export function EChartPanel({ rows, chartKind, spec }: EChartPanelProps) {
  const groups = useMemo(() => groupBySeries(rows, spec, chartKind), [rows, spec, chartKind]);
  const groupNames = useMemo(() => groups.map((group) => group.name), [groups]);
  const legendColorMap = useMemo(() => buildLegendColorMap(groupNames, spec), [groupNames, spec]);
  const showLegend = spec.showLegend ?? true;
  const customLegendPanel = useMemo(() => renderCustomLegendPanel(spec, groupNames), [groupNames, spec]);
  const defaultHighlightSeries = useMemo(
    () => (spec.highlightSeries ?? []).map(String),
    [spec.highlightSeries],
  );
  const activeHighlightSet = useMemo(() => new Set(defaultHighlightSeries), [defaultHighlightSeries]);
  const availableSeries = useMemo<ChartSeriesOption[]>(
    () => spec.availableSeries ?? groupNames.map((name) => ({ id: name })),
    [groupNames, spec.availableSeries],
  );
  const loadedSeriesIds = useMemo(() => new Set((spec.loadedSeriesIds ?? groupNames).map(String)), [spec.loadedSeriesIds, groupNames]);
  const defaultSeriesIds = useMemo(() => new Set((spec.defaultSeriesIds ?? []).map(String)), [spec.defaultSeriesIds]);
  const [selectedBumpSeries, setSelectedBumpSeries] = useState<string | null>(null);
  const [bumpSearch, setBumpSearch] = useState("");
  const isControlledBump = spec.selectedBumpSeries !== undefined;
  const effectiveSelectedBumpSeries = isControlledBump ? (spec.selectedBumpSeries ?? null) : selectedBumpSeries;
  const setEffectiveSelectedBump = (value: string | null) => {
    spec.onHighlightSeries?.(value);
    if (!isControlledBump) {
      setSelectedBumpSeries(value);
    }
  };

  useEffect(() => {
    if (chartKind !== "bump") {
      return;
    }
    const availableSeriesIds = new Set(availableSeries.map((series) => series.id));
    if (effectiveSelectedBumpSeries && !availableSeriesIds.has(effectiveSelectedBumpSeries) && !groups.some((group) => group.name === effectiveSelectedBumpSeries)) {
      setEffectiveSelectedBump(null);
    }
  }, [availableSeries, chartKind, groups, effectiveSelectedBumpSeries]);

  useEffect(() => {
    if (chartKind !== "bump") {
      return;
    }
    if (!isControlledBump) {
      setSelectedBumpSeries(null);
    }
    setBumpSearch("");
  }, [chartKind, spec.resetToken]);

  if (!rows.length) {
    return <div className="empty-state">표시할 차트 데이터가 없습니다.</div>;
  }

  const hasTimeAxis = chartKind === "line" && rows.some((row) => isDateLike(row[spec.x as keyof typeof row]));
  const xAxisType = chartKind === "scatter" ? "value" : hasTimeAxis ? "time" : "category";
  const timeAxisLabel = spec.timeAxisLabel ?? "datetime";
  const categories = spec.xDomain ?? Array.from(
    new Set(rows.map((row) => String(row[spec.x as keyof typeof row] ?? ""))),
  );
  const markLines = buildMarkLines(spec);
  const quadrantHints = buildQuadrantHints(spec);
  const chartEvents = buildChartEvents(spec);

  if (chartKind === "bump") {
    const effectiveHighlightSeries = effectiveSelectedBumpSeries ? [effectiveSelectedBumpSeries] : defaultHighlightSeries;
    const option = buildBumpOption(rows, { ...spec, highlightSeries: effectiveHighlightSeries }, groups);
    const loadedSeries = availableSeries.filter((series) => loadedSeriesIds.has(series.id));
    const normalizedSearch = bumpSearch.trim().toLowerCase();
    const visibleSeries = availableSeries.filter((series) => {
      if (!normalizedSearch) {
        return true;
      }
      const searchableText = [series.id, series.label, series.brand].filter(Boolean).join(" ").toLowerCase();
      return searchableText.includes(normalizedSearch);
    });
    const orderByAvailable = (id: string) => {
      const i = availableSeries.findIndex((s) => s.id === id);
      return i >= 0 ? i : 9999;
    };
    const sortedSeries = [...visibleSeries].sort((left, right) => {
      const leftPriority = left.id === effectiveSelectedBumpSeries ? 0 : defaultSeriesIds.has(left.id) ? 1 : 2;
      const rightPriority = right.id === effectiveSelectedBumpSeries ? 0 : defaultSeriesIds.has(right.id) ? 1 : 2;
      if (leftPriority !== rightPriority) {
        return leftPriority - rightPriority;
      }
      return orderByAvailable(left.id) - orderByAvailable(right.id);
    });
    const extraLoadedCount = Math.max(loadedSeriesIds.size - defaultSeriesIds.size, 0);
    const handleSelectSeries = (series: ChartSeriesOption) => {
      setEffectiveSelectedBump(series.id);
      if (!loadedSeriesIds.has(series.id)) {
        spec.onSelectSeries?.(series.id);
      }
    };

    return (
      <div className="bump-chart-panel">
        <div className="bump-chart-panel__toolbar">
          <div className="bump-chart-panel__summary">
            <strong>상품번호</strong>
            <span>
              {effectiveSelectedBumpSeries
                ? `${effectiveSelectedBumpSeries} 라인 강조 중`
                : `전체 ${availableSeries.length}개 상품, 기본 ${defaultSeriesIds.size}개 + 추가 ${extraLoadedCount}개`}
            </span>
          </div>
          <div className="bump-chart-panel__controls">
            <input
              type="search"
              value={bumpSearch}
              onChange={(event) => setBumpSearch(event.target.value)}
              placeholder="상품번호, 상품명, 브랜드 검색"
            />
            {spec.onClearSelectedSeries ? (
              <button
                type="button"
                className="ghost-button"
                onClick={() => {
                  setEffectiveSelectedBump(null);
                  spec.onClearSelectedSeries?.();
                }}
              >
                추가 상품 초기화
              </button>
            ) : null}
          </div>
          {loadedSeries.length ? (
            <div className="bump-chart-panel__selected">
              {loadedSeries.map((series) => {
                const isDefault = defaultSeriesIds.has(series.id);
                const isActive = effectiveSelectedBumpSeries === series.id;
                return (
                  <div key={`selected-${series.id}`} className="bump-chart-panel__selected-chip-wrap">
                    <button
                      type="button"
                      className={`bump-chart-panel__selected-chip${isDefault ? " is-default" : ""}${isActive ? " is-active" : ""}`}
                      onClick={() => {
                        if (isActive) {
                          setEffectiveSelectedBump(null);
                          return;
                        }
                        setEffectiveSelectedBump(series.id);
                      }}
                    >
                      <span>{series.id}</span>
                      <small>{isDefault ? "기본 제외" : isActive ? "강조 중" : "강조"}</small>
                    </button>
                    <button
                      type="button"
                      className="bump-chart-panel__selected-remove"
                      aria-label={`${series.id} 제거`}
                      title={`${series.id} 제거`}
                      onClick={(event) => {
                        event.stopPropagation();
                        if (isActive) {
                          setEffectiveSelectedBump(null);
                        }
                        spec.onRemoveSeries?.(series.id);
                      }}
                    >
                      x
                    </button>
                  </div>
                );
              })}
            </div>
          ) : null}
          <div className="bump-chart-panel__list">
            {sortedSeries.length ? sortedSeries.map((series) => {
              const isActive = series.id === effectiveSelectedBumpSeries;
              const isDefault = !effectiveSelectedBumpSeries && defaultSeriesIds.has(series.id);
              const isLoaded = loadedSeriesIds.has(series.id);
              const detailBits = [
                series.label,
                series.brand,
                series.priceBand ? `판매가 ${series.priceBand}` : null,
                series.estimatedOriginalPriceBand ? `정가 ${series.estimatedOriginalPriceBand}` : null,
                series.latestRank ? `최근 ${series.latestRank}위` : null,
                series.latestMomentum != null ? `모멘텀 ${formatNumber(series.latestMomentum, "number")}` : null,
              ].filter(Boolean);
              return (
                <button
                  key={series.id}
                  type="button"
                  title={detailBits.join(" | ")}
                  className={`bump-chart-panel__chip${isActive ? " is-active" : isDefault ? " is-default" : ""}${isLoaded ? " is-loaded" : ""}`}
                  onClick={() => handleSelectSeries(series)}
                >
                  <span>{series.id}</span>
                  {!isLoaded ? <small>추가</small> : null}
                </button>
              );
            }) : <div className="bump-chart-panel__empty">검색 결과가 없습니다.</div>}
          </div>
        </div>
        <ReactECharts
          key={`bump-chart-${spec.resetToken ?? 0}`}
          option={option}
          onEvents={chartEvents}
          notMerge
          lazyUpdate
          style={{ height: 420 }}
        />
      </div>
    );
  }

  if (chartKind === "heatmap") {
    const yCategories = spec.yDomain ?? Array.from(
      new Set(rows.map((row) => String(row[spec.y as keyof typeof row] ?? ""))),
    );
    const heatNumeric = rows.map((row) => toNumeric(row[(spec.value ?? "value") as keyof typeof row]) ?? 0);
    const heatMin = heatNumeric.length ? Math.min(0, ...heatNumeric) : 0;
    const heatMax = heatNumeric.length ? Math.max(1, ...heatNumeric) : 1;
    const option = {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        backgroundColor: chartTooltipBackground,
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: chartTextColor, fontSize: chartTooltipFontSize },
        formatter: (params: { data?: { meta?: Record<string, unknown>; value: [string, string, number] } }) => {
          const raw = params.data?.meta ?? {};
          const value = params.data?.value?.[2] ?? 0;
          const baseLines = [
            `${spec.xLabel ?? spec.x}: ${formatDimensionValue(params.data?.value?.[0] ?? "-", spec.x)}`,
            `${spec.yLabel ?? spec.y}: ${formatDimensionValue(params.data?.value?.[1] ?? "-", spec.y)}`,
            `${spec.value ?? "value"}: ${formatNumber(value, spec.valueFormat ?? "integer", spec.value ?? "value")}`,
          ];
          return [...baseLines, ...buildTooltipFieldLines(raw, spec)].join("<br/>");
        },
      },
      grid: { left: 80, right: 92, top: 60, bottom: 96 },
      xAxis: {
        type: "category",
        name: spec.xLabel ?? spec.x,
        data: categories,
        axisLabel: {
          color: chartTextColor,
          fontSize: chartAxisLabelFontSize,
          margin: 14,
          formatter: (value: string | number) => formatDimensionValue(value, spec.x),
        },
        nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
        axisLine: { lineStyle: { color: chartAxisColor } },
      },
      yAxis: {
        type: "category",
        name: spec.yLabel ?? spec.y,
        data: yCategories,
        axisLabel: {
          color: chartTextColor,
          fontSize: chartAxisLabelFontSize,
          formatter: (value: string | number) => formatDimensionValue(value, spec.y),
        },
        nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
        axisLine: { lineStyle: { color: chartAxisColor } },
      },
      visualMap: {
        type: "continuous",
        min: heatMin,
        max: heatMax,
        calculable: false,
        orient: "vertical",
        right: 0,
        top: "middle",
        itemWidth: 14,
        itemHeight: 180,
        textGap: 10,
        text: ["높음", "낮음"],
        textStyle: { color: chartTextColor, fontSize: chartAxisLabelFontSize },
        inRange: {
          color: ["#111214", "#1f2937", "#374151", "#4b5563", "#64748b", "#93c5fd"],
        },
      },
      series: [
        {
          type: "heatmap",
          data: rows.map((row) => ({
            value: [
              String(row[spec.x as keyof typeof row] ?? ""),
              String(row[spec.y as keyof typeof row] ?? ""),
              toNumeric(row[(spec.value ?? "value") as keyof typeof row]) ?? 0,
            ],
            meta: row,
          })),
          label: {
            show: true,
            color: "#f5f5f5",
            fontSize: chartAuxiliaryFontSize,
            formatter: (params: { value?: [string, string, number] }) =>
              formatNumber(params.value?.[2] ?? 0, spec.valueFormat ?? "number", spec.value ?? "value"),
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 12,
              shadowColor: "rgba(0,0,0,0.36)",
            },
          },
        },
      ],
    };
    return <ReactECharts option={option} onEvents={chartEvents} style={{ height: 400 }} />;
  }

  if (chartKind === "scatter") {
    const collectAxis = (key: string) => {
      const out: number[] = [];
      rows.forEach((row) => {
        const v = toNumeric(row[key as keyof typeof row]);
        if (v != null) {
          out.push(v);
        }
      });
      return out;
    };
    const equalExtents = spec.scatterSquareEqualScale
      ? computeEqualScaleScatterExtents(collectAxis(spec.x), collectAxis(spec.y))
      : null;
    const scatterSizeRange = spec.scatterSizeRange ?? [8, 24];
    const scatterMinSize = Math.min(scatterSizeRange[0], scatterSizeRange[1]);
    const scatterMaxSize = Math.max(scatterSizeRange[0], scatterSizeRange[1]);
    const scatterSizeExponent = Math.max(spec.scatterSizeExponent ?? 1, 0.1);
    const scatterSizeFallback = spec.scatterSizeFallback ?? 11;
    const scatterSizeValues = spec.scatterSizeBy
      ? rows
          .map((row) => toNumeric(row[spec.scatterSizeBy as keyof typeof row]))
          .filter((value): value is number => value !== null)
      : [];
    const scatterSizeMin = scatterSizeValues.length ? Math.min(...scatterSizeValues) : null;
    const scatterSizeMax = scatterSizeValues.length ? Math.max(...scatterSizeValues) : null;
    const resolveScatterSize = (row: Record<string, unknown>) => {
      if (!spec.scatterSizeBy) {
        return scatterSizeFallback;
      }
      const raw = toNumeric(row[spec.scatterSizeBy as keyof typeof row]);
      if (raw === null || scatterSizeMin === null || scatterSizeMax === null) {
        return scatterSizeFallback;
      }
      if (scatterSizeMin === scatterSizeMax) {
        return (scatterMinSize + scatterMaxSize) / 2;
      }
      const ratio = (raw - scatterSizeMin) / (scatterSizeMax - scatterSizeMin);
      const curvedRatio = Math.pow(Math.max(0, Math.min(1, ratio)), scatterSizeExponent);
      return scatterMinSize + curvedRatio * (scatterMaxSize - scatterMinSize);
    };
    const xAxisExtent = equalExtents ? { min: equalExtents.xMin, max: equalExtents.xMax } : {};
    const yAxisExtent = equalExtents ? { min: equalExtents.yMin, max: equalExtents.yMax } : {};
    const scatterOption = {
      backgroundColor: "transparent",
      grid: { left: 56, right: 24, top: 56, bottom: 44 },
      tooltip: {
        trigger: "item",
        renderMode: "html",
        appendToBody: true,
        confine: false,
        extraCssText: "max-width: 320px; white-space: normal;",
        backgroundColor: chartTooltipBackground,
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: chartTextColor, fontSize: chartTooltipFontSize },
        formatter: (params: { seriesName: string; value: [number, number, Record<string, unknown>] }) => {
          const raw = params.value?.[2] ?? {};
          const name = raw.name ?? "-";
          const brand = raw.brand ?? "-";
          const rank = raw.rank ?? "-";
          const discountPct = raw.discountPct ?? "-";
          const price = raw.price ?? "-";
          const mainImage = typeof raw.mainImage === "string" && raw.mainImage.trim() ? raw.mainImage.trim() : null;
          const imgHtml = mainImage
            ? `<br/><img src="${productImageApiUrl(mainImage)}" alt="" style="max-width:168px;max-height:168px;display:block;margin-top:8px;border-radius:6px;object-fit:contain;" />`
            : "";
          const mainImageNote =
            spec.scatterMainImageTooltip && !mainImage
              ? `<span style="color:#a1a1aa;">메인 이미지 없음</span>`
              : "";
          const extraLines = buildTooltipFieldLines(raw, spec);
          return [
            `<strong>${escapeHtml(String(name))}</strong>`,
            `그룹: ${escapeHtml(params.seriesName)}`,
            `브랜드: ${escapeHtml(String(brand))}`,
            `현재 순위: ${formatNumber(rank, "integer")}`,
            `${spec.xLabel ?? spec.x}: ${formatNumber(params.value?.[0] ?? "-", spec.xFormat ?? "auto", spec.x)}`,
            `${spec.yLabel ?? spec.y}: ${formatNumber(params.value?.[1] ?? "-", spec.yFormat ?? "auto", spec.y)}`,
            `현재 할인율: ${formatNumber(discountPct, "percent", "discountPct")}`,
            `가격: ${formatNumber(price, "price", "price")}`,
            ...extraLines,
            ...(mainImageNote ? [mainImageNote] : []),
            ...(imgHtml ? [imgHtml] : []),
          ].join("<br/>");
        },
      },
      legend: {
        show: showLegend,
        top: 0,
        formatter: (name: string) => formatSeriesLabel(name, spec),
        textStyle: { color: chartTextColor, fontSize: chartLegendFontSize },
      },
      color: getPalette(spec, groupNames),
      graphic: quadrantHints,
      xAxis: {
        type: "value",
        name: spec.xLabel ?? spec.x,
        nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
        axisLabel: {
          color: chartTextColor,
          fontSize: chartAxisLabelFontSize,
          formatter: (value: number) => formatNumber(value, spec.xFormat ?? "auto", spec.x),
        },
        axisLine: { lineStyle: { color: chartAxisColor } },
        splitLine: { lineStyle: { color: chartSplitLineColor } },
        ...xAxisExtent,
      },
      yAxis: {
        type: "value",
        name: spec.yLabel ?? spec.y,
        nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
        axisLabel: {
          color: chartTextColor,
          fontSize: chartAxisLabelFontSize,
          formatter: (value: number) => formatNumber(value, spec.yFormat ?? "auto", spec.y),
        },
        axisLine: { lineStyle: { color: chartAxisColor } },
        splitLine: { lineStyle: { color: chartSplitLineColor } },
        inverse: spec.yAxisInverse ?? false,
        ...yAxisExtent,
      },
      series: [
        ...buildScatterRegionSeries(spec),
        ...groups.map((group) => ({
          name: group.name,
          type: "scatter",
          animationDurationUpdate: 600,
          color: legendColorMap.get(group.name),
          itemStyle: {
            color: legendColorMap.get(group.name),
            opacity: activeHighlightSet.size && !activeHighlightSet.has(group.name) ? 0.16 : 0.92,
          },
          emphasis: {
            focus: "series",
          },
          markLine: markLines ? { silent: true, symbol: "none", data: markLines } : undefined,
          data: group.rows.map((row) => ({
            value: [
              row[spec.x as keyof typeof row] ?? 0,
              row[spec.y as keyof typeof row] ?? 0,
              row,
            ],
            meta: row,
            symbolSize: resolveScatterSize(row),
          })),
        })),
        ...buildScatterAnnotationSeries(spec),
      ],
    };
    const scatterChart = spec.scatterSquareEqualScale
      ? <ScatterSquareChart option={scatterOption} onEvents={chartEvents} />
      : <ReactECharts option={scatterOption} onEvents={chartEvents} style={{ height: 360 }} />;
    return (
      <div className="chart-panel">
        {customLegendPanel}
        {scatterChart}
      </div>
    );
  }

  const option =
    {
          backgroundColor: "transparent",
          grid: { left: 56, right: 24, top: 56, bottom: 44 },
          tooltip: {
            trigger: hasTimeAxis ? "item" : "axis",
            backgroundColor: chartTooltipBackground,
            borderColor: "rgba(255,255,255,0.08)",
            textStyle: { color: chartTextColor, fontSize: chartTooltipFontSize },
            formatter: hasTimeAxis
              ? (params: {
                  seriesName: string;
                  value: [string | number, number];
                  data?: { meta?: Record<string, unknown> };
                }) => {
                  const raw = params.data?.meta ?? {};
                  return [
                    formatSeriesLabel(params.seriesName, spec),
                    formatDateLabel(params.value[0]),
                    `${spec.yLabel ?? spec.y}: ${formatNumber(params.value[1], spec.yFormat ?? "auto", spec.y)}`,
                    ...buildTooltipFieldLines(raw, spec),
                  ].join("<br/>");
                }
              : undefined,
          },
          legend: {
            show: showLegend,
            top: 0,
            formatter: (name: string) => formatSeriesLabel(name, spec),
            textStyle: { color: chartTextColor, fontSize: chartLegendFontSize },
          },
          color: getPalette(spec, groupNames),
          xAxis: {
            type: xAxisType,
            name: spec.xLabel ?? spec.x,
            data: hasTimeAxis ? undefined : categories,
            nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
            axisLabel: hasTimeAxis
              ? {
                  formatter: (value: string | number) =>
                    timeAxisLabel === "day" ? formatDateDayLabel(value) : formatDateLabel(value),
                  color: chartTextColor,
                  fontSize: chartAxisLabelFontSize,
                  hideOverlap: true,
                }
              : {
                  color: chartTextColor,
                  fontSize: chartAxisLabelFontSize,
                  hideOverlap: true,
                  formatter: (value: string | number) => formatDimensionValue(value, spec.x),
                },
            axisLine: { lineStyle: { color: chartAxisColor } },
          },
          yAxis: {
            type: "value",
            name: spec.yLabel ?? spec.y,
            inverse: spec.yAxisInverse ?? false,
            nameTextStyle: { color: chartTextColor, fontSize: chartBaseFontSize },
            axisLabel: {
              color: chartTextColor,
              fontSize: chartAxisLabelFontSize,
              formatter: (value: number) => formatNumber(value, spec.yFormat ?? "auto", spec.y),
            },
            axisLine: { lineStyle: { color: chartAxisColor } },
            splitLine: { lineStyle: { color: chartSplitLineColor } },
          },
          series: groups.map((group) => ({
            name: formatSeriesLabel(group.name, spec),
            type: chartKind,
            smooth: chartKind === "line" ? (spec.lineSmooth ?? true) : undefined,
            animationDurationUpdate: 600,
            universalTransition: true,
            color: legendColorMap.get(group.name),
            itemStyle:
              chartKind === "bar" && spec.color
                ? {
                    color:
                      spec.palette === "semantic"
                        ? semanticColorMap[group.name] ?? undefined
                        : legendColorMap.get(group.name),
                    borderColor: legendColorMap.get(group.name),
                    borderWidth: legendColorMap.get(group.name) ? 1 : undefined,
                    opacity: activeHighlightSet.size && !activeHighlightSet.has(group.name) ? 0.18 : 1,
                  }
                : chartKind === "line"
                  ? {
                      color: legendColorMap.get(group.name),
                      opacity: activeHighlightSet.size && !activeHighlightSet.has(group.name) ? 0.22 : 1,
                    }
                  : undefined,
            showSymbol: chartKind === "line" ? (spec.lineShowSymbol ?? true) : undefined,
            showAllSymbol: chartKind === "line" ? (spec.lineShowAllSymbol ?? true) : undefined,
            symbolSize: chartKind === "line" ? (spec.lineSymbolSize ?? 10) : undefined,
            step: chartKind === "line" ? spec.lineStep : undefined,
            sampling: chartKind === "line" ? spec.lineSampling : undefined,
            areaStyle:
              chartKind === "line" && spec.lineArea
                ? {
                    opacity: 0.22,
                    color: legendColorMap.get(group.name),
                  }
                : undefined,
            lineStyle:
              chartKind === "line" && group.rows.length === 1
                ? { width: 0 }
                : chartKind === "line"
                  ? {
                      width: activeHighlightSet.size && activeHighlightSet.has(group.name) ? 3.1 : 2.5,
                      opacity: activeHighlightSet.size && !activeHighlightSet.has(group.name) ? 0.18 : 0.95,
                      cap: "round",
                      join: "round",
                    }
                  : undefined,
            emphasis:
              chartKind === "line"
                ? {
                    focus: "series",
                    itemStyle: {
                      borderColor: "#ffffff",
                      borderWidth: 2,
                    },
                  }
                : undefined,
            markLine:
              chartKind === "line" && markLines
                ? { silent: true, symbol: "none", data: markLines }
                : undefined,
            data: hasTimeAxis
              ? applyDiscreteHoldCurve(
                  applyMovingAverage(
                    flattenSmallLineChanges(
                      [...group.rows].sort((left, right) => {
                        const leftTime = Date.parse(String(left[spec.x as keyof typeof left] ?? ""));
                        const rightTime = Date.parse(String(right[spec.x as keyof typeof right] ?? ""));
                        return leftTime - rightTime;
                      }),
                      spec.y,
                      chartKind === "line" ? spec.lineSmallChangeThreshold : undefined,
                    ),
                    spec.y,
                    chartKind === "line" ? spec.lineMovingAverageWindow : undefined,
                  ),
                  spec.x,
                  spec.y,
                  chartKind === "line" ? spec.lineDiscreteHoldRatio : undefined,
                )
                  .map((row) => ({
                    value: [
                      row[spec.x as keyof typeof row] ?? "",
                      toNumeric(row[spec.y as keyof typeof row]) ?? 0,
                    ],
                    meta: row,
                  }))
              : categories.map((category) => {
                  const matchedRow = group.rows.find(
                    (row) => String(row[spec.x as keyof typeof row] ?? "") === category,
                  );
                  if (!matchedRow) {
                    if (chartKind === "line") {
                      return { value: 0, meta: {} };
                    }
                    return null;
                  }
                  const numericValue = toNumeric(matchedRow[spec.y as keyof typeof matchedRow]) ?? 0;
                  if (chartKind === "bar" && !spec.seriesBy) {
                    const dimmed = activeHighlightSet.size && !activeHighlightSet.has(category);
                    return {
                      value: numericValue,
                      meta: matchedRow,
                      itemStyle: {
                        color: resolveCategoryColor(category, categories.indexOf(category), spec),
                        opacity: dimmed ? 0.18 : 1,
                      },
                    };
                  }
                  return {
                    value: numericValue,
                    meta: matchedRow,
                  };
                }),
          })),
        };

  return (
    <div className="chart-panel">
      {customLegendPanel}
      <ReactECharts option={option} onEvents={chartEvents} style={{ height: spec.chartHeight ?? 360 }} />
    </div>
  );
}
