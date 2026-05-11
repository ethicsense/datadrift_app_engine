import { useLayoutEffect, useMemo, useState } from "react";

import ReactECharts from "echarts-for-react";

import { formatNumber } from "../../lib/formatters";
import { EChartPanel } from "./EChartPanel";

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

const chartTextColor = "#d4d4d8";
const chartAxisColor = "#71717a";
const chartSplitLineColor = "rgba(255,255,255,0.1)";
const chartTooltipBackground = "rgba(12,12,14,0.96)";

const ROW_H = 30;
const ROW_GAP = 5;
const GRID_LEFT = 148;
const GRID_RIGHT = 20;
const TOP_PAD = 6;
const BOTTOM_AXIS = 26;
const MAX_COMPARE = 6;
const DEFAULT_SELECT = 3;
const MAX_SPARK_HEIGHT = 400;

type Point = { t: number; v: number };

function toNumeric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function formatDay(ts: number) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) {
    return "";
  }
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function truncateLabel(s: string, max: number) {
  if (s.length <= max) {
    return s;
  }
  return `${s.slice(0, Math.max(0, max - 1))}…`;
}

function buildModel(rows: Record<string, unknown>[]) {
  const byKw = new Map<string, Point[]>();
  let tMin = Infinity;
  let tMax = -Infinity;

  for (const row of rows) {
    const kw = String(row.keyword ?? "").trim() || "기타";
    const raw = row.snapshotDate;
    const t = typeof raw === "string" || typeof raw === "number" ? Date.parse(String(raw)) : NaN;
    const v = toNumeric(row.mentionCount);
    if (!Number.isFinite(t) || v === null) {
      continue;
    }
    tMin = Math.min(tMin, t);
    tMax = Math.max(tMax, t);
    const list = byKw.get(kw) ?? [];
    list.push({ t, v });
    byKw.set(kw, list);
  }

  for (const [, list] of byKw) {
    list.sort((a, b) => a.t - b.t);
  }

  const scoreKw = (k: string) => {
    const list = byKw.get(k)!;
    const latestT = Math.max(...list.map((x) => x.t));
    const latest = list.filter((x) => x.t === latestT).reduce((m, x) => Math.max(m, x.v), 0);
    const sum = list.reduce((s, x) => s + x.v, 0);
    return latest * 1e12 + sum;
  };

  const keywords = Array.from(byKw.keys()).sort((a, b) => scoreKw(b) - scoreKw(a));

  return { byKw, keywords, tMin: Number.isFinite(tMin) ? tMin : 0, tMax: Number.isFinite(tMax) ? tMax : 0 };
}

type KeywordTimeseriesSparklinePanelProps = {
  rows: Record<string, unknown>[];
};

export function KeywordTimeseriesSparklinePanel({ rows }: KeywordTimeseriesSparklinePanelProps) {
  const model = useMemo(() => buildModel(rows), [rows]);
  const { byKw, keywords, tMin, tMax } = model;

  const [filterQuery, setFilterQuery] = useState("");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());

  const normalizedQuery = filterQuery.trim().toLowerCase();
  const visibleKeywords = useMemo(() => {
    if (!normalizedQuery) {
      return keywords;
    }
    return keywords.filter((k) => k.toLowerCase().includes(normalizedQuery));
  }, [keywords, normalizedQuery]);

  const keywordSig = keywords.join("|");
  useLayoutEffect(() => {
    setSelected(new Set(keywords.slice(0, Math.min(DEFAULT_SELECT, keywords.length))));
  }, [keywordSig]);

  const sparkHeight = useMemo(() => {
    const n = visibleKeywords.length;
    if (n === 0) {
      return 120;
    }
    const body = TOP_PAD + n * (ROW_H + ROW_GAP) - ROW_GAP + BOTTOM_AXIS;
    return Math.min(MAX_SPARK_HEIGHT, Math.max(body, 160));
  }, [visibleKeywords.length]);

  const sparkOption = useMemo(() => {
    const n = visibleKeywords.length;
    if (n === 0) {
      return null;
    }

    const grids = visibleKeywords.map((_, i) => ({
      left: GRID_LEFT,
      right: GRID_RIGHT,
      top: TOP_PAD + i * (ROW_H + ROW_GAP),
      height: ROW_H,
    }));

    const xAxes = visibleKeywords.map((_, i) => ({
      type: "time" as const,
      gridIndex: i,
      min: tMin,
      max: tMax,
      axisLine: { show: i === n - 1, lineStyle: { color: chartAxisColor } },
      axisTick: { show: i === n - 1 },
      axisLabel: {
        show: i === n - 1,
        color: chartTextColor,
        fontSize: 13,
        hideOverlap: true,
        formatter: (value: number) => formatDay(value),
      },
      splitLine: { show: false },
    }));

    const yAxes = visibleKeywords.map((kw, i) => {
      const pts = byKw.get(kw) ?? [];
      const vmax = pts.length ? Math.max(...pts.map((p) => p.v), 0) : 0;
      const yMax = vmax <= 0 ? 1 : vmax * 1.08;
      const dimmed = selected.size > 0 && !selected.has(kw);
      return {
        type: "value" as const,
        gridIndex: i,
        min: 0,
        max: yMax,
        name: truncateLabel(kw, 18),
        nameLocation: "middle" as const,
        nameGap: 36,
        nameRotate: 0,
        nameTextStyle: {
          color: dimmed ? "rgba(212,212,216,0.38)" : chartTextColor,
          fontSize: 13,
          align: "right" as const,
          verticalAlign: "middle" as const,
        },
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: {
          show: true,
          lineStyle: { color: chartSplitLineColor, width: 1 },
        },
      };
    });

    const series = visibleKeywords.map((kw, i) => {
      const pts = byKw.get(kw) ?? [];
      const color = categoricalPalette[keywords.indexOf(kw) % categoricalPalette.length];
      const dimmed = selected.size > 0 && !selected.has(kw);
      return {
        name: kw,
        type: "line" as const,
        xAxisIndex: i,
        yAxisIndex: i,
        triggerLineEvent: true,
        smooth: false,
        showSymbol: pts.length <= 12,
        showAllSymbol: false,
        symbolSize: 5,
        sampling: "lttb" as const,
        lineStyle: {
          width: selected.has(kw) ? 2.4 : 1.6,
          opacity: dimmed ? 0.28 : 0.95,
        },
        itemStyle: { color },
        emphasis: { focus: "series" as const },
        data: pts.map((p) => [p.t, p.v]),
      };
    });

    return {
      backgroundColor: "transparent",
      animationDurationUpdate: 400,
      tooltip: {
        trigger: "axis" as const,
        axisPointer: {
          type: "line" as const,
          lineStyle: { color: "rgba(148,163,184,0.45)", width: 1 },
        },
        backgroundColor: chartTooltipBackground,
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: chartTextColor, fontSize: 14 },
        formatter: (params: unknown) => {
          const list = Array.isArray(params) ? params : [params];
          const first = list[0] as { value?: [number, number]; seriesName?: string } | undefined;
          const v = first?.value;
          if (!v || v.length < 2) {
            return "";
          }
          const dated = formatDay(v[0]);
          const ranked = [...list]
            .map((p) => {
              const item = p as { seriesName?: string; value?: [number, number] };
              const val = item.value?.[1] ?? 0;
              return { name: item.seriesName ?? "", val };
            })
            .sort((a, b) => b.val - a.val);
          const cap = 8;
          const head = ranked.slice(0, cap);
          const rest = ranked.length - head.length;
          const lines = [
            dated,
            ...head.map((row) => `${truncateLabel(row.name, 40)}: ${formatNumber(row.val, "integer", "mentionCount")}`),
            rest > 0 ? `… 외 ${rest}개 키워드(값 순 상위 ${cap}개만 표시)` : "",
          ].filter(Boolean);
          return lines.join("<br/>");
        },
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
    };
  }, [byKw, visibleKeywords, keywords, tMin, tMax, selected]);

  const detailRows = useMemo(() => {
    const out: Record<string, unknown>[] = [];
    const ordered = [...selected].filter((k) => byKw.has(k));
    for (const kw of ordered) {
      const pts = byKw.get(kw);
      if (!pts) {
        continue;
      }
      for (const p of pts) {
        out.push({
          snapshotDate: new Date(p.t).toISOString(),
          keyword: kw,
          mentionCount: p.v,
        });
      }
    }
    return out;
  }, [selected, byKw]);

  const handleSparkClick = (params: { seriesName?: string }) => {
    const name = params.seriesName;
    if (!name || !byKw.has(name)) {
      return;
    }
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
        return next;
      }
      if (next.size >= MAX_COMPARE) {
        return next;
      }
      next.add(name);
      return next;
    });
  };

  const clearSelection = () => {
    setSelected(new Set());
  };

  if (!keywords.length) {
    return <div className="empty-state">표시할 키워드 시계열 데이터가 없습니다.</div>;
  }

  return (
    <div className="keyword-spark-panel">
      <div className="keyword-spark-panel__toolbar">
        <label className="keyword-spark-panel__search">
          <span>키워드 필터</span>
          <input
            type="search"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="부분 일치 검색…"
            aria-label="키워드 필터"
          />
        </label>
        <span className="keyword-spark-panel__meta">
          표시 {visibleKeywords.length} / 전체 {keywords.length} · 행 클릭으로 아래 비교 대상에 추가·해제 (최대 {MAX_COMPARE}개)
        </span>
        {selected.size > 0 ? (
          <button type="button" className="ghost-button keyword-spark-panel__clear" onClick={clearSelection}>
            비교 선택 초기화
          </button>
        ) : null}
      </div>

      <p className="keyword-spark-panel__hint">
        모든 행이 같은 시간 범위를 공유합니다. 왼쪽 이름·오른쪽 선은 키워드별 스케일이며, 아래는 선택한 키워드만 동일 Y축으로 비교합니다.
      </p>

      <div className="keyword-spark-panel__spark-wrap" style={{ maxHeight: MAX_SPARK_HEIGHT }}>
        {visibleKeywords.length === 0 ? (
          <div className="empty-state">필터와 일치하는 키워드가 없습니다.</div>
        ) : sparkOption ? (
          <ReactECharts
            option={sparkOption}
            style={{ height: sparkHeight, width: "100%" }}
            notMerge
            lazyUpdate
            onEvents={{ click: handleSparkClick }}
          />
        ) : null}
      </div>

      {detailRows.length > 0 ? (
        <div className="keyword-spark-panel__detail">
          <div className="keyword-spark-panel__detail-title">선택 키워드 비교 (절대 언급 수)</div>
          <EChartPanel
            rows={detailRows}
            chartKind="line"
            spec={{
              x: "snapshotDate",
              y: "mentionCount",
              seriesBy: "keyword",
              xLabel: "스냅샷 날짜",
              yLabel: "언급 수",
              yFormat: "integer",
              palette: "categorical",
              lineSmooth: false,
              lineShowSymbol: false,
              lineShowAllSymbol: false,
              lineSampling: "lttb",
              timeAxisLabel: "day",
            }}
          />
        </div>
      ) : (
        <div className="keyword-spark-panel__empty-detail">비교할 키워드를 위 스파크라인에서 선택하면 이 영역에 확대 그래프가 표시됩니다.</div>
      )}
    </div>
  );
}
