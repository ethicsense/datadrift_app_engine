import { useEffect, useState } from "react";

import ReactECharts from "echarts-for-react";

import { formatNumber } from "../../lib/formatters";

type RankRaceFrame = {
  snapshotId: string;
  label: string;
  bars: Array<{
    entityId: string;
    entityLabel: string;
    rank?: number | null;
    rankDelta?: number | null;
    momentumScore?: number | null;
  }>;
};

type RankRaceChartProps = {
  frames: RankRaceFrame[];
};

function rankColor(rankDelta: number | null | undefined) {
  if (rankDelta === null || rankDelta === undefined) {
    return "#a1a1aa";
  }
  if (rankDelta > 0) {
    return rankDelta >= 3 ? "#22c55e" : "#4ade80";
  }
  if (rankDelta < 0) {
    return rankDelta <= -3 ? "#ef4444" : "#fb7185";
  }
  return "#a1a1aa";
}

export function RankRaceChart({ frames }: RankRaceChartProps) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing || frames.length <= 1) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setIndex((current) => (current + 1) % frames.length);
    }, 1400);
    return () => window.clearInterval(timer);
  }, [frames.length, playing]);

  const currentFrame = frames[index];
  if (!frames.length || !currentFrame) {
    return <div className="empty-state">순위 애니메이션 데이터가 없습니다.</div>;
  }

  const sorted = [...currentFrame.bars].sort((left, right) => (left.rank ?? 9999) - (right.rank ?? 9999));
  const option = {
    backgroundColor: "transparent",
    grid: { left: 110, right: 24, top: 28, bottom: 44 },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(12,12,14,0.96)",
      borderColor: "rgba(255,255,255,0.08)",
      textStyle: { color: "#d4d4d8" },
      formatter: (params: { data: { meta?: RankRaceFrame["bars"][number] } }) => {
        const bar = params.data?.meta;
        if (!bar) {
          return "";
        }
        return [
          `<strong>${bar.entityLabel}</strong>`,
          `현재 순위: ${formatNumber(bar.rank ?? "-", "integer")}`,
          `순위 변화: ${formatNumber(bar.rankDelta ?? "-", "number")}`,
          `모멘텀: ${formatNumber(bar.momentumScore ?? "-", "number")}`,
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      inverse: true,
      name: "순위",
      nameTextStyle: { color: "#d4d4d8" },
      axisLabel: { color: "#d4d4d8", formatter: (value: number) => formatNumber(value, "integer") },
      axisLine: { lineStyle: { color: "#71717a" } },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: sorted.map((bar) => bar.entityLabel),
      axisLabel: { color: "#d4d4d8" },
      axisLine: { lineStyle: { color: "#71717a" } },
    },
    series: [
      {
        type: "bar",
        animationDurationUpdate: 700,
        universalTransition: true,
        label: {
          show: true,
          position: "right",
          color: "#f4f4f5",
          formatter: (params: { data: { meta?: RankRaceFrame["bars"][number] } }) => {
            const bar = params.data?.meta;
            return bar ? `#${formatNumber(bar.rank ?? "-", "integer")}` : "";
          },
        },
        data: sorted.map((bar) => ({
          value: bar.rank ?? 0,
          itemStyle: {
            color: rankColor(bar.rankDelta),
          },
          meta: bar,
        })),
      },
    ],
  };

  return (
    <div>
      <div className="playback-row">
        <button type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? "일시정지" : "재생"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 0)}
          value={index}
          onChange={(event) => setIndex(Number(event.target.value))}
        />
        <span>{currentFrame.label}</span>
      </div>
      <ReactECharts option={option} style={{ height: 420 }} />
    </div>
  );
}
