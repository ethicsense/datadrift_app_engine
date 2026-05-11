import { useMemo } from "react";

import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";
import "echarts-wordcloud";

type WordCloudPanelProps = {
  rows: Record<string, unknown>[];
  wordKey: string;
  valueKey: string;
  maxWords?: number;
};

type WordFrequency = {
  name: string;
  value: number;
};

function toNumeric(value: unknown) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

/** 다크 배경에서 가독성 있는 스펙트럼 분산 팔레트 */
const palette = [
  "#93c5fd", "#60a5fa", "#38bdf8", "#22d3ee", "#2dd4bf", "#34d399", "#a3e635", "#fbbf24",
  "#fb923c", "#f97316", "#f472b6", "#fb7185", "#fda4af", "#f0abfc", "#d8b4fe", "#c4b5fd",
  "#a78bfa", "#818cf8", "#7dd3fc", "#5eead4", "#86efac", "#fde047", "#fcd34d", "#fdba74",
  "#fca5a5", "#f9a8d4", "#e879f9", "#a5b4fc", "#67e8f9", "#bef264", "#fde68a", "#fecdd3",
  "#ddd6fe", "#bae6fd", "#99f6e4", "#bbf7d0", "#fef08a", "#fed7aa", "#fecaca", "#e9d5ff",
];

export function WordCloudPanel({ rows, wordKey, valueKey, maxWords = 40 }: WordCloudPanelProps) {
  const option = useMemo(() => {
    const normalized: WordFrequency[] = rows
      .map((row) => {
        const name = String(row[wordKey as keyof typeof row] ?? "").trim();
        const value = toNumeric(row[valueKey as keyof typeof row]) ?? 0;
        return { name, value };
      })
      .filter((row) => row.name.length > 0 && row.value > 0)
      .sort((left, right) => right.value - left.value)
      .slice(0, maxWords);

    if (!normalized.length) {
      return null;
    }

    const pickColor = (index: number, name: string) => {
      const salt = name.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
      const i = (index * 17 + salt) % palette.length;
      return palette[i];
    };

    const data = normalized.map((item, index) => ({
      name: item.name,
      value: item.value,
      textStyle: {
        color: pickColor(index, item.name),
      },
    }));

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item" as const,
        backgroundColor: "rgba(12,12,14,0.96)",
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: "#d4d4d8", fontSize: 15 },
        formatter: (params: { name?: string; value?: number }) => {
          const v = typeof params.value === "number" ? params.value : 0;
          return [`<strong>${params.name ?? "-"}</strong>`, `빈도: ${v.toLocaleString("ko-KR")}`].join("<br/>");
        },
      },
      series: [
        {
          type: "wordCloud" as const,
          shape: "circle" as const,
          left: "center" as const,
          top: "center" as const,
          width: "90%",
          height: "90%",
          sizeRange: [16, 56] as [number, number],
          rotationRange: [-45, 45] as [number, number],
          rotationStep: 15,
          gridSize: 6,
          drawOutOfBound: false,
          layoutAnimation: true,
          textStyle: {
            fontFamily: "Inter, system-ui, sans-serif",
            fontWeight: 700 as const,
          },
          emphasis: {
            textStyle: {
              shadowBlur: 14,
              shadowColor: "rgba(255,255,255,0.22)",
            },
          },
          data,
        },
      ],
    };
  }, [rows, wordKey, valueKey, maxWords]);

  if (!option) {
    return <div className="empty-state">표시할 단어 빈도 데이터가 없습니다.</div>;
  }

  return (
    <div className="word-cloud-echarts">
      <ReactECharts
        echarts={echarts}
        option={option}
        notMerge
        lazyUpdate
        style={{ height: 400, width: "100%", minHeight: 320 }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
