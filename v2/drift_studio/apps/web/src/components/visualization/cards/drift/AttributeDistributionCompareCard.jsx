import React from "react";
import CardContainer from "../common/CardContainer";
import BarChart from "../../widgets/BarChart";
import { formatMetricLabel, formatMetricValue, formatNumber } from "../../utils";

export default function AttributeDistributionCompareCard({ data }) {
  const metrics = data?.metrics;
  if (!metrics || typeof metrics !== "object") return null;

  const formatRangeLabel = (name, min) => {
    if (min === null || min === undefined) return "";
    return `${formatMetricValue(name, min)}`;
  };

  const tooltipRenderer = (name) => ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;
    const item = payload[0]?.payload;
    if (!item) return null;
    const rawMin = item.rawMin;
    const rawMax = item.rawMax;
    return (
      <div className="bg-white border rounded p-2 text-xs shadow">
        <div className="font-medium">{formatMetricLabel(name)}</div>
        {rawMin !== undefined && rawMax !== undefined && (
          <>
            <div>raw: {formatNumber(rawMin)} - {formatNumber(rawMax)}</div>
            <div>
              range: {formatMetricValue(name, rawMin)} - {formatMetricValue(name, rawMax)}
            </div>
          </>
        )}
        <div>base: {item.base}</div>
        <div>target: {item.target}</div>
      </div>
    );
  };

  return (
    <CardContainer title="Attribute Distributions (Base vs Target)">
      <div className="space-y-6">
        {Object.entries(metrics).map(([name, metric]) => {
          const base = metric?.base;
          const target = metric?.target;
          if (!base?.bins || !base?.counts || !target?.counts) return null;
          const bins = base.bins;
          const baseCounts = base.counts || [];
          const targetCounts = target.counts || [];
          const baseData = baseCounts.map((count, idx) => {
            const rawMin = bins[idx];
            const rawMax = bins[idx + 1];
            return {
              name: formatRangeLabel(name, rawMin),
              value: count,
              rawMin,
              rawMax,
            };
          });
          const targetData = targetCounts.map((count, idx) => {
            const rawMin = bins[idx];
            const rawMax = bins[idx + 1];
            return {
              name: formatRangeLabel(name, rawMin),
              value: count,
              rawMin,
              rawMax,
            };
          });
          const tickStep = Math.max(1, Math.ceil(baseData.length / 6));
          return (
            <div key={name}>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs font-medium text-gray-600">
                  {formatMetricLabel(name)} (histogram)
                </div>
                <div className="text-[11px] text-gray-500">
                  score: {metric?.score ?? "-"}
                  {metric?.method ? ` · ${metric.method}` : ""}
                </div>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <div className="text-[11px] text-gray-500 mb-1">base</div>
                  <BarChart
                    data={baseData}
                    height={220}
                    tickFormatter={(label, idx) => (idx % tickStep === 0 ? label : "")}
                    tooltipRenderer={tooltipRenderer(name)}
                    color="#7c3aed"
                  />
                </div>
                <div>
                  <div className="text-[11px] text-gray-500 mb-1">target</div>
                  <BarChart
                    data={targetData}
                    height={220}
                    tickFormatter={(label, idx) => (idx % tickStep === 0 ? label : "")}
                    tooltipRenderer={tooltipRenderer(name)}
                    color="#84cc16"
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </CardContainer>
  );
}

