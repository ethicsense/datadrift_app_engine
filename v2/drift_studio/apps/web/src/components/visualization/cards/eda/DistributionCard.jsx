import React, { useState } from "react";
import CardContainer from "../common/CardContainer";
import BarChart from "../../widgets/BarChart";
import { toChartData, formatNumber, formatMetricValue, formatMetricLabel } from "../../utils";

function toTopN(obj, n = 12) {
  const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
  return Object.fromEntries(entries.slice(0, n));
}

export default function DistributionCard({ data, cardConfig, title }) {
  const cardTitle = title || cardConfig?.title || cardConfig?.name || "Distributions";
  const histDistributions = data?.distributions;
  const labelDistributions =
    data?.summary?.label_distributions || data?.summary?.distributions;
  const hasLabelDistributions =
    labelDistributions && typeof labelDistributions === "object";
  const hasHistograms =
    histDistributions && typeof histDistributions === "object";

  if (!hasLabelDistributions && !hasHistograms) return null;

  const [binCount, setBinCount] = useState(20);

  const formatRangeLabel = (name, min) => {
    if (min === null || min === undefined) return "";
    return `${formatMetricValue(name, min)}`;
  };

  const histogramToChart = (name, hist) => {
    const normalized = hist?.histogram || hist;
    if (!normalized?.bins || !normalized?.counts) return [];
    const bins = normalized.bins;
    const counts = normalized.counts;
    const isSharpness = String(name || "").toLowerCase().includes("sharpness");
    return counts.map((count, idx) => {
      const rawMin = bins[idx];
      const rawMax = bins[idx + 1];
      const label = formatRangeLabel(name, rawMin);
      return {
        name: label,
        value: count,
        rawMin,
        rawMax,
      };
    });
  };

  const rebinSamples = (name, samples, bins) => {
    if (!samples || samples.length === 0) return [];
    const isSharpness = String(name || "").toLowerCase().includes("sharpness");
    const normalized = isSharpness
      ? samples
          .filter((v) => typeof v === "number" && v > 0)
          .map((v) => Math.log10(v))
      : samples;
    const sorted = [...normalized].sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    if (min === max) {
      return [
        {
          name: formatRangeLabel(name, min),
          value: sorted.length,
          rawMin: min,
          rawMax: min,
        },
      ];
    }
    const step = (max - min) / bins;
    const counts = new Array(bins).fill(0);
    for (const v of sorted) {
      const idx = Math.min(Math.floor((v - min) / step), bins - 1);
      counts[idx] += 1;
    }
    return counts.map((count, idx) => {
      const rawMin = min + step * idx;
      const rawMax = min + step * (idx + 1);
      return {
        name: formatRangeLabel(name, rawMin),
        value: count,
        rawMin,
        rawMax,
      };
    });
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
            <div>range: {formatMetricValue(name, rawMin)} - {formatMetricValue(name, rawMax)}</div>
          </>
        )}
        <div>count: {item.value}</div>
      </div>
    );
  };

  return (
    <CardContainer title={cardTitle}>
      <div className="space-y-6">
        {hasHistograms && (
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span>bins</span>
            <select
              className="border rounded px-2 py-1 text-xs"
              value={binCount}
              onChange={(e) => setBinCount(Number(e.target.value))}
            >
              {[10, 20, 30, 40].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
        )}
        {hasHistograms &&
          Object.entries(histDistributions).map(([name, hist]) => {
            const categoricalFreqs = hist?.frequencies;
            if (categoricalFreqs && typeof categoricalFreqs === "object") {
              const top = toTopN(categoricalFreqs);
              const chartData = toChartData(top);
              if (!chartData.length) return null;
              return (
                <div key={name}>
                  <div className="text-xs font-medium text-gray-600 mb-2">
                    {formatMetricLabel(name)} (category frequencies)
                  </div>
                  <BarChart data={chartData} height={240} />
                </div>
              );
            }
            const isSharpness = String(name || "").toLowerCase().includes("sharpness");
            const histogram = hist?.histogram || hist;
            const chartData = histogram?.samples
              ? rebinSamples(name, histogram.samples, binCount)
              : histogramToChart(name, hist);
            if (!chartData.length) return null;
            const tickStep = Math.max(1, Math.ceil(chartData.length / 6));
            return (
              <div key={name}>
                <div className="text-xs font-medium text-gray-600 mb-2">
                  {formatMetricLabel(name)} {isSharpness ? "(log10 histogram)" : "(histogram)"}
                </div>
                <BarChart
                  data={chartData}
                  height={240}
                  tickFormatter={(label, idx) => (idx % tickStep === 0 ? label : "")}
                  tooltipRenderer={tooltipRenderer(name)}
                />
              </div>
            );
          })}
        {hasLabelDistributions &&
          Object.entries(labelDistributions).map(([name, dist]) => {
            const top = toTopN(dist);
            const chartData = toChartData(top);
            if (!chartData.length) return null;
            return (
              <div key={name}>
                <div className="text-xs font-medium text-gray-600 mb-2">{name}</div>
                <BarChart data={chartData} height={240} />
              </div>
            );
          })}
      </div>
    </CardContainer>
  );
}
