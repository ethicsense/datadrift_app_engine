import React, { useMemo, useState } from "react";
import CardContainer from "../common/CardContainer";
import MultiLineChart from "../../widgets/MultiLineChart";

const MAX_OVERLAY = 20;

function sampleRuns(runs) {
  if (runs.length <= MAX_OVERLAY) return runs;
  const latest = runs.slice(0, Math.floor(MAX_OVERLAY / 2));
  const rest = runs.slice(Math.floor(MAX_OVERLAY / 2));
  const sampled = rest.slice(0, MAX_OVERLAY - latest.length);
  return [...latest, ...sampled];
}

export default function TrainlogMetricsOverlayCard({ data }) {
  const runs = Array.isArray(data?.runs) ? data.runs : [];
  const metricsIndex = data?.metrics || {};
  const overlayDefaults = Array.isArray(data?.summary?.overlay_defaults)
    ? data.summary.overlay_defaults
    : [];

  const metricKeys = useMemo(() => {
    const all = new Set(overlayDefaults);
    Object.values(metricsIndex || {}).forEach((runMetrics) => {
      Object.keys(runMetrics || {}).forEach((key) => all.add(key));
    });
    return Array.from(all);
  }, [metricsIndex, overlayDefaults]);

  const [selectedMetric, setSelectedMetric] = useState(metricKeys[0] || "");
  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => (b.start_time || 0) - (a.start_time || 0)),
    [runs]
  );
  const overlayRuns = sampleRuns(sortedRuns);

  const series = overlayRuns.map((run) => {
    const metrics = metricsIndex?.[run.run_id] || {};
    const points = (metrics[selectedMetric] || []).map((item) => ({
      step: item.step,
      value: item.value,
    }));
    return { name: run.run_name || run.run_id, data: points };
  });

  if (!selectedMetric || !series.some((item) => item.data && item.data.length)) return null;

  return (
    <CardContainer title="Metrics Overlay">
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-600">metric</span>
          <select
            className="border rounded px-2 py-1 text-xs"
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            {metricKeys.map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
          {runs.length > MAX_OVERLAY && (
            <span className="text-gray-500">
              {overlayRuns.length} / {runs.length} runs (샘플링)
            </span>
          )}
        </div>
        <MultiLineChart series={series} height={320} xKey="step" />
      </div>
    </CardContainer>
  );
}
