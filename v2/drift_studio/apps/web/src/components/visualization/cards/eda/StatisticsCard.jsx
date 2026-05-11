import React, { useMemo, useState } from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import BarChart from "../../widgets/BarChart";
import { pickNumeric, omitKeys, formatMetricLabel } from "../../utils";

function toTopN(obj, n = 12) {
  const entries = Object.entries(obj || {}).sort((a, b) => {
    const va = Math.abs(a[1] ?? 0);
    const vb = Math.abs(b[1] ?? 0);
    return vb - va;
  });
  return Object.fromEntries(entries.slice(0, n));
}

function splitColumnMetric(key) {
  const raw = String(key || "");
  const idx = raw.lastIndexOf(".");
  if (idx <= 0 || idx >= raw.length - 1) return null;
  return {
    column: raw.slice(0, idx),
    metric: raw.slice(idx + 1),
  };
}

function buildColumnMetrics(numeric) {
  const grouped = {};
  Object.entries(numeric || {}).forEach(([key, value]) => {
    const parsed = splitColumnMetric(key);
    if (!parsed) return;
    const { column, metric } = parsed;
    if (!grouped[column]) grouped[column] = {};
    grouped[column][metric] = value;
  });

  return Object.entries(grouped)
    .map(([column, metrics]) => ({
      column,
      metrics,
    }))
    .sort((a, b) => a.column.localeCompare(b.column));
}

export default function StatisticsCard({ data }) {
  const stats = data?.stats || null;
  if (!stats && data?.summary) return null;
  if (!stats) return null;
  const numeric = pickNumeric(
    omitKeys(stats, ["label_distributions", "distributions", "summary"])
  );
  if (!Object.keys(numeric).length) return null;
  const chartData = Object.entries(toTopN(numeric)).map(([key, value]) => ({
    name: formatMetricLabel(key),
    value,
  }));
  const columnMetrics = useMemo(() => buildColumnMetrics(numeric), [numeric]);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const activeColumn = selectedColumn || columnMetrics[0]?.column || null;
  const activeMetrics = useMemo(() => {
    const found = columnMetrics.find((item) => item.column === activeColumn);
    return found?.metrics || {};
  }, [columnMetrics, activeColumn]);
  const activeChartData = Object.entries(activeMetrics).map(([metric, value]) => ({
    name: formatMetricLabel(metric),
    value,
  }));

  return (
    <CardContainer title="통계 지표">
      <div className="space-y-4">
        {columnMetrics.length > 0 && (
          <div className="space-y-4">
            <div className="text-xs text-gray-600">컬럼별 통계</div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {columnMetrics.map(({ column }) => {
                const isActive = column === activeColumn;
                return (
                  <button
                    key={column}
                    type="button"
                    onClick={() => setSelectedColumn(column)}
                    className={`px-3 py-1.5 rounded-full border text-xs whitespace-nowrap ${
                      isActive
                        ? "bg-indigo-600 text-white border-indigo-600"
                        : "bg-white text-gray-700 border-gray-300 hover:border-gray-400"
                    }`}
                  >
                    {column}
                  </button>
                );
              })}
            </div>
            {activeColumn && (
              <div className="space-y-3">
                <div className="text-xs text-gray-600">
                  선택 컬럼: <span className="font-medium text-gray-800">{activeColumn}</span>
                </div>
                <MetricGrid data={activeMetrics} />
                {activeChartData.length > 0 && <BarChart data={activeChartData} height={220} />}
              </div>
            )}
          </div>
        )}
        {chartData.length > 0 && (
          <div>
            <div className="text-xs text-gray-600 mb-2">통계 그래프</div>
            <BarChart data={chartData} height={240} />
          </div>
        )}
      </div>
    </CardContainer>
  );
}
