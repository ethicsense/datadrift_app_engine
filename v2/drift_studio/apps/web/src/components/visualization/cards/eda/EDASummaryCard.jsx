import React from "react";
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

export default function EDASummaryCard({ data }) {
  if (!data?.summary) return null;
  const numeric = pickNumeric(
    omitKeys(data.summary, ["label_distributions", "distributions"])
  );
  if (!Object.keys(numeric).length) return null;
  const chartData = Object.entries(toTopN(numeric)).map(([key, value]) => ({
    name: formatMetricLabel(key),
    value,
  }));
  return (
    <CardContainer title="EDA 요약">
      <div className="space-y-4">
        <MetricGrid data={numeric} />
        {chartData.length > 0 && (
          <div>
            <div className="text-xs text-gray-600 mb-2">요약 그래프</div>
            <BarChart data={chartData} height={240} />
          </div>
        )}
      </div>
    </CardContainer>
  );
}
