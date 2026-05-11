import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import { pickNumeric } from "../../utils";

function formatTime(ts) {
  if (ts === null || ts === undefined) return "-";
  try {
    return new Date(Number(ts)).toLocaleString();
  } catch {
    return String(ts);
  }
}

export default function TrainlogSummaryCard({ data }) {
  const summary = data?.summary;
  if (!summary) return null;
  const display = { ...pickNumeric(summary) };
  if (summary?.earliest_start_time !== undefined) {
    display.earliest_start_time = formatTime(summary.earliest_start_time);
  }
  if (summary?.latest_end_time !== undefined) {
    display.latest_end_time = formatTime(summary.latest_end_time);
  }
  return (
    <CardContainer title="MLflow Summary">
      <div className="space-y-4">
        <MetricGrid data={display} />
        {summary?.earliest_run_id && (
          <div className="text-xs text-gray-600">earliest_run_id: {summary.earliest_run_id}</div>
        )}
        {summary?.latest_run_id && (
          <div className="text-xs text-gray-600">latest_run_id: {summary.latest_run_id}</div>
        )}
        {Array.isArray(summary?.overlay_defaults) && summary.overlay_defaults.length > 0 && (
          <div className="text-xs text-gray-600">
            기본 오버레이 메트릭: {summary.overlay_defaults.join(", ")}
          </div>
        )}
      </div>
    </CardContainer>
  );
}
