import React from "react";
import { formatMetricLabel, formatMetricValue } from "../utils";

export default function MetricGrid({ data }) {
  if (!data || !Object.keys(data).length) return null;
  return (
    <div className="grid grid-cols-2 gap-3 text-sm">
      {Object.entries(data).map(([key, value]) => (
        <div key={key} className="flex items-center justify-between">
          <span className="text-gray-600">{formatMetricLabel(key)}</span>
          <span className="font-medium">{formatMetricValue(key, value)}</span>
        </div>
      ))}
    </div>
  );
}
