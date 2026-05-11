import React from "react";
import {
  BarChart as RCBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function CompareBarChart({
  data,
  height = 280,
  baseKey = "base",
  targetKey = "target",
  nameKey = "name",
  tickFormatter,
  tooltipRenderer,
}) {
  if (!data || !data.length) return null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RCBarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey={nameKey}
          interval={0}
          angle={-25}
          textAnchor="end"
          height={60}
          tickFormatter={tickFormatter}
        />
        <YAxis />
        <Tooltip content={tooltipRenderer} />
        <Legend />
        <Bar dataKey={baseKey} fill="#2563eb" fillOpacity={0.6} />
        <Bar dataKey={targetKey} fill="#ef4444" fillOpacity={0.6} />
      </RCBarChart>
    </ResponsiveContainer>
  );
}

