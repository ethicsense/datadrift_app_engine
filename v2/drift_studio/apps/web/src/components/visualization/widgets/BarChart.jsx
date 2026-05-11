import React from "react";
import {
  BarChart as RCBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function BarChart({
  data,
  height = 280,
  valueKey = "value",
  nameKey = "name",
  tickFormatter,
  tooltipRenderer,
  color = "#6366f1",
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
        <Bar dataKey={valueKey} fill={color} />
      </RCBarChart>
    </ResponsiveContainer>
  );
}
