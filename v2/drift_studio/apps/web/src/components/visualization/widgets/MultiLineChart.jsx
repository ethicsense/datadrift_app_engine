import React from "react";
import {
  LineChart as RCLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const PALETTE = ["#2563eb", "#ef4444", "#10b981", "#f97316", "#8b5cf6", "#14b8a6"];

export default function MultiLineChart({
  series = [],
  height = 320,
  xKey = "step",
}) {
  if (!Array.isArray(series) || series.length === 0) return null;

  const keys = series.map((item, idx) => ({
    key: `s${idx}`,
    name: item?.name || `run-${idx + 1}`,
    color: item?.color || PALETTE[idx % PALETTE.length],
    data: Array.isArray(item?.data) ? item.data : [],
  }));

  const steps = new Set();
  keys.forEach((item) => {
    item.data.forEach((point) => {
      if (point?.[xKey] !== undefined) {
        steps.add(point[xKey]);
      }
    });
  });
  const sortedSteps = Array.from(steps).sort((a, b) => Number(a) - Number(b));

  const data = sortedSteps.map((step) => {
    const row = { [xKey]: step };
    keys.forEach((item) => {
      const match = item.data.find((p) => p?.[xKey] === step);
      if (match && match.value !== undefined) {
        row[item.key] = match.value;
      }
    });
    return row;
  });

  if (!data.length) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RCLineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {keys.map((item) => (
          <Line
            key={item.key}
            type="monotone"
            dataKey={item.key}
            name={item.name}
            stroke={item.color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </RCLineChart>
    </ResponsiveContainer>
  );
}
