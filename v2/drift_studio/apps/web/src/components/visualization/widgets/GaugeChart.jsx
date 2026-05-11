import React from "react";
import { PieChart as RCPieChart, Pie, Cell, ResponsiveContainer } from "recharts";

export default function GaugeChart({ value, max = 1, height = 180 }) {
  if (value === null || value === undefined) return null;
  const safeMax = max || 1;
  const clamped = Math.max(0, Math.min(value, safeMax));
  const data = [
    { name: "value", value: clamped },
    { name: "rest", value: Math.max(safeMax - clamped, 0) },
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RCPieChart>
        <Pie
          data={data}
          dataKey="value"
          startAngle={180}
          endAngle={0}
          innerRadius={60}
          outerRadius={80}
          paddingAngle={2}
        >
          <Cell fill="#3b82f6" />
          <Cell fill="#e5e7eb" />
        </Pie>
      </RCPieChart>
    </ResponsiveContainer>
  );
}
