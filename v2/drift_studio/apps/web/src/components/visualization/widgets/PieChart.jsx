import React from "react";
import {
  PieChart as RCPieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const COLORS = ["#2563eb", "#16a34a", "#f97316", "#8b5cf6", "#ef4444", "#14b8a6"];

export default function PieChart({ data, height = 260, valueKey = "value" }) {
  if (!data || !data.length) return null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RCPieChart>
        <Pie
          data={data}
          dataKey={valueKey}
          nameKey="name"
          outerRadius={90}
          innerRadius={40}
          label
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${entry.name}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </RCPieChart>
    </ResponsiveContainer>
  );
}
