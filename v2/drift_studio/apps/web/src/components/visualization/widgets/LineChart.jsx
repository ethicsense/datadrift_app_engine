import React from "react";
import {
  LineChart as RCLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function LineChart({ data, height = 280, valueKey = "value" }) {
  if (!data || !data.length) return null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RCLineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey={valueKey} stroke="#0ea5e9" strokeWidth={2} />
      </RCLineChart>
    </ResponsiveContainer>
  );
}
