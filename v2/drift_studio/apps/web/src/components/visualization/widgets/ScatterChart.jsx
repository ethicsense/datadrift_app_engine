import React from "react";
import {
  ScatterChart as RCScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function ScatterChart({
  data,
  series,
  height = 280,
  xKey = "x",
  yKey = "y",
  square = false,
}) {
  const hasSeries = Array.isArray(series) && series.length > 0;
  const totalPoints = hasSeries
    ? series.reduce((sum, item) => sum + (item?.data?.length || 0), 0)
    : data?.length || 0;
  if (!totalPoints) return null;

  const allPoints = hasSeries
    ? series.flatMap((item) => item?.data || [])
    : Array.isArray(data)
      ? data
      : [];
  const xValues = allPoints.map((p) => Number(p?.[xKey])).filter(Number.isFinite);
  const yValues = allPoints.map((p) => Number(p?.[yKey])).filter(Number.isFinite);
  const minX = xValues.length ? Math.min(...xValues) : null;
  const maxX = xValues.length ? Math.max(...xValues) : null;
  const minY = yValues.length ? Math.min(...yValues) : null;
  const maxY = yValues.length ? Math.max(...yValues) : null;
  const spanX = minX !== null && maxX !== null ? maxX - minX : 0;
  const spanY = minY !== null && maxY !== null ? maxY - minY : 0;
  const maxSpan = Math.max(spanX, spanY, 1e-6);
  const cx = minX !== null && maxX !== null ? (minX + maxX) / 2 : 0;
  const cy = minY !== null && maxY !== null ? (minY + maxY) / 2 : 0;
  const half = maxSpan / 2;
  const xDomain = square ? [cx - half, cx + half] : ["auto", "auto"];
  const yDomain = square ? [cy - half, cy + half] : ["auto", "auto"];

  return (
    <ResponsiveContainer width="100%" height={square ? undefined : height} aspect={square ? 1 : undefined}>
      <RCScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey} type="number" domain={xDomain} />
        <YAxis dataKey={yKey} type="number" domain={yDomain} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
        {hasSeries ? (
          <>
            {series.map((item, idx) => (
              <Scatter
                key={item?.name || idx}
                name={item?.name}
                data={item?.data || []}
                fill={item?.color || "#22c55e"}
              />
            ))}
            <Legend />
          </>
        ) : (
          <Scatter data={data} fill="#22c55e" />
        )}
      </RCScatterChart>
    </ResponsiveContainer>
  );
}
