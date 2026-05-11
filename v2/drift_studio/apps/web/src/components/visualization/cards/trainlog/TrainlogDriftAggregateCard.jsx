import React, { useMemo } from "react";
import CardContainer from "../common/CardContainer";
import { formatNumber } from "../../utils";

export default function TrainlogDriftAggregateCard({ data }) {
  if (!data?.metrics) return null;
  const metrics = data.metrics || {};
  const rows = useMemo(() => {
    return Object.entries(metrics)
      .map(([name, item]) => ({
        name,
        normalized: item?.normalized_delta ?? 0,
        base: item?.base || {},
        target: item?.target || {},
        deltaMean: item?.delta_mean,
      }))
      .sort((a, b) => Math.abs(b.normalized) - Math.abs(a.normalized))
      .slice(0, 12);
  }, [metrics]);

  return (
    <CardContainer title="MLflow Drift (Aggregate)">
      <div className="space-y-3">
        <div className="text-xs text-gray-600">
          base runs: {data?.base?.runs ?? 0}, target runs: {data?.target?.runs ?? 0}
        </div>
        <div className="overflow-auto border rounded">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-2 py-1">metric</th>
                <th className="text-left px-2 py-1">base mean</th>
                <th className="text-left px-2 py-1">target mean</th>
                <th className="text-left px-2 py-1">delta mean</th>
                <th className="text-left px-2 py-1">norm delta</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name} className="border-t">
                  <td className="px-2 py-1">{row.name}</td>
                  <td className="px-2 py-1">{formatNumber(row.base?.mean)}</td>
                  <td className="px-2 py-1">{formatNumber(row.target?.mean)}</td>
                  <td className="px-2 py-1">{formatNumber(row.deltaMean)}</td>
                  <td className="px-2 py-1">{formatNumber(row.normalized)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data?.params && (
          <div className="text-xs text-gray-600">
            params changed ratio: {formatNumber(data.params.changed_ratio, 3)}
          </div>
        )}
      </div>
    </CardContainer>
  );
}
