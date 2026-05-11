import React, { useMemo } from "react";
import CardContainer from "../common/CardContainer";
import { formatNumber } from "../../utils";

export default function TrainlogDriftPairsCard({ data }) {
  const pairs = Array.isArray(data?.pairs) ? data.pairs : [];
  if (!pairs.length) return null;
  const rows = useMemo(() => pairs.slice(0, 10), [pairs]);
  return (
    <CardContainer title="MLflow Drift (Matched Pairs)">
      <div className="space-y-3">
        <div className="text-xs text-gray-600">
          pairs: {pairs.length} (showing {rows.length})
        </div>
        <div className="overflow-auto border rounded">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-2 py-1">signature</th>
                <th className="text-left px-2 py-1">base</th>
                <th className="text-left px-2 py-1">target</th>
                <th className="text-left px-2 py-1">delta metrics</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.signature}-${row.base_run_id}-${row.target_run_id}`} className="border-t">
                  <td className="px-2 py-1 font-mono">{row.signature}</td>
                  <td className="px-2 py-1">{row.base_run_name || row.base_run_id}</td>
                  <td className="px-2 py-1">{row.target_run_name || row.target_run_id}</td>
                  <td className="px-2 py-1">
                    {row.delta_final_metrics
                      ? Object.entries(row.delta_final_metrics)
                          .slice(0, 3)
                          .map(([key, value]) => `${key}: ${formatNumber(value)}`)
                          .join(", ")
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </CardContainer>
  );
}
