import React, { useMemo, useState } from "react";
import CardContainer from "../common/CardContainer";

function formatTime(ts) {
  if (!ts) return "-";
  try {
    return new Date(Number(ts)).toLocaleString();
  } catch {
    return String(ts);
  }
}

export default function TrainlogRunListCard({ data }) {
  const runs = Array.isArray(data?.runs) ? data.runs : [];
  if (!runs.length) return null;
  const [limit, setLimit] = useState(10);
  const sorted = useMemo(
    () => [...runs].sort((a, b) => (b.start_time || 0) - (a.start_time || 0)),
    [runs]
  );
  const showRuns = sorted.slice(0, limit);
  return (
    <CardContainer title="MLflow Runs">
      <div className="space-y-3">
        <div className="text-xs text-gray-600">
          total: {runs.length} (showing {showRuns.length})
        </div>
        <div className="overflow-auto border rounded">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-2 py-1">run</th>
                <th className="text-left px-2 py-1">name</th>
                <th className="text-left px-2 py-1">user</th>
                <th className="text-left px-2 py-1">status</th>
                <th className="text-left px-2 py-1">start</th>
                <th className="text-left px-2 py-1">end</th>
              </tr>
            </thead>
            <tbody>
              {showRuns.map((run) => (
                <tr key={run.run_id} className="border-t">
                  <td className="px-2 py-1 font-mono">{run.run_id}</td>
                  <td className="px-2 py-1">{run.run_name || "-"}</td>
                  <td className="px-2 py-1">{run.user_id || "-"}</td>
                  <td className="px-2 py-1">{run.status ?? "-"}</td>
                  <td className="px-2 py-1">{formatTime(run.start_time)}</td>
                  <td className="px-2 py-1">{formatTime(run.end_time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {limit < runs.length && (
          <button
            type="button"
            onClick={() => setLimit((prev) => Math.min(prev + 10, runs.length))}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            더 보기
          </button>
        )}
      </div>
    </CardContainer>
  );
}
