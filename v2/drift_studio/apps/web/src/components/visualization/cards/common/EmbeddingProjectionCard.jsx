import React, { useMemo } from "react";
import CardContainer from "./CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import BarChart from "../../widgets/BarChart";
import ScatterChart from "../../widgets/ScatterChart";

const COLORS = [
  "#7c3aed",
  "#22c55e",
  "#06b6d4",
  "#f59e0b",
  "#ef4444",
  "#3b82f6",
  "#a855f7",
  "#84cc16",
  "#14b8a6",
  "#e11d48",
  "#6366f1",
  "#f97316",
];

function normalizeClusterId(value) {
  if (value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : String(value);
}

function clusterLabel(clusterId) {
  if (clusterId === -1) return "noise";
  if (typeof clusterId === "number") return `cluster_${clusterId}`;
  return String(clusterId);
}

export default function EmbeddingProjectionCard({ data }) {
  const projection = data?.projection;
  const clustering = data?.clustering;
  const clusterProjection = clustering?.projection;
  const points = clusterProjection?.points || projection?.points || [];
  const clusters = Array.isArray(clustering?.clusters) ? clustering.clusters : [];

  if (!points.length) return null;

  const nClusters =
    typeof clustering?.n_clusters === "number" ? clustering.n_clusters : clusters.length || null;

  const metrics = {
    method: clustering?.method || "-",
    n_clusters: nClusters ?? "-",
    total_points: clusterProjection?.sampling?.total ?? projection?.points?.length ?? points.length,
    sampled_points: points.length,
  };

  const sizeChartData = useMemo(() => {
    if (!clusters.length) return [];
    return clusters
      .map((c) => ({
        name: clusterLabel(normalizeClusterId(c?.id)),
        value: c?.size ?? 0,
      }))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
  }, [clusters]);

  const clusteredSeries = useMemo(() => {
    const hasCluster = points.some((p) => p?.cluster !== undefined && p?.cluster !== null);
    if (!hasCluster) return [];

    const map = new Map();
    for (const p of points) {
      const cid = normalizeClusterId(p?.cluster);
      if (cid === null) continue;
      if (!map.has(cid)) map.set(cid, []);
      map.get(cid).push(p);
    }

    const keys = Array.from(map.keys()).sort((a, b) => {
      if (a === -1) return 1;
      if (b === -1) return -1;
      if (typeof a === "number" && typeof b === "number") return a - b;
      return String(a).localeCompare(String(b));
    });

    return keys.map((cid, idx) => ({
      name: clusterLabel(cid),
      data: map.get(cid) || [],
      color: COLORS[idx % COLORS.length],
    }));
  }, [points]);

  return (
    <CardContainer title="Embedding Projection">
      <div className="space-y-4">
        <div className="text-xs text-gray-600 mb-2">
          embedding projection {projection?.method ? `(${projection.method})` : ""}
          {clusterProjection?.sampling?.n !== undefined && clusterProjection?.sampling?.cap !== undefined && (
            <span className="ml-2">
              samples: {clusterProjection.sampling.n} / cap {clusterProjection.sampling.cap}
            </span>
          )}
        </div>
        {clusteredSeries.length > 0 ? (
          <ScatterChart series={clusteredSeries} square />
        ) : (
          <ScatterChart data={points} square />
        )}

        {clusters.length > 0 && (
          <>
            <MetricGrid data={metrics} />
            {sizeChartData.length > 0 && (
              <div>
                <div className="text-xs text-gray-600 mb-2">cluster sizes</div>
                <BarChart data={sizeChartData} height={240} />
              </div>
            )}
            <details className="text-xs">
              <summary className="cursor-pointer text-gray-600">클러스터 상세(대표 샘플/유사도)</summary>
              <div className="mt-2 space-y-2">
                {clusters
                  .slice()
                  .sort((a, b) => {
                    const aa = normalizeClusterId(a?.id);
                    const bb = normalizeClusterId(b?.id);
                    if (aa === -1) return 1;
                    if (bb === -1) return -1;
                    if (typeof aa === "number" && typeof bb === "number") return aa - bb;
                    return String(aa).localeCompare(String(bb));
                  })
                  .map((c) => (
                    <div key={String(c?.id)} className="border rounded p-2 bg-gray-50">
                      <div className="font-medium">
                        {clusterLabel(normalizeClusterId(c?.id))} · size {c?.size ?? "-"}
                      </div>
                      <div className="text-gray-600">
                        avg_similarity: {c?.avg_similarity ?? "-"} / min: {c?.min_similarity ?? "-"} / max:{" "}
                        {c?.max_similarity ?? "-"}
                      </div>
                    </div>
                  ))}
              </div>
            </details>
          </>
        )}
      </div>
    </CardContainer>
  );
}
