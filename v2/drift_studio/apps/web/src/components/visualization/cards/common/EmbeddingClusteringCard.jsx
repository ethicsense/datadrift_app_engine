import React, { useMemo } from "react";
import CardContainer from "./CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import BarChart from "../../widgets/BarChart";
import ScatterChart from "../../widgets/ScatterChart";

const COLORS = [
  "#7c3aed", // violet
  "#22c55e", // green
  "#06b6d4", // cyan
  "#f59e0b", // amber
  "#ef4444", // red
  "#3b82f6", // blue
  "#a855f7", // purple
  "#84cc16", // lime
  "#14b8a6", // teal
  "#e11d48", // rose
  "#6366f1", // indigo
  "#f97316", // orange
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

export default function EmbeddingClusteringCard({ data }) {
  const clustering = data?.clustering || data;
  const projection = clustering?.projection;
  const points = projection?.points || [];
  const clusters = Array.isArray(clustering?.clusters) ? clustering.clusters : [];

  if (!projection && !clusters.length) return null;

  const nClusters =
    typeof clustering?.n_clusters === "number"
      ? clustering.n_clusters
      : clusters.length || null;

  const metrics = {
    method: clustering?.method || "kmeans",
    n_clusters: nClusters ?? "-",
    total_points: projection?.sampling?.total ?? (points?.length || 0),
    sampled_points: points?.length || 0,
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

  const series = useMemo(() => {
    if (!points?.length) return [];
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
    <CardContainer title="Embedding Clustering">
      <div className="space-y-4">
        <MetricGrid data={metrics} />

        {sizeChartData.length > 0 && (
          <div>
            <div className="text-xs text-gray-600 mb-2">cluster sizes</div>
            <BarChart data={sizeChartData} height={240} />
          </div>
        )}

        {series.length > 0 && (
          <div>
            <div className="text-xs text-gray-600 mb-2">
              embedding projection + clustering ({projection?.method || "pca"})
              {projection?.sampling?.n !== undefined && projection?.sampling?.cap !== undefined && (
                <span className="ml-2">
                  samples: {projection.sampling.n} / cap {projection.sampling.cap}
                </span>
              )}
            </div>
            <ScatterChart series={series} square />
          </div>
        )}

        {clusters.length > 0 && (
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-600">
              클러스터 상세(대표 샘플/유사도)
            </summary>
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
                    {Array.isArray(c?.top_similar_files) && c.top_similar_files.length > 0 && (
                      <div className="mt-1 text-gray-700">
                        top_similar_files:
                        <ul className="list-disc ml-5">
                          {c.top_similar_files.slice(0, 5).map((item, idx) => (
                            <li key={idx}>
                              {item?.file} ({item?.similarity})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
            </div>
          </details>
        )}
      </div>
    </CardContainer>
  );
}

