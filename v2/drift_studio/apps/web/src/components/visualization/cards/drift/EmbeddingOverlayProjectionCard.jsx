import React from "react";
import CardContainer from "../common/CardContainer";
import ScatterChart from "../../widgets/ScatterChart";

export default function EmbeddingOverlayProjectionCard({ data }) {
  const points = data?.points || [];
  if (!points.length) return null;

  const basePoints = points.filter((p) => p?.split === "base");
  const targetPoints = points.filter((p) => p?.split === "target");

  const sampling = data?.sampling;

  return (
    <CardContainer title="Embedding Projection (Base vs Target)">
      <div className="text-xs text-gray-600 mb-2">
        {data?.method ? `method: ${data.method}` : "method: pca"}
        {sampling?.n !== undefined && sampling?.cap !== undefined && (
          <span className="ml-2">
            samples: {sampling.n} / cap {sampling.cap}
          </span>
        )}
      </div>
      <ScatterChart
        series={[
          { name: "base", data: basePoints, color: "#7c3aed" },
          { name: "target", data: targetPoints, color: "#84cc16" },
        ]}
        height={320}
      />
    </CardContainer>
  );
}

