import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import BarChart from "../../widgets/BarChart";
import { toChartData } from "../../utils";

export default function EmbeddingDriftCard({ data }) {
  if (!data?.embeddingDriftDetailed && data?.embeddingDrift === undefined) {
    return null;
  }

  const detail = data.embeddingDriftDetailed;
  const normalizedScores = detail?.normalized_scores;

  return (
    <CardContainer title="Embedding Drift">
      {data.embeddingDrift !== undefined && (
        <div className="text-sm mb-3">
          embedding_drift: {data.embeddingDrift}
        </div>
      )}
      {detail && (
        <div className="space-y-4">
          {normalizedScores && (
            <div>
              <div className="text-xs text-gray-600 mb-2">normalized_scores</div>
              <BarChart data={toChartData(normalizedScores)} />
            </div>
          )}
          {detail.weights && (
            <div>
              <div className="text-xs text-gray-600 mb-2">weights</div>
              <MetricGrid data={detail.weights} />
            </div>
          )}
        </div>
      )}
    </CardContainer>
  );
}
