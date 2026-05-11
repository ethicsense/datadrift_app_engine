import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import BarChart from "../../widgets/BarChart";
import ScatterChart from "../../widgets/ScatterChart";
import { pickNumeric, toChartData } from "../../utils";

export default function VisionImageCard({ data }) {
  if (!data) return null;
  const summary = data.summary || data.stats || data;
  const attributeDrifts = data.attributeDrifts;
  const numeric = pickNumeric(summary);
  const projection = data.embedding_projection;

  return (
    <CardContainer title="Vision Image">
      {Object.keys(numeric).length > 0 && (
        <div className="mb-4">
          <MetricGrid data={numeric} />
        </div>
      )}
      {attributeDrifts && (
        <div>
          <div className="text-xs text-gray-600 mb-2">attribute_drifts</div>
          <BarChart data={toChartData(attributeDrifts)} />
        </div>
      )}
      {projection?.points?.length ? (
        <div className="mt-4">
          <div className="text-xs text-gray-600 mb-2">
            embedding_projection ({projection.method})
          </div>
          <ScatterChart data={projection.points} />
        </div>
      ) : null}
    </CardContainer>
  );
}
