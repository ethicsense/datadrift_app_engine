import React from "react";
import CardContainer from "../common/CardContainer";
import BarChart from "../../widgets/BarChart";
import ScatterChart from "../../widgets/ScatterChart";
import MetricGrid from "../../widgets/MetricGrid";
import { toChartData, pickNumeric, omitKeys } from "../../utils";

export default function TextCard({ data }) {
  if (!data) return null;
  const attributeDrifts = data.attributeDrifts || data.attribute_drifts;
  const summary = data.summary || data.stats || data;
  const numeric = pickNumeric(omitKeys(summary, ["label_distributions", "distributions"]));
  const projection = data.embedding_projection;

  return (
    <CardContainer title="Text">
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
