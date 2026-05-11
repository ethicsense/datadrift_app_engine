import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import { pickNumeric } from "../../utils";

export default function TimeSeriesCard({ data }) {
  if (!data) return null;
  const summary = data.summary || data.stats || data;
  const numeric = pickNumeric(summary);
  if (!Object.keys(numeric).length) return null;
  return (
    <CardContainer title="TimeSeries">
      <MetricGrid data={numeric} />
    </CardContainer>
  );
}
