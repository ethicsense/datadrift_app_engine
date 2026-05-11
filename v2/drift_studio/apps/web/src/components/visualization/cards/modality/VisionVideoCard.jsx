import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import { pickNumeric } from "../../utils";

export default function VisionVideoCard({ data }) {
  if (!data) return null;
  const summary = data.summary || data.stats || data;
  const numeric = pickNumeric(summary);
  if (!Object.keys(numeric).length) return null;
  return (
    <CardContainer title="Vision Video">
      <MetricGrid data={numeric} />
    </CardContainer>
  );
}
