import React from "react";
import CardContainer from "./CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import { omitKeys, pickNumeric } from "../../utils";

export default function SummaryCard({ data, title = "요약" }) {
  if (!data || typeof data !== "object") return null;
  const numeric = pickNumeric(omitKeys(data, ["label_distributions", "distributions"]));
  if (!Object.keys(numeric).length) return null;
  return (
    <CardContainer title={title}>
      <MetricGrid data={numeric} />
    </CardContainer>
  );
}
