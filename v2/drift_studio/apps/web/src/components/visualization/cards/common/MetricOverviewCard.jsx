import React from "react";
import CardContainer from "./CardContainer";
import MetricGrid from "../../widgets/MetricGrid";

export default function MetricOverviewCard({ title, data }) {
  if (!data || !Object.keys(data).length) return null;
  return (
    <CardContainer title={title}>
      <MetricGrid data={data} />
    </CardContainer>
  );
}
