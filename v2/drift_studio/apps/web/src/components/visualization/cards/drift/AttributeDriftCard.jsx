import React from "react";
import CardContainer from "../common/CardContainer";
import BarChart from "../../widgets/BarChart";
import { toChartData } from "../../utils";

export default function AttributeDriftCard({ data }) {
  if (!data?.attributeDrifts) return null;
  const chartData = toChartData(data.attributeDrifts);
  if (!chartData.length) return null;
  return (
    <CardContainer title="Attribute Drifts">
      <BarChart data={chartData} />
    </CardContainer>
  );
}
