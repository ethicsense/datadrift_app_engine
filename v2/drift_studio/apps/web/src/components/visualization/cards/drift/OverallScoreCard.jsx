import React from "react";
import CardContainer from "../common/CardContainer";
import GaugeChart from "../../widgets/GaugeChart";
import { formatNumber } from "../../utils";

export default function OverallScoreCard({ data }) {
  if (data?.overallScore === undefined) return null;
  const max = data.overallScore > 1 ? data.overallScore : 1;
  return (
    <CardContainer title="Overall Score">
      <div className="flex items-center justify-between gap-4">
        <div className="text-2xl font-semibold">{formatNumber(data.overallScore)}</div>
        <div className="w-40">
          <GaugeChart value={data.overallScore} max={max} />
        </div>
      </div>
    </CardContainer>
  );
}
