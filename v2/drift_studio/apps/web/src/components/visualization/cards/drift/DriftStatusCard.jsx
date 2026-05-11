import React from "react";
import CardContainer from "../common/CardContainer";
import GaugeChart from "../../widgets/GaugeChart";
import { formatNumber } from "../../utils";

const STATUS_STYLES = {
  CRITICAL: "bg-red-50 border-red-200 text-red-800",
  WARNING: "bg-yellow-50 border-yellow-200 text-yellow-800",
  NORMAL: "bg-green-50 border-green-200 text-green-800",
};

export default function DriftStatusCard({ data }) {
  if (!data) return null;
  const status = data.status;
  const overallScore = data.overallScore;
  const modality = data.modality;
  const style = STATUS_STYLES[status] || "bg-gray-50 border-gray-200 text-gray-700";

  return (
    <CardContainer title="Drift 요약" className={style}>
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="text-lg font-semibold">{status || "UNKNOWN"}</div>
          {modality && (
            <div className="text-xs mt-1 opacity-80">modality: {modality}</div>
          )}
          {overallScore !== undefined && (
            <div className="text-2xl font-semibold mt-3 tabular-nums">
              {formatNumber(overallScore)}
              <span className="text-sm font-normal ml-1 opacity-80">overall</span>
            </div>
          )}
        </div>
        {overallScore !== undefined && (
          <div className="w-40 flex-shrink-0">
            <GaugeChart value={overallScore} max={overallScore > 1 ? overallScore : 1} />
          </div>
        )}
      </div>
    </CardContainer>
  );
}
