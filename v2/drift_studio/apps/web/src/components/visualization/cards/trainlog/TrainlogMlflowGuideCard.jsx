import React from "react";
import CardContainer from "../common/CardContainer";

export default function TrainlogMlflowGuideCard({ data }) {
  if (!data?.command && !data?.tracking_dir) return null;
  return (
    <CardContainer title="MLflow UI 가이드">
      <div className="space-y-2 text-sm">
        {data?.note && <div className="text-gray-700">{data.note}</div>}
        {data?.tracking_dir && (
          <div className="text-xs text-gray-600">tracking_dir: {data.tracking_dir}</div>
        )}
        {data?.command && (
          <pre className="bg-gray-50 border rounded p-2 text-xs overflow-auto">
            {data.command}
          </pre>
        )}
      </div>
    </CardContainer>
  );
}
