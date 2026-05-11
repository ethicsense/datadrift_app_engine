import React from "react";
import CardContainer from "../common/CardContainer";

export default function TrainlogPreviewImageCard({ data }) {
  if (!data?.data || !data?.mime) return null;
  const src = `data:${data.mime};base64,${data.data}`;
  return (
    <CardContainer title="MLflow Preview">
      <div className="space-y-2">
        {data?.path && <div className="text-xs text-gray-600">{data.path}</div>}
        <img src={src} alt="mlflow preview" className="max-w-full rounded border" />
      </div>
    </CardContainer>
  );
}
