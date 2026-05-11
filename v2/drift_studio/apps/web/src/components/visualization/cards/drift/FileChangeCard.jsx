import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";

export default function FileChangeCard({ data }) {
  if (
    data?.filesAdded === undefined &&
    data?.filesRemoved === undefined &&
    data?.filesCommon === undefined
  ) {
    return null;
  }
  const metrics = {
    files_added: data.filesAdded,
    files_removed: data.filesRemoved,
    files_common: data.filesCommon,
  };
  return (
    <CardContainer title="파일 변화">
      <MetricGrid data={metrics} />
    </CardContainer>
  );
}
