import React from "react";
import CardContainer from "./CardContainer";

const STATUS_STYLES = {
  CRITICAL: "bg-red-50 border-red-200 text-red-800",
  WARNING: "bg-yellow-50 border-yellow-200 text-yellow-800",
  NORMAL: "bg-green-50 border-green-200 text-green-800",
};

export default function StatusCard({ status, title = "상태" }) {
  if (!status) return null;
  const style = STATUS_STYLES[status] || "bg-gray-50 border-gray-200 text-gray-700";
  return (
    <CardContainer title={title} className={style}>
      <div className="text-lg font-semibold">{status}</div>
    </CardContainer>
  );
}
