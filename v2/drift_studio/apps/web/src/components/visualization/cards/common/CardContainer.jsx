import React from "react";

export default function CardContainer({ title, children, className = "" }) {
  return (
    <div
      className={`p-4 border border-gray-200 rounded-lg bg-white shadow-sm ${className}`}
    >
      {title && <div className="text-sm font-semibold mb-3">{title}</div>}
      {children}
    </div>
  );
}
