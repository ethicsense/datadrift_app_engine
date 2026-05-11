import type { KpiResponse } from "../types";
import { formatNumber } from "../lib/formatters";

type KpiGridProps = {
  data: KpiResponse;
};

export function KpiGrid({ data }: KpiGridProps) {
  const items = [
    { label: "스냅샷 수", value: data.full.snapshotCount },
    { label: "관측 레코드 수", value: data.full.recordCount },
    { label: "상품 수", value: data.full.productCount },
    { label: "브랜드 수", value: data.full.brandCount },
  ];

  return (
    <div>
      <div className="hero-block">
        <h1>{data.title}</h1>
        <p>{data.subtitle}</p>
      </div>
      <div className="kpi-grid">
        {items.map((item) => (
          <div key={item.label} className="kpi-card">
            <span>{item.label}</span>
            <strong>{formatNumber(item.value, "integer")}</strong>
          </div>
        ))}
      </div>
      <div className="kpi-meta">
        <span>
          현재 보기 {formatNumber(data.filtered.recordCount, "integer")}건 / 스냅샷 {formatNumber(data.filtered.snapshotCount, "integer")}개
        </span>
        <span>
          전체 데이터 범위: {data.dateRange?.min ?? "-"} ~ {data.dateRange?.max ?? "-"}
        </span>
      </div>
    </div>
  );
}
