import type { DashboardFilters, ExplainabilityFact } from "../types";

function summarizeSelection(values: string[]) {
  if (!values.length) {
    return "전체";
  }
  if (values.length === 1) {
    return values[0];
  }
  return `${values.length}개 선택`;
}

export function describeDateWindow(filters: DashboardFilters) {
  if (filters.dateFrom && filters.dateTo) {
    return `${filters.dateFrom} ~ ${filters.dateTo}`;
  }
  if (filters.dateFrom) {
    return `${filters.dateFrom} 이후`;
  }
  if (filters.dateTo) {
    return `${filters.dateTo} 이전`;
  }
  return "전체 기간";
}

export function buildDashboardFilterBadges(filters: DashboardFilters): ExplainabilityFact[] {
  return [
    { label: "데이터셋", value: filters.dataset ?? "analytics" },
    { label: "브랜드", value: summarizeSelection(filters.brands) },
    { label: "소스", value: summarizeSelection(filters.sourceDatasets) },
    { label: "플랫폼", value: summarizeSelection(filters.platforms) },
    { label: "스키마", value: summarizeSelection(filters.schemaVersions) },
    { label: "기간", value: describeDateWindow(filters) },
    { label: "스냅샷 윈도우", value: `${filters.snapshotWindow ?? 300}개` },
  ];
}

export function describeDashboardFilterScope(filters: DashboardFilters) {
  const source = filters.sourceDatasets.length ? `${filters.sourceDatasets.length}개 소스 데이터셋` : "전체 소스 데이터셋";
  const brands = filters.brands.length ? `${filters.brands.length}개 브랜드` : "전체 브랜드";
  const platforms = filters.platforms.length ? `${filters.platforms.length}개 플랫폼` : "전체 플랫폼";
  return `${brands}, ${platforms}, ${source} 기준으로 현재 보기 범위를 구성합니다.`;
}
