import type { DashboardFilters } from "../types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

/** 로컬 파일 경로를 시각화 API 이미지 엔드포인트 URL로 바꾼다. */
export function productImageApiUrl(fsPath: string): string {
  return `${API_BASE_URL}/api/files/product-image?path=${encodeURIComponent(fsPath)}`;
}

function buildParams(filters: DashboardFilters, extra?: Record<string, string | string[] | number | undefined>) {
  const params = new URLSearchParams();
  if (filters.dataset) {
    params.set("dataset", filters.dataset);
  }
  if (filters.snapshotWindow) {
    params.set("snapshot_window", String(filters.snapshotWindow));
  }
  if (filters.dateFrom) {
    params.set("date_from", filters.dateFrom);
  }
  if (filters.dateTo) {
    params.set("date_to", filters.dateTo);
  }
  filters.brands.forEach((brand) => params.append("brands", brand));
  filters.sourceDatasets.forEach((sourceDataset) => params.append("source_datasets", sourceDataset));
  filters.platforms.forEach((platform) => params.append("platforms", platform));
  filters.schemaVersions.forEach((schemaVersion) => params.append("schema_versions", schemaVersion));
  Object.entries(extra ?? {}).forEach(([key, value]) => {
    if (value === undefined) {
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((entry) => params.append(key, entry));
      return;
    }
    params.set(key, String(value));
  });
  return params;
}

export async function apiGet<T>(
  path: string,
  filters: DashboardFilters,
  extra?: Record<string, string | string[] | number | undefined>,
): Promise<T> {
  const params = buildParams(filters, extra);
  const url = `${API_BASE_URL}${path}${params.toString() ? `?${params.toString()}` : ""}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API 요청 실패: ${response.status}`);
  }
  return response.json() as Promise<T>;
}
