import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";

import type { DashboardFilters } from "../types";

export function useDashboardFilters() {
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<DashboardFilters>(() => {
    const snapshotWindowRaw = searchParams.get("snapshotWindow");
    return {
      dataset: searchParams.get("dataset") ?? "analytics",
      brands: searchParams.getAll("brand"),
      sourceDatasets: searchParams.getAll("sourceDataset"),
      platforms: searchParams.getAll("platform"),
      schemaVersions: searchParams.getAll("schemaVersion"),
      snapshotWindow: snapshotWindowRaw ? Number(snapshotWindowRaw) : 300,
      dateFrom: searchParams.get("dateFrom") ?? undefined,
      dateTo: searchParams.get("dateTo") ?? undefined,
    };
  }, [searchParams]);

  const update = (next: Partial<DashboardFilters>) => {
    const merged: DashboardFilters = {
      ...filters,
      ...next,
      brands: next.brands ?? filters.brands,
      sourceDatasets: next.sourceDatasets ?? filters.sourceDatasets,
      platforms: next.platforms ?? filters.platforms,
      schemaVersions: next.schemaVersions ?? filters.schemaVersions,
    };
    const params = new URLSearchParams();
    if (merged.snapshotWindow) {
      params.set("snapshotWindow", String(merged.snapshotWindow));
    }
    if (merged.dateFrom) {
      params.set("dateFrom", merged.dateFrom);
    }
    if (merged.dateTo) {
      params.set("dateTo", merged.dateTo);
    }
    merged.brands.forEach((brand) => params.append("brand", brand));
    merged.sourceDatasets.forEach((sourceDataset) => params.append("sourceDataset", sourceDataset));
    merged.platforms.forEach((platform) => params.append("platform", platform));
    merged.schemaVersions.forEach((schemaVersion) => params.append("schemaVersion", schemaVersion));
    setSearchParams(params, { replace: true });
  };

  return { filters, update };
}
