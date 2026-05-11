import { useQuery } from "@tanstack/react-query";

import { apiGet } from "../lib/api";
import type { DashboardFilters } from "../types";

type FilterPanelProps = {
  filters: DashboardFilters;
  onChange: (next: Partial<DashboardFilters>) => void;
};

type FilterResponse = {
  availableBrands: string[];
  availableSourceDatasets: string[];
  availablePlatforms: string[];
  availableSchemaVersions: string[];
};

export function FilterPanel({ filters, onChange }: FilterPanelProps) {
  const filterQuery = useQuery({
    queryKey: ["filters", filters],
    queryFn: () =>
      apiGet<FilterResponse>("/api/filters", {
        ...filters,
        dataset: "analytics",
      }),
  });

  return (
    <div className="filter-panel">
      <label>
        최근 스냅샷 수
        <input
          type="number"
          min={1}
          value={filters.snapshotWindow ?? 300}
          onChange={(event) => onChange({ snapshotWindow: Number(event.target.value) })}
        />
      </label>
      <label>
        시작일
        <input type="date" value={filters.dateFrom ?? ""} onChange={(event) => onChange({ dateFrom: event.target.value || undefined })} />
      </label>
      <label>
        종료일
        <input type="date" value={filters.dateTo ?? ""} onChange={(event) => onChange({ dateTo: event.target.value || undefined })} />
      </label>
      <label>
        출처 데이터셋
        <select
          multiple
          size={4}
          value={filters.sourceDatasets}
          onChange={(event) =>
            onChange({
              sourceDatasets: Array.from(event.target.selectedOptions).map((option) => option.value),
            })
          }
        >
          {(filterQuery.data?.availableSourceDatasets ?? []).map((sourceDataset) => (
            <option key={sourceDataset} value={sourceDataset}>
              {sourceDataset}
            </option>
          ))}
        </select>
      </label>
      <label>
        플랫폼
        <select
          multiple
          size={3}
          value={filters.platforms}
          onChange={(event) =>
            onChange({
              platforms: Array.from(event.target.selectedOptions).map((option) => option.value),
            })
          }
        >
          {(filterQuery.data?.availablePlatforms ?? []).map((platform) => (
            <option key={platform} value={platform}>
              {platform}
            </option>
          ))}
        </select>
      </label>
      <label>
        스키마 버전
        <select
          multiple
          size={3}
          value={filters.schemaVersions}
          onChange={(event) =>
            onChange({
              schemaVersions: Array.from(event.target.selectedOptions).map((option) => option.value),
            })
          }
        >
          {(filterQuery.data?.availableSchemaVersions ?? []).map((schemaVersion) => (
            <option key={schemaVersion} value={schemaVersion}>
              {schemaVersion}
            </option>
          ))}
        </select>
      </label>
      <label>
        브랜드
        <select
          multiple
          size={8}
          value={filters.brands}
          onChange={(event) =>
            onChange({
              brands: Array.from(event.target.selectedOptions).map((option) => option.value),
            })
          }
        >
          {(filterQuery.data?.availableBrands ?? []).map((brand) => (
            <option key={brand} value={brand}>
              {brand}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        className="ghost-button"
        onClick={() =>
          onChange({
            brands: [],
            sourceDatasets: [],
            platforms: [],
            schemaVersions: [],
            dateFrom: undefined,
            dateTo: undefined,
            snapshotWindow: 300,
          })
        }
      >
        필터 초기화
      </button>
    </div>
  );
}
