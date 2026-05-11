import { useEffect, useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { DataTable } from "../components/DataTable";
import { PageContainer } from "../components/PageContainer";
import { SectionCard } from "../components/SectionCard";
import { apiGet } from "../lib/api";
import { describeDashboardFilterScope } from "../lib/explainability";
import { formatNumber } from "../lib/formatters";
import { useDashboardFilters } from "../lib/filters";

type SchemaField = {
  field: string;
  scope: "raw" | "normalized";
  sourcePath?: string;
  observedCount?: number;
  observedRatePct?: number;
  nonNullCount?: number;
  nonNullRatePct?: number;
  inferredType?: string;
  sampleValue?: string | null;
};

type SchemaArtifact = {
  rowCount: number;
  fieldCount: number;
  fields: SchemaField[];
};

type SchemaInventoryResponse = {
  dataset: string;
  raw: SchemaArtifact;
  normalized: SchemaArtifact;
  availableSourceDatasets?: string[];
  availablePlatforms?: string[];
  availableSchemaVersions?: string[];
  filteredRecordCount?: number;
  diff: {
    summary?: Record<string, unknown>;
    rawOnlyFields?: SchemaField[];
    explorationRawOnlyFields?: SchemaField[];
    normalizedOnlyFields?: SchemaField[];
    sourceMappings?: Record<string, unknown>[];
  };
  report?: Record<string, unknown>;
};

type SchemaDiffResponse = {
  scope: "raw" | "normalized";
  dataset: string;
  currentSourceDataset?: string | null;
  compareSourceDataset?: string | null;
  summary?: Record<string, unknown>;
  addedFields?: SchemaField[];
  removedFields?: SchemaField[];
  explorationAddedFields?: SchemaField[];
  explorationRemovedFields?: SchemaField[];
  typeChangedFields?: Record<string, unknown>[];
  storedDiff?: {
    summary?: Record<string, unknown>;
    rawOnlyFields?: SchemaField[];
    explorationRawOnlyFields?: SchemaField[];
    normalizedOnlyFields?: SchemaField[];
    sourceMappings?: Record<string, unknown>[];
  };
};

type SchemaPairRow = {
  schemaKey: string;
  rawField: string | null;
  normalizedField: string | null;
  rawType: string | null;
  normalizedType: string | null;
  rawSampleValue: string | null;
  normalizedSampleValue: string | null;
  rawObservedRatePct: number | null;
  normalizedObservedRatePct: number | null;
  rawNonNullRatePct: number | null;
  normalizedNonNullRatePct: number | null;
  rawSourcePath: string | null;
  normalizedSourcePath: string | null;
  valueStatus: "same" | "changed" | "rawOnly" | "normalizedOnly";
  pairingStatus: "paired" | "rawOnly" | "normalizedOnly";
};

type SchemaPairSample = {
  snapshotId: string | null;
  productId: string | null;
  rawValue: unknown;
  normalizedValue: unknown;
};

type SchemaPairSampleGroup = {
  field: string;
  rawField: string;
  normalizedField: string;
  rawInferredType?: string | null;
  normalizedInferredType?: string | null;
  matchCount?: number;
  diffCount?: number;
  samples: SchemaPairSample[];
};

type SchemaPairSamplesResponse = {
  dataset: string;
  sampleSize: number;
  pairs: SchemaPairSampleGroup[];
};

type PairSampleRow = {
  field: string;
  rawType: string | null;
  normalizedType: string | null;
  rawValue: string;
  normalizedValue: string;
  valueStatus: "same" | "changed";
  snapshotId: string | null;
  productId: string | null;
};

type RawOnlyFieldRow = {
  field: string;
  rawType: string | null;
  rawSampleValue: string | null;
  rawSourcePath: string | null;
  rawObservedRatePct: number | null;
  rawNonNullRatePct: number | null;
};

function formatPairValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function parseSourceLabel(sourceDataset: string): string {
  const lower = sourceDataset.toLowerCase();
  if (lower.includes("musinsa")) {
    return "무신사";
  }
  const bits = sourceDataset.split("_").filter(Boolean);
  if (bits.length >= 2) {
    return bits[1];
  }
  return sourceDataset;
}

export function SchemaExplorerPage() {
  const { filters } = useDashboardFilters();
  const [scope, setScope] = useState<"raw" | "normalized">("raw");
  const [currentSourceDataset, setCurrentSourceDataset] = useState<string>("");
  const [compareSourceDataset, setCompareSourceDataset] = useState<string>("");
  const inventoryQuery = useQuery({
    queryKey: ["schema-inventory", filters],
    queryFn: () => apiGet<SchemaInventoryResponse>("/api/meta/schema/inventory", filters),
  });
  const pairSamplesQuery = useQuery({
    queryKey: ["schema-pair-samples", filters],
    queryFn: () =>
      apiGet<SchemaPairSamplesResponse>("/api/meta/schema/pair-samples", filters, { sample_size: 5 }),
  });
  const diffQuery = useQuery({
    queryKey: ["schema-diff", filters, scope, currentSourceDataset, compareSourceDataset],
    queryFn: () =>
      apiGet<SchemaDiffResponse>("/api/meta/schema/diff", filters, {
        scope,
        current_source_dataset: currentSourceDataset || undefined,
        compare_source_dataset: compareSourceDataset || undefined,
      }),
    enabled: compareSourceDataset.length > 0,
  });

  const availableSources = useMemo(() => inventoryQuery.data?.availableSourceDatasets ?? [], [inventoryQuery.data?.availableSourceDatasets]);
  const compareOptions = useMemo(
    () => availableSources.filter((sourceDataset) => sourceDataset !== currentSourceDataset),
    [availableSources, currentSourceDataset],
  );

  useEffect(() => {
    if (currentSourceDataset && !availableSources.includes(currentSourceDataset)) {
      const fallback = filters.sourceDatasets[0] && availableSources.includes(filters.sourceDatasets[0]) ? filters.sourceDatasets[0] : (availableSources[0] ?? "");
      setCurrentSourceDataset(fallback);
      return;
    }
    if (!currentSourceDataset) {
      const initial = filters.sourceDatasets[0] && availableSources.includes(filters.sourceDatasets[0]) ? filters.sourceDatasets[0] : (availableSources[0] ?? "");
      setCurrentSourceDataset(initial);
    }
  }, [availableSources, currentSourceDataset, filters.sourceDatasets]);

  useEffect(() => {
    if (compareSourceDataset && !compareOptions.includes(compareSourceDataset)) {
      setCompareSourceDataset("");
      return;
    }
  }, [compareOptions, compareSourceDataset]);

  const currentSourceLabel = currentSourceDataset || "선택 필요";
  const inventory = inventoryQuery.data;
  const rawOnlyFields = inventory?.diff?.explorationRawOnlyFields ?? inventory?.diff?.rawOnlyFields ?? [];
  const normalizedOnlyFields = inventory?.diff?.normalizedOnlyFields ?? [];
  const addedFields = diffQuery.data?.explorationAddedFields ?? diffQuery.data?.addedFields ?? [];
  const removedFields = diffQuery.data?.explorationRemovedFields ?? diffQuery.data?.removedFields ?? [];

  const rawArtifact = inventory?.raw;
  const normalizedArtifact = inventory?.normalized;
  const schemaPairRows = useMemo(() => {
    const rawMap = new Map((rawArtifact?.fields ?? []).map((field) => [field.field, field]));
    const normalizedMap = new Map((normalizedArtifact?.fields ?? []).map((field) => [field.field, field]));
    const keys = Array.from(new Set([...rawMap.keys(), ...normalizedMap.keys()]));
    return keys.map((schemaKey): SchemaPairRow => {
      const rawField = rawMap.get(schemaKey);
      const normalizedField = normalizedMap.get(schemaKey);
      return {
        schemaKey,
        rawField: rawField?.field ?? null,
        normalizedField: normalizedField?.field ?? null,
        rawType: rawField?.inferredType ?? null,
        normalizedType: normalizedField?.inferredType ?? null,
        rawSampleValue: rawField?.sampleValue ?? null,
        normalizedSampleValue: normalizedField?.sampleValue ?? null,
        rawObservedRatePct: rawField?.observedRatePct ?? null,
        normalizedObservedRatePct: normalizedField?.observedRatePct ?? null,
        rawNonNullRatePct: rawField?.nonNullRatePct ?? null,
        normalizedNonNullRatePct: normalizedField?.nonNullRatePct ?? null,
        rawSourcePath: rawField?.sourcePath ?? null,
        normalizedSourcePath: normalizedField?.sourcePath ?? null,
        valueStatus: rawField && normalizedField
          ? String(rawField.sampleValue ?? "") === String(normalizedField.sampleValue ?? "")
            ? "same"
            : "changed"
          : rawField
            ? "rawOnly"
            : "normalizedOnly",
        pairingStatus: rawField && normalizedField ? "paired" : rawField ? "rawOnly" : "normalizedOnly",
      };
    });
  }, [rawArtifact?.fields, normalizedArtifact?.fields]);
  void schemaPairRows;
  const rawOnlyPairRows = useMemo<RawOnlyFieldRow[]>(
    () =>
      schemaPairRows
        .filter((row) => row.pairingStatus === "rawOnly")
        .map((row) => ({
          field: row.rawField ?? row.schemaKey,
          rawType: row.rawType,
          rawSampleValue: row.rawSampleValue,
          rawSourcePath: row.rawSourcePath,
          rawObservedRatePct: row.rawObservedRatePct,
          rawNonNullRatePct: row.rawNonNullRatePct,
        })),
    [schemaPairRows],
  );
  const rawOnlyPairRowsTable = rawOnlyPairRows as unknown as Record<string, unknown>[];
  const pairSampleGroups = pairSamplesQuery.data?.pairs ?? [];
  const pairSampleRows = useMemo<PairSampleRow[]>(() => {
    const rows: PairSampleRow[] = [];
    for (const group of pairSampleGroups) {
      for (const sample of group.samples) {
        const rawValue = formatPairValue(sample.rawValue);
        const normalizedValue = formatPairValue(sample.normalizedValue);
        rows.push({
          field: group.field,
          rawType: group.rawInferredType ?? null,
          normalizedType: group.normalizedInferredType ?? null,
          rawValue,
          normalizedValue,
          valueStatus: rawValue === normalizedValue ? "same" : "changed",
          snapshotId: sample.snapshotId,
          productId: sample.productId,
        });
      }
    }
    return rows;
  }, [pairSampleGroups]);
  const pairSampleRowsTable = pairSampleRows as unknown as Record<string, unknown>[];
  const pairSampleFieldCount = pairSampleGroups.length;
  const pairSampleChangedCount = useMemo(
    () => pairSampleRows.filter((row) => row.valueStatus === "changed").length,
    [pairSampleRows],
  );
  const totalSchemaRows = (rawArtifact?.rowCount ?? 0) + (normalizedArtifact?.rowCount ?? 0);

  const rawOnlyFieldNames = useMemo(() => new Set(rawOnlyFields.map((f) => f.field)), [rawOnlyFields]);
  const normalizedOnlyFieldNames = useMemo(
    () => new Set(normalizedOnlyFields.map((f) => f.field)),
    [normalizedOnlyFields],
  );

  void rawOnlyFieldNames;
  void normalizedOnlyFieldNames;
  const pairSampleColumns = [
    "field",
    "rawValue",
    "normalizedValue",
    "valueStatus",
    "rawType",
    "normalizedType",
    "snapshotId",
    "productId",
  ];
  const rawOnlyColumns = [
    "field",
    "rawType",
    "rawSampleValue",
    "rawSourcePath",
  ];

  const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
  const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
  const targetSummary = targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";

  if (inventoryQuery.isLoading) {
    return <div className="loading-state">스키마 탐색 데이터를 불러오는 중입니다.</div>;
  }

  return (
    <PageContainer
      title="스키마 탐색"
      description="필드 정의와 수집 형태를 원문·정규화·데이터셋 비교로 확인합니다."
    >
      <section className="overview-story-hero schema-explorer-hero">
        <small>SCHEME</small>
        <h2>데이터의 가공 방식</h2>
        <p>
          {targetSummary} 데이터에서 <strong>원문(raw)</strong>과 <strong>정규화</strong> 범위를 고르면, 인벤토리에 잡힌 필드
          이름·타입·관측 비율을 한눈에 볼 수 있습니다.
        </p>
        <p>
          아래 두 패널로 나눠서 봅니다. 위쪽에서는 같은 상품·스냅샷에서 뽑은 <strong>원문–정규화 값 쌍 샘플</strong>을
          좌우로 비교하고, 아래쪽에서는 아직 정규화되지 않은 원문 필드만 모아 확인합니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>정의</span>
          <strong>필드·타입</strong>
          <small>이름, 추론 타입, 관측·결측 비율로 스키마를 읽습니다.</small>
        </article>
        <article className="overview-story-card">
          <span>형태</span>
          <strong>원문 vs 정규화</strong>
          <small>같은 데이터셋도 스코프에 따라 필드 집합이 달라질 수 있습니다.</small>
        </article>
        <article className="overview-story-card">
          <span>비교</span>
          <strong>소스 쌍</strong>
          <small>기준·비교 데이터셋을 고르면 단독 필드 diff를 봅니다.</small>
        </article>
      </section>

      <section className="schema-explorer-toolbar" aria-label="탐색 범위">
        <p className="schema-explorer-toolbar__scope">{describeDashboardFilterScope(filters)}</p>
        <div className="legend-filter__controls schema-explorer-toolbar__controls">
          <label>
            범위
            <select value={scope} onChange={(event) => setScope(event.target.value as "raw" | "normalized")}>
              <option value="raw">원문</option>
              <option value="normalized">정규화</option>
            </select>
          </label>
          <label>
            기준 데이터셋
            <select value={currentSourceDataset} onChange={(event) => setCurrentSourceDataset(event.target.value)}>
              <option value="">선택 필요</option>
              {availableSources.map((sourceDataset) => (
                <option key={sourceDataset} value={sourceDataset}>
                  {sourceDataset}
                </option>
              ))}
            </select>
          </label>
          <label>
            비교 데이터셋
            <select value={compareSourceDataset} onChange={(event) => setCompareSourceDataset(event.target.value)}>
              <option value="">비교 안 함</option>
              {compareOptions.map((sourceDataset) => (
                <option key={sourceDataset} value={sourceDataset}>
                  {sourceDataset}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <SectionCard
        title="정규화 필드와 원문 값 비교"
        description={`동일 행(snapshot_id · product_id)에서 뽑은 원문–정규화 값 쌍을 보여줍니다. 관측 행 ${formatNumber(totalSchemaRows, "integer")}행 중 쌍을 추출할 수 있는 필드는 ${formatNumber(pairSampleFieldCount, "integer")}개입니다.`}
        section="summary"
        explainability={{
          context: [
            { label: "쌍 추출 필드", value: `${formatNumber(pairSampleFieldCount, "integer")}개` },
            { label: "값 변경 샘플", value: `${formatNumber(pairSampleChangedCount, "integer")}건` },
            { label: "표시 샘플", value: `${formatNumber(pairSampleRows.length, "integer")}건` },
          ],
          readingGuide: [
            {
              text: "같은 상품·스냅샷에서 원문(rawValue)과 정규화(normalizedValue) 값이 어떻게 달라졌는지 좌우로 비교하세요.",
            },
            {
              text: "가져오기 어렵거나 파생·불리언처럼 대응 raw 값이 의미 없는 필드는 제외했습니다.",
            },
          ],
        }}
      >
        {pairSamplesQuery.isLoading ? (
          <div className="loading-state">원문–정규화 값 쌍을 불러오는 중입니다.</div>
        ) : pairSampleRowsTable.length ? (
          <DataTable
            rows={pairSampleRowsTable}
            includeColumns={pairSampleColumns}
            initialSorting={[{ id: "field", desc: false }]}
          />
        ) : (
          <div className="empty-state">현재 필터 범위에서는 추출 가능한 원문–정규화 값 쌍이 없습니다.</div>
        )}
      </SectionCard>

      <SectionCard
        title="정규화 미적용 원문 필드"
        description="원문(raw)에는 있지만 정규화 컬럼이 아직 없는 항목을 모아 봅니다."
        section="result"
        explainability={{
          context: [
            { label: "필드", value: `${formatNumber(rawOnlyPairRows.length, "integer")}개` },
          ],
          readingGuide: [{ text: "원천 값의 형태·결측을 보고 정규화 우선순위를 판단하세요." }],
        }}
      >
        {rawOnlyPairRowsTable.length ? (
          <DataTable
            rows={rawOnlyPairRowsTable}
            includeColumns={rawOnlyColumns}
            initialSorting={[{ id: "field", desc: false }]}
          />
        ) : (
          <div className="empty-state">현재 필터 범위에서는 정규화 미적용 원문 필드가 없습니다.</div>
        )}
      </SectionCard>

      <SectionCard
        title="데이터셋별 단독 필드"
        description="기준과 비교 소스를 고르면, 한쪽에만 존재하는 필드를 나란히 봅니다."
        section="interpretation"
        explainability={{
          context: [
            { label: `${currentSourceLabel} 전용`, value: `${addedFields.length}개` },
            { label: `${compareSourceDataset || "비교"} 전용`, value: `${removedFields.length}개` },
          ],
          readingGuide: [{ text: "비교 데이터셋을 선택해야 diff가 활성화됩니다." }],
        }}
      >
        {compareSourceDataset ? (
          <div className="schema-dual-grid">
            <div>
              <h3>{currentSourceLabel}에만 있는 필드</h3>
              <DataTable rows={addedFields as Record<string, unknown>[]} />
            </div>
            <div>
              <h3>{compareSourceDataset}에만 있는 필드</h3>
              <DataTable rows={removedFields as Record<string, unknown>[]} />
            </div>
          </div>
        ) : (
          <p className="schema-explorer-placeholder">비교 데이터셋을 선택하면 단독 필드가 표시됩니다.</p>
        )}
      </SectionCard>
    </PageContainer>
  );
}
