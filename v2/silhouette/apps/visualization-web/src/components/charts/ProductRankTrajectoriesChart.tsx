import { useCallback, useMemo, useState } from "react";

import { SectionCard } from "../SectionCard";
import { EChartPanel } from "./EChartPanel";
import { formatNumber } from "../../lib/formatters";
import type { ChartSeriesOption, ChartSpec, NarrativeSection, WidgetExplainability } from "../../types";

export type RankTrajectoryPoint = {
  snapshotId: string;
  crawlDatetime: string | null;
  entityId: string;
  entityLabel: string;
  brand?: string | null;
  rank: number | null;
  rankDelta: number | null;
  momentumScore: number | null;
  rankEnergy?: number | null;
  energyVelocity?: number | null;
  energyAcceleration?: number | null;
  persistence?: number | null;
  eventState?: string | null;
  eventLabel?: string | null;
  recordCount: number | null;
  priceBand?: string | null;
  estimatedOriginalPriceBand?: string | null;
};

type ProductRankTrajectoriesChartProps = {
  title: string;
  description?: string;
  section?: string;
  takeaway?: string;
  explainability?: WidgetExplainability;
  bodyCollapsible?: boolean;
  defaultBodyExpanded?: boolean;
  bodyToggleLabel?: string;
  rows: Record<string, unknown>[];
  baseSpec: Omit<
    ChartSpec,
    "loadedSeriesIds" | "defaultSeriesIds" | "onRemoveSeries" | "onSelectSeries" | "selectedBumpSeries" | "onHighlightSeries" | "highlightSeries"
  >;
  defaultSeriesIds: string[];
  availableSeries: ChartSeriesOption[];
  series: RankTrajectoryPoint[];
};

export function ProductRankTrajectoriesChart({
  title,
  description,
  section,
  takeaway,
  explainability,
  bodyCollapsible,
  defaultBodyExpanded,
  bodyToggleLabel,
  rows,
  baseSpec,
  defaultSeriesIds,
  availableSeries,
  series,
}: ProductRankTrajectoriesChartProps) {
  const [displayedIds, setDisplayedIds] = useState<string[]>(() => defaultSeriesIds);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [addProductInput, setAddProductInput] = useState("");

  const allEntityIds = useMemo(() => new Set(rows.map((r) => String(r.entityId ?? ""))), [rows]);

  const resetToDefault = useCallback(() => {
    setDisplayedIds(defaultSeriesIds);
    setSelectedProductId(null);
  }, [defaultSeriesIds]);

  const addProduct = useCallback(() => {
    const id = addProductInput.trim();
    if (!id || !allEntityIds.has(id)) {
      return;
    }
    setDisplayedIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setAddProductInput("");
  }, [addProductInput, allEntityIds]);

  const removeProduct = useCallback((id: string) => {
    setDisplayedIds((prev) => prev.filter((x) => x !== id));
    setSelectedProductId((current) => (current === id ? null : current));
  }, []);

  const addProductIfInList = useCallback((id: string) => {
    if (!allEntityIds.has(id)) {
      return;
    }
    setDisplayedIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
  }, [allEntityIds]);

  const spec: ChartSpec = useMemo(
    () => ({
      ...baseSpec,
      loadedSeriesIds: displayedIds,
      defaultSeriesIds,
      onRemoveSeries: removeProduct,
      onSelectSeries: addProductIfInList,
      selectedBumpSeries: selectedProductId ?? undefined,
      onHighlightSeries: setSelectedProductId,
      highlightSeries: selectedProductId ? [selectedProductId] : [],
    }),
    [baseSpec, defaultSeriesIds, displayedIds, removeProduct, addProductIfInList, selectedProductId],
  );

  const selectedDetail = useMemo(() => {
    if (!selectedProductId) return null;
    const points = series.filter((p) => p.entityId === selectedProductId);
    if (!points.length) return null;
    const sorted = [...points].sort((a, b) => (b.crawlDatetime ?? "").localeCompare(a.crawlDatetime ?? ""));
    const latest = sorted[0];
    const first = sorted[sorted.length - 1];
    const rankChange = latest?.rank != null && first?.rank != null ? (first.rank ?? 0) - (latest.rank ?? 0) : null;
    return {
      entityId: selectedProductId,
      entityLabel: latest?.entityLabel ?? selectedProductId,
      brand: latest?.brand ?? null,
      latestRank: latest?.rank ?? null,
      rankDelta: latest?.rankDelta ?? null,
      rankChange,
      momentumScore: latest?.momentumScore ?? null,
      rankEnergy: latest?.rankEnergy ?? null,
      energyVelocity: latest?.energyVelocity ?? null,
      energyAcceleration: latest?.energyAcceleration ?? null,
      persistence: latest?.persistence ?? null,
      eventLabel: latest?.eventLabel ?? null,
      observationCount: points.length,
      priceBand: latest?.priceBand ?? null,
      estimatedOriginalPriceBand: latest?.estimatedOriginalPriceBand ?? null,
    };
  }, [series, selectedProductId]);

  if (!rows.length) {
    return (
      <SectionCard
        title={title}
        description={description}
        section={section as NarrativeSection | undefined}
        takeaway={takeaway}
        explainability={explainability}
        bodyCollapsible={bodyCollapsible}
        defaultBodyExpanded={defaultBodyExpanded}
        bodyToggleLabel={bodyToggleLabel}
      >
        <div className="empty-state">
          선택한 필터에 해당하는 순위 시계열이 없습니다. 데이터셋에 스냅샷(날짜·순위) 데이터가 있는지, 기간·브랜드 필터를 완화해 보세요.
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      title={title}
      description={description}
      section={section as NarrativeSection | undefined}
      takeaway={takeaway}
      explainability={explainability}
      bodyCollapsible={bodyCollapsible}
      defaultBodyExpanded={defaultBodyExpanded}
      bodyToggleLabel={bodyToggleLabel}
    >
      <div className="rank-trajectories-panel">
        <div className="rank-trajectories-panel__add-row">
          <label className="rank-trajectories-panel__add-label" htmlFor="rank-traj-add-product">
            제품번호 추가
          </label>
          <div className="rank-trajectories-panel__add-controls">
            <input
              id="rank-traj-add-product"
              type="text"
              value={addProductInput}
              onChange={(e) => setAddProductInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addProduct()}
              placeholder="제품 번호 입력 후 추가"
              className="rank-trajectories-panel__add-input"
            />
            <button
              type="button"
              className="rank-trajectories-panel__add-btn"
              onClick={addProduct}
              disabled={!addProductInput.trim() || !allEntityIds.has(addProductInput.trim())}
            >
              추가
            </button>
            <button
              type="button"
              className="rank-trajectories-panel__reset-btn"
              onClick={resetToDefault}
              title="표시 제품을 모멘텀 상위 3개로 되돌립니다"
            >
              표시 제품 초기화
            </button>
          </div>
          {addProductInput.trim() && !allEntityIds.has(addProductInput.trim()) && (
            <span className="rank-trajectories-panel__add-hint">데이터에 없는 제품번호입니다.</span>
          )}
        </div>

        {selectedDetail ? (
          <div className="rank-trajectories-panel__detail">
            <div className="rank-trajectories-panel__detail-header">
              <span className="rank-trajectories-panel__detail-title">선택 제품 요약</span>
              <button
                type="button"
                className="rank-trajectories-panel__detail-close"
                onClick={() => setSelectedProductId(null)}
                aria-label="강조 해제"
              >
                닫기
              </button>
            </div>
            <dl className="rank-trajectories-panel__detail-grid">
              <dt>제품번호</dt>
              <dd>{selectedDetail.entityId}</dd>
              <dt>상품명</dt>
              <dd className="rank-trajectories-panel__detail-label">{selectedDetail.entityLabel}</dd>
              <dt>브랜드</dt>
              <dd>{selectedDetail.brand ?? "-"}</dd>
              <dt>최근 순위</dt>
              <dd>{selectedDetail.latestRank != null ? `${formatNumber(selectedDetail.latestRank, "integer")}위` : "-"}</dd>
              <dt>순위 변화(구간)</dt>
              <dd>
                {selectedDetail.rankChange != null
                  ? `${selectedDetail.rankChange > 0 ? "↑" : selectedDetail.rankChange < 0 ? "↓" : ""} ${formatNumber(Math.abs(selectedDetail.rankChange), "integer")}`
                  : "-"}
              </dd>
              <dt>모멘텀 점수</dt>
              <dd>{selectedDetail.momentumScore != null ? formatNumber(selectedDetail.momentumScore, "number") : "-"}</dd>
              <dt>순위 에너지</dt>
              <dd>{selectedDetail.rankEnergy != null ? formatNumber(selectedDetail.rankEnergy, "number") : "-"}</dd>
              <dt>에너지 가속도</dt>
              <dd>{selectedDetail.energyAcceleration != null ? formatNumber(selectedDetail.energyAcceleration, "number") : "-"}</dd>
              <dt>움직임 상태</dt>
              <dd>{selectedDetail.eventLabel ?? "-"}</dd>
              <dt>판매가 기준 가격대</dt>
              <dd>{selectedDetail.priceBand ?? "-"}</dd>
              <dt>추정 정가 기준 가격대</dt>
              <dd>{selectedDetail.estimatedOriginalPriceBand ?? "-"}</dd>
              <dt>관측 수</dt>
              <dd>{selectedDetail.observationCount != null ? formatNumber(selectedDetail.observationCount, "integer") : "-"}</dd>
            </dl>
          </div>
        ) : null}

        <EChartPanel
          rows={rows}
          chartKind="bump"
          spec={spec}
        />
      </div>
    </SectionCard>
  );
}
