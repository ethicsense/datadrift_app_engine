import { useState } from "react";

import { DataTable } from "../components/DataTable";
import { KpiGrid } from "../components/KpiGrid";
import { SectionCard } from "../components/SectionCard";
import { AnimatedScatterChart } from "../components/charts/AnimatedScatterChart";
import { EChartPanel } from "../components/charts/EChartPanel";
import { LocationMapPanel } from "../components/charts/LocationMapPanel";
import { ProductRankTrajectoriesChart } from "../components/charts/ProductRankTrajectoriesChart";
import { RankRaceChart } from "../components/charts/RankRaceChart";
import { KeywordTimeseriesSparklinePanel } from "../components/charts/KeywordTimeseriesSparklinePanel";
import { BrandImageProfilePanel } from "../components/charts/BrandImageProfilePanel";
import { WordCloudPanel } from "../components/charts/WordCloudPanel";
import type { WidgetConfig } from "../types";

type TableWidgetConfig = Extract<WidgetConfig, { type: "table" }>;

function TableWidgetCard({ widget }: { widget: TableWidgetConfig }) {
  const previewLimit = typeof widget.rowPreviewLimit === "number" ? widget.rowPreviewLimit : null;
  const hasPreview = previewLimit !== null && previewLimit > 0 && widget.rows.length > previewLimit;
  const [expanded, setExpanded] = useState(widget.rowPreviewDefaultExpanded ?? false);
  const visibleRows = hasPreview && !expanded ? widget.rows.slice(0, previewLimit) : widget.rows;
  const toggleLabel = widget.rowPreviewToggleLabel ?? "행";
  return (
    <SectionCard
      title={widget.title}
      description={widget.description}
      section={widget.section}
      takeaway={widget.takeaway}
      explainability={widget.explainability}
      bodyCollapsible={widget.bodyCollapsible}
      defaultBodyExpanded={widget.defaultBodyExpanded}
      bodyToggleLabel={widget.bodyToggleLabel}
    >
      <div className={`table-preview-wrap${hasPreview ? " has-preview-toggle" : ""}`}>
        <DataTable
          rows={visibleRows}
          highlightKey={widget.highlightKey}
          highlightValue={widget.highlightValue}
          onRowSelect={widget.onRowSelect}
        />
        {hasPreview ? (
          <button
            type="button"
            className="section-card__toggle section-card__toggle--body table-preview-wrap__toggle"
            onClick={() => setExpanded((value) => !value)}
            aria-expanded={expanded}
          >
            {expanded ? `${toggleLabel} 접기` : `${toggleLabel} 더보기 (${widget.rows.length - visibleRows.length}개)`}
          </button>
        ) : null}
      </div>
    </SectionCard>
  );
}

export function renderWidget(widget: WidgetConfig) {
  if (widget.type === "kpis") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <KpiGrid data={widget.payload} />
      </SectionCard>
    );
  }
  if (widget.type === "chart") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <EChartPanel rows={widget.rows} chartKind={widget.chartKind} spec={widget.spec} />
      </SectionCard>
    );
  }
  if (widget.type === "table") {
    return <TableWidgetCard widget={widget} />;
  }
  if (widget.type === "keywordSparkTimeseries") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <KeywordTimeseriesSparklinePanel rows={widget.rows} />
      </SectionCard>
    );
  }
  if (widget.type === "brandImageProfile") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <BrandImageProfilePanel
          profileRows={widget.profileRows}
          matrixRows={widget.matrixRows}
          scoringMethod={widget.scoringMethod}
          embeddingMeta={widget.embeddingMeta}
          evidenceRows={widget.evidenceRows}
        />
      </SectionCard>
    );
  }
  if (widget.type === "wordCloud") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <WordCloudPanel
          rows={widget.rows}
          wordKey={widget.wordKey}
          valueKey={widget.valueKey}
          maxWords={widget.maxWords}
        />
      </SectionCard>
    );
  }
  if (widget.type === "map") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <LocationMapPanel rows={widget.rows} />
      </SectionCard>
    );
  }
  if (widget.type === "rankTrajectories") {
    const p = widget.payload;
    return (
      <ProductRankTrajectoriesChart
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
        rows={p.rows}
        baseSpec={p.baseSpec}
        defaultSeriesIds={p.defaultSeriesIds}
        availableSeries={p.availableSeries}
        series={p.series}
      />
    );
  }
  if (widget.animationKind === "scatterMotion") {
    return (
      <SectionCard
        title={widget.title}
        description={widget.description}
        section={widget.section}
        takeaway={widget.takeaway}
        explainability={widget.explainability}
        bodyCollapsible={widget.bodyCollapsible}
        defaultBodyExpanded={widget.defaultBodyExpanded}
        bodyToggleLabel={widget.bodyToggleLabel}
      >
        <AnimatedScatterChart frames={(widget.payload.frames as never[]) ?? []} />
      </SectionCard>
    );
  }
  return (
    <SectionCard
      title={widget.title}
      description={widget.description}
      section={widget.section}
      takeaway={widget.takeaway}
      explainability={widget.explainability}
      bodyCollapsible={widget.bodyCollapsible}
      defaultBodyExpanded={widget.defaultBodyExpanded}
      bodyToggleLabel={widget.bodyToggleLabel}
    >
      <RankRaceChart frames={(widget.payload.frames as never[]) ?? []} />
    </SectionCard>
  );
}
