import { useEffect, useMemo, useState } from "react";

import ReactECharts from "echarts-for-react";

import { formatNumber } from "../../lib/formatters";

const chartTextColor = "#d4d4d8";
const chartAxisColor = "#71717a";
const chartTooltipBackground = "rgba(12,12,14,0.96)";
const chartFontSize = 14;

function toNum(v: unknown): number {
  if (typeof v === "number" && Number.isFinite(v)) {
    return v;
  }
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

function str(v: unknown): string {
  if (v === null || v === undefined) {
    return "";
  }
  return String(v);
}

export type BrandImageProfilePanelProps = {
  profileRows: Record<string, unknown>[];
  matrixRows: Record<string, unknown>[];
  scoringMethod?: string;
  embeddingMeta?: Record<string, unknown> | null;
  evidenceRows?: Record<string, unknown>[];
};

type ChartMode = "bars" | "radar";

export function BrandImageProfilePanel({
  profileRows,
  matrixRows,
  scoringMethod,
  embeddingMeta,
  evidenceRows = [],
}: BrandImageProfilePanelProps) {
  const brands = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const row of profileRows) {
      const b = str(row.brand).trim();
      if (b && !seen.has(b)) {
        seen.add(b);
        out.push(b);
      }
    }
    return out;
  }, [profileRows]);

  const [pickedBrand, setPickedBrand] = useState<string | null>(null);
  const [mode, setMode] = useState<ChartMode>("bars");

  const brand = pickedBrand && brands.includes(pickedBrand) ? pickedBrand : brands[0] ?? "";

  useEffect(() => {
    if (!brands.length) {
      return;
    }
    if (pickedBrand && !brands.includes(pickedBrand)) {
      setPickedBrand(null);
    }
  }, [brands, pickedBrand]);

  const profile = useMemo(
    () => profileRows.find((r) => str(r.brand) === brand),
    [profileRows, brand],
  );

  const axisRows = useMemo(() => {
    const rows = matrixRows.filter((r) => str(r.brand) === brand);
    rows.sort((a, b) => str(a.style).localeCompare(str(b.style)));
    return rows;
  }, [matrixRows, brand]);

  const styleLabels = useMemo(() => axisRows.map((r) => str(r.styleLabel || r.style)), [axisRows]);

  const intentPct = useMemo(() => axisRows.map((r) => toNum(r.intentShare) * 100), [axisRows]);
  const perceivedPct = useMemo(() => axisRows.map((r) => toNum(r.perceivedShare) * 100), [axisRows]);

  const alignment = profile ? toNum(profile.imageAlignment) : null;
  const claimMass = profile ? toNum(profile.claimStyleMass) : null;
  const reviewMass = profile ? toNum(profile.reviewStyleMass) : null;
  const customerNote = profile ? str(profile.customerLedImageNote ?? "") : "";
  const brandNote = profile ? str(profile.brandLedImageNote ?? "") : "";

  const scoringLabel =
    scoringMethod === "embedding" ? "임베딩·프로토타입" : scoringMethod === "keyword" ? "키워드 사전" : null;
  const modelHint =
    embeddingMeta && typeof embeddingMeta.model_name === "string"
      ? embeddingMeta.model_name
      : null;

  const brandEvidence = useMemo(() => {
    const rows = evidenceRows ?? [];
    return rows
      .filter((r) => str(r.brand) === brand)
      .sort((a, b) => {
        const sa = str(a.styleLabel ?? a.style);
        const sb = str(b.styleLabel ?? b.style);
        if (sa !== sb) {
          return sa.localeCompare(sb);
        }
        return toNum(a.rank) - toNum(b.rank);
      })
      .slice(0, 16);
  }, [evidenceRows, brand]);

  const barOption = useMemo(() => {
    if (!styleLabels.length) {
      return null;
    }
    return {
      backgroundColor: "transparent",
      grid: { left: 120, right: 28, top: 48, bottom: 28 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: chartTooltipBackground,
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: chartTextColor, fontSize: chartFontSize },
        formatter: (params: Array<{ seriesName: string; value: number; dataIndex: number }>) => {
          if (!params?.length) {
            return "";
          }
          const idx = params[0]!.dataIndex;
          const row = axisRows[idx];
          const gap = row ? toNum(row.styleGap) : 0;
          const lines = [
            `<strong>${styleLabels[idx] ?? ""}</strong>`,
            ...params.map(
              (p) => `${p.seriesName}: ${formatNumber(p.value, "number", "share")}%`,
            ),
            `갭(지각−의도): ${formatNumber(gap * 100, "number", "gap")}%p`,
          ];
          return lines.join("<br/>");
        },
      },
      legend: {
        top: 0,
        textStyle: { color: chartTextColor, fontSize: chartFontSize },
        data: ["의도(상품 카피)", "지각(리뷰)"],
      },
      xAxis: {
        type: "value",
        max: 100,
        name: "비중(%)",
        nameTextStyle: { color: chartTextColor },
        axisLabel: {
          color: chartTextColor,
          formatter: (v: number) => `${v}`,
        },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
        axisLine: { lineStyle: { color: chartAxisColor } },
      },
      yAxis: {
        type: "category",
        data: styleLabels,
        inverse: true,
        axisLabel: {
          color: chartTextColor,
          fontSize: 13,
          width: 112,
          overflow: "truncate",
        },
        axisLine: { lineStyle: { color: chartAxisColor } },
      },
      series: [
        {
          name: "의도(상품 카피)",
          type: "bar",
          data: intentPct,
          barMaxWidth: 14,
          itemStyle: { color: "#93c5fd", borderRadius: [0, 4, 4, 0] },
        },
        {
          name: "지각(리뷰)",
          type: "bar",
          data: perceivedPct,
          barMaxWidth: 14,
          itemStyle: { color: "#6ee7b7", borderRadius: [0, 4, 4, 0] },
        },
      ],
    };
  }, [axisRows, intentPct, perceivedPct, styleLabels]);

  const radarOption = useMemo(() => {
    if (!styleLabels.length) {
      return null;
    }
    const maxV = Math.max(100, ...intentPct, ...perceivedPct, 1);
    const indicators = styleLabels.map((name) => ({ name, max: maxV }));
    return {
      backgroundColor: "transparent",
      tooltip: {
        backgroundColor: chartTooltipBackground,
        borderColor: "rgba(255,255,255,0.08)",
        textStyle: { color: chartTextColor, fontSize: chartFontSize },
      },
      legend: {
        bottom: 0,
        textStyle: { color: chartTextColor, fontSize: chartFontSize },
        data: ["의도(상품 카피)", "지각(리뷰)"],
      },
      radar: {
        indicator: indicators,
        radius: "58%",
        center: ["50%", "46%"],
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
        splitArea: { show: true, areaStyle: { color: ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.06)"] } },
        axisName: {
          color: chartTextColor,
          fontSize: 12,
        },
      },
      series: [
        {
          type: "radar",
          data: [
            {
              name: "의도(상품 카피)",
              value: intentPct,
              areaStyle: { color: "rgba(147, 197, 253, 0.35)" },
              lineStyle: { color: "#93c5fd", width: 2 },
              itemStyle: { color: "#93c5fd" },
            },
            {
              name: "지각(리뷰)",
              value: perceivedPct,
              areaStyle: { color: "rgba(110, 231, 183, 0.32)" },
              lineStyle: { color: "#6ee7b7", width: 2 },
              itemStyle: { color: "#6ee7b7" },
            },
          ],
        },
      ],
    };
  }, [intentPct, perceivedPct, styleLabels]);

  if (!profileRows.length) {
    return <div className="empty-state">브랜드 이미지 프로파일을 만들 데이터가 없습니다.</div>;
  }

  const activeOption = mode === "bars" ? barOption : radarOption;

  return (
    <div className="brand-image-profile-panel">
      <div className="brand-image-profile-panel__toolbar">
        <label className="brand-image-profile-panel__field">
          <span className="brand-image-profile-panel__field-label">브랜드</span>
          <select
            className="brand-image-profile-panel__select"
            value={brand}
            onChange={(e) => setPickedBrand(e.target.value)}
          >
            {brands.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>
        <div className="brand-image-profile-panel__modes" role="tablist" aria-label="차트 유형">
          <button
            type="button"
            role="tab"
            className={`brand-image-profile-panel__mode${mode === "bars" ? " is-active" : ""}`}
            onClick={() => setMode("bars")}
          >
            막대 비교
          </button>
          <button
            type="button"
            role="tab"
            className={`brand-image-profile-panel__mode${mode === "radar" ? " is-active" : ""}`}
            onClick={() => setMode("radar")}
          >
            레이더
          </button>
        </div>
        {scoringLabel ? (
          <div className="brand-image-profile-panel__scoring" title={modelHint ?? undefined}>
            <span className="brand-image-profile-panel__scoring-label">산출 방식</span>
            <span className="brand-image-profile-panel__scoring-value">{scoringLabel}</span>
            {modelHint ? <small className="brand-image-profile-panel__scoring-model">{modelHint}</small> : null}
          </div>
        ) : null}
      </div>

      <div className="brand-image-profile-panel__kpis" aria-label="요약 지표">
        <div className="brand-image-profile-panel__kpi">
          <span className="brand-image-profile-panel__kpi-label">이미지 정렬도</span>
          <strong className="brand-image-profile-panel__kpi-value">
            {alignment !== null && Number.isFinite(alignment) ? formatNumber(alignment, "number", "imageAlignment") : "—"}
          </strong>
          <span className="brand-image-profile-panel__kpi-hint">1에 가까울수록 의도·지각 분포가 유사</span>
        </div>
        <div className="brand-image-profile-panel__kpi">
          <span className="brand-image-profile-panel__kpi-label">카피 신호량</span>
          <strong className="brand-image-profile-panel__kpi-value">
            {claimMass !== null && claimMass > 0 ? formatNumber(claimMass, "integer", "claimStyleMass") : "—"}
          </strong>
        </div>
        <div className="brand-image-profile-panel__kpi">
          <span className="brand-image-profile-panel__kpi-label">리뷰 신호량</span>
          <strong className="brand-image-profile-panel__kpi-value">
            {reviewMass !== null && reviewMass > 0 ? formatNumber(reviewMass, "integer", "reviewStyleMass") : "—"}
          </strong>
        </div>
        {customerNote ? (
          <div className="brand-image-profile-panel__kpi brand-image-profile-panel__kpi--wide">
            <span className="brand-image-profile-panel__kpi-label">고객 인식 우세</span>
            <span className="brand-image-profile-panel__kpi-note">{customerNote}</span>
          </div>
        ) : null}
        {brandNote ? (
          <div className="brand-image-profile-panel__kpi brand-image-profile-panel__kpi--wide">
            <span className="brand-image-profile-panel__kpi-label">브랜드 카피 우세</span>
            <span className="brand-image-profile-panel__kpi-note">{brandNote}</span>
          </div>
        ) : null}
      </div>

      {brandEvidence.length > 0 ? (
        <div className="brand-image-profile-panel__evidence" aria-label="축별 근거 스니펫">
          <div className="brand-image-profile-panel__evidence-title">상위 근거 스니펫 (임베딩 경로)</div>
          <ul className="brand-image-profile-panel__evidence-list">
            {brandEvidence.map((row, idx) => (
              <li key={`${str(row.style)}-${str(row.source)}-${idx}`} className="brand-image-profile-panel__evidence-item">
                <span className="brand-image-profile-panel__evidence-meta">
                  {str(row.styleLabel ?? row.style)} · {str(row.source) === "intent" ? "카피" : "리뷰"} · #
                  {str(row.rank ?? "")}
                </span>
                <span className="brand-image-profile-panel__evidence-snippet">{str(row.snippet)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {activeOption ? (
        <div className="brand-image-profile-panel__chart">
          <ReactECharts
            option={activeOption}
            notMerge
            lazyUpdate
            style={{ height: mode === "bars" ? Math.max(380, styleLabels.length * 36) : 440 }}
          />
        </div>
      ) : (
        <div className="empty-state">선택한 브랜드에 스타일 축 데이터가 없습니다.</div>
      )}
    </div>
  );
}
