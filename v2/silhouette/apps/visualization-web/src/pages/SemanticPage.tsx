import { useCallback, useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { apiGet } from "../lib/api";
import { describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type { WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

type SemanticResponse = {
  itemDistribution: Record<string, unknown>[];
  coverageRows: Record<string, unknown>[];
  colorDistribution: Record<string, unknown>[];
  materialDistribution: Record<string, unknown>[];
  originCountryDistribution: Record<string, unknown>[];
  districtDistribution: Record<string, unknown>[];
  dongDistribution: Record<string, unknown>[];
  locationMapPoints: Record<string, unknown>[];
  materialPriceBandHeatmap: Record<string, unknown>[];
  rows: Record<string, unknown>[];
};

const PRICE_BAND_ORDER = ["~3만", "3-7만", "7-12만", "12-20만", "20-50만", "50만+", "미분류"];
const EMPTY_ROWS: Record<string, unknown>[] = [];

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

function toNumber(value: unknown) {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function SemanticPage() {
  const { filters } = useDashboardFilters();
  const [attributeSort, setAttributeSort] = useState<"recordCount" | "avgRank" | "avgPrice">("recordCount");
  const query = useQuery({
    queryKey: ["semantic", filters],
    queryFn: () => apiGet<SemanticResponse>("/api/semantic/text-features", filters),
  });

  const sortAttributeRows = useCallback(
    (rows: Record<string, unknown>[]) =>
      [...rows].sort((left, right) => {
        if (attributeSort === "avgRank") {
          return (toNumber(left.avgRank) ?? Number.MAX_SAFE_INTEGER) - (toNumber(right.avgRank) ?? Number.MAX_SAFE_INTEGER);
        }
        if (attributeSort === "avgPrice") {
          return (toNumber(right.avgPrice) ?? -Infinity) - (toNumber(left.avgPrice) ?? -Infinity);
        }
        return (toNumber(right.recordCount) ?? -Infinity) - (toNumber(left.recordCount) ?? -Infinity);
      }),
    [attributeSort],
  );

  const colorDistribution = useMemo(
    () => sortAttributeRows(query.data?.colorDistribution ?? EMPTY_ROWS),
    [query.data?.colorDistribution, sortAttributeRows],
  );
  const materialDistribution = useMemo(
    () => sortAttributeRows(query.data?.materialDistribution ?? EMPTY_ROWS),
    [query.data?.materialDistribution, sortAttributeRows],
  );
  const originCountryDistribution = useMemo(
    () => sortAttributeRows(query.data?.originCountryDistribution ?? EMPTY_ROWS),
    [query.data?.originCountryDistribution, sortAttributeRows],
  );
  const districtDistribution = useMemo(
    () => sortAttributeRows(query.data?.districtDistribution ?? EMPTY_ROWS),
    [query.data?.districtDistribution, sortAttributeRows],
  );
  const dongDistribution = useMemo(
    () => sortAttributeRows(query.data?.dongDistribution ?? EMPTY_ROWS),
    [query.data?.dongDistribution, sortAttributeRows],
  );
  const materialYDomain = useMemo(
    () => materialDistribution.map((row) => String(row.materialValue ?? "미입력")),
    [materialDistribution],
  );

  const targetSummary = useMemo(() => {
    const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
    const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
    return targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";
  }, [filters.sourceDatasets]);

  const sharedContext = useMemo(
    () => [
      {
        label: "정렬 기준",
        value: attributeSort === "recordCount" ? "상품 수" : attributeSort === "avgRank" ? "평균 순위" : "평균 가격",
      },
      { label: "사례 표본", value: `${formatNumber((query.data?.rows ?? EMPTY_ROWS).length, "integer")}행` },
    ],
    [attributeSort, query.data?.rows],
  );

  const widgets = useMemo((): WidgetConfig[] => {
    const data = query.data;
    if (!data) {
      return [];
    }
    return [
      {
        id: "product-info-location-map",
        type: "map",
        title: "영업소재지 분포도",
        description: "구/동 집계 위치를 지도에 표시해 공간 분포를 봅니다.",
        section: "result",
        takeaway: "위치 점은 상세 주소가 아니라 구/동 집계 중심점입니다.",
        rows: data.locationMapPoints ?? EMPTY_ROWS,
      },
      {
        id: "product-info-district",
        type: "chart",
        title: "영업소재지 시군구 분포",
        description: "영업소재지의 시군구 기준으로 상품 수와 성과 차이를 봅니다.",
        section: "result",
        takeaway: "지역 집중도를 보면 가격대/브랜드 패턴 해석이 쉬워집니다.",
        chartKind: "bar",
        rows: districtDistribution,
        spec: { x: "businessDistrict", y: "recordCount", palette: "categorical", yLabel: "상품 수", yFormat: "integer" },
      },
      {
        id: "product-info-dong",
        type: "chart",
        title: "영업소재지 동 분포",
        description: "영업소재지 중 동 단위 파싱이 가능한 주소만 모아 집중도를 봅니다.",
        section: "result",
        takeaway: "동 분포는 주소 품질이 좋은 레코드에서만 유효한 보조 뷰입니다.",
        chartKind: "bar",
        rows: dongDistribution,
        spec: { x: "businessDong", y: "recordCount", palette: "categorical", yLabel: "상품 수", yFormat: "integer" },
      },
      {
        id: "semantic-distribution",
        type: "chart",
        title: "상품 유형",
        description: "정규화된 제품명 기준 상품 유형 분포를 봅니다.",
        section: "summary",
        takeaway: "상품 유형(`nameItem`)을 임시 축으로 써 속성/가격 관계를 빠르게 볼 수 있습니다.",
        chartKind: "bar",
        rows: data.itemDistribution ?? EMPTY_ROWS,
        spec: { x: "nameItem", y: "count", palette: "categorical", yLabel: "건수", yFormat: "integer" },
        explainability: {
          context: sharedContext,
          readingGuide: [{ text: "막대 1개는 상품 유형 1개이며, 높이는 상품 수입니다." }],
          drilldown: [{ text: "아래 색상/소재/공간 분포와 사례 표본을 함께 보세요." }],
        },
      },
      {
        id: "product-info-color",
        type: "chart",
        title: "색상 분포",
        description: "색상별 상품 수와 평균 순위/가격을 비교합니다.",
        section: "interpretation",
        takeaway: "정렬 변경으로 빈도 중심/성과 중심 해석을 나눠 볼 수 있습니다.",
        chartKind: "bar",
        rows: colorDistribution,
        spec: { x: "colorValue", y: "recordCount", palette: "categorical", yLabel: "상품 수", yFormat: "integer" },
      },
      {
        id: "product-info-material",
        type: "chart",
        title: "소재 분포",
        description: "소재별 상품 수와 평균 순위 차이를 봅니다.",
        section: "interpretation",
        takeaway: "소재는 빈도보다 가격대/브랜드와의 교차로 읽는 편이 유효합니다.",
        chartKind: "bar",
        rows: materialDistribution,
        spec: { x: "materialValue", y: "recordCount", palette: "categorical", yLabel: "상품 수", yFormat: "integer" },
      },
      {
        id: "product-info-origin-country",
        type: "chart",
        title: "제조국 분포",
        description: "제조국 입력 상품 기준으로 국가별 분포와 평균 순위/가격을 봅니다.",
        section: "interpretation",
        takeaway: "제조국은 단독 해석보다 브랜드/가격 포지셔닝 보조 축으로 보세요.",
        chartKind: "bar",
        rows: originCountryDistribution,
        spec: { x: "originCountry", y: "recordCount", palette: "categorical", yLabel: "상품 수", yFormat: "integer" },
      },
      {
        id: "product-info-material-price-band",
        type: "chart",
        title: "소재-가격대 분포",
        description: "소재와 추정 정가 가격대의 교차 분포를 봅니다.",
        section: "result",
        takeaway: "같은 소재라도 가격대 집중이 다르면 포지셔닝 차이를 시사합니다.",
        chartKind: "heatmap",
        rows: data.materialPriceBandHeatmap ?? EMPTY_ROWS,
        spec: {
          x: "priceBand",
          y: "materialValue",
          value: "count",
          valueFormat: "integer",
          xLabel: "추정 정가 가격대",
          yLabel: "소재",
          palette: "sequential",
          xDomain: PRICE_BAND_ORDER,
          yDomain: materialYDomain,
          tooltipFields: [
            { key: "count", label: "상품 수", format: "integer" },
            { key: "avgRank", label: "평균 순위", format: "number", fallback: "-" },
          ],
        },
        explainability: {
          context: sharedContext,
          readingGuide: [{ text: "셀 1개는 소재 1개·가격대 1개 조합이며, 진할수록 상품 수가 많습니다." }],
          interpretationRules: [{ text: "핵심은 소재 빈도보다 가격대 집중 구간입니다." }],
        },
      },
      {
        id: "product-info-coverage",
        type: "table",
        title: "수집 범위",
        description: "전체 원문이 아니라, 파이프라인에 보존된 필드만 사용합니다.",
        section: "summary",
        takeaway: "먼저 실제 확보된 필드를 확인해 해석 범위를 맞춥니다.",
        explainability: {
          context: sharedContext,
          readingGuide: [{ text: "행 1개는 속성 축 1개의 수집 범위와 사용 범위를 뜻합니다." }],
          caveats: [{ text: "보존된 필드만 포함하므로 원문 전체를 대표하지는 않습니다.", tone: "warning" }],
        },
        rows: data.coverageRows ?? EMPTY_ROWS,
      },
      {
        id: "semantic-table",
        type: "table",
        title: "사례 표본",
        description: "색상/소재/가격대/순위를 함께 보며 실제 사례를 확인합니다.",
        section: "examples",
        takeaway: "분포에서 본 패턴을 상품 행 단위로 검증하는 구간입니다.",
        explainability: {
          context: sharedContext,
          readingGuide: [{ text: "행 1개는 상품 1개의 속성 조합입니다." }],
          drilldown: [{ text: "원형 필드/이미지는 `섬네일` 탭에서 같은 상품으로 확인하세요." }],
        },
        rows: data.rows ?? EMPTY_ROWS,
      },
    ];
  }, [
    query.data,
    districtDistribution,
    dongDistribution,
    colorDistribution,
    materialDistribution,
    originCountryDistribution,
    materialYDomain,
    sharedContext,
  ]);

  if (query.isLoading) {
    return <div className="loading-state">상품 정보 속성 데이터를 불러오는 중입니다.</div>;
  }

  return (
    <PageContainer
      title="상품 정보"
      description="상품이 어떤 속성으로 채워져 있는지, 분포와 교차 패턴, 실제 사례까지 한 흐름으로 확인합니다."
    >
      <section className="overview-story-hero">
        <small>DETAILS</small>
        <h2>상품 고시 정보를 통해 알 수 있는 것들</h2>
        <p>
          {targetSummary} 상품 등록 시 작성하는 <strong>상품 고시 정보 테이블</strong>에서 실제로 어떤 정보가 채워지는지,
          어떤 항목은 잘 기입되지 않는지 확인합니다.
        </p>
        <p>
          이 테이블에서 얻을 수 있는 신호를 영업소재지 분포, 정합성 점검, 표본 사례까지 이어서 읽습니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>분포</span>
          <strong>영업소재지 분포</strong>
          <small>영업소재지 지도와 시군구·동 분포를 먼저 보고, 지역 정보가 실제로 어떻게 들어와 있는지 확인합니다.</small>
        </article>
        <article className="overview-story-card">
          <span>품질</span>
          <strong>정보의 정합성</strong>
          <small>속성 분포와 소재-가격대 교차를 보며 값이 비었는지, 특정 항목에 과도하게 몰리는지 점검합니다.</small>
        </article>
        <article className="overview-story-card">
          <span>검증</span>
          <strong>데이터 표본 사례</strong>
          <small>마지막 표본 카드에서 실제 행을 보며, 위에서 본 패턴이 현장 데이터에서도 맞는지 검증합니다.</small>
        </article>
      </section>

      <section className="semantic-page-toolbar" aria-label="필터 요약">
        <p className="semantic-page-toolbar__scope">{describeDashboardFilterScope(filters)}</p>
        <p className="semantic-page-toolbar__selection">지도 점은 상세 주소가 아닌 구/동 집계 중심점입니다.</p>
        <div className="legend-filter__controls">
          <label>
            속성 막대 정렬
            <select value={attributeSort} onChange={(event) => setAttributeSort(event.target.value as typeof attributeSort)}>
              <option value="recordCount">상품 수 순</option>
              <option value="avgRank">평균 순위 순</option>
              <option value="avgPrice">평균 가격 순</option>
            </select>
          </label>
        </div>
      </section>
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
