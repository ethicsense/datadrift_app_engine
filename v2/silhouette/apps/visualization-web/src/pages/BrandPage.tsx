import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { PageContextBar } from "../components/PageContextBar";
import { apiGet } from "../lib/api";
import { buildDashboardFilterBadges, describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type { WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

type BrandResponse = {
  chart: Record<string, unknown>[];
  rows: Record<string, unknown>[];
};

type AnimationResponse = {
  frames: Record<string, unknown>[];
};

export function BrandPage() {
  const { filters } = useDashboardFilters();
  const indexQuery = useQuery({
    queryKey: ["brand-index", filters],
    queryFn: () => apiGet<BrandResponse>("/api/brand/index", filters),
  });
  const raceQuery = useQuery({
    queryKey: ["brand-rank-race", filters],
    queryFn: () => apiGet<AnimationResponse>("/api/animation/rank-race", filters, { entity_type: "brand" }),
  });

  if (indexQuery.isLoading) {
    return <div className="loading-state">브랜드 분석 데이터를 불러오는 중입니다.</div>;
  }

  const filterBadges = buildDashboardFilterBadges(filters);
  const sharedContext = [
    { label: "브랜드 수", value: `${formatNumber((indexQuery.data?.rows ?? []).length, "integer")}개` },
    { label: "레이스 프레임", value: `${formatNumber((raceQuery.data?.frames ?? []).length, "integer")}개` },
  ];

  const widgets: WidgetConfig[] = [
    {
      id: "brand-index-chart",
      type: "chart",
      title: "브랜드 평균 순위",
      description: "브랜드 단위에서 평균 순위와 모멘텀의 차이가 어떻게 나타나는지 확인합니다.",
      section: "interpretation",
      takeaway: "브랜드는 개별 상품 사례보다 더 안정적인 속성 단위라서, 해석의 기준 축으로 쓰기 좋습니다.",
      chartKind: "bar",
      rows: indexQuery.data?.chart ?? [],
      spec: { x: "brand", y: "avgRank", yAxisInverse: true, palette: "brand", yLabel: "평균 순위", yFormat: "number" },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "막대 1개는 브랜드 1개이며, 낮은 평균 순위일수록 더 강한 성과를 뜻합니다." },
        ],
      },
    },
    {
      id: "brand-index-table",
      type: "table",
      title: "브랜드 분석 표",
      section: "examples",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "행 1개는 브랜드 1개입니다. 차트에서 본 순위 격차를 수치로 다시 확인하는 표입니다." },
        ],
      },
      rows: indexQuery.data?.rows ?? [],
    },
    {
      id: "brand-race",
      type: "animation",
      title: "브랜드 순위 레이스",
      description: "브랜드 평균 순위가 시점에 따라 위아래로 이동합니다.",
      section: "examples",
      takeaway: "브랜드 평균 순위의 시간 흐름을 함께 보면, 정적인 집계 결과를 동적인 변화로 다시 해석할 수 있습니다.",
      animationKind: "rankRace",
      payload: { frames: raceQuery.data?.frames ?? [] },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "프레임 1개는 시점 1개이며, 브랜드 간 평균 순위의 상대 위치 변화를 시간축으로 읽습니다." },
        ],
      },
    },
  ];

  return (
    <PageContainer title="브랜드" description="브랜드를 하나의 속성 축으로 보고, 평균 순위와 시간 흐름을 함께 해석합니다.">
      <PageContextBar
        title="브랜드 해석 기준"
        summary="개별 상품보다 변동성이 낮은 브랜드 단위에서 평균 순위와 시간 흐름을 보면, 반복적으로 강한 속성 축을 더 쉽게 읽을 수 있습니다."
        badges={filterBadges}
        notes={[
          { text: describeDashboardFilterScope(filters) },
          { label: "읽는 순서", text: "평균 순위 막대로 전체 분포를 먼저 보고, 분석 표와 순위 레이스로 시간 흐름을 이어서 확인하세요." },
        ]}
      />
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
