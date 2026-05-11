import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { PageContextBar } from "../components/PageContextBar";
import { apiGet } from "../lib/api";
import { buildDashboardFilterBadges, describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type { WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

type PerformanceResponse = {
  chart: Record<string, unknown>[];
  rows: Record<string, unknown>[];
};

export function PerformancePage() {
  const { filters } = useDashboardFilters();
  const query = useQuery({
    queryKey: ["performance", filters],
    queryFn: () => apiGet<PerformanceResponse>("/api/performance/tag-correlation", filters),
  });

  if (query.isLoading) {
    return <div className="loading-state">성과 상관 데이터를 불러오는 중입니다.</div>;
  }

  const filterBadges = buildDashboardFilterBadges(filters);
  const sharedContext = [
    { label: "태그 수", value: `${formatNumber((query.data?.rows ?? []).length, "integer")}개` },
  ];

  const widgets: WidgetConfig[] = [
    {
      id: "performance-chart",
      type: "chart",
      title: "태그별 평균 순위",
      description: "속성 그룹별로 평균 순위가 어떻게 달라지는지 먼저 확인합니다.",
      section: "interpretation",
      takeaway: "속성 탭에서는 개별 속성 분포를 본 뒤, 그 속성이 성과 차이와 연결되는지 확인하는 단계가 중요합니다.",
      chartKind: "bar",
      rows: query.data?.chart ?? [],
      spec: { x: "tag", y: "avgRank", yAxisInverse: true, palette: "categorical", yLabel: "평균 순위", yFormat: "number" },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "막대 1개는 태그 그룹 1개이며, 낮은 평균 순위일수록 더 강한 성과를 뜻합니다." },
        ],
      },
    },
    {
      id: "performance-table",
      type: "table",
      title: "태그 성과 표",
      section: "examples",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "행 1개는 태그 1개입니다. 차트에서 본 차이가 표본 수나 추가 지표와 함께 어떻게 나타나는지 확인합니다." },
        ],
      },
      rows: query.data?.rows ?? [],
    },
  ];

  return (
    <PageContainer title="태그 성과" description="태그 속성이 순위, 모멘텀, 안정성과 어떤 차이를 보이는지 비교합니다.">
      <PageContextBar
        title="태그 성과 비교"
        summary="이 화면은 태그라는 속성 축이 실제 순위 차이와 연결되는지 보는 비교 화면입니다. 빈도만 보는 것이 아니라 성과와 함께 읽는 것이 핵심입니다."
        badges={filterBadges}
        notes={[
          { text: describeDashboardFilterScope(filters) },
          { label: "읽는 순서", text: "태그별 평균 순위를 먼저 보고, 아래 표에서 표본 수와 세부 수치를 같이 확인하세요." },
        ]}
      />
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
