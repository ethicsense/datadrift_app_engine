import { useMemo } from "react";

import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { ReviewsPerProductDistributionPanel } from "../components/charts/ReviewsPerProductDistributionPanel";
import { WordCloudPanel } from "../components/charts/WordCloudPanel";
import { SectionCard } from "../components/SectionCard";
import { apiGet } from "../lib/api";
import { describeDashboardFilterScope } from "../lib/explainability";
import { useDashboardFilters } from "../lib/filters";
import type { WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

type TextOverviewResponse = {
  kpiRows: Record<string, unknown>[];
  sentimentDistribution: Record<string, unknown>[];
  reviewTypeDistribution: Record<string, unknown>[];
  aspectVolume: Record<string, unknown>[];
  qualityRows: Record<string, unknown>[];
  tpoDistribution: Record<string, unknown>[];
  reviewsPerProductDistribution: Record<string, unknown>[];
  reviewsPerProductBins: Record<string, unknown>[];
  reviewsPerProductStats: Record<string, unknown>[];
};

type TextAspectsResponse = {
  aspectSentiment: Record<string, unknown>[];
  aspectByCategory: Record<string, unknown>[];
  aspectByBrand: Record<string, unknown>[];
  rows: Record<string, unknown>[];
};

type TextFusionResponse = {
  fusionByAspect: Record<string, unknown>[];
  fusionByCategory: Record<string, unknown>[];
  fusionByBrand: Record<string, unknown>[];
  fusionTopProducts: Record<string, unknown>[];
};

type TextUnmetNeedsResponse = {
  summary: Record<string, unknown>[];
  byAspect: Record<string, unknown>[];
  topSentences: Record<string, unknown>[];
};

type TextSizeGuideResponse = {
  productSizeTendency: Record<string, unknown>[];
};

type TextTrendsResponse = {
  keywordTimeseries: Record<string, unknown>[];
  risingKeywords: Record<string, unknown>[];
};

type TextWordFrequencyResponse = {
  words: Record<string, unknown>[];
};

type TextBrandImageResponse = {
  brandProfile: Record<string, unknown>[];
  brandStyleMatrix: Record<string, unknown>[];
  scoringMethod?: string;
  embeddingMeta?: Record<string, unknown> | null;
  brandImageEvidence?: Record<string, unknown>[];
};

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

export function TextPage() {
  const { filters } = useDashboardFilters();
  const overviewQuery = useQuery({
    queryKey: ["text-overview", filters],
    queryFn: () => apiGet<TextOverviewResponse>("/api/text/overview", filters),
  });
  const aspectsQuery = useQuery({
    queryKey: ["text-aspects", filters],
    queryFn: () => apiGet<TextAspectsResponse>("/api/text/aspects", filters),
  });
  const fusionQuery = useQuery({
    queryKey: ["text-fusion", filters],
    queryFn: () => apiGet<TextFusionResponse>("/api/text/fusion", filters),
  });
  const wordFrequencyQuery = useQuery({
    queryKey: ["text-word-frequency", filters],
    queryFn: () => apiGet<TextWordFrequencyResponse>("/api/text/word-frequency", filters),
  });

  const primaryDataReady = overviewQuery.isSuccess && aspectsQuery.isSuccess && fusionQuery.isSuccess;

  const unmetNeedsQuery = useQuery({
    queryKey: ["text-unmet-needs", filters],
    queryFn: () => apiGet<TextUnmetNeedsResponse>("/api/text/unmet-needs", filters),
    enabled: primaryDataReady,
  });
  const sizeGuideQuery = useQuery({
    queryKey: ["text-size-guide", filters],
    queryFn: () => apiGet<TextSizeGuideResponse>("/api/text/size-guide", filters),
    enabled: primaryDataReady,
  });
  const trendsQuery = useQuery({
    queryKey: ["text-trends", filters],
    queryFn: () => apiGet<TextTrendsResponse>("/api/text/trends", filters),
    enabled: primaryDataReady,
  });
  const brandImageQuery = useQuery({
    queryKey: ["text-brand-image", filters],
    queryFn: () => apiGet<TextBrandImageResponse>("/api/text/brand-image", filters),
    enabled: primaryDataReady,
  });

  const aspectDomain = useMemo(
    () =>
      Array.from(
        new Set(
          (aspectsQuery.data?.aspectSentiment ?? EMPTY_ROWS)
            .map((row) => String(row.aspect ?? "general"))
            .filter((value) => value.length > 0),
        ),
      ),
    [aspectsQuery.data?.aspectSentiment],
  );
  const categoryDomain = useMemo(
    () =>
      Array.from(
        new Set(
          (fusionQuery.data?.fusionByCategory ?? EMPTY_ROWS)
            .map((row) => String(row.category ?? "미분류"))
            .filter((value) => value.length > 0),
        ),
      ),
    [fusionQuery.data?.fusionByCategory],
  );
  const brandStyleDomain = useMemo(
    () =>
      Array.from(
        new Set(
          (brandImageQuery.data?.brandStyleMatrix ?? EMPTY_ROWS)
            .map((row) => String(row.styleLabel ?? row.style ?? ""))
            .filter((v) => v.length > 0),
        ),
      ),
    [brandImageQuery.data?.brandStyleMatrix],
  );
  const brandDomain = useMemo(
    () =>
      Array.from(
        new Set(
          (brandImageQuery.data?.brandStyleMatrix ?? EMPTY_ROWS)
            .map((row) => String(row.brand ?? ""))
            .filter((v) => v.length > 0),
        ),
      ),
    [brandImageQuery.data?.brandStyleMatrix],
  );

  const keywordTimeseriesRows = useMemo(
    () => trendsQuery.data?.keywordTimeseries ?? EMPTY_ROWS,
    [trendsQuery.data?.keywordTimeseries],
  );

  const hasTimeseries = useMemo(
    () =>
      keywordTimeseriesRows.length > 0
      && new Set(keywordTimeseriesRows.map((r) => String(r.snapshotDate ?? ""))).size >= 2,
    [keywordTimeseriesRows],
  );

  const isLoading =
    overviewQuery.isLoading || aspectsQuery.isLoading || fusionQuery.isLoading || wordFrequencyQuery.isLoading;

  const overviewData = overviewQuery.data;
  const aspectsData = aspectsQuery.data;
  const fusionData = fusionQuery.data;

  const sharedContext = useMemo(
    () => [
      {
        label: "리뷰 문장",
        value: `${(overviewData?.kpiRows ?? EMPTY_ROWS).find((row) => row.metric === "리뷰 문장 수")?.value ?? "-"}문장`,
      },
      {
        label: "클레임 문장",
        value: `${(overviewData?.kpiRows ?? EMPTY_ROWS).find((row) => row.metric === "클레임 문장 수")?.value ?? "-"}문장`,
      },
    ],
    [overviewData?.kpiRows],
  );

  const widgets = useMemo((): WidgetConfig[] => {
    if (!overviewData || !aspectsData || !fusionData) {
      return [];
    }
    return [
      {
        id: "text-sentiment-distribution",
        type: "chart",
        title: "사용자 감성 분포",
        description: "리뷰 문장 단위의 긍정·중립·부정 반응을 비교합니다.",
        section: "summary",
        takeaway: "평점과 감성은 항상 같이 움직이지 않으니, 문장 반응 신호를 따로 확인하세요.",
        chartKind: "bar",
        rows: overviewData.sentimentDistribution ?? EMPTY_ROWS,
        spec: { x: "sentiment", y: "count", palette: "categorical", yLabel: "문장 수", yFormat: "integer" },
      },
      {
        id: "text-unmet-needs-by-aspect",
        type: "chart",
        title: "속성별 미충족 비율",
        description: "어떤 속성에서 불만/아쉬움이 집중되는지 비교합니다.",
        section: "result",
        takeaway: "미충족 비율이 높은 속성부터 개선 우선순위 후보로 보세요.",
        chartKind: "bar",
        rows: unmetNeedsQuery.data?.byAspect ?? EMPTY_ROWS,
        spec: { x: "aspect", y: "unmetRatio", palette: "categorical", yLabel: "미충족 비율(%)", yFormat: "number" },
      },
      {
        id: "text-size-guide",
        type: "table",
        title: "상품별 사이즈 반응",
        description: "리뷰 기반으로 '작다/정사이즈/크다' 반응을 상품 단위로 집계합니다.",
        section: "result",
        takeaway: "사이즈 경향이 뚜렷하면 상세페이지 가이드 보강 우선순위로 잡기 좋습니다.",
        rows: sizeGuideQuery.data?.productSizeTendency ?? EMPTY_ROWS,
        rowPreviewLimit: 12,
        rowPreviewToggleLabel: "사이즈 반응",
      },
      ...(hasTimeseries
        ? [
            {
              id: "text-keyword-timeseries",
              type: "keywordSparkTimeseries" as const,
              title: "키워드 시계열",
              description: "키워드별 스냅샷 추이를 행 단위로 보고, 선택한 키워드는 동일 축에서 절대 언급 수를 비교합니다.",
              section: "interpretation" as const,
              takeaway: "행 스파크라인은 추세 형태를, 아래 비교 차트는 규모 차이를 동시에 읽을 수 있습니다.",
              rows: keywordTimeseriesRows,
            },
          ]
        : []),
      {
        id: "text-rising-keywords",
        type: "table",
        title: "급상승 키워드",
        description: "색상/스타일 키워드 중 최근 급증한 키워드를 확인합니다.",
        section: "result",
        takeaway: "급상승 키워드는 SEO 태그, 캠페인 카피, 큐레이션 테마의 후보입니다.",
        rows: trendsQuery.data?.risingKeywords ?? EMPTY_ROWS,
        rowPreviewLimit: 12,
        rowPreviewToggleLabel: "키워드",
      },
      {
        id: "text-overview-kpis",
        type: "table",
        title: "반응 신호 요약",
        description: "리뷰와 클레임에서 잡힌 핵심 신호를 먼저 확인합니다.",
        section: "summary",
        takeaway: "한 숫자만 보지 말고, 점수·신뢰도·합의도를 같이 봐야 해석이 덜 흔들립니다.",
        explainability: {
          context: sharedContext,
          readingGuide: [{ text: "지표는 현재 필터 범위의 상품 집합을 기준으로 집계합니다." }],
        },
        rows: overviewData.kpiRows ?? EMPTY_ROWS,
      },
      {
        id: "text-tpo-distribution",
        type: "chart",
        title: "착용 상황(TPO) 분포",
        description: "리뷰에서 어떤 상황으로 언급되는지 빈도를 확인합니다.",
        section: "result",
        takeaway: "언급 상황이 뚜렷한 제품은 마케팅 소구 문구를 상황 중심으로 잡기 쉽습니다.",
        chartKind: "bar",
        rows: overviewData.tpoDistribution ?? EMPTY_ROWS,
        spec: { x: "tpo", y: "count", palette: "categorical", yLabel: "문장 수", yFormat: "integer" },
      },
      {
        id: "text-review-type",
        type: "chart",
        title: "리뷰 타입 구성",
        description: "일반/스타일/한달/뷰티 타입별 리뷰 규모를 비교합니다.",
        section: "interpretation",
        takeaway: "리뷰 타입 비중이 다르면 반응 톤도 달라져, 해석할 때 분리해서 보는 게 안전합니다.",
        chartKind: "bar",
        rows: overviewData.reviewTypeDistribution ?? EMPTY_ROWS,
        spec: { x: "reviewType", y: "count", palette: "categorical", yLabel: "리뷰 수", yFormat: "integer" },
      },
      {
        id: "text-unmet-needs-summary",
        type: "table",
        title: "미충족 니즈 요약",
        description: "리뷰에서 아쉬움·개선 요청 문장의 전체 비율을 확인합니다.",
        section: "interpretation",
        takeaway: "불편 신호가 높은 영역은 상세페이지 개선과 기획 보완의 직접 단서가 됩니다.",
        rows: unmetNeedsQuery.data?.summary ?? EMPTY_ROWS,
      },
      {
        id: "text-unmet-needs-sentences",
        type: "table",
        title: "미충족 니즈 대표 문장",
        description: "아쉬움/불만/개선 요청이 포함된 실제 리뷰 문장을 확인합니다.",
        section: "examples",
        takeaway: "숫자에서 본 신호가 실제 문장에서도 반복되는지 최종 검증하는 구간입니다.",
        rows: unmetNeedsQuery.data?.topSentences ?? EMPTY_ROWS,
      },
      {
        id: "text-fusion-quality",
        type: "table",
        title: "신호 품질 지표",
        description: "해석에 쓰이는 근거 커버리지와 밀도를 확인합니다.",
        section: "summary",
        takeaway: "신뢰도와 커버리지를 함께 보면, 결론의 안정성을 더 높일 수 있습니다.",
        rows: overviewData.qualityRows ?? EMPTY_ROWS,
      },
      {
        id: "text-reviews-per-product-stats",
        type: "table",
        title: "상품당 리뷰 수 요약",
        description: "리뷰 분포 차트와 함께 보면 쏠림(초과 건) 규모를 빠르게 파악할 수 있습니다.",
        section: "summary",
        takeaway: "중앙값·최대·300건 초과 상품 수를 보면 반응 데이터의 밀도 편차를 점검하기 좋습니다.",
        rows: overviewData.reviewsPerProductStats ?? EMPTY_ROWS,
      },
      {
        id: "text-aspect-sentiment-heatmap",
        type: "chart",
        title: "속성-감성 반응 히트맵",
        description: "속성(aspect)별 감성 분포를 교차로 봅니다.",
        section: "result",
        takeaway: "특정 속성에서 부정이 집중되면 개선 우선순위 후보입니다.",
        chartKind: "heatmap",
        rows: aspectsData.aspectSentiment ?? EMPTY_ROWS,
        spec: {
          x: "sentiment",
          y: "aspect",
          value: "sentenceCount",
          xLabel: "감성",
          yLabel: "속성",
          valueFormat: "integer",
          xDomain: ["positive", "neutral", "negative"],
          yDomain: aspectDomain,
          palette: "sequential",
        },
      },
      {
        id: "text-fusion-by-aspect",
        type: "chart",
        title: "속성별 융합 점수",
        description: "속성별 평균 융합 점수를 비교합니다. (리뷰/클레임/OCR 등 보조 신호를 합산)",
        section: "result",
        takeaway: "핵심은 사용자 반응 신호이고, OCR 텍스트는 해석을 더 풍부하게 만드는 보강 재료로 함께 활용합니다.",
        chartKind: "bar",
        rows: fusionData.fusionByAspect ?? EMPTY_ROWS,
        spec: { x: "aspect", y: "avgFusionScore", palette: "categorical", yLabel: "평균 융합 점수", yFormat: "number" },
      },
      {
        id: "text-fusion-by-category",
        type: "chart",
        title: "카테고리-속성 융합 히트맵",
        description: "카테고리별 어떤 속성이 강하게 관찰되는지 확인합니다.",
        section: "interpretation",
        takeaway: "값이 높은 셀은 강점 신호, 상대적으로 낮은 셀은 개선 기회 신호로 해석할 수 있습니다.",
        chartKind: "heatmap",
        rows: fusionData.fusionByCategory ?? EMPTY_ROWS,
        spec: {
          x: "category",
          y: "aspect",
          value: "avgFusionScore",
          xLabel: "카테고리",
          yLabel: "속성",
          valueFormat: "number",
          xDomain: categoryDomain,
          yDomain: aspectDomain,
          palette: "sequential",
        },
      },
      {
        id: "text-brand-profile",
        type: "brandImageProfile",
        title: "브랜드 이미지 프로파일",
        description:
          "상품명·태그·소재 등 브랜드 측 카피(클레임)에서 추정한 스타일 의도와, 리뷰 문장에서 추정한 고객 지각을 같은 스타일 축으로 비교합니다. 브랜드를 바꿔 프로파일 형태를 확인하세요.",
        section: "result",
        takeaway:
          "막대·레이더는 해석을 넓혀 주는 보강 도구입니다. 사용자 반응 신호(감성/불만/사이즈)와 함께 보면 해석이 더 선명해집니다.",
        profileRows: brandImageQuery.data?.brandProfile ?? EMPTY_ROWS,
        matrixRows: brandImageQuery.data?.brandStyleMatrix ?? EMPTY_ROWS,
        scoringMethod: brandImageQuery.data?.scoringMethod,
        embeddingMeta: brandImageQuery.data?.embeddingMeta ?? null,
        evidenceRows: brandImageQuery.data?.brandImageEvidence ?? EMPTY_ROWS,
      },
      {
        id: "text-brand-style-gap-heatmap",
        type: "chart",
        title: "브랜드×스타일 이미지 갭",
        description: "축별 지각 비중 − 의도 비중(−1~1에 가깝게). 양수는 리뷰에서 해당 이미지가 카피보다 더 두드러질 때입니다.",
        section: "interpretation",
        takeaway: "브랜드 메시지와 체감 차이를 넓게 이해하는 확장 뷰로 활용하세요.",
        chartKind: "heatmap",
        rows: brandImageQuery.data?.brandStyleMatrix ?? EMPTY_ROWS,
        spec: {
          x: "brand",
          y: "styleLabel",
          value: "styleGap",
          xLabel: "브랜드",
          yLabel: "스타일 축",
          valueFormat: "number",
          xDomain: brandDomain,
          yDomain: brandStyleDomain,
          palette: "sequential",
        },
      },
    ];
  }, [
    overviewData,
    aspectsData,
    fusionData,
    unmetNeedsQuery.data,
    sizeGuideQuery.data,
    trendsQuery.data,
    brandImageQuery.data,
    aspectDomain,
    categoryDomain,
    brandDomain,
    brandStyleDomain,
    hasTimeseries,
    keywordTimeseriesRows,
    sharedContext,
  ]);

  const targetSummary = useMemo(() => {
    const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
    const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
    return targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";
  }, [filters.sourceDatasets]);

  const reviewBins = overviewData?.reviewsPerProductBins ?? EMPTY_ROWS;
  const reviewStats = overviewData?.reviewsPerProductStats ?? EMPTY_ROWS;

  if (isLoading) {
    return <div className="loading-state">텍스트 통합 분석 데이터를 불러오는 중입니다.</div>;
  }

  return (
    <PageContainer
      title="리뷰"
      description="리뷰 중심의 사용자 반응 신호를 먼저 보고, OCR·상품 메타를 함께 활용해 해석을 더 풍부하게 만듭니다."
    >
      <section className="overview-story-hero">
        <small>VOICE</small>
        <h2>사용자 반응 신호는 무엇을 말하나</h2>
        <p>
          {targetSummary} 데이터에서 리뷰 문장을 중심으로 <strong>사용자 경험과 반응 신호</strong>를 먼저 읽습니다.
          감성, 불만, 사이즈, 상황 신호를 통해 제품 경험의 결을 파악합니다.
        </p>
        <p>
          해석을 보강하기 위해 OCR 텍스트와 상품 메타를 함께 수집하고 사용합니다. 결론은 사용자 반응 신호에서 먼저 찾습니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>반응 요약</span>
          <strong>감성·불만 신호</strong>
          <small>핵심 지표, 감성 분포, 미충족 비율로 사용자 반응의 방향을 먼저 확인합니다.</small>
        </article>
        <article className="overview-story-card">
          <span>경험 단서</span>
          <strong>사이즈·상황·유형</strong>
          <small>사이즈 반응, 착용 상황, 리뷰 타입을 묶어 실제 사용 경험 맥락을 읽습니다.</small>
        </article>
        <article className="overview-story-card">
          <span>보조 해석</span>
          <strong>키워드·브랜드 이미지</strong>
          <small>OCR/메타를 포함한 추가 신호를 마지막에 함께 확인해 해석의 완성도를 높입니다.</small>
        </article>
      </section>

      <section className="text-page-toolbar" aria-label="필터 요약">
        <p className="text-page-toolbar__scope">{describeDashboardFilterScope(filters)}</p>
        <p className="text-page-toolbar__selection">해석 우선순위: 사용자 반응 신호(리뷰) + OCR·메타 보강 신호</p>
      </section>
      <SectionCard
        title="리뷰 워드클라우드"
        description="리뷰 문장에서 형태소 단위로 집계한 빈도 상위 단어입니다. 불용어는 제외합니다."
        section="result"
        takeaway="자주 등장하는 표현은 고객 언어와 관심 주제를 빠르게 스캔하는 데 도움이 됩니다."
      >
        <WordCloudPanel
          rows={wordFrequencyQuery.data?.words ?? EMPTY_ROWS}
          wordKey="word"
          valueKey="frequency"
          maxWords={60}
        />
      </SectionCard>
      <ReviewsPerProductDistributionPanel bins={reviewBins} statsRows={reviewStats} sharedContext={sharedContext} />
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
