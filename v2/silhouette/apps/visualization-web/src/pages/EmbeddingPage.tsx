import { useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { PageContainer } from "../components/PageContainer";
import { apiGet } from "../lib/api";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type { EmbeddingAnimationResponse, EmbeddingOverviewResponse, WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

function pickPreviewLabel(value?: string | null) {
  if (!value || value === "-") {
    return "";
  }
  return value
    .split(" / ")[0]
    .replace(/\s+\d+%$/, "")
    .trim();
}

function buildClusterAnnotationLabel(cluster: EmbeddingOverviewResponse["clusters"][number]) {
  return pickPreviewLabel(cluster.l3Preview) || pickPreviewLabel(cluster.itemPreview) || cluster.dominantCategory || cluster.clusterId;
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

function buildLegendDisplay(label: string) {
  const parts = label.split(">").map((part) => part.trim()).filter(Boolean);
  if (parts.length >= 3) {
    return {
      group: `${parts[0]} > ${parts[1]}`,
      label: parts.slice(2).join(" > "),
    };
  }
  if (parts.length === 2) {
    return {
      group: parts[0],
      label: parts[1],
    };
  }
  return {
    group: "기타",
    label,
  };
}

function cross(
  origin: [number, number],
  pointA: [number, number],
  pointB: [number, number],
) {
  return (pointA[0] - origin[0]) * (pointB[1] - origin[1]) - (pointA[1] - origin[1]) * (pointB[0] - origin[0]);
}

function buildConvexHull(points: Array<[number, number]>) {
  if (points.length <= 3) {
    return points;
  }
  const sorted = [...points].sort((left, right) => {
    if (left[0] !== right[0]) {
      return left[0] - right[0];
    }
    return left[1] - right[1];
  });
  const lower: Array<[number, number]> = [];
  sorted.forEach((point) => {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  });
  const upper: Array<[number, number]> = [];
  [...sorted].reverse().forEach((point) => {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  });
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

export function EmbeddingPage() {
  const { filters } = useDashboardFilters();
  const [hoveredClusterId, setHoveredClusterId] = useState<string | null>(null);
  const embeddingQuery = useQuery({
    queryKey: ["embedding-projection", filters],
    queryFn: () => apiGet<EmbeddingAnimationResponse>("/api/animation/embedding-projection", filters),
  });
  const overviewQuery = useQuery({
    queryKey: ["embedding-overview", filters],
    queryFn: () => apiGet<EmbeddingOverviewResponse>("/api/animation/embedding-overview", filters),
  });

  const summaryRows = useMemo(() => {
    const frames = embeddingQuery.data?.frames ?? [];
    const uniqueProducts = new Set<string>();
    let newCount = 0;
    let retainedCount = 0;
    let exitedCount = 0;
    frames.forEach((frame) => {
      newCount += frame.newCount ?? 0;
      retainedCount += frame.retainedCount ?? 0;
      exitedCount += frame.exitedCount ?? 0;
      frame.points.forEach((point) => {
        uniqueProducts.add(point.productId);
      });
    });
    return [
      { metric: "프레임 수", value: frames.length, unit: "개" },
      { metric: "고유 상품 수", value: uniqueProducts.size, unit: "개" },
      { metric: "총 포인트 수", value: frames.reduce((sum, frame) => sum + frame.points.length, 0), unit: "개" },
      { metric: "신규(직전 2프레임 밖)", value: newCount, unit: "개" },
      { metric: "유지(직전 2프레임 안)", value: retainedCount, unit: "개" },
      { metric: "이탈 잔상(직전 2프레임 기준)", value: exitedCount, unit: "개" },
    ];
  }, [embeddingQuery.data?.frames]);

  const densityRows = useMemo(() => {
    return (overviewQuery.data?.clusters ?? [])
      .slice()
      .sort((left, right) => right.pointCount - left.pointCount)
      .slice(0, 8)
      .map((cluster, index) => ({
        순위: index + 1,
        밀집구간: buildClusterAnnotationLabel(cluster),
        포인트: cluster.pointCount,
        평균순위: typeof cluster.avgRank === "number" ? Number(cluster.avgRank.toFixed(0)) : "-",
        지배카테고리: cluster.dominantCategory,
        지배비중: `${formatNumber(cluster.dominantSharePct, "number")}%`,
      }));
  }, [overviewQuery.data?.clusters]);

  const clusterRows = useMemo(() => {
    const overviewClusters = (overviewQuery.data?.clusters ?? [])
      .slice()
      .sort((left, right) => right.pointCount - left.pointCount)
      .slice(0, 12)
      .map((cluster) => ({
        클러스터: cluster.clusterId,
        규모: formatNumber(cluster.pointCount, "integer"),
        상위카테고리: cluster.l3Preview && cluster.l3Preview !== "-" ? cluster.l3Preview : cluster.dominantCategory,
        주요색상: cluster.colorPreview ?? "-",
        주요소재: cluster.materialPreview ?? "-",
        평균순위: typeof cluster.avgRank === "number" ? formatNumber(cluster.avgRank, "integer") : "-",
        대표비중: `${formatNumber(cluster.dominantSharePct, "number")}%`,
      }));
    if (overviewClusters.length > 0) {
      return overviewClusters;
    }

    const counts = new Map<string, { frames: number; shareSum: number; dominantCategory: string; dominantSharePct: number }>();
    (embeddingQuery.data?.frames ?? []).forEach((frame) => {
      (frame.clusterShareTopN ?? []).forEach((cluster) => {
        const key = String(cluster.clusterId ?? "unknown");
        const current = counts.get(key) ?? {
          frames: 0,
          shareSum: 0,
          dominantCategory: String(cluster.dominantCategory ?? "미분류"),
          dominantSharePct: Number(cluster.dominantSharePct ?? 0),
        };
        current.frames += 1;
        current.shareSum += Number(cluster.sharePct ?? 0);
        if ((cluster.dominantSharePct ?? 0) > current.dominantSharePct) {
          current.dominantSharePct = Number(cluster.dominantSharePct ?? 0);
          current.dominantCategory = String(cluster.dominantCategory ?? "미분류");
        }
        counts.set(key, current);
      });
    });
    return Array.from(counts.entries())
      .map(([clusterId, value]) => ({
        클러스터: clusterId,
        관측프레임: formatNumber(value.frames, "integer"),
        평균점유율: `${formatNumber(value.frames > 0 ? value.shareSum / value.frames : 0, "number")}%`,
        대표카테고리: value.dominantCategory,
        대표비중: `${formatNumber(value.dominantSharePct, "number")}%`,
      }))
      .sort((left, right) => {
        const leftValue = Number(String(left.평균점유율).replace("%", ""));
        const rightValue = Number(String(right.평균점유율).replace("%", ""));
        return rightValue - leftValue;
      })
      .slice(0, 12);
  }, [embeddingQuery.data?.frames, overviewQuery.data?.clusters]);

  const clusterAnnotations = useMemo(() => {
    const points = overviewQuery.data?.points ?? [];
    const clusters = (overviewQuery.data?.clusters ?? [])
      .slice()
      .sort((left, right) => right.pointCount - left.pointCount)
      .slice(0, 5);
    if (!points.length || !clusters.length) {
      return [];
    }

    const centroids = new Map<string, { xSum: number; ySum: number; count: number }>();
    points.forEach((point) => {
      const clusterId = point.clusterId ? String(point.clusterId) : "";
      if (!clusterId || typeof point.x !== "number" || typeof point.y !== "number") {
        return;
      }
      const current = centroids.get(clusterId) ?? { xSum: 0, ySum: 0, count: 0 };
      current.xSum += point.x;
      current.ySum += point.y;
      current.count += 1;
      centroids.set(clusterId, current);
    });

    return clusters.flatMap((cluster, index) => {
      const centroid = centroids.get(cluster.clusterId);
      if (!centroid || centroid.count === 0) {
        return [];
      }
      const tone: "accent" | "default" = index < 2 ? "accent" : "default";
      const colorHint = pickPreviewLabel(cluster.colorPreview);
      const itemHint = pickPreviewLabel(cluster.itemPreview);
      const label = buildClusterAnnotationLabel(cluster);
      const subHints = [
        colorHint && colorHint !== label ? colorHint : "",
        !colorHint && itemHint && itemHint !== label ? itemHint : "",
        `${formatNumber(cluster.pointCount, "integer")}개`,
      ].filter(Boolean);
      return [{
        key: cluster.clusterId,
        x: centroid.xSum / centroid.count,
        y: centroid.ySum / centroid.count,
        label,
        subLabel: subHints.join(" · "),
        tone,
      }];
    });
  }, [overviewQuery.data?.clusters, overviewQuery.data?.points]);

  const clusterRegions = useMemo(() => {
    const points = overviewQuery.data?.points ?? [];
    const clusters = (overviewQuery.data?.clusters ?? [])
      .slice()
      .sort((left, right) => right.pointCount - left.pointCount)
      .slice(0, 5);
    if (!points.length || !clusters.length) {
      return [];
    }

    const pointMap = new Map<string, Array<[number, number]>>();
    points.forEach((point) => {
      const clusterId = point.clusterId ? String(point.clusterId) : "";
      if (!clusterId || typeof point.x !== "number" || typeof point.y !== "number") {
        return;
      }
      const current = pointMap.get(clusterId) ?? [];
      current.push([point.x, point.y]);
      pointMap.set(clusterId, current);
    });

    return clusters.flatMap((cluster, index) => {
      const clusterPoints = pointMap.get(cluster.clusterId) ?? [];
      if (clusterPoints.length < 3) {
        return [];
      }
      const hull = buildConvexHull(clusterPoints);
      if (hull.length < 3) {
        return [];
      }
      return [{
        key: cluster.clusterId,
        label: buildClusterAnnotationLabel(cluster),
        points: hull,
        tone: (index < 2 ? "accent" : "default") as "accent" | "default",
      }];
    });
  }, [overviewQuery.data?.clusters, overviewQuery.data?.points]);

  const categoryLegendItems = useMemo(() => {
    const counts = new Map<string, number>();
    (overviewQuery.data?.points ?? []).forEach((point) => {
      const key = String(point.categoryLabel ?? "").trim();
      if (!key) {
        return;
      }
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return Array.from(counts.entries())
      .sort((left, right) => {
        if (right[1] !== left[1]) {
          return right[1] - left[1];
        }
        return left[0].localeCompare(right[0], "ko");
      })
      .map(([key, count]) => {
        const display = buildLegendDisplay(key);
        return {
          key,
          group: display.group,
          label: display.label,
          description: `${key} · ${formatNumber(count, "integer")}개`,
        };
      });
  }, [overviewQuery.data?.points]);

  if (embeddingQuery.isLoading || overviewQuery.isLoading) {
    return <div className="loading-state">임베딩 데이터를 불러오는 중입니다.</div>;
  }

  const frameCount = embeddingQuery.data?.frames.length ?? 0;
  const pointCount = overviewQuery.data?.points.length ?? 0;
  const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
  const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
  const targetSummary = targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";
  const sharedContext = [
    { label: "프레임", value: `${formatNumber(frameCount, "integer")}개` },
    { label: "고유 상품", value: `${formatNumber(pointCount, "integer")}개` },
    { label: "전략", value: overviewQuery.data?.strategy ?? "기본 투영" },
  ];

  const lifecycleLegendRows = [
    { 상태: "신규", 기준: "현재 프레임에만 새로 들어온 상품" },
    { 상태: "유지", 기준: "현재 프레임과 직전 1~2개 프레임에 함께 있는 상품" },
    { 상태: "이탈", 기준: "현재 프레임에는 없고 직전 1~2개 프레임에는 있던 상품" },
  ];

  const widgets: WidgetConfig[] = [
    {
      id: "embedding-animation",
      type: "animation",
      title: "제품의 생애주기",
      description: "시간 흐름에 따라 어떤 스타일 구간이 유입·유지·이탈하는지 애니메이션으로 확인합니다.",
      section: "result",
      animationKind: "scatterMotion",
      payload: { frames: embeddingQuery.data?.frames ?? [] },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "프레임 1개는 한 시점의 임베딩 분포이며, `new/retained/exited`는 좌표 이동보다 구성 변화 해석에 초점을 둡니다." },
        ],
        drilldown: [
          { text: "애니메이션에서 본 프레임 변화는 밀집 구간 표와 군집 단서를 함께 보며 해석하세요." },
        ],
      },
    },
    {
      id: "embedding-overview-scatter",
      type: "chart",
      chartKind: "scatter",
      title: "전체 고유상품 분포도",
      description: "고유 상품을 2차원 좌표에 펼쳐, 어떤 스타일이 어디에 몰려 있는지 먼저 확인합니다.",
      section: "result",
      takeaway: "먼저 점이 많이 모인 구간을 찾고, 그 주변 상품 이미지를 보며 인기 스타일 후보를 빠르게 좁힙니다.",
      rows: overviewQuery.data?.points ?? [],
      spec: {
        x: "x",
        y: "y",
        seriesBy: "categoryLabel",
        xLabel: "projection-x",
        yLabel: "projection-y",
        palette: "categorical",
        showLegend: false,
        customLegendTitle: "카테고리 범례",
        customLegendItems: categoryLegendItems,
        scatterSquareEqualScale: true,
        scatterMainImageTooltip: true,
        scatterAnnotations: clusterAnnotations,
        scatterRegions: clusterRegions,
        scatterActiveRegionKey: hoveredClusterId,
        onHoverDatum: (row) => {
          if (!row || !row.__scatterAnnotation) {
            setHoveredClusterId(null);
            return;
          }
          setHoveredClusterId(String(row.clusterId ?? ""));
        },
        tooltipFields: [
          { key: "categoryLabel", label: "카테고리", format: "string", fallback: "-" },
          { key: "clusterId", label: "클러스터", format: "string", fallback: "-" },
          { key: "productId", label: "상품 ID", format: "string" },
        ],
      },
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "점 1개는 고유 상품 1개입니다. 가까울수록 이미지 표현이 비슷한 편으로 해석합니다." },
        ],
        interpretationRules: [
          { text: "밀집 영역의 군집 라벨과 평균순위를 함께 읽어, 어떤 스타일 구간이 인기 신호를 만드는지 확인하세요." },
        ],
      },
    },
    {
      id: "embedding-density-hotspots",
      type: "table",
      title: "지금 많이 모이는 스타일 구간 TOP",
      description: "좌표 공간에서 포인트가 많이 모인 구간을 상위부터 보여줍니다.",
      section: "summary",
      takeaway: "포인트가 높은 구간은 해당 시각 특징 조합을 가진 상품이 상대적으로 많이 관측된 영역입니다.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "밀집구간은 클러스터 중심 라벨이며, 평균순위를 함께 보면서 관심도 신호를 함께 판단합니다." },
        ],
      },
      rows: densityRows,
    },
    {
      id: "embedding-clusters",
      type: "table",
      title: "구간별 스타일 단서",
      description: "밀집 구간을 읽기 위해 군집별 카테고리·색상·소재 단서를 함께 보여줍니다.",
      section: "interpretation",
      takeaway: "군집 이름은 고정 의미가 없으므로, 단서 조합을 통해 스타일/상품군 의미를 읽습니다.",
      rows: clusterRows,
    },
    {
      id: "embedding-lifecycle-legend",
      type: "table",
      title: "생애주기 읽는 법",
      description: "애니메이션에 표시되는 상태를 간단한 기준으로 정리한 표입니다.",
      section: "interpretation",
      takeaway: "이 화면의 움직임은 동일 상품 좌표 이동보다 유입/유지/이탈 구성 변화를 강조합니다.",
      rows: lifecycleLegendRows,
    },
    {
      id: "embedding-overview-summary",
      type: "table",
      title: "분포 한눈에 요약",
      description: "시간축을 제거하고 고유 상품 기준 임베딩 2차원 분포를 집계합니다. `미분류`는 제외하고, 카테고리 축은 기본적으로 상위/중간 분류를 사용하되 의류는 표본이 충분하면 하위 분류까지 확장합니다.",
      section: "summary",
      takeaway: "색상 수를 과도하게 늘리지 않으면서, 내부 구조가 중요한 구간은 더 세부적으로 읽을 수 있게 했습니다.",
      rows: overviewQuery.data?.summary ?? [],
    },
    {
      id: "embedding-overview-clusters",
      type: "table",
      title: "스타일 묶음의 카테고리 성격",
      description: "전체 분포에 대해 비지도 클러스터를 만들고, 각 클러스터의 지배 카테고리 비중을 계산합니다.",
      section: "interpretation",
      takeaway: "지배 카테고리 비중이 높을수록 임베딩 군집이 실제 카테고리 체계와 맞닿아 있을 가능성이 큽니다.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "행 1개는 비지도 군집 1개입니다. 지배 카테고리는 군집의 의미를 읽기 위한 보조 라벨입니다." },
        ],
      },
      rows: overviewQuery.data?.clusters ?? [],
    },
    {
      id: "embedding-summary",
      type: "table",
      title: "분석 대상 요약",
      description: "현재 필터에서 임베딩 해석에 사용할 시점·상품 규모를 정리합니다.",
      section: "summary",
      takeaway: "이 탭은 카테고리 분류보다, 시각적으로 비슷한 상품이 어디에 모이는지를 읽는 데 초점을 둡니다.",
      explainability: {
        context: sharedContext,
        readingGuide: [
          { text: "신규·유지·이탈은 모두 현재 프레임을 직전 1~2개 프레임과 비교한 결과입니다." },
        ],
      },
      rows: summaryRows,
    },
  ];

  return (
    <PageContainer
      title="임베딩"
      description="이미지 임베딩 좌표에서 어떤 시각적 특징 구간이 밀집되는지 확인하고, 그 구간의 상품 단서를 해석합니다."
    >
      <section className="overview-story-hero">
        <small>IMAGE FEATURES</small>
        <h2>어떻게 생긴 제품이 인기가 많을까</h2>
        <p>
          {targetSummary} 상품 이미지를 한 화면에 펼쳐서, 비슷한 느낌의 제품이 어디에 많이 모이는지 먼저 봅니다.
          많이 모인 구간일수록 사람들이 자주 보는 스타일일 가능성이 큽니다.
        </p>
        <p>
          먼저 제품의 생애주기와 전체 분포도를 보고, 아래 근거 표에서 왜 그런 흐름이 나왔는지 확인해보세요.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>첫 화면</span>
          <strong>제품의 생애주기</strong>
          <small>시간이 지나면서 어떤 스타일이 새로 뜨고, 계속 인기인지, 빠지는지 한눈에 봅니다.</small>
        </article>
        <article className="overview-story-card">
          <span>두 번째 화면</span>
          <strong>전체 고유상품 분포도</strong>
          <small>상품들이 어디에 많이 모여 있는지 보면서, 비슷한 느낌의 제품군을 빠르게 찾습니다.</small>
        </article>
        <article className="overview-story-card">
          <span>해석 근거</span>
          <strong>추가 근거 패널</strong>
          <small>아래 표들을 같이 보면서 “왜 그렇게 보이는지”를 근거로 확인합니다.</small>
        </article>
      </section>
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
