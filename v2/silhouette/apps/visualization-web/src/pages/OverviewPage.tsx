import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { PageContainer } from "../components/PageContainer";
import { apiGet } from "../lib/api";
import { useDashboardFilters } from "../lib/filters";
import { formatNumber } from "../lib/formatters";
import type { KpiResponse, WidgetConfig } from "../types";
import { WidgetRenderer } from "../widgets/WidgetRenderer";

type DatasetProfileResponse = {
  profileRows: Record<string, unknown>[];
  grainRows: Record<string, unknown>[];
};

type OverviewSchemaResponse = {
  rows: Record<string, unknown>[];
};

type OverviewCaveatsResponse = {
  rows: Record<string, unknown>[];
};

function summarizeFilteredSources(sources: string[] | undefined): {
  cardHeadline: string;
  cardDetail: string;
  heroScopePhrase: string;
  cardTitleAttr: string;
} {
  const list = sources?.filter(Boolean) ?? [];
  if (list.length === 0) {
    return {
      cardHeadline: "—",
      cardDetail: "fact에 source_dataset 없거나 필터 결과가 비어 있습니다.",
      heroScopePhrase: "현재 필터 범위의 소스",
      cardTitleAttr: "",
    };
  }
  if (list.length === 1) {
    return {
      cardHeadline: list[0],
      cardDetail: "source_dataset · 필터 반영",
      heroScopePhrase: list[0],
      cardTitleAttr: list[0],
    };
  }
  const joined = list.join(", ");
  const detail =
    list.length <= 3 ? list.join(" · ") : `${list.slice(0, 2).join(" · ")} 외 ${list.length - 2}개`;
  return {
    cardHeadline: `${list.length}개 소스`,
    cardDetail: detail,
    heroScopePhrase: `${list.length}개의 수집 소스`,
    cardTitleAttr: joined,
  };
}

function toDayCountInclusive(minDate?: string, maxDate?: string): number | null {
  if (!minDate || !maxDate) {
    return null;
  }
  const min = new Date(`${minDate}T00:00:00`);
  const max = new Date(`${maxDate}T00:00:00`);
  if (Number.isNaN(min.getTime()) || Number.isNaN(max.getTime())) {
    return null;
  }
  const diffDays = Math.floor((max.getTime() - min.getTime()) / (1000 * 60 * 60 * 24)) + 1;
  return diffDays > 0 ? diffDays : null;
}

export function OverviewPage() {
  const { filters } = useDashboardFilters();
  const kpiQuery = useQuery({
    queryKey: ["overview-kpis", filters],
    queryFn: () => apiGet<KpiResponse>("/api/overview/kpis", filters),
  });
  const profileQuery = useQuery({
    queryKey: ["overview-dataset-profile", filters],
    queryFn: () => apiGet<DatasetProfileResponse>("/api/overview/dataset-profile", filters),
  });
  const schemaQuery = useQuery({
    queryKey: ["overview-schema", filters],
    queryFn: () => apiGet<OverviewSchemaResponse>("/api/overview/schema", filters),
  });
  const caveatsQuery = useQuery({
    queryKey: ["overview-caveats", filters],
    queryFn: () => apiGet<OverviewCaveatsResponse>("/api/overview/caveats", filters),
  });

  if (kpiQuery.isLoading || profileQuery.isLoading || schemaQuery.isLoading || caveatsQuery.isLoading) {
    return <div className="loading-state">개요 데이터를 불러오는 중입니다.</div>;
  }

  const filteredRecordCount = kpiQuery.data?.filtered.recordCount ?? 0;
  const filteredProductCount = kpiQuery.data?.filtered.productCount ?? 0;
  const filteredSnapshotCount = kpiQuery.data?.filtered.snapshotCount ?? 0;
  const sharedWidgetContext = [
    { label: "관측 행", value: `${formatNumber(filteredRecordCount, "integer")}행` },
    { label: "고유 상품", value: `${formatNumber(filteredProductCount, "integer")}개` },
    { label: "스냅샷", value: `${formatNumber(filteredSnapshotCount, "integer")}회` },
  ];

  const dateMin = kpiQuery.data?.dateRange?.min;
  const dateMax = kpiQuery.data?.dateRange?.max;
  const trackedDays = toDayCountInclusive(dateMin, dateMax);
  const avgSnapshotsPerDay =
    trackedDays && filteredSnapshotCount > 0 ? Number((filteredSnapshotCount / trackedDays).toFixed(2)) : null;
  const avgRowsPerSnapshot =
    filteredSnapshotCount > 0 ? Number((filteredRecordCount / filteredSnapshotCount).toFixed(1)) : null;
  const targetMeta = summarizeFilteredSources(kpiQuery.data?.filteredSourceDatasets);

  const widgets: WidgetConfig[] = [];
  if (profileQuery.data) {
    widgets.push({
      id: "overview-dataset-profile",
      type: "table",
      title: "데이터 프로필",
      description: "상단 요약과 같은 숫자를 표로 풀어, 범위와 품질을 함께 확인할 수 있습니다.",
      section: "summary",
      takeaway: "숫자가 의미하는 단위(행·상품·회차)를 먼저 맞춰 두면 이후 탭 해석이 쉬워집니다.",
      explainability: {
        context: sharedWidgetContext,
        readingGuide: [
          { text: "각 행은 한 가지 지표에 대해 전체 범위와 현재 필터 범위를 나란히 보여 줍니다." },
        ],
        caveats: [
          { text: "수집 범위가 낮은 속성은 이후 탭의 차이가 데이터 부재 영향일 수 있습니다.", tone: "warning" },
        ],
      },
      rows: profileQuery.data.profileRows,
    });
    widgets.push({
      id: "overview-data-grain-guide",
      type: "table",
      title: "읽는 단위 안내",
      description: "행(스냅샷×상품), 고유 상품, 계산된 지표가 각각 무엇을 가리키는지 정리합니다.",
      section: "summary",
      takeaway: "같은 숫자라도 행 기준인지 상품 기준인지에 따라 읽는 방법이 달라집니다.",
      explainability: {
        context: sharedWidgetContext,
        readingGuide: [
          { text: "이 표는 각 탭이 어떤 단위의 데이터를 쌓고 비교하는지 해설하는 사전 역할을 합니다." },
        ],
        drilldown: [
          { text: "`섬네일` 탭으로 이동하면 이 단위가 실제 행과 스냅샷에서 어떻게 보이는지 확인할 수 있습니다." },
        ],
      },
      rows: profileQuery.data.grainRows,
    });
  }
  if (caveatsQuery.data) {
    widgets.push({
      id: "overview-caveats",
      type: "table",
      title: "해석 주의사항",
      description: "이 데이터셋을 읽을 때 자주 헷갈리는 기준과 제약을 정리했습니다.",
      section: "summary",
      takeaway: "가격대 체계, 파생 지표, 공간 좌표 같은 규칙을 먼저 이해하면 이후 분석 탭의 해석 오류를 크게 줄일 수 있습니다.",
      explainability: {
        context: sharedWidgetContext,
        readingGuide: [
          { text: "이 블록은 개별 차트의 수치를 읽기 전에 공통 제약을 확인하는 체크리스트입니다." },
        ],
        caveats: [
          { text: "여기 적힌 제한조건은 이후 가격·모멘텀·공간 탭 전반에 동일하게 적용됩니다.", tone: "warning" },
        ],
      },
      rows: caveatsQuery.data.rows,
    });
  }
  if (schemaQuery.data) {
    widgets.push({
      id: "overview-schema",
      type: "table",
      title: "컬럼 사전",
      description: "현재 필터 기준으로 실제 존재하는 컬럼의 타입, 범주, 결측률, 예시값, 의미를 정리한 표입니다.",
      section: "summary",
      takeaway: "원천값/파생값 구분과 수집 제약을 한 번에 파악할 수 있습니다.",
      explainability: {
        context: sharedWidgetContext,
        readingGuide: [
          { text: "컬럼 하나가 한 행이며, 값 형식과 결측률을 함께 읽는 참고 사전입니다." },
        ],
        caveats: [
          { text: "데이터 형식을 직접 해석해야 할 때 확인하는 참고 블록입니다.", tone: "warning" },
        ],
      },
      bodyCollapsible: true,
      defaultBodyExpanded: false,
      bodyToggleLabel: "컬럼 사전",
      rows: schemaQuery.data.rows,
    });
  }

  return (
    <PageContainer
      title="개요"
      description="패션 산업 데이터를 추적 관측하고 분석한 프로젝트의 범위를 한눈에 파악하는 페이지입니다."
    >
      <section className="overview-story-hero">
        <small>OUTLINE</small>
        <h2>패션산업 데이터의 추적 관측 및 분석</h2>
        <p>
          마케팅·커머스 관점에서 수집한 스냅샷 데이터를 시간축으로 정리해, 채널과 상품 단위의 변화를 비교·해석할 수 있는 관측
          데이터셋으로 구성했습니다. 현재 화면의 범위는 필터에 따라 {targetMeta.heroScopePhrase}에 맞춰 좁혀집니다.
        </p>
      </section>
      <section className="overview-story-grid">
        <article className="overview-story-card" title={targetMeta.cardTitleAttr || undefined}>
          <span>타겟</span>
          <strong>{targetMeta.cardHeadline}</strong>
          <small>{targetMeta.cardDetail}</small>
        </article>
        <article className="overview-story-card">
          <span>기간</span>
          <strong className="overview-story-card__dates">
            {dateMin && dateMax ? (
              <>
                <span className="overview-story-card__date-line">{dateMin}</span>
                <span className="overview-story-card__date-line">{dateMax}</span>
              </>
            ) : (
              "기간 정보 없음"
            )}
          </strong>
          <small>{trackedDays ? `총 ${formatNumber(trackedDays, "integer")}일` : "기간 계산 불가"}</small>
        </article>
        <article className="overview-story-card">
          <span>총 스냅샷</span>
          <strong>{formatNumber(filteredSnapshotCount, "integer")}회</strong>
          <small>{avgSnapshotsPerDay ? `하루 평균 ${formatNumber(avgSnapshotsPerDay, "number")}회` : "하루 평균 계산 불가"}</small>
        </article>
        <article className="overview-story-card">
          <span>관측 행</span>
          <strong>{formatNumber(filteredRecordCount, "integer")}행</strong>
          <small>{avgRowsPerSnapshot ? `회당 평균 ${formatNumber(avgRowsPerSnapshot, "number")}행` : "회당 평균 계산 불가"}</small>
        </article>
        <article className="overview-story-card">
          <span>고유 상품</span>
          <strong>{formatNumber(filteredProductCount, "integer")}개</strong>
          <small>중복 관측 제거 기준</small>
        </article>
        <article className="overview-story-card">
          <span>고유 브랜드</span>
          <strong>{formatNumber(kpiQuery.data?.filtered.brandCount ?? 0, "integer")}개</strong>
          <small>브랜드 분포 폭 확인 지표</small>
        </article>
      </section>
      <section className="overview-cta-card">
        <div>
          <small>Next Step</small>
          <h2>요약 뒤에는 근거를 확인하세요.</h2>
          <p>
            개요가 관측 범위를 보여줬다면, `섬네일` 탭은 각 스냅샷의 실제 행 데이터와 이미지/상세정보 구조를 원본에 가깝게
            검증하는 화면입니다.
          </p>
        </div>
        <Link className="ghost-button overview-cta-card__link" to="/thumbnails">
          섬네일 탭 열기
        </Link>
      </section>
      <WidgetRenderer widgets={widgets} />
    </PageContainer>
  );
}
