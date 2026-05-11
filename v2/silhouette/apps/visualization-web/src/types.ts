export type NavigationItem = {
  key: string;
  label: string;
  path: string;
};

export type FormatKind = "auto" | "number" | "percent" | "integer" | "price" | "string";

export type NarrativeSection = "summary" | "input" | "formula" | "result" | "interpretation" | "examples";

export type ExplainabilityTone = "neutral" | "accent" | "warning";

export type ExplainabilityFact = {
  label: string;
  value: string;
  tone?: ExplainabilityTone;
};

export type ExplainabilityNote = {
  label?: string;
  text: string;
  tone?: ExplainabilityTone;
};

export type WidgetExplainability = {
  context?: ExplainabilityFact[];
  readingGuide?: ExplainabilityNote[];
  interpretationRules?: ExplainabilityNote[];
  caveats?: ExplainabilityNote[];
  drilldown?: ExplainabilityNote[];
};

export type DashboardFilters = {
  dataset?: string;
  brands: string[];
  sourceDatasets: string[];
  platforms: string[];
  schemaVersions: string[];
  snapshotWindow?: number;
  dateFrom?: string;
  dateTo?: string;
};

export type KpiResponse = {
  title: string;
  subtitle: string;
  full: {
    snapshotCount: number;
    recordCount: number;
    productCount: number;
    brandCount: number;
  };
  filtered: {
    snapshotCount: number;
    recordCount: number;
    productCount: number;
    brandCount: number;
  };
  dateRange?: {
    min?: string;
    max?: string;
  };
  /** 필터가 적용된 fact 행에 등장하는 source_dataset 값(관측 데이터 기준). */
  filteredSourceDatasets?: string[];
};

export type ValueShapeKind = "number" | "text" | "image" | "kv" | "list" | "datetime" | "row";

export type ThumbnailImage = {
  path: string;
  isMainImage: boolean;
  mainImageSource?: string | null;
  isExplicitMainImage?: boolean;
};

export type ThumbnailFieldCell = {
  key: string;
  label: string;
  shape: ValueShapeKind;
  grain: string;
  value: unknown;
};

export type ThumbnailDetailItem = {
  key: string;
  label: string;
  value: string;
  shape: ValueShapeKind;
};

export type ThumbnailShapeLegendItem = {
  key: string;
  label: string;
  shape: ValueShapeKind;
  grain: string;
  meaning: string;
};

export type ThumbnailSnapshotSummary = {
  snapshotId: string;
  label: string;
  crawlDatetime?: string | null;
  snapshotDate?: string | null;
  snapshotTime?: string | null;
  snapshotCount?: number;
  startSnapshotId?: string | null;
  endSnapshotId?: string | null;
  recordCount: number;
  productCount: number;
  brandCount: number;
  avgPrice?: number | null;
  mainImageCoveragePct?: number | null;
  detailInfoCoveragePct?: number | null;
  categoryCoveragePct?: number | null;
  topBrand?: string | null;
  topCategory?: string | null;
};

export type ThumbnailRecordRow = {
  snapshotId: string;
  snapshotLabel: string;
  snapshotDate?: string | null;
  snapshotTime?: string | null;
  crawlDatetime?: string | null;
  productId: string;
  brand?: string | null;
  name: string;
  rank?: number | null;
  price?: number | null;
  discountPct?: number | null;
  categoryLabel?: string | null;
  categoryPath: string[];
  tags: string[];
  mainImagePath?: string | null;
  mainImageSource?: string | null;
  hasMainImage: boolean;
  hasExplicitMainImage: boolean;
  imageCount: number;
  images: ThumbnailImage[];
  detailInfoCount: number;
  detailInfoPreview: ThumbnailDetailItem[];
  detailInfoRows: ThumbnailDetailItem[];
  categorySource?: string | null;
  categoryStatus?: string | null;
  productUrl?: string | null;
  sourceDataset?: string | null;
  platform?: string | null;
  schemaVersion?: string | null;
  fieldCells: ThumbnailFieldCell[];
  rawCategoryRows: ThumbnailDetailItem[];
};

export type ThumbnailSnapshotsResponse = {
  defaultSnapshotId?: string | null;
  snapshots: ThumbnailSnapshotSummary[];
};

export type ThumbnailRecordsResponse = {
  windowMode?: "point" | "range";
  selectedSnapshotId?: string | null;
  selectedSnapshotIds?: string[];
  snapshotSummary?: ThumbnailSnapshotSummary | null;
  shapeLegend: ThumbnailShapeLegendItem[];
  rows: ThumbnailRecordRow[];
};

export type CoreEntityRow = {
  productId?: string;
  snapshotId?: string;
  sourceDataset?: string;
  schemaVersion?: string;
  categoryLabel?: string | null;
  mainImage?: string | null;
};

export type PriceDistributionResponse = {
  priceBandDistribution: Record<string, unknown>[];
  discountBandDistribution: Record<string, unknown>[];
  priceBandPerformance: Record<string, unknown>[];
  priceBandCategoryHeatmap: Record<string, unknown>[];
  summaryRows: Record<string, unknown>[];
};

export type PriceTimeseriesResponse = {
  topRankedProducts: Record<string, unknown>[];
};

export type DiscountEffectsSummary = {
  eventCount: number;
  confidentEventCount: number;
  improvedCount: number;
  neutralCount: number;
  worsenedCount: number;
  improvementRate: number | null;
  medianAbnormalRankDelta: number | null;
  velocityThreshold: number;
  controlVelocityThreshold: number;
  preWindowDays: number;
  postWindowDays: number;
  minObsPerSide: number;
  minControlSamples: number;
};

export type DiscountEffectsResponse = {
  summary: DiscountEffectsSummary;
  events: Record<string, unknown>[];
  eventStudyCurves: Record<string, unknown>[];
  effectScatter: Record<string, unknown>[];
};

export type DiscountDrilldownResponse = {
  product: Record<string, unknown> | null;
  timeline: Record<string, unknown>[];
  events: Record<string, unknown>[];
  summary: { velocityThreshold: number; eventCount: number };
};

export type MomentumInputsResponse = {
  rankVelocityDistribution: Record<string, unknown>[];
  rankAccelerationDistribution: Record<string, unknown>[];
  energyVelocityDistribution?: Record<string, unknown>[];
  energyAccelerationDistribution?: Record<string, unknown>[];
  discountVelocityDistribution: Record<string, unknown>[];
  stabilityDistribution: Record<string, unknown>[];
};

export type MomentumDistributionResponse = {
  momentumBandDistribution: Record<string, unknown>[];
  eventStateDistribution?: Record<string, unknown>[];
  brandMomentum: Record<string, unknown>[];
  priceBandMomentum: Record<string, unknown>[];
  topMomentum: Record<string, unknown>[];
};

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

export type RankTrajectoriesResponse = {
  entityType: string;
  series: RankTrajectoryPoint[];
};

export type EmbeddingAnimationResponse = {
  frames: Array<{
    snapshotId: string;
    label: string;
    newCount?: number;
    retainedCount?: number;
    exitedCount?: number;
    clusterShareTopN?: Array<{
      clusterId: string;
      pointCount: number;
      sharePct: number;
      dominantCategory: string;
      dominantSharePct: number;
    }>;
    points: Array<{
      productId: string;
      brand?: string;
      name?: string;
      rank?: number | null;
      rankVelocity?: number | null;
      movementGroup?: string | null;
      clusterId?: string | null;
      dominantCategory?: string | null;
      x?: number | null;
      y?: number | null;
      mainImage?: string | null;
      lifecycleState?: "new" | "retained" | "exited" | null;
      isGhost?: boolean;
    }>;
  }>;
};

export type EmbeddingOverviewResponse = {
  points: Array<{
    productId: string;
    snapshotId?: string;
    name?: string;
    brand?: string;
    rank?: number | null;
    categoryLabel?: string | null;
    clusterId?: string | null;
    x?: number | null;
    y?: number | null;
    mainImage?: string | null;
  }>;
  clusters: Array<{
    clusterId: string;
    pointCount: number;
    dominantCategory: string;
    dominantSharePct: number;
    categoryCount: number;
    l3Preview?: string;
    itemPreview?: string;
    colorPreview?: string;
    materialPreview?: string;
    avgRank?: number | null;
  }>;
  summary: Array<{
    metric: string;
    value: number | string;
    unit: string;
  }>;
  strategy?: string;
};

type BaseWidgetConfig = {
  id: string;
  title: string;
  description?: string;
  section?: NarrativeSection;
  takeaway?: string;
  explainability?: WidgetExplainability;
  bodyCollapsible?: boolean;
  defaultBodyExpanded?: boolean;
  bodyToggleLabel?: string;
};

export type RankTrajectoriesPayload = {
  rows: Record<string, unknown>[];
  baseSpec: Omit<
    ChartSpec,
    "loadedSeriesIds" | "defaultSeriesIds" | "onRemoveSeries" | "onSelectSeries" | "selectedBumpSeries" | "onHighlightSeries" | "highlightSeries"
  >;
  defaultSeriesIds: string[];
  availableSeries: ChartSeriesOption[];
  series: Array<{
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
  }>;
};

export type WidgetConfig =
  | (BaseWidgetConfig & { type: "kpis"; payload: KpiResponse })
  | (BaseWidgetConfig & { type: "chart"; chartKind: "bar" | "line" | "scatter" | "heatmap" | "bump"; rows: Record<string, unknown>[]; spec: ChartSpec })
  | (BaseWidgetConfig & {
      type: "table";
      rows: Record<string, unknown>[];
      highlightKey?: string;
      highlightValue?: string | null;
      rowPreviewLimit?: number;
      rowPreviewDefaultExpanded?: boolean;
      rowPreviewToggleLabel?: string;
      onRowSelect?: (row: Record<string, unknown>) => void;
    })
  | (BaseWidgetConfig & {
      type: "wordCloud";
      rows: Record<string, unknown>[];
      wordKey: string;
      valueKey: string;
      maxWords?: number;
    })
  | (BaseWidgetConfig & {
      type: "brandImageProfile";
      profileRows: Record<string, unknown>[];
      matrixRows: Record<string, unknown>[];
      scoringMethod?: string;
      embeddingMeta?: Record<string, unknown> | null;
      evidenceRows?: Record<string, unknown>[];
    })
  | (BaseWidgetConfig & { type: "keywordSparkTimeseries"; rows: Record<string, unknown>[] })
  | (BaseWidgetConfig & { type: "map"; rows: Record<string, unknown>[] })
  | (BaseWidgetConfig & { type: "rankTrajectories"; payload: RankTrajectoriesPayload })
  | (BaseWidgetConfig & { type: "animation"; animationKind: "rankRace" | "scatterMotion"; payload: Record<string, unknown> });

export type ChartSeriesOption = {
  id: string;
  label?: string;
  brand?: string;
  latestRank?: number | null;
  latestMomentum?: number | null;
  observationCount?: number | null;
  priceBand?: string | null;
  estimatedOriginalPriceBand?: string | null;
};

export type ChartSpec = {
  x: string;
  y: string;
  value?: string;
  color?: string;
  seriesBy?: string;
  xFormat?: FormatKind;
  yFormat?: FormatKind;
  valueFormat?: FormatKind;
  xDomain?: string[];
  yDomain?: string[];
  yAxisInverse?: boolean;
  xLabel?: string;
  yLabel?: string;
  palette?: "categorical" | "semantic" | "sequential" | "brand";
  markLines?: Array<{
    axis: "x" | "y";
    value: number;
    label?: string;
    /** 라벨/라인 색을 의미 톤으로 지정한다. */
    tone?: "rising" | "falling" | "neutral";
    /** 라벨 위치(기본: end = 상단). */
    labelPosition?: "start" | "middle" | "end";
  }>;
  quadrantHints?: boolean | Array<{ x: "left" | "right"; y: "top" | "bottom"; text: string }>;
  tooltipFields?: Array<{ key: string; label: string; format?: FormatKind; fallback?: string }>;
  showLegend?: boolean;
  customLegendTitle?: string;
  customLegendItems?: Array<{
    key: string;
    label: string;
    group?: string;
    description?: string;
  }>;
  /** 산점도: 플롯 영역을 정사각형에 가깝게 두고 x/y 축에 동일한 수치 범위를 사용한다(임베딩 투영 등). */
  scatterSquareEqualScale?: boolean;
  /** 산점도: 점 크기를 특정 수치 필드에 비례해 표현한다. */
  scatterSizeBy?: string;
  /** 산점도: 최소/최대 점 크기(px). */
  scatterSizeRange?: [number, number];
  /** 산점도: 점 크기 곡률(1보다 크면 상위값을 더 크게 강조). */
  scatterSizeExponent?: number;
  /** 산점도: 크기 필드가 비어 있을 때 사용할 점 크기(px). */
  scatterSizeFallback?: number;
  /** 산점도: 대표 이미지(mainImage)가 없을 때 툴팁에 안내 문구를 넣는다(임베딩 등). */
  scatterMainImageTooltip?: boolean;
  /** 산점도: 좌표 위에 군집/구간 같은 보조 라벨을 오버레이한다. */
  scatterAnnotations?: Array<{
    key: string;
    x: number;
    y: number;
    label: string;
    subLabel?: string;
    tone?: "default" | "accent";
  }>;
  /** 산점도: 군집 영역을 아주 옅은 윤곽/채움으로 표시한다. */
  scatterRegions?: Array<{
    key: string;
    label?: string;
    points: Array<[number, number]>;
    tone?: "default" | "accent";
  }>;
  /** 산점도: 현재 강조 중인 영역 키(hover/selection 연동). */
  scatterActiveRegionKey?: string | null;
  highlightSeries?: string[];
  maxVisiblePointCount?: number;
  fadeNonHighlighted?: boolean;
  availableSeries?: ChartSeriesOption[];
  loadedSeriesIds?: string[];
  defaultSeriesIds?: string[];
  resetToken?: number;
  onSelectSeries?: (seriesId: string) => void;
  onRemoveSeries?: (seriesId: string) => void;
  onClearSelectedSeries?: () => void;
  /** 제품 강조 시 호출 (bump 차트). controlled 모드일 때 선택값으로 사용 */
  selectedBumpSeries?: string | null;
  onHighlightSeries?: (seriesId: string | null) => void;
  onSelectDatum?: (row: Record<string, unknown>) => void;
  onHoverDatum?: (row: Record<string, unknown> | null) => void;
  lineSmooth?: boolean;
  /** 라인 차트 아래 영역 채움(분포 곡선 등). */
  lineArea?: boolean;
  lineShowSymbol?: boolean;
  lineShowAllSymbol?: boolean;
  lineSymbolSize?: number;
  /** 라인 차트: 값 유지 구간을 계단형으로 표현(start/middle/end). */
  lineStep?: "start" | "middle" | "end";
  /** 라인 차트: 이전 값과 차이가 임계값 이하이면 동일 값으로 간주한다. */
  lineSmallChangeThreshold?: number;
  /** 라인 차트: 이동평균 윈도우 크기(클수록 추세 위주로 매끈해짐). */
  lineMovingAverageWindow?: number;
  /**
   * 라인 차트: 값을 변경 직전까지 수평으로 유지하고, 변곡 구간에서만 짧게 전환한다.
   * 0~1 사이 비율로, 값이 클수록 수평 유지 구간이 길어지고 전환 구간이 짧아진다.
   */
  lineDiscreteHoldRatio?: number;
  lineSampling?: "lttb" | "average" | "max" | "min" | "sum";
  timeAxisLabel?: "day" | "datetime";
  /** 기본 360px 높이 대신 사용(라인/막대 공통 하단 차트). */
  chartHeight?: number;
};
