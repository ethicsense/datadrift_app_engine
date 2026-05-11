import type { FormatKind } from "../types";

const COLUMN_LABEL_MAP: Record<string, string> = {
  기준순위: "기준 순위",
  metric: "지표",
  whyItMatters: "설명",
  fullValue: "전체 범위",
  filteredValue: "필터 범위",
  scope: "기준",
  concept: "개념",
  definition: "정의",
  examples: "예시 필드",
  howToRead: "읽는 법",
  value: "값",
  unit: "단위",
  categoryLabel: "카테고리",
  productCount: "상품 수",
  brandCount: "브랜드 수",
  recordCount: "레코드 수",
  shareOfCatalog: "카탈로그 점유율",
  avgRank: "평균 순위",
  avgMomentumScore: "평균 모멘텀",
  avgFusionScore: "평균 융합 점수",
  fusionScore: "융합 점수",
  confidenceScore: "신뢰도",
  avgConfidenceScore: "평균 신뢰도",
  agreementRate: "합의도",
  avgAgreementRate: "평균 합의도",
  evidenceCount: "근거 수",
  evidenceDensity: "근거 밀도",
  claimSignal: "클레임 신호",
  reviewSignal: "리뷰 신호",
  contributionSharePct: "기여 비중",
  sourceField: "소스 필드",
  sourceFieldCount: "소스 수",
  schemaKey: "스키마 키",
  field: "필드 명",
  rawField: "Raw 필드",
  normalizedField: "Normalized 필드",
  rawValue: "원문 값",
  normalizedValue: "정규화 값",
  rawType: "Raw 타입",
  normalizedType: "Normalized 타입",
  rawSampleValue: "Raw 예시 값",
  normalizedSampleValue: "Normalized 예시 값",
  rawObservedRatePct: "Raw 관측 비율",
  normalizedObservedRatePct: "Normalized 관측 비율",
  rawNonNullRatePct: "Raw 비결측 비율",
  normalizedNonNullRatePct: "Normalized 비결측 비율",
  rawSourcePath: "Raw 원천 경로",
  normalizedSourcePath: "Normalized 원천 경로",
  valueStatus: "값 비교",
  pairingStatus: "매칭 상태",
  representativeClaim: "대표 클레임",
  representativeClaimType: "대표 클레임 유형",
  representativeClaimSource: "대표 클레임 소스",
  representativeReviewSentence: "대표 리뷰 문장",
  representativeReviewSignal: "대표 리뷰 신호",
  representativeReviewType: "대표 리뷰 유형",
  avgPrice: "평균 가격",
  avgDiscountPct: "평균 할인율",
  fallbackCount: "보완 분류 건수",
  status: "상태",
  source: "분류 소스",
  count: "건수",
  snapshotDate: "스냅샷 날짜",
  snapshotId: "스냅샷 번호",
  dataset: "데이터셋",
  platform: "플랫폼",
  schemaVersion: "스키마 버전",
  priceBand: "판매가 가격대",
  estimatedOriginalPriceBand: "추정 정가 가격대",
  materialValue: "소재",
  colorValue: "색상",
  name: "상품명",
  productId: "상품 번호",
  rank: "순위",
  rankVelocity: "순위 변화 속도",
  rankAcceleration: "순위 변화 가속도",
  standardScore: "기본 점수",
  rankEnergy: "순위 에너지",
  energyVelocity: "에너지 속도",
  energyAcceleration: "에너지 가속도",
  entryScore: "진입 강도",
  exitScore: "탈락 충격",
  actionScore: "액션 점수",
  observationCount: "관측 수",
  presenceRatio: "순위권 등장 비율",
  cumulativeRankEnergy: "누적 순위 에너지",
  avgRankEnergy: "평균 순위 에너지",
  bestRank: "최고 순위",
  bestRankEnergy: "최고 순위 에너지",
  sustainedRankEnergy: "지속 순위 에너지",
  totalSustainedEnergy: "총 지속 순위 에너지",
  momentumScore: "모멘텀 점수",
  persistence: "지속성",
  eventLabel: "움직임 상태",
  eventState: "움직임 상태",
  discountPct: "할인율",
  price: "가격",
  tpo: "TPO(상황)",
  avgSentimentScore: "평균 감성 점수",
  unmetCount: "미충족 문장 수",
  totalCount: "전체 문장 수",
  unmetRatio: "미충족 비율",
  sentence: "문장",
  smallCount: "작다 수",
  trueCount: "정사이즈 수",
  largeCount: "크다 수",
  dominantTendency: "대표 경향",
  confidenceLevel: "신뢰도",
  keyword: "키워드",
  keywordType: "유형",
  mentionCount: "언급 수",
  recentCount: "최근 언급",
  priorCount: "이전 언급",
  growthRate: "증가율",
  avgSentiment: "평균 감성",
  topColors: "주요 색상",
  topMaterials: "주요 소재",
  aspectStrengths: "강점 속성",
  intentStyleTop: "의도 이미지(상품 카피)",
  perceivedStyleTop: "지각 이미지(리뷰)",
  imageAlignment: "이미지 정렬도",
  claimStyleMass: "카피 스타일 신호량",
  reviewStyleMass: "리뷰 스타일 신호량",
  customerLedImageNote: "고객 인식 우세(지각−의도)",
  brandLedImageNote: "브랜드 카피 우세(의도−지각)",
  dominantTpo: "대표 TPO",
  sentenceCount: "문장 수",
  brand: "브랜드",
};

const VALUE_LABEL_MAP_BY_KEY: Record<string, Record<string, string>> = {
  sentiment: {
    positive: "긍정",
    neutral: "중립",
    negative: "부정",
    mixed: "복합",
    unknown: "미분류",
  },
  reviewType: {
    general: "일반 리뷰",
    style: "스타일 리뷰",
    monthly: "한달 리뷰",
    month: "한달 사용 리뷰",
    experience: "체험단 리뷰",
    goods: "굿즈/사은품 리뷰",
    beauty: "뷰티 리뷰",
    photo: "포토 리뷰",
    unknown: "미분류",
  },
  review_type: {
    general: "일반 리뷰",
    style: "스타일 리뷰",
    monthly: "한달 리뷰",
    month: "한달 사용 리뷰",
    experience: "체험단 리뷰",
    goods: "굿즈/사은품 리뷰",
    beauty: "뷰티 리뷰",
    photo: "포토 리뷰",
    unknown: "미분류",
  },
  sourceField: {
    name: "상품명",
    material: "소재",
    color: "색상",
    tags: "태그",
    ocr_text_joined: "OCR 텍스트",
    detail_info: "상세정보",
    detailInfo: "상세정보",
    product_name: "상품명",
    productName: "상품명",
    category: "카테고리",
    ocr: "OCR",
    ingredient: "전성분",
    unknown: "미분류",
  },
  pairingStatus: {
    paired: "쌍 매칭",
    rawOnly: "Raw 전용",
    normalizedOnly: "Normalized 전용",
  },
  valueStatus: {
    same: "동일",
    changed: "변환됨",
    rawOnly: "Raw 전용",
    normalizedOnly: "Normalized 전용",
  },
  aspect: {
    size_fit: "사이즈/핏",
    comfort: "착용감",
    design: "디자인",
    quality: "품질",
    price_value: "가격/가성비",
    delivery_service: "배송/서비스",
    scent_beauty: "향/뷰티",
    color_accuracy: "색상 정확도",
    thickness: "두께/비침",
    durability: "내구성",
    stretch: "신축성",
    general: "일반",
    unknown: "미분류",
  },
  tpo: {
    commute: "출근룩",
    wedding_guest: "하객룩",
    casual: "데일리",
    date: "데이트",
    travel: "여행",
    exercise: "운동",
    school: "학교/캠퍼스",
    season_summer: "여름",
    season_winter: "겨울",
    unspecified: "미지정",
  },
  dominantTendency: {
    small: "작게 나옴",
    true_to_size: "정사이즈",
    large: "크게 나옴",
    mixed: "혼재",
  },
  size_tendency: {
    small: "작게 나옴",
    true_to_size: "정사이즈",
    large: "크게 나옴",
    mixed: "혼재",
  },
  keywordType: {
    color: "색상",
    style: "스타일",
  },
  eventState: {
    first_seen: "첫 관측",
    chart_in_spike: "순위권 진입",
    chart_out_drop: "순위권 탈락",
    out_of_chart: "순위권 밖",
    breakout: "가속 상승",
    sustained_growth: "지속 상승",
    cooling: "상승 둔화",
    reversal: "하락 전환",
    steady: "정체",
  },
  momentumEventState: {
    first_seen: "첫 관측",
    chart_in_spike: "순위권 진입",
    chart_out_drop: "순위권 탈락",
    out_of_chart: "순위권 밖",
    breakout: "가속 상승",
    sustained_growth: "지속 상승",
    cooling: "상승 둔화",
    reversal: "하락 전환",
    steady: "정체",
  },
};

const decimalFormatter = new Intl.NumberFormat("ko-KR", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("ko-KR", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

function isNumeric(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function splitCamelCase(value: string) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .trim();
}

export function formatColumnLabel(key: string): string {
  const mapped = COLUMN_LABEL_MAP[key];
  if (mapped) {
    return mapped;
  }
  const normalized = splitCamelCase(key);
  return normalized || key;
}

export function formatDimensionValue(value: unknown, key?: string): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const normalizedKey = (key ?? "").trim();
  if (normalizedKey) {
    let mappedByKey = VALUE_LABEL_MAP_BY_KEY[normalizedKey];
    if (!mappedByKey && normalizedKey === "dominantTpo") {
      mappedByKey = VALUE_LABEL_MAP_BY_KEY.tpo;
    }
    if (mappedByKey) {
      const mapped = mappedByKey[String(value)];
      if (mapped) {
        return mapped;
      }
    }
  }
  return String(value);
}

export function inferFormatKind(key?: string, value?: unknown): FormatKind {
  const normalizedKey = (key ?? "").toLowerCase();
  if (!normalizedKey && typeof value === "string") {
    return "string";
  }
  if (normalizedKey.includes("pct") || normalizedKey.includes("percent") || normalizedKey.includes("rate")) {
    return "percent";
  }
  if (normalizedKey.includes("price") || normalizedKey.includes("krw")) {
    return "price";
  }
  if (normalizedKey.endsWith("id")) {
    return "string";
  }
  if (
    normalizedKey.includes("count")
    || normalizedKey.includes("rank")
    || normalizedKey.includes("snapshot")
  ) {
    return "integer";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? "integer" : "number";
  }
  return "auto";
}

export function formatNumber(value: unknown, kind: FormatKind = "auto", key?: string): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "string" && kind === "string") {
    return value;
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  const normalizedKind = kind === "auto" ? inferFormatKind(key, value) : kind;
  if (normalizedKind === "string") {
    return String(value);
  }
  const numericValue = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numericValue)) {
    return String(value);
  }
  if (normalizedKind === "percent") {
    return `${decimalFormatter.format(numericValue)}%`;
  }
  if (normalizedKind === "price" || normalizedKind === "integer") {
    return integerFormatter.format(numericValue);
  }
  return decimalFormatter.format(numericValue);
}

/** [{ value, count }, …] → "값1 ×12 · 값2 ×3" */
function formatValueCountPairs(items: unknown[]): string | null {
  if (!items.length) {
    return null;
  }
  const parts: string[] = [];
  for (const item of items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return null;
    }
    const o = item as Record<string, unknown>;
    if (!("value" in o) || !("count" in o)) {
      return null;
    }
    const v = o.value;
    const c = o.count;
    const countNum = typeof c === "number" ? c : Number(c);
    if (!Number.isFinite(countNum)) {
      return null;
    }
    const label = v === null || v === undefined ? "—" : String(v).trim() || "—";
    parts.push(`${label} ×${integerFormatter.format(countNum)}`);
  }
  return parts.join(" · ");
}

/** [{ styleLabel, sharePct }, …] → "라벨 12.3% · …" */
function formatStyleTopPairs(items: unknown[]): string | null {
  if (!items.length) {
    return null;
  }
  const parts: string[] = [];
  for (const item of items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return null;
    }
    const o = item as Record<string, unknown>;
    if (!("styleLabel" in o) || !("sharePct" in o)) {
      return null;
    }
    const label = String(o.styleLabel ?? "").trim() || "—";
    const p = o.sharePct;
    const num = typeof p === "number" ? p : Number(p);
    const pctStr = Number.isFinite(num) ? `${decimalFormatter.format(num)}%` : String(p ?? "-");
    parts.push(`${label} ${pctStr}`);
  }
  return parts.join(" · ");
}

/** [{ aspect, score }, …] → 한글 속성명과 점수 */
function formatAspectScorePairs(items: unknown[]): string | null {
  if (!items.length) {
    return null;
  }
  const parts: string[] = [];
  for (const item of items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return null;
    }
    const o = item as Record<string, unknown>;
    if (!("aspect" in o) || !("score" in o)) {
      return null;
    }
    const label = formatDimensionValue(o.aspect, "aspect");
    const s = o.score;
    const num = typeof s === "number" ? s : Number(s);
    const scoreStr = Number.isFinite(num) ? decimalFormatter.format(num) : String(s ?? "-");
    parts.push(`${label} ${scoreStr}`);
  }
  return parts.join(" · ");
}

/** 단일 레코드 객체를 한 줄 요약 (테이블 셀용) */
function formatPlainObjectCell(obj: Record<string, unknown>): string {
  const entries = Object.entries(obj).filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (!entries.length) {
    return "-";
  }
  return entries
    .map(([k, v]) => {
      const label = formatColumnLabel(k);
      if (typeof v === "number") {
        return `${label} ${decimalFormatter.format(v)}`;
      }
      if (typeof v === "string" || typeof v === "boolean") {
        return `${label} ${formatDimensionValue(v, k)}`;
      }
      if (Array.isArray(v)) {
        const inner = formatValueCountPairs(v) ?? formatAspectScorePairs(v);
        if (inner !== null) {
          return `${label}: ${inner}`;
        }
      }
      return `${label} ${JSON.stringify(v)}`;
    })
    .join(" · ");
}

export function formatCellValue(value: unknown, key?: string): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "-";
    }
    const k = key ?? "";
    if (k === "topColors" || k === "topMaterials") {
      const line = formatValueCountPairs(value);
      if (line !== null) {
        return line;
      }
    }
    if (k === "intentStyleTop" || k === "perceivedStyleTop") {
      const line = formatStyleTopPairs(value);
      if (line !== null) {
        return line;
      }
    }
    if (k === "aspectStrengths") {
      const line = formatAspectScorePairs(value);
      if (line !== null) {
        return line;
      }
    }
    const asValueCount = formatValueCountPairs(value);
    if (asValueCount !== null) {
      return asValueCount;
    }
    const asAspect = formatAspectScorePairs(value);
    if (asAspect !== null) {
      return asAspect;
    }
    return value.map((row) => (typeof row === "object" && row && !Array.isArray(row)
      ? formatPlainObjectCell(row as Record<string, unknown>)
      : String(row))).join(" │ ");
  }
  if (typeof value === "object") {
    return formatPlainObjectCell(value as Record<string, unknown>);
  }
  if (isNumeric(value)) {
    return formatNumber(value, "auto", key);
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && value.trim() !== "") {
      return formatNumber(parsed, "auto", key);
    }
  }
  if (typeof value === "string") {
    return formatDimensionValue(value, key);
  }
  return String(value);
}
