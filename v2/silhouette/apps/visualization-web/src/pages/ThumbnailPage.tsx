import { useEffect, useMemo, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import { PageContainer } from "../components/PageContainer";
import { apiGet, productImageApiUrl } from "../lib/api";
import { useDashboardFilters } from "../lib/filters";
import type { ThumbnailRecordsResponse, ThumbnailRecordRow, ThumbnailSnapshotSummary, ThumbnailSnapshotsResponse } from "../types";

const numberFormatter = new Intl.NumberFormat("ko-KR");
const currencyFormatter = new Intl.NumberFormat("ko-KR");
const INITIAL_VISIBLE_RECORDS = 24;
const VISIBLE_RECORDS_STEP = 24;

function formatNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return numberFormatter.format(value);
}

function formatPrice(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${currencyFormatter.format(Math.round(value))}원`;
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(value % 1 === 0 ? 0 : 1)}%`;
}

function formatFieldValue(key: string, value: unknown) {
  if (key === "price") {
    return typeof value === "number" ? formatPrice(value) : "-";
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(" / ") : "-";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  return String(value ?? "-");
}

function getSnapshotMonth(snapshot: ThumbnailSnapshotSummary) {
  return snapshot.snapshotDate?.slice(0, 7) ?? snapshot.label.slice(0, 7);
}

function getSnapshotDay(snapshot: ThumbnailSnapshotSummary) {
  return snapshot.snapshotDate ?? snapshot.label.slice(0, 10);
}

function getSnapshotTime(snapshot: ThumbnailSnapshotSummary) {
  return snapshot.snapshotTime ?? snapshot.label.slice(11, 16);
}

function uniq(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
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

type SelectionState = {
  selectedMonth: string | null;
  selectedDay: string | null;
  selectedSnapshotId: string | null;
  monthOptions: string[];
  dayOptions: string[];
  snapshotOptions: ThumbnailSnapshotSummary[];
};

function resolveSelection(
  snapshots: ThumbnailSnapshotSummary[],
  monthParam: string | null,
  dayParam: string | null,
  snapshotParam: string | null,
  fallbackSnapshotId?: string | null,
): SelectionState {
  const monthOptions = uniq(snapshots.map(getSnapshotMonth));
  const explicitSnapshot = snapshotParam ? snapshots.find((snapshot) => snapshot.snapshotId === snapshotParam) ?? null : null;
  let fallbackSnapshot: ThumbnailSnapshotSummary | null = explicitSnapshot;
  if (!fallbackSnapshot && fallbackSnapshotId) {
    fallbackSnapshot = snapshots.find((snapshot) => snapshot.snapshotId === fallbackSnapshotId) ?? null;
  }
  if (!fallbackSnapshot && snapshots.length > 0) {
    fallbackSnapshot = snapshots[0];
  }

  const selectedMonth = explicitSnapshot
    ? getSnapshotMonth(explicitSnapshot)
    : monthParam && monthOptions.includes(monthParam)
      ? monthParam
      : fallbackSnapshot
        ? getSnapshotMonth(fallbackSnapshot)
        : null;
  const monthScoped = selectedMonth ? snapshots.filter((snapshot) => getSnapshotMonth(snapshot) === selectedMonth) : [];
  const dayOptions = uniq(monthScoped.map(getSnapshotDay));
  const selectedDay = explicitSnapshot
    ? getSnapshotDay(explicitSnapshot)
    : dayParam && dayOptions.includes(dayParam)
      ? dayParam
      : monthScoped[0]
        ? getSnapshotDay(monthScoped[0])
        : null;
  const snapshotOptions = monthScoped.filter((snapshot) => !selectedDay || getSnapshotDay(snapshot) === selectedDay);
  const selectedSnapshotId =
    explicitSnapshot && snapshotOptions.some((snapshot) => snapshot.snapshotId === explicitSnapshot.snapshotId)
      ? explicitSnapshot.snapshotId
      : snapshotOptions[0]?.snapshotId ?? null;
  return { selectedMonth, selectedDay, selectedSnapshotId, monthOptions, dayOptions, snapshotOptions };
}

function SelectionCard({
  title,
  selection,
  onMonthChange,
  onDayChange,
  onSnapshotChange,
}: {
  title: string;
  selection: SelectionState;
  onMonthChange: (value: string) => void;
  onDayChange: (value: string) => void;
  onSnapshotChange: (value: string) => void;
}) {
  const currentLabel =
    selection.selectedSnapshotId
      ? selection.snapshotOptions.find((snapshot) => snapshot.snapshotId === selection.selectedSnapshotId)?.label ?? "시간 선택"
      : "시간 선택";

  return (
    <section className="thumbnail-selector-card">
      <div className="thumbnail-selector-card__header">
        <div>
          <small>{title}</small>
          <h3>{currentLabel}</h3>
        </div>
      </div>
      <div className="thumbnail-selector-card__controls">
        <label>
          월
          <select value={selection.selectedMonth ?? ""} onChange={(event) => onMonthChange(event.target.value)}>
            {selection.monthOptions.map((month) => (
              <option key={month} value={month}>
                {month}
              </option>
            ))}
          </select>
        </label>
        <label>
          일
          <select value={selection.selectedDay ?? ""} onChange={(event) => onDayChange(event.target.value)} disabled={!selection.dayOptions.length}>
            {selection.dayOptions.map((day) => (
              <option key={day} value={day}>
                {day}
              </option>
            ))}
          </select>
        </label>
        <label>
          시간
          <select
            value={selection.selectedSnapshotId ?? ""}
            onChange={(event) => onSnapshotChange(event.target.value)}
            disabled={!selection.snapshotOptions.length}
          >
            {selection.snapshotOptions.map((snapshot) => (
              <option key={snapshot.snapshotId} value={snapshot.snapshotId}>
                {getSnapshotTime(snapshot)} · {formatNumber(snapshot.recordCount)}행
              </option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}

function SnapshotMetrics({ summary }: { summary: ThumbnailRecordsResponse["snapshotSummary"] }) {
  if (!summary) {
    return null;
  }
  const cards = [
    { label: "스냅샷", value: `${formatNumber(summary.snapshotCount ?? 1)}회` },
    { label: "관측 행", value: `${formatNumber(summary.recordCount)}행` },
    { label: "고유 상품", value: `${formatNumber(summary.productCount)}개` },
    { label: "고유 브랜드", value: `${formatNumber(summary.brandCount)}개` },
    { label: "대표 이미지", value: formatPercent(summary.mainImageCoveragePct) },
    { label: "상세 스펙", value: formatPercent(summary.detailInfoCoveragePct) },
    { label: "카테고리", value: formatPercent(summary.categoryCoveragePct) },
    { label: "평균 가격", value: formatPrice(summary.avgPrice) },
  ];
  return (
    <div className="thumbnail-metric-grid">
      {cards.map((card) => (
        <article key={card.label} className="thumbnail-metric-card">
          <span>{card.label}</span>
          <strong>{card.value}</strong>
        </article>
      ))}
    </div>
  );
}

function LoadingOverlay({ visible, message }: { visible: boolean; message: string }) {
  if (!visible) {
    return null;
  }
  return (
    <div className="thumbnail-loading-overlay" aria-live="polite" aria-busy="true">
      <div className="thumbnail-loading-overlay__content">
        <div className="thumbnail-loading-overlay__spinner" />
        <strong>로딩 중</strong>
        <p>{message}</p>
      </div>
    </div>
  );
}

function ThumbnailPageRecordCard({
  row,
  active,
  onSelect,
  showSnapshotLabel,
}: {
  row: ThumbnailRecordRow;
  active: boolean;
  onSelect: () => void;
  showSnapshotLabel: boolean;
}) {
  const previewImages = row.images.slice(0, 3);
  return (
    <button type="button" className={`thumbnail-record ${active ? "is-active" : ""}`} onClick={onSelect}>
      <div className="thumbnail-record__media">
        {previewImages.length ? (
          <div className="thumbnail-image-stack">
            {previewImages.map((image, index) => (
              <div key={`${row.productId}-${image.path}-${index}`} className={`thumbnail-image-stack__item ${image.isMainImage ? "is-main" : ""}`}>
                <img src={productImageApiUrl(image.path)} alt={`${row.name} 썸네일 ${index + 1}`} />
                {image.isMainImage ? <span className="thumbnail-main-badge">대표</span> : null}
              </div>
            ))}
            {row.imageCount > previewImages.length ? <span className="thumbnail-image-stack__more">+{row.imageCount - previewImages.length}</span> : null}
          </div>
        ) : (
          <div className="thumbnail-image-stack thumbnail-image-stack--empty">이미지 없음</div>
        )}
      </div>
      <div className="thumbnail-record__body">
        <div className="thumbnail-record__header">
          <div>
            <div className="thumbnail-record__eyebrow">
              {showSnapshotLabel ? <span>{row.snapshotLabel}</span> : null}
              <span>#{row.rank ?? "-"}</span>
              <span>{row.brand ?? "브랜드 미입력"}</span>
            </div>
            <strong>{row.name}</strong>
          </div>
          <span className={`thumbnail-record__main-indicator ${row.hasExplicitMainImage ? "" : "is-missing"}`}>
            {row.hasExplicitMainImage ? "대표 이미지 지정" : "메인 이미지 없음"}
          </span>
        </div>
        <div className="thumbnail-record__fields">
          {row.fieldCells.map((cell) => (
            <div key={`${row.productId}-${cell.key}`} className={`thumbnail-record__field thumbnail-record__field--${cell.key}`}>
              <span className={`thumbnail-record__field-key thumbnail-record__field-key--${cell.key}`}>{cell.label}</span>
              <strong>{formatFieldValue(cell.key, cell.value)}</strong>
            </div>
          ))}
        </div>
        {row.detailInfoPreview.length ? (
          <div className="thumbnail-kv-preview">
            {row.detailInfoPreview.map((item) => (
              <span key={`${row.productId}-${item.key}`}>
                <strong>{item.label}</strong>
                {item.value}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      <div className="thumbnail-record__action">상세 보기</div>
    </button>
  );
}

function ImageLightbox({
  imagePath,
  imageAlt,
  onClose,
}: {
  imagePath: string | null;
  imageAlt: string;
  onClose: () => void;
}) {
  if (!imagePath) {
    return null;
  }
  return (
    <>
      <button type="button" className="thumbnail-lightbox-backdrop" aria-label="이미지 뷰어 닫기" onClick={onClose} />
      <div className="thumbnail-lightbox" role="dialog" aria-modal="true" aria-label="확대 이미지">
        <button type="button" className="thumbnail-lightbox__close" onClick={onClose}>
          닫기
        </button>
        <img src={productImageApiUrl(imagePath)} alt={imageAlt} />
      </div>
    </>
  );
}

function RecordDetailDrawer({
  row,
  onClose,
  onOpenImage,
}: {
  row: ThumbnailRecordRow | null;
  onClose: () => void;
  onOpenImage: (imagePath: string, imageAlt: string) => void;
}) {
  if (!row) {
    return null;
  }
  return (
    <>
      <button type="button" className="thumbnail-drawer-backdrop" aria-label="상세 닫기" onClick={onClose} />
      <aside className="thumbnail-drawer">
        <div className="thumbnail-drawer__header">
          <div>
            <small>{row.snapshotLabel}</small>
            <h2>{row.name}</h2>
            <p>
              {row.brand ?? "브랜드 미입력"} · 상품 ID `{row.productId}`
            </p>
          </div>
          <button type="button" className="thumbnail-drawer__close" onClick={onClose}>
            닫기
          </button>
        </div>
        <div className="thumbnail-drawer__content">
          <section className="thumbnail-drawer__section">
            <h3>이미지</h3>
            {row.images.length ? (
              <div className="thumbnail-drawer__gallery">
                {row.images.map((image, index) => (
                  <button
                    type="button"
                    key={`${row.productId}-${image.path}-${index}`}
                    className={`thumbnail-drawer__gallery-item ${image.isMainImage ? "is-main" : ""}`}
                    onClick={() => onOpenImage(image.path, `${row.name} 이미지 ${index + 1}`)}
                  >
                    <img src={productImageApiUrl(image.path)} alt={`${row.name} 이미지 ${index + 1}`} />
                    {image.isMainImage ? <span className="thumbnail-main-badge">대표 이미지</span> : null}
                    <span className="thumbnail-drawer__gallery-zoom">확대</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty-state">이 행에는 연결된 이미지가 없습니다.</div>
            )}
          </section>
          <section className="thumbnail-drawer__section">
            <h3>요약 정보</h3>
            <dl className="thumbnail-drawer__facts">
              <div><dt>스냅샷</dt><dd>{row.snapshotLabel}</dd></div>
              <div><dt>순위</dt><dd>{row.rank ?? "-"}</dd></div>
              <div><dt>가격</dt><dd>{formatPrice(row.price)}</dd></div>
              <div><dt>할인율</dt><dd>{formatPercent(row.discountPct)}</dd></div>
              <div><dt>카테고리 경로</dt><dd>{row.categoryPath.length ? row.categoryPath.join(" > ") : "-"}</dd></div>
              <div><dt>이미지 상태</dt><dd>{row.hasExplicitMainImage ? "명시적 대표 이미지 수집" : "메인 이미지 없음"}</dd></div>
              <div><dt>이미지 출처</dt><dd>{row.mainImageSource ?? "-"}</dd></div>
              <div><dt>카테고리 소스</dt><dd>{row.categorySource ?? "-"}</dd></div>
              <div><dt>카테고리 상태</dt><dd>{row.categoryStatus ?? "-"}</dd></div>
              <div><dt>원천 데이터셋</dt><dd>{row.sourceDataset ?? "-"}</dd></div>
              <div><dt>플랫폼</dt><dd>{row.platform ?? "-"}</dd></div>
              <div><dt>스키마 버전</dt><dd>{row.schemaVersion ?? "-"}</dd></div>
            </dl>
            {row.productUrl ? <a className="thumbnail-open-link" href={row.productUrl} target="_blank" rel="noreferrer">원본 상품 페이지 열기</a> : null}
          </section>
          <section className="thumbnail-drawer__section">
            <h3>상품 상세 스펙</h3>
            {row.detailInfoRows.length ? (
              <div className="thumbnail-detail-list">
                {row.detailInfoRows.map((item) => (
                  <div key={`${row.productId}-${item.key}`} className="thumbnail-detail-list__item">
                    <div><strong>{item.label}</strong></div>
                    <p>{item.value}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">이 행에는 펼쳐 볼 상품 상세 스펙이 없습니다.</div>
            )}
          </section>
        </div>
      </aside>
    </>
  );
}

export function ThumbnailPage() {
  const { filters } = useDashboardFilters();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedRecord, setSelectedRecord] = useState<ThumbnailRecordRow | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_RECORDS);
  const [lightboxImagePath, setLightboxImagePath] = useState<string | null>(null);
  const [lightboxImageAlt, setLightboxImageAlt] = useState("");

  const mode = searchParams.get("thumbnailMode") === "range" ? "range" : "point";
  const pointMonthParam = searchParams.get("thumbnailMonth");
  const pointDayParam = searchParams.get("thumbnailDay");
  const pointSnapshotParam = searchParams.get("thumbnailSnapshot");
  const startMonthParam = searchParams.get("thumbnailStartMonth");
  const startDayParam = searchParams.get("thumbnailStartDay");
  const startSnapshotParam = searchParams.get("thumbnailStartSnapshot");
  const endMonthParam = searchParams.get("thumbnailEndMonth");
  const endDayParam = searchParams.get("thumbnailEndDay");
  const endSnapshotParam = searchParams.get("thumbnailEndSnapshot");
  const thumbnailProductIdParam = (searchParams.get("thumbnailProductId") ?? "").trim();
  const thumbnailOpenParam = searchParams.get("thumbnailOpen") === "1";

  const snapshotsQuery = useQuery({
    queryKey: ["thumbnails-snapshots", filters],
    queryFn: () => apiGet<ThumbnailSnapshotsResponse>("/api/thumbnails/snapshots", filters),
  });

  const snapshots = snapshotsQuery.data?.snapshots ?? [];
  const defaultSnapshotId = snapshotsQuery.data?.defaultSnapshotId ?? snapshots[0]?.snapshotId ?? null;
  const pointSelection = useMemo(
    () => resolveSelection(snapshots, pointMonthParam, pointDayParam, pointSnapshotParam, defaultSnapshotId),
    [defaultSnapshotId, pointDayParam, pointMonthParam, pointSnapshotParam, snapshots],
  );
  const startSelection = useMemo(
    () => resolveSelection(snapshots, startMonthParam, startDayParam, startSnapshotParam, snapshots[snapshots.length - 1]?.snapshotId ?? defaultSnapshotId),
    [defaultSnapshotId, snapshots, startDayParam, startMonthParam, startSnapshotParam],
  );
  const endSelection = useMemo(
    () => resolveSelection(snapshots, endMonthParam, endDayParam, endSnapshotParam, defaultSnapshotId),
    [defaultSnapshotId, endDayParam, endMonthParam, endSnapshotParam, snapshots],
  );

  const effectivePointSnapshotId = pointSelection.selectedSnapshotId;
  const effectiveStartSnapshotId = startSelection.selectedSnapshotId;
  const effectiveEndSnapshotId = endSelection.selectedSnapshotId;

  const recordsQuery = useQuery({
    enabled: Boolean(mode === "point" ? effectivePointSnapshotId : effectiveStartSnapshotId && effectiveEndSnapshotId),
    queryKey: ["thumbnails-records", filters, mode, effectivePointSnapshotId, effectiveStartSnapshotId, effectiveEndSnapshotId],
    queryFn: () =>
      apiGet<ThumbnailRecordsResponse>("/api/thumbnails/records", filters, {
        snapshot_id: mode === "point" ? (effectivePointSnapshotId ?? undefined) : undefined,
        start_snapshot_id: mode === "range" ? (effectiveStartSnapshotId ?? undefined) : undefined,
        end_snapshot_id: mode === "range" ? (effectiveEndSnapshotId ?? undefined) : undefined,
      }),
  });

  const rows = recordsQuery.data?.rows ?? [];
  const visibleRows = useMemo(() => rows.slice(0, visibleCount), [rows, visibleCount]);
  const remainingRowCount = Math.max(rows.length - visibleRows.length, 0);
  const summary = recordsQuery.data?.snapshotSummary ?? null;
  const showSnapshotLabel = (recordsQuery.data?.selectedSnapshotIds?.length ?? 0) > 1;
  const isOutputLoading = recordsQuery.isFetching;

  const sourceDatasetList = filters.sourceDatasets.length ? filters.sourceDatasets : ["20260415_musinsa_34"];
  const targetLabels = Array.from(new Set(sourceDatasetList.map(parseSourceLabel)));
  const targetSummary = targetLabels.length > 1 ? `${targetLabels.length}개 채널` : targetLabels[0] ?? "선택 채널";

  useEffect(() => {
    setSelectedRecord(null);
    setIsDetailOpen(false);
    setVisibleCount(INITIAL_VISIBLE_RECORDS);
    setLightboxImagePath(null);
    setLightboxImageAlt("");
  }, [mode, effectivePointSnapshotId, effectiveStartSnapshotId, effectiveEndSnapshotId]);

  useEffect(() => {
    if (!rows.length) {
      setSelectedRecord(null);
      return;
    }
    setSelectedRecord((current) => rows.find((row) => row.productId === current?.productId && row.snapshotId === current?.snapshotId) ?? rows[0]);
  }, [rows]);

  useEffect(() => {
    if (!thumbnailProductIdParam || !rows.length) {
      return;
    }
    const matchedIndex = rows.findIndex((row) => row.productId === thumbnailProductIdParam);
    if (matchedIndex < 0) {
      return;
    }
    const matched = rows[matchedIndex];
    setSelectedRecord(matched);
    if (thumbnailOpenParam) {
      setIsDetailOpen(true);
    }
    if (matchedIndex + 1 > visibleCount) {
      const nextVisibleCount = Math.ceil((matchedIndex + 1) / VISIBLE_RECORDS_STEP) * VISIBLE_RECORDS_STEP;
      setVisibleCount(nextVisibleCount);
    }
  }, [rows, thumbnailOpenParam, thumbnailProductIdParam, visibleCount]);

  const updateParams = (mutate: (next: URLSearchParams) => void) => {
    const next = new URLSearchParams(searchParams);
    mutate(next);
    setSearchParams(next, { replace: true });
  };

  const setMode = (nextMode: "point" | "range") => {
    updateParams((next) => {
      next.set("thumbnailMode", nextMode);
    });
  };

  const bindSelection =
    (prefix: "thumbnail" | "thumbnailStart" | "thumbnailEnd", selection: SelectionState) =>
    ({
      onMonthChange: (value: string) =>
        updateParams((next) => {
          next.set(`${prefix}Month`, value);
          next.delete(`${prefix}Day`);
          next.delete(`${prefix}Snapshot`);
        }),
      onDayChange: (value: string) =>
        updateParams((next) => {
          next.set(`${prefix}Month`, selection.selectedMonth ?? "");
          next.set(`${prefix}Day`, value);
          next.delete(`${prefix}Snapshot`);
        }),
      onSnapshotChange: (value: string) =>
        updateParams((next) => {
          next.set(`${prefix}Month`, selection.selectedMonth ?? "");
          next.set(`${prefix}Day`, selection.selectedDay ?? "");
          next.set(`${prefix}Snapshot`, value);
        }),
    });

  if (snapshotsQuery.isLoading) {
    return <div className="loading-state">섬네일 스냅샷 목록을 불러오는 중입니다.</div>;
  }

  if (!snapshots.length) {
    return (
      <PageContainer
        title="섬네일"
        description="수집해 둔 스냅샷을 골라, 실제로 어떤 행이 쌓였는지 확인하는 화면입니다. 현재 필터에서는 볼 수 있는 스냅샷이 없습니다."
      >
        <div className="empty-state">현재 필터 기준으로 표시할 스냅샷이 없습니다.</div>
      </PageContainer>
    );
  }

  const pointBinding = bindSelection("thumbnail", pointSelection);
  const startBinding = bindSelection("thumbnailStart", startSelection);
  const endBinding = bindSelection("thumbnailEnd", endSelection);

  return (
    <PageContainer
      title="섬네일"
      description="수집해 둔 제품 상세 데이터가 화면에 어떻게 보이는지 확인합니다."
    >
      <section className="overview-story-hero">
        <small>PREVIEW</small>
        <h2>무엇을 가져와 담았나</h2>
        <p>
          {targetSummary} 제품 상세 화면을 <strong>크롤링</strong>으로 모으고, 이미지 속 텍스트는 <strong>OCR</strong>로 읽으며,{" "}
          <strong>카테고리 분류</strong>로 경로를 맞춥니다.
        </p>
      </section>

      <section className="overview-story-grid">
        <article className="overview-story-card">
          <span>원천</span>
          <strong>제품 상세 HTML</strong>
          <small>랭킹에 노출된 상품의 상세 페이지에서 필드와 이미지를 함께 가져옵니다.</small>
        </article>
        <article className="overview-story-card">
          <span>기록</span>
          <strong>스냅샷마다 상품 페이지 전체</strong>
          <small>시점이 바뀌면 같은 상품도 다시 읽어 변화를 남깁니다.</small>
        </article>
        <article className="overview-story-card overview-story-card--includes" aria-labelledby="thumbnail-includes-heading">
          <span>데이터 구성</span>
          <strong id="thumbnail-includes-heading">포함된 정보</strong>
          <ul className="overview-story-card__includes-list">
            <li>순위, 브랜드, 제품명</li>
            <li>가격, 할인율</li>
            <li>카테고리 경로, 태그</li>
            <li>제품 이미지·갤러리·대표 이미지</li>
            <li>고시 정보</li>
            <li>원본 상품 URL</li>
            <li>스냅샷·크롤 시각, 플랫폼·데이터셋·스키마 메타</li>
          </ul>
        </article>
      </section>

      <section className="thumbnail-topology-card">
        <div>
          <small>관측 범위 고르기</small>
          <h2>같은 상품도 시점마다 다시 읽습니다</h2>
          <p>
            <strong>포인트</strong>는 한 시점만, <strong>범위</strong>는 여러 스냅샷을 묶어 변화와 반복을 봅니다.
          </p>
        </div>
        <div className="thumbnail-topology-card__guide">
          <div>
            <strong>포인트</strong>
            <span>특정 시점에 읽어 온 목록을 그대로 봅니다.</span>
          </div>
          <div>
            <strong>범위</strong>
            <span>여러 시점을 묶어 변화와 반복을 봅니다.</span>
          </div>
          <div>
            <strong>관측 행</strong>
            <span>스냅샷과 상품 조합마다, 그때 읽어 온 HTML에서 뽑은 필드가 한 묶음으로 붙습니다.</span>
          </div>
        </div>
      </section>

      <section className="thumbnail-mode-toggle">
        <button type="button" className={mode === "point" ? "is-active" : ""} onClick={() => setMode("point")}>
          포인트 관측
        </button>
        <button type="button" className={mode === "range" ? "is-active" : ""} onClick={() => setMode("range")}>
          범위 관측
        </button>
      </section>

      {mode === "point" ? (
        <section className="thumbnail-selector-grid thumbnail-selector-grid--single">
          <SelectionCard title="볼 스냅샷" selection={pointSelection} {...pointBinding} />
        </section>
      ) : (
        <section className="thumbnail-selector-grid">
          <SelectionCard title="시작" selection={startSelection} {...startBinding} />
          <SelectionCard title="끝" selection={endSelection} {...endBinding} />
        </section>
      )}

      <section className={`thumbnail-output-shell ${isOutputLoading ? "is-loading" : ""}`}>
        <div className="thumbnail-output-shell__content">
          <section className="thumbnail-hero">
            <div className="thumbnail-hero__summary">
              <small>{mode === "point" ? "선택한 시점" : "선택한 구간"}</small>
              <h2>{summary?.label ?? "-"}</h2>
              <p>
                {mode === "point"
                  ? `이 시점 기준으로 관측 행 ${formatNumber(summary?.recordCount ?? 0)}행, 고유 상품 ${formatNumber(summary?.productCount ?? 0)}개가 묶여 있습니다.`
                  : `이 구간에는 스냅샷 ${formatNumber(summary?.snapshotCount ?? 0)}회분이 포함되고, 관측 행은 총 ${formatNumber(summary?.recordCount ?? 0)}행입니다.`}
              </p>
            </div>
            <SnapshotMetrics summary={summary} />
          </section>

          <section className="thumbnail-record-panel">
            <div className="thumbnail-record-section__header">
              <div>
                <h2>{mode === "point" ? "이 시점의 관측 행" : "이 구간의 관측 행"}</h2>
                <p>
                  {mode === "point"
                    ? "선택한 시점에 읽어 온 상품별 데이터가 카드로 정리됩니다. 각 카드는 그 시점의 관측 결과입니다."
                    : "선택한 기간에 읽어 온 제품 페이지들을 시간순으로 모아 둡니다. 스냅샷이 여러 개면 같은 상품이 여러 번 등장할 수 있습니다."}
                </p>
              </div>
              <strong>
                {recordsQuery.isLoading ? "불러오는 중" : `${formatNumber(visibleRows.length)} / ${formatNumber(rows.length)}행`}
              </strong>
            </div>

            {recordsQuery.isLoading && !rows.length ? (
              <div className="loading-state">선택한 범위의 관측 행을 불러오는 중입니다.</div>
            ) : rows.length ? (
              <>
                <div className="thumbnail-record-list">
                  {visibleRows.map((row) => (
                    <ThumbnailPageRecordCard
                      key={`${row.snapshotId}-${row.productId}`}
                      row={row}
                      active={selectedRecord?.snapshotId === row.snapshotId && selectedRecord?.productId === row.productId}
                      showSnapshotLabel={showSnapshotLabel}
                      onSelect={() => {
                        setSelectedRecord(row);
                        setIsDetailOpen(true);
                      }}
                    />
                  ))}
                </div>
                {remainingRowCount > 0 ? (
                  <div className="thumbnail-record-list__footer">
                    <span>아래에 관측 행이 {formatNumber(remainingRowCount)}행 더 있습니다.</span>
                    <button
                      type="button"
                      className="thumbnail-more-button"
                      onClick={() => setVisibleCount((current) => current + VISIBLE_RECORDS_STEP)}
                    >
                      더보기
                    </button>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="empty-state">선택한 범위에 표시할 관측 행이 없습니다.</div>
            )}
          </section>
        </div>
        <LoadingOverlay
          visible={isOutputLoading}
          message={mode === "point" ? "선택한 시점의 관측 행을 다시 불러오는 중입니다." : "선택한 구간의 관측 행과 요약을 다시 계산하는 중입니다."}
        />
      </section>

      <RecordDetailDrawer
        row={isDetailOpen ? selectedRecord : null}
        onClose={() => setIsDetailOpen(false)}
        onOpenImage={(imagePath, imageAlt) => {
          setLightboxImagePath(imagePath);
          setLightboxImageAlt(imageAlt);
        }}
      />
      <ImageLightbox imagePath={lightboxImagePath} imageAlt={lightboxImageAlt} onClose={() => setLightboxImagePath(null)} />
    </PageContainer>
  );
}
