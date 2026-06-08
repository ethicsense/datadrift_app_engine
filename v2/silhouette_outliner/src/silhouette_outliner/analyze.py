"""MVP v2 analysis for a single category (see docs/mvp-v2-plan.md).

The output dict shape is the public contract with the report template; field
names map 1:1 to the Jinja template.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta, timezone
from statistics import median
from typing import Any

from .bcave_portfolio import (
    BCAVE_BRAND_PANES,
    BCAVE_PORTFOLIO,
    STYLE_SECTION_OVERALL,
    BcaveBrandPane,
    BcaveBrandSpec,
    match_bcave_brand,
)

BRAND_PORTFOLIO_NAV_KEY = "__brand__"
CUSTOMER_SIGNALS_NAV_KEY = "__cs__"
BRAND_PORTFOLIO_REPORT_FILENAME = "report_brand.html"
CUSTOMER_SIGNALS_REPORT_FILENAME = "report_customer_signals.html"
MAIN_REPORT_FILENAME = "report.html"
from .config import (
    DEFAULT_DEMOGRAPHICS_WINDOW,
    PERIOD_AGE_BAND_CODE,
    PERIOD_GENDER_CODE,
)
from .customer_signals import fetch_goods_catalog, join_content_products
from .demographics import (
    DEFAULT_AGE_BANDS,
    DEFAULT_GENDER_FILTERS,
    age_band_index,
    gender_key,
    gender_label_for_code,
)
from .models import (
    CustomerSignalDataset,
    NormalizedDataset,
    RankingItem,
    RawCollection,
    normalize_category_code,
    utc_timestamp,
)

_SUB_PAN_BRANDS = frozenset({"brand", "brands"})

# Heat = viewers + buyers * BUYERS. A buy-intent signal weighs more than a passive view.
HEAT_WEIGHT_BUYERS = 2

# HHI thresholds borrowed from US antitrust convention.
HHI_DIFFUSE_MAX = 1500
HHI_CONCENTRATED_MIN = 2500

# Validation Score weights. Built from signals that the platform cannot easily
# inflate or moderate, in descending order of population coverage:
#   - cumulative review count (every product has one, decades of spread)
#   - real-time buyers ("N명이 구매 중") - cannot be faked, but sample is partial
#   - sales-volume labels ("판매 N만개" / "누적 판매 N만 돌파") - strongest signal
#     but only shown above a threshold, so we treat it as a bonus.
VALIDATION_WEIGHT_REVIEW = 0.5
VALIDATION_WEIGHT_BUYERS = 0.3
VALIDATION_WEIGHT_SALES_LABEL = 0.2

# Sales-label parsers. We grade in "log10(units)" space so a 100K label scores
# +2.0, a 10K label scores +1.0, etc. This keeps the bonus on the same scale
# as the other log-based weights.
_SALES_QUANTITY_RE = re.compile(r"판매\s*([\d,]+)(?:\.(\d+))?\s*만개")
_SALES_MILESTONE_RE = re.compile(r"누적\s*판매\s*([\d,]+)\s*만")
_BEST_HINT = ("BEST", "베스트")

PRICE_BUCKETS = [
    ("~3만", 0, 30_000),
    ("3~5만", 30_000, 50_000),
    ("5~10만", 50_000, 100_000),
    ("10~20만", 100_000, 200_000),
    ("20~50만", 200_000, 500_000),
    ("50만~", 500_000, None),
]

_REALTIME_WINDOW_IDS = frozenset({"rt", "realtime"})

RANK_BANDS = [(i, i + 9) for i in range(1, 100, 10)]  # (1,10), (11,20), ..., (91,100)

KST = timezone(timedelta(hours=9))

# Scatter plot canvas (logical units; CSS scales).
SCATTER_W = 720
SCATTER_H = 480
SCATTER_PAD = {"top": 24, "right": 24, "bottom": 32, "left": 40}
SCATTER_DEFAULT_PRESET = "top20"
SCATTER_OUTLIER_N = 12

DOT_STRIP_W = 720
DOT_STRIP_H = 60


def _collections_for_window(
    collections: list[RawCollection],
    window_id: str,
) -> tuple[list[RawCollection], list[RawCollection]]:
    successful = [c for c in collections if c.ok and c.target.ranking_window.id == window_id]
    failed = [c for c in collections if not c.ok and c.target.ranking_window.id == window_id]
    return successful, failed


def _top10_card(item: RankingItem, validation: float) -> dict[str, Any]:
    return {
        "rank": item.rank,
        "brand": item.brand or "-",
        "product": item.product_clean or item.product or "-",
        "price": item.price,
        "discount_rate": item.discount_rate,
        "url": item.product_url,
        "image_url": item.image_url,
        "is_sold_out": item.is_sold_out,
        "validation": round(validation, 2),
        "review_count": item.review_count,
        "buyers_now": item.buyers_now,
    }


def _item_has_discount(item: RankingItem) -> bool:
    return item.discount_rate is not None and item.discount_rate > 0


def _top10_for_window(
    items: list[RankingItem],
    validation_scores: dict[int, float],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (item for item in items if item.rank is not None),
        key=lambda item: item.rank or 9999,
    )
    return [
        _top10_card(item, validation_scores.get(id(item), 0.0))
        for item in ranked[:top_n]
    ]


def _analyze_slice(
    items: list[RankingItem],
    successful: list[RawCollection],
    failed: list[RawCollection],
) -> dict[str, Any]:
    prices = [item.price for item in items if item.price is not None]
    discounted = [
        item.discount_rate
        for item in items
        if _item_has_discount(item) and item.discount_rate is not None
    ]
    sold_out_count = sum(1 for item in items if item.is_sold_out)
    validation_scores = {id(item): _validation_score(item) for item in items}
    return {
        "kpis": _kpis(items, prices, discounted, sold_out_count),
        "heat_top": _heat_top(items),
        "scatter": _ranking_validation_scatter(items, validation_scores),
        "brand_concentration": _brand_concentration(items),
        "outliers": _outlier_lists(items, validation_scores),
        "price_dot_strip": _price_dot_strip(items),
        "quality": _quality_oneliner(items, successful, failed),
        "top10": _top10_for_window(items, validation_scores, top_n=10),
    }


def _aggregate_headline(
    items: list[RankingItem],
    primary_window_id: str,
    collections: list[RawCollection],
) -> dict[str, Any]:
    if not items:
        now = utc_timestamp()
        fallback_collection = next(
            (
                collection
                for collection in collections
                if collection.target.ranking_window.id == primary_window_id
            ),
            collections[0] if collections else None,
        )
        fallback_category = fallback_collection.target.category.label if fallback_collection else "데이터 없음"
        fallback_label = fallback_collection.target.ranking_window.label if fallback_collection else ""
        fallback_limit = fallback_collection.target.limit if fallback_collection else 0
        return {
            "category": fallback_category,
            "collected_at_utc": now,
            "collected_at_kst_pretty": _format_kst(now),
            "item_count": 0,
            "total_item_count": 0,
            "limit_target": fallback_limit,
            "primary_window_id": primary_window_id,
            "primary_window_label": fallback_label,
        }

    primary_items = [item for item in items if item.ranking_window_id == primary_window_id]
    effective_primary_id = primary_window_id
    if not primary_items:
        primary_items = items
        effective_primary_id = items[0].ranking_window_id

    collected_candidates = [item.collected_at for item in items if item.collected_at]
    collected_at = max(collected_candidates) if collected_candidates else utc_timestamp()

    limit_target = sum(
        collection.target.limit
        for collection in collections
        if collection.target.ranking_window.id == effective_primary_id
    ) or len(primary_items)

    return {
        "category": primary_items[0].category_label,
        "collected_at_utc": collected_at,
        "collected_at_kst_pretty": _format_kst(collected_at),
        "item_count": len(primary_items),
        "total_item_count": len(items),
        "limit_target": limit_target,
        "primary_window_id": effective_primary_id,
        "primary_window_label": primary_items[0].ranking_window_label,
    }


def _product_join_key(item: RankingItem) -> str | None:
    if item.product_id:
        return f"id:{item.product_id}"
    if item.product_url:
        return f"url:{item.product_url}"
    brand = (item.brand or "").strip()
    name = (item.product_clean or item.product or "").strip()
    if brand and name:
        return f"name:{brand}|{name}"
    return None


def _limit_for_window(collections: list[RawCollection], window_id: str) -> int:
    # Use the largest single-collection limit so duplicated (gender × age) targets
    # do not inflate the y-axis. Multi-section configs that intentionally union
    # different limits still get the most permissive ceiling.
    limits = [
        c.target.limit
        for c in collections
        if c.target.ranking_window.id == window_id and c.target.limit > 0
    ]
    return max(limits) if limits else 100


def _cross_row(
    data: dict[str, Any],
    rank_a: int,
    rank_b: int,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    delta = rank_b - rank_a
    return {
        "brand": data.get("brand") or "-",
        "product": data.get("product") or "-",
        "url": data.get("url"),
        "rank_a": rank_a,
        "rank_b": rank_b,
        "label_a": label_a,
        "label_b": label_b,
        "delta": delta,
        "delta_label": f"{label_a}→{label_b} 순위차 {delta:+d} (양수=숫자가 커짐=하락)",
    }


# Rank-line chart canvas (logical units; CSS scales).
RANK_CHART_W = 720
RANK_CHART_H = 480
RANK_CHART_PAD = {"top": 24, "right": 24, "bottom": 36, "left": 44}

# How many top products by sustained energy to plot as lines.
RANK_CHART_TOP_N = 30
RANK_CHART_DEFAULT_PRESET = "top12"
RANK_CHART_TOP12_N = 12
RANK_CHART_MOVERS_N = 8
RANK_CHART_Y_PAD = 5

# Insight preview rows per card; full lists are rendered separately in the report.
INSIGHT_PREVIEW_COUNT = 3

# Top / bottom N by momentum_span (rank_energy newest − oldest) for the report table.
MOMENTUM_EXTREMA_COUNT = 5

# Momentum distribution bin edges. Values are momentum_span = rank_energy(newest) - rank_energy(oldest).
MOMENTUM_BIN_EDGES = [-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0]


def _momentum_bin_index(value: float, edges: list[float] | None = None) -> int | None:
    """Return bin index for *value*, or None if outside edges."""

    edges = edges or MOMENTUM_BIN_EDGES
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if (value >= lo and value < hi) or (i == n_bins - 1 and value <= hi):
            return i
    return None


def _momentum_bin_label(lo: float, hi: float, is_last: bool) -> str:
    bracket = "]" if is_last else ")"
    return f"[{lo:+.2f}, {hi:+.2f}{bracket}"


def _momentum_bin_subtitle(
    lo: float,
    hi: float,
    is_last: bool,
    chart_order: list[str],
    labels: dict[str, str],
) -> str:
    if len(chart_order) < 2:
        bracket = "]" if is_last else ")"
        return f"momentum_span이 [{lo:+.2f}, {hi:+.2f}{bracket} 구간에 해당합니다."
    oldest = chart_order[0]
    newest = chart_order[-1]
    oldest_l = labels.get(oldest, oldest)
    newest_l = labels.get(newest, newest)
    bracket = "]" if is_last else ")"
    return (
        f"momentum_span = rank_energy({newest_l}) − rank_energy({oldest_l})가 "
        f"[{lo:+.2f}, {hi:+.2f}{bracket} 구간에 해당합니다."
    )


def _compute_momentum_span(
    chart_order: list[str],
    rank_energy: dict[str, float],
) -> float | None:
    """Endpoint span only: rank_energy(newest) − rank_energy(oldest) when both exist."""

    if not rank_energy or len(chart_order) < 2:
        return None
    first_w, last_w = chart_order[0], chart_order[-1]
    e_first = rank_energy.get(first_w)
    e_last = rank_energy.get(last_w)
    if e_first is not None and e_last is not None:
        return round(e_last - e_first, 4)
    return None


# Event clusters for products without a comparable endpoint span.
EVENT_CLUSTER_SPECS: list[tuple[str, str]] = [
    ("entry_latest_only", "신규 진입"),
    ("exit_oldest_only", "이탈"),
    ("middle_only", "중간 단독"),
    ("transient_new", "신규 쪽 부분"),
    ("transient_old", "과거 쪽 부분"),
    ("partial_observed", "부분 관측"),
]

# Primary event clusters (shown first in the report UI).
EVENT_CLUSTER_PRIMARY = frozenset(
    {"entry_latest_only", "exit_oldest_only", "middle_only"},
)


def _classify_event_cluster(
    chart_order: list[str],
    rank_energy: dict[str, float],
    momentum_span: float | None,
) -> str | None:
    """Classify observation-pattern events when endpoint span is unavailable."""

    if momentum_span is not None:
        return "none"
    if not rank_energy or len(chart_order) < 2:
        return None

    first_w, last_w = chart_order[0], chart_order[-1]
    present = [wid for wid in chart_order if wid in rank_energy]
    present_set = set(present)
    if not present:
        return None

    if present_set == {last_w}:
        return "entry_latest_only"
    if present_set == {first_w}:
        return "exit_oldest_only"
    if len(present) == 1 and present[0] not in (first_w, last_w):
        return "middle_only"
    if first_w not in present_set and last_w in present_set:
        return "transient_new"
    if last_w not in present_set and first_w in present_set:
        return "transient_old"
    if len(present) >= 2:
        return "partial_observed"
    return None


def _compute_event_strength(
    chart_order: list[str],
    rank_energy: dict[str, float],
    event_cluster: str | None,
) -> float | None:
    """Internal sort/display strength within an event cluster (not momentum_span)."""

    if not event_cluster or event_cluster in ("none",) or not rank_energy:
        return None

    first_w, last_w = chart_order[0], chart_order[-1]
    present = [wid for wid in chart_order if wid in rank_energy]
    if not present:
        return None

    if event_cluster == "entry_latest_only":
        return round(rank_energy[last_w], 4)
    if event_cluster == "exit_oldest_only":
        return round(rank_energy[first_w], 4)
    if event_cluster == "middle_only":
        return round(rank_energy[present[0]], 4)
    return round(max(rank_energy[w] for w in present), 4)


def _event_cluster_subtitle(
    cluster_id: str,
    chart_order: list[str],
    labels: dict[str, str],
) -> str:
    if not chart_order:
        return "momentum_span이 없는 관측 패턴입니다."
    oldest = chart_order[0]
    newest = chart_order[-1]
    oldest_l = labels.get(oldest, oldest)
    newest_l = labels.get(newest, newest)
    if cluster_id == "entry_latest_only":
        return f"{newest_l}에만 순위가 있습니다. 스팬 분포와 직접 비교하지 마세요."
    if cluster_id == "exit_oldest_only":
        return f"{oldest_l}에만 순위가 있습니다. 스팬 분포와 직접 비교하지 마세요."
    if cluster_id == "middle_only" and len(chart_order) >= 3:
        mid = chart_order[len(chart_order) // 2]
        return f"{labels.get(mid, mid)}에만 순위가 있습니다. 스팬 분포와 직접 비교하지 마세요."
    if cluster_id == "transient_new":
        return f"{oldest_l} 없이 {newest_l} 쪽에만 관측됩니다."
    if cluster_id == "transient_old":
        return f"{newest_l} 없이 {oldest_l} 쪽에만 관측됩니다."
    if cluster_id == "partial_observed":
        return "양 끝이 아닌 구간에서만 순위가 비교 가능합니다."
    return "momentum_span이 없는 관측 패턴입니다."


def _serialize_event_insight_row(
    product: dict[str, Any],
    chart_order: list[str],
    labels: dict[str, str],
) -> dict[str, Any]:
    row = _serialize_insight_row(product, chart_order, labels)
    row["event_strength"] = product.get("event_strength")
    return row


def _event_distribution(
    products: list[dict[str, Any]],
    chart_order: list[str],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Aggregate join products without endpoint span into event clusters."""

    by_key = {product["key"]: product for product in products}
    keys_by_cluster: dict[str, list[str]] = {cid: [] for cid, _ in EVENT_CLUSTER_SPECS}
    for product in products:
        cluster = product.get("event_cluster")
        if cluster and cluster != "none":
            keys_by_cluster.setdefault(cluster, []).append(product["key"])

    max_count = max((len(keys_by_cluster.get(cluster_id, [])) for cluster_id, _ in EVENT_CLUSTER_SPECS), default=0)
    cells: list[dict[str, Any]] = []
    for cluster_id, label in EVENT_CLUSTER_SPECS:
        keys = keys_by_cluster.get(cluster_id, [])
        bucket = [by_key[key] for key in keys if key in by_key]
        bucket.sort(key=lambda p: -(p.get("event_strength") or 0.0))
        rows = [_serialize_event_insight_row(p, chart_order, labels) for p in bucket]
        count = len(rows)
        cells.append(
            {
                "id": cluster_id,
                "label": label,
                "is_primary": cluster_id in EVENT_CLUSTER_PRIMARY,
                "subtitle": _event_cluster_subtitle(cluster_id, chart_order, labels),
                "count": count,
                "product_keys": keys,
                "picks": rows[:INSIGHT_PREVIEW_COUNT],
                "rows": rows,
                "pct": round(count / max_count * 100, 1) if max_count else 0,
            }
        )
    total = sum(c["count"] for c in cells)
    join_size = len(products)
    span_count = sum(1 for p in products if p.get("momentum_span") is not None)
    return {
        "has_data": bool(total),
        "total": total,
        "join_size": join_size,
        "span_count": span_count,
        "cells": cells,
    }


# Report / analysis.meta: 일간 → 주간 → 월간 (알파벳 정렬은 1d, 1m, 1w가 되어 깨짐).
_REPORT_WINDOW_RANK = {"1d": 0, "1w": 1, "1m": 2}


def _period_baseline_items(items: list[RankingItem]) -> list[RankingItem]:
    """Items from the period track (전체 성별 · 전체 연령)."""
    return [
        item
        for item in items
        if item.gender_filter == PERIOD_GENDER_CODE and item.age_band == PERIOD_AGE_BAND_CODE
    ]


def _period_momentum_items(items: list[RankingItem]) -> list[RankingItem]:
    """일/주/월 모멘텀용. 연령별 실시간(`age_rankings_window`)은 별도 섹션이므로 제외."""
    return [
        item
        for item in _period_baseline_items(items)
        if not _is_realtime_window(item.ranking_window_id, item.ranking_window_label)
    ]


def _period_momentum_window_ids(
    items: list[RankingItem],
    collections: list[RawCollection],
) -> list[str]:
    """Window ids for cross-window momentum (no realtime duplicate of 1d)."""
    from_items = {item.ranking_window_id for item in _period_momentum_items(items)}
    if from_items:
        return _ordered_window_ids_for_report(from_items)
    from_collections = {
        collection.target.ranking_window.id
        for collection in collections
        if collection.target.gender_filter == PERIOD_GENDER_CODE
        and collection.target.age_band == PERIOD_AGE_BAND_CODE
        and not _is_realtime_window(
            collection.target.ranking_window.id,
            collection.target.ranking_window.label,
            collection.target.ranking_window.query_params,
        )
    }
    return _ordered_window_ids_for_report(from_collections) if from_collections else []


def _demographics_window_from_collections(
    collections: list[RawCollection],
) -> tuple[str, str]:
    """Window used for gender/age snapshots; matches config `demographics_window`."""
    for collection in collections:
        target = collection.target
        if (
            target.gender_filter != PERIOD_GENDER_CODE
            or target.age_band != PERIOD_AGE_BAND_CODE
        ):
            return target.ranking_window.id, target.ranking_window.label
    return DEFAULT_DEMOGRAPHICS_WINDOW.id, DEFAULT_DEMOGRAPHICS_WINDOW.label


def _ordered_window_ids_for_report(window_ids: set[str]) -> list[str]:
    return sorted(window_ids, key=lambda wid: (_REPORT_WINDOW_RANK.get(wid, 99), wid))


def _window_meta(
    items: list[RankingItem],
    collections: list[RawCollection],
    window_ids: list[str],
) -> tuple[list[str], dict[str, str], dict[str, int | None], dict[str, int]]:
    """Resolve labels, days and limits per window; order from oldest to newest.

    The chart x-axis flows past -> present so the line slope mirrors what the
    reference image shows (older accumulation on the left, today's pulse on the right).
    Windows with unknown `days` are placed at the ends in their original order.
    """

    labels: dict[str, str] = {}
    days: dict[str, int | None] = {}
    for item in items:
        wid = item.ranking_window_id
        labels.setdefault(wid, item.ranking_window_label or wid)
        if days.get(wid) is None and item.ranking_window_days is not None:
            days[wid] = item.ranking_window_days
    for collection in collections:
        wid = collection.target.ranking_window.id
        labels.setdefault(wid, collection.target.ranking_window.label or wid)
        if days.get(wid) is None:
            days.setdefault(wid, collection.target.ranking_window.days_effective)
    for wid in window_ids:
        labels.setdefault(wid, wid)
        days.setdefault(wid, None)

    def order_key(wid: str) -> tuple[int, float, str]:
        d = days.get(wid)
        # Larger `days` first (oldest aggregation = left). Unknown days sink to right.
        return (0 if d is not None else 1, -float(d) if d is not None else 0.0, wid)

    chart_order = sorted(window_ids, key=order_key)
    limits = {wid: _limit_for_window(collections, wid) for wid in window_ids}
    return chart_order, labels, days, limits


def _rank_energy(rank: int | None, limit: int) -> float | None:
    if rank is None or limit <= 0:
        return None
    if rank < 1:
        rank = 1
    if rank > limit:
        return 0.0
    return round((limit + 1 - rank) / limit, 6)


def _window_weight(days: int | None) -> float:
    """sqrt(days) weighting; 1 for unknown, 1/sqrt(7)/sqrt(30) for daily/weekly/monthly."""
    if days is None or days <= 0:
        return 1.0
    return math.sqrt(days)


def _classify_pattern(
    present: list[str],
    chart_order: list[str],
    energy_by_window: dict[str, float],
    velocities: list[dict[str, Any]],
    momentum_span: float | None,
    sustained_rank_energy: float,
    sustained_top_threshold: float,
) -> str:
    """Bucket each product into a single label used for cards/colors."""

    expected = set(chart_order)
    present_set = set(present)
    newest = chart_order[-1] if chart_order else None
    oldest = chart_order[0] if chart_order else None

    if len(chart_order) >= 2 and present_set == {newest}:
        return "entry_breakout"
    if len(chart_order) >= 2 and present_set == {oldest}:
        return "classic_drop"
    if len(chart_order) >= 3 and present_set == {chart_order[1]}:
        return "mid_blip"
    if len(chart_order) >= 3:
        if present_set == {chart_order[1], chart_order[2]}:
            return "transient_dw"
        if present_set == {chart_order[0], chart_order[1]}:
            return "transient_wm"

    # Need full coverage and momentum_span for the remaining buckets.
    if present_set != expected or momentum_span is None:
        return "mixed"

    velocity_values = [v["value"] for v in velocities if v.get("value") is not None]
    if not velocity_values:
        return "mixed"

    all_up = all(v >= 0 for v in velocity_values) and any(v > 0 for v in velocity_values)
    all_down = all(v < 0 for v in velocity_values)
    if all_up and momentum_span >= 0.2:
        return "steady_climb"
    if all_down and momentum_span <= -0.2:
        return "fading"
    last_v = velocity_values[-1]
    if last_v >= 0.3 and momentum_span >= 0.1:
        return "breakout"
    if abs(momentum_span) < 0.1 and sustained_rank_energy >= sustained_top_threshold:
        return "stable_top"
    return "mixed"


def _rank_line_chart(
    products: list[dict[str, Any]],
    chart_order: list[str],
    labels: dict[str, str],
    days: dict[str, int | None],
    limits: dict[str, int],
) -> dict[str, Any]:
    """Build SVG-ready coordinates for the rank-over-windows line chart."""

    if not products or len(chart_order) < 2:
        return {"has_data": False}

    pad = RANK_CHART_PAD
    inner_w = RANK_CHART_W - pad["left"] - pad["right"]
    inner_h = RANK_CHART_H - pad["top"] - pad["bottom"]

    # X positions: log10(1 + days) with larger days on the left.
    def log_days(wid: str) -> float:
        d = days.get(wid)
        return math.log10(1 + d) if d and d > 0 else 0.0

    log_values = [log_days(wid) for wid in chart_order]
    # If multiple windows share the same effective day count, pure log-days projection
    # collapses them onto one vertical axis. Spread ties by a tiny deterministic offset.
    if len(set(log_values)) < len(log_values) and len(chart_order) > 1:
        by_lv: dict[float, list[int]] = {}
        for idx, lv in enumerate(log_values):
            by_lv.setdefault(lv, []).append(idx)
        eps = 1e-3
        adjusted = list(log_values)
        for indices in by_lv.values():
            if len(indices) <= 1:
                continue
            center = (len(indices) - 1) / 2
            for offset_i, src_idx in enumerate(indices):
                adjusted[src_idx] = log_values[src_idx] + (offset_i - center) * eps
        log_values = adjusted
    log_min, log_max = min(log_values), max(log_values)
    span = (log_max - log_min) or 1.0

    axis_windows: list[dict[str, Any]] = []
    x_by_window: dict[str, float] = {}
    for wid, lv in zip(chart_order, log_values):
        # invert so largest days maps to left (x_pct=0).
        x_pct = (log_max - lv) / span if log_max != log_min else (
            chart_order.index(wid) / max(1, len(chart_order) - 1)
        )
        px = pad["left"] + x_pct * inner_w
        x_by_window[wid] = px
        axis_windows.append(
            {
                "id": wid,
                "label": labels.get(wid, wid),
                "days": days.get(wid),
                "x_pct": round(x_pct, 4),
                "x": round(px, 2),
            }
        )

    max_limit = max(limits.get(wid) or 1 for wid in chart_order)
    # Y ticks: 1, then nice subdivisions of max_limit.
    nice_ranks = [1, 10, 25, 50]
    if max_limit not in nice_ranks:
        nice_ranks.append(max_limit)
    nice_ranks = sorted({r for r in nice_ranks if 1 <= r <= max_limit})

    def y_for(rank: int) -> float:
        if max_limit <= 1:
            return pad["top"]
        return pad["top"] + (rank - 1) / (max_limit - 1) * inner_h

    y_ticks = [{"rank": r, "y": round(y_for(r), 2)} for r in nice_ranks]

    # Sort products by sustained_rank_energy descending; plot every join row with data.
    ranked = sorted(
        products,
        key=lambda p: p.get("sustained_rank_energy", 0.0),
        reverse=True,
    )

    movers_pool = [p for p in ranked if p.get("momentum_span") is not None]
    movers_pool.sort(key=lambda p: abs(p["momentum_span"]), reverse=True)
    top_mover_keys = {p["key"] for p in movers_pool[:RANK_CHART_MOVERS_N]}
    span_ranked = [p for p in ranked if p.get("momentum_span") is not None]
    top12_keys = {p["key"] for p in span_ranked[:RANK_CHART_TOP12_N]}

    # Plot every join row with at least one observed rank (span + event products).
    lines: list[dict[str, Any]] = []
    for product in ranked:
        observed_pts = []
        all_pts = []
        for wid in chart_order:
            rank = product["ranks"].get(wid)
            observed = rank is not None
            point = {
                "wid": wid,
                "x": round(x_by_window[wid], 2),
                "y": round(y_for(rank), 2) if observed else None,
                "rank": rank,
                "observed": observed,
            }
            all_pts.append(point)
            if observed:
                observed_pts.append(point)
        if not observed_pts:
            continue
        observed_coords = [
            {"x": p["x"], "y": p["y"], "rank": p["rank"], "wid": p["wid"]}
            for p in observed_pts
        ]
        polyline = " ".join(f"{p['x']},{p['y']}" for p in observed_pts)
        observed_ranks = [p["rank"] for p in observed_pts if p["rank"] is not None]
        span_value = product.get("momentum_span")
        event_cluster = product.get("event_cluster")
        if span_value is not None:
            if span_value > 0.1:
                color = "up"
            elif span_value < -0.1:
                color = "down"
            else:
                color = "flat"
        elif event_cluster and event_cluster != "none":
            color = "event"
        else:
            color = "flat"
        product_key = product["key"]
        lines.append(
            {
                "key": product_key,
                "brand": product["brand"],
                "product": product["product"],
                "url": product["url"],
                "image_url": product.get("image_url"),
                "polyline": polyline,
                "observed_coords": observed_coords,
                "points": all_pts,
                "color": color,
                "pattern": product.get("pattern", "mixed"),
                "momentum_span": span_value,
                "event_cluster": event_cluster,
                "event_strength": product.get("event_strength"),
                "sustained_rank_energy": product.get("sustained_rank_energy"),
                "highlight": product_key in top_mover_keys,
                "in_top12": product_key in top12_keys,
                "in_movers": product_key in top_mover_keys,
                "rank_min": min(observed_ranks) if observed_ranks else None,
                "rank_max": max(observed_ranks) if observed_ranks else None,
            }
        )

    color_counts = {"up": 0, "down": 0, "flat": 0, "event": 0}
    for line in lines:
        color_counts[line["color"]] = color_counts.get(line["color"], 0) + 1

    span_lines = [line for line in lines if line.get("momentum_span") is not None]
    all_line_keys = [line["key"] for line in lines]
    presets = {
        "top12": {
            "label": f"상위 {RANK_CHART_TOP12_N}",
            "keys": [k for k in all_line_keys if k in top12_keys],
            "default": RANK_CHART_DEFAULT_PRESET == "top12",
        },
        "movers8": {
            "label": f"모멘텀 {RANK_CHART_MOVERS_N}",
            "keys": [k for k in all_line_keys if k in top_mover_keys],
            "default": RANK_CHART_DEFAULT_PRESET == "movers8",
        },
        "all": {
            "label": f"전체 {len(lines)}",
            "keys": all_line_keys,
            "default": RANK_CHART_DEFAULT_PRESET == "all",
        },
    }

    return {
        "has_data": True,
        "canvas": {
            "w": RANK_CHART_W,
            "h": RANK_CHART_H,
            "pad_left": pad["left"],
            "pad_right": pad["right"],
            "pad_top": pad["top"],
            "pad_bottom": pad["bottom"],
            "inner_w": inner_w,
            "inner_h": inner_h,
        },
        "axis_windows": axis_windows,
        "y_ticks": y_ticks,
        "max_limit": max_limit,
        "y_pad": RANK_CHART_Y_PAD,
        "lines": lines,
        "all_lines": lines,
        "presets": presets,
        "default_preset": RANK_CHART_DEFAULT_PRESET,
        "top_mover_keys": list(top_mover_keys),
        "plotted_count": len(lines),
        "momentum_cohort_size": len(span_lines),
        "event_cohort_size": len(lines) - len(span_lines),
        "candidate_count": len(products),
        "color_counts": color_counts,
    }


def _momentum_distribution(
    products: list[dict[str, Any]],
    chart_order: list[str],
    labels: dict[str, str],
) -> dict[str, Any]:
    edges = MOMENTUM_BIN_EDGES
    n_bins = len(edges) - 1
    bins = [0] * n_bins
    keys_by_bin: list[list[str]] = [[] for _ in range(n_bins)]
    for product in products:
        value = product.get("momentum_span")
        if value is None:
            continue
        idx = _momentum_bin_index(value, edges)
        if idx is None:
            continue
        bins[idx] += 1
        keys_by_bin[idx].append(product["key"])
    by_key = {product["key"]: product for product in products}
    max_count = max(bins) if bins else 0
    cells = []
    for i, count in enumerate(bins):
        lo, hi = edges[i], edges[i + 1]
        center = (lo + hi) / 2
        is_last = i == n_bins - 1
        keys = keys_by_bin[i]
        bucket = [by_key[key] for key in keys if key in by_key]
        bucket.sort(key=lambda p: p.get("momentum_span") or 0.0, reverse=True)
        rows = [_serialize_insight_row(p, chart_order, labels) for p in bucket]
        cells.append(
            {
                "index": i,
                "lo": lo,
                "hi": hi,
                "center": round(center, 3),
                "label": _momentum_bin_label(lo, hi, is_last),
                "subtitle": _momentum_bin_subtitle(lo, hi, is_last, chart_order, labels),
                "count": count,
                "product_keys": keys,
                "picks": rows[:INSIGHT_PREVIEW_COUNT],
                "rows": rows,
                "pct": round(count / max_count * 100, 1) if max_count else 0,
                "side": "neg" if hi <= 0 else ("pos" if lo >= 0 else "mid"),
            }
        )
    total = sum(bins)
    join_size = len(products)
    unmapped_count = join_size - total
    pos_count = sum(c["count"] for c in cells if c["side"] == "pos")
    neg_count = sum(c["count"] for c in cells if c["side"] == "neg")
    mid_count = sum(c["count"] for c in cells if c["side"] == "mid")
    return {
        "has_data": bool(total),
        "total": total,
        "join_size": join_size,
        "unmapped_count": unmapped_count,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "mid_count": mid_count,
        "max_count": max_count,
        "cells": cells,
    }


def _serialize_insight_row(
    product: dict[str, Any],
    chart_order: list[str],
    labels: dict[str, str],
) -> dict[str, Any]:
    ranks_chain = [
        {
            "wid": wid,
            "label": labels.get(wid, wid),
            "rank": product["ranks"].get(wid),
            "observed": wid in product["ranks"],
        }
        for wid in chart_order
    ]
    return {
        "key": product["key"],
        "brand": product["brand"] or "-",
        "product": product["product"] or "-",
        "url": product["url"],
        "image_url": product.get("image_url"),
        "is_sold_out": product.get("is_sold_out", False),
        "ranks_chain": ranks_chain,
        "momentum_span": product.get("momentum_span"),
        "sustained_rank_energy": product.get("sustained_rank_energy"),
        "pattern": product.get("pattern"),
    }


def _momentum_span_extrema(
    products: list[dict[str, Any]],
    n: int,
    chart_order: list[str],
    labels: dict[str, str],
    span_first: str,
    span_last: str,
) -> dict[str, Any]:
    pool = [p for p in products if p.get("momentum_span") is not None]
    if not pool:
        return {
            "has_data": False,
            "n": n,
            "from_label": labels.get(span_first, span_first),
            "to_label": labels.get(span_last, span_last),
            "top": [],
            "bottom": [],
        }
    by_desc = sorted(pool, key=lambda p: p["momentum_span"], reverse=True)
    by_asc = sorted(pool, key=lambda p: p["momentum_span"])
    top_rows = [_serialize_insight_row(p, chart_order, labels) for p in by_desc[:n]]
    bottom_rows = [_serialize_insight_row(p, chart_order, labels) for p in by_asc[:n]]
    return {
        "has_data": True,
        "n": n,
        "from_label": labels.get(span_first, span_first),
        "to_label": labels.get(span_last, span_last),
        "top": top_rows,
        "bottom": bottom_rows,
    }


# Each insight card aggregates one or more pattern labels.
_INSIGHT_DEFINITIONS = [
    ("breakout", "급부상", "1주 → 1일에서 순위가 크게 올라온 신규/돌발 진입", ("breakout", "entry_breakout")),
    ("steady_climb", "꾸준한 상승", "월 → 주 → 일 모두 한 방향으로 개선", ("steady_climb",)),
    ("stable_top", "스테디 톱", "모든 윈도우 상위에서 안정 유지", ("stable_top",)),
    ("fading", "식어가는", "월 → 일로 갈수록 순위가 떨어짐", ("fading",)),
    ("classic_drop", "월간 잔존", "월간에만 살아 있음 (현재 펄스 약함)", ("classic_drop",)),
]


def _insight_cards(
    products: list[dict[str, Any]],
    chart_order: list[str],
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        pattern = product.get("pattern", "mixed")
        by_pattern[pattern].append(product)

    newest = chart_order[-1] if chart_order else None
    oldest = chart_order[0] if chart_order else None

    sort_keys: dict[str, Any] = {
        "breakout": lambda p: -(p.get("rank_energy", {}).get(newest, 0.0) if newest else 0.0),
        "steady_climb": lambda p: -(p.get("momentum_span") or 0.0),
        "stable_top": lambda p: -(p.get("sustained_rank_energy") or 0.0),
        "fading": lambda p: (p.get("momentum_span") or 0.0),
        "classic_drop": lambda p: -(p.get("rank_energy", {}).get(oldest, 0.0) if oldest else 0.0),
    }

    cards: list[dict[str, Any]] = []
    for key, title, subtitle, pattern_keys in _INSIGHT_DEFINITIONS:
        bucket: list[dict[str, Any]] = []
        for pk in pattern_keys:
            bucket.extend(by_pattern.get(pk, []))
        sort_fn = sort_keys.get(key)
        if sort_fn is not None:
            bucket.sort(key=sort_fn)
        rows = [_serialize_insight_row(p, chart_order, labels) for p in bucket]
        picks = rows[:INSIGHT_PREVIEW_COUNT]
        cards.append(
            {
                "key": key,
                "title": title,
                "subtitle": subtitle,
                "count": len(bucket),
                "rows": rows,
                "picks": picks,
            }
        )
    return cards


def _pair_summaries(
    by_key: dict[str, dict[str, Any]],
    chart_order: list[str],
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    """Legacy adjacent-pair tables (kept under cross_window.legacy.pair_summaries)."""

    summaries: list[dict[str, Any]] = []
    for index in range(len(chart_order) - 1):
        window_a = chart_order[index]
        window_b = chart_order[index + 1]
        set_a = {key for key, row in by_key.items() if window_a in row["ranks"]}
        set_b = {key for key, row in by_key.items() if window_b in row["ranks"]}
        only_a = set_a - set_b
        only_b = set_b - set_a
        both = set_a & set_b
        deltas: list[tuple[int, str, int, int]] = []
        for key in both:
            ra = by_key[key]["ranks"][window_a]
            rb = by_key[key]["ranks"][window_b]
            deltas.append((abs(rb - ra), key, ra, rb))
        deltas.sort(key=lambda row: row[0], reverse=True)
        top_rows = [
            _cross_row(
                by_key[key],
                ra,
                rb,
                labels.get(window_a, window_a),
                labels.get(window_b, window_b),
            )
            for _, key, ra, rb in deltas[:10]
        ]
        summaries.append(
            {
                "from_window": window_a,
                "to_window": window_b,
                "from_label": labels.get(window_a, window_a),
                "to_label": labels.get(window_b, window_b),
                "only_from_count": len(only_a),
                "only_to_count": len(only_b),
                "both_count": len(both),
                "top_delta_rows": top_rows,
            }
        )
    return summaries


def _cross_window_analysis(
    items: list[RankingItem],
    window_ids: list[str],
    collections: list[RawCollection],
) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "has_data": False,
        "disclaimer": "",
        "window_ids": window_ids,
        "chart_window_order": list(window_ids),
        "axis_labels": {},
        "axis_days": {},
        "limits_by_window": {},
        "products": [],
        "momentum_chart": {"has_data": False},
        "momentum_distribution": {"has_data": False},
        "event_distribution": {"has_data": False},
        "pattern_counts": {},
        "insights": [],
        "momentum_extrema": {
            "has_data": False,
            "n": MOMENTUM_EXTREMA_COUNT,
            "from_label": "",
            "to_label": "",
            "top": [],
            "bottom": [],
        },
        "join_size": 0,
        "weighting": {
            "energy": "rank_energy(w) = (limit+1-rank)/limit",
            "window_weight": "sqrt(days)",
            "momentum_span": "rank_energy(newest) - rank_energy(oldest)",
            "no_weight_for_velocity": True,
        },
        "legacy": {"pair_summaries": [], "momentum_span_rows": []},
    }
    if len(window_ids) < 2:
        return empty

    chart_order, labels, days, limits = _window_meta(items, collections, window_ids)
    if len(chart_order) < 2:
        return empty

    disclaimer = (
        "일·주·월은 시계열 세 점이 아니라 동일 시점에서의 집계 길이가 다른 윈도우입니다. "
        "momentum_span은 양 끝 윈도우가 모두 있을 때만 rank_energy(newest)−rank_energy(oldest)입니다. "
        "진입·이탈·부분 관측은 event_cluster로 분리합니다. sustained는 sqrt(days) 가중 합입니다."
    )

    # ------------------------------------------------------------------
    # Build per-product buckets.
    # ------------------------------------------------------------------
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        join_key = _product_join_key(item)
        if join_key is None:
            continue
        bucket = by_key.setdefault(
            join_key,
            {
                "key": join_key,
                "product_id": item.product_id,
                "brand": item.brand,
                "product": item.product_clean or item.product,
                "url": item.product_url,
                "image_url": item.image_url,
                "is_sold_out": item.is_sold_out,
                "ranks": {},
                "rank_energy": {},
            },
        )
        if not bucket.get("image_url") and item.image_url:
            bucket["image_url"] = item.image_url
        if item.rank is not None:
            wid = item.ranking_window_id
            limit_w = limits.get(wid, 100) or 100
            bucket["ranks"][wid] = item.rank
            energy = _rank_energy(item.rank, limit_w)
            if energy is not None:
                bucket["rank_energy"][wid] = energy

    # ------------------------------------------------------------------
    # Compute per-product analytics (sustained, velocities, momentum_span).
    # ------------------------------------------------------------------
    products: list[dict[str, Any]] = []
    for bucket in by_key.values():
        ranks = bucket["ranks"]
        rank_energy = bucket["rank_energy"]
        present = [wid for wid in chart_order if wid in rank_energy]

        sustained = 0.0
        for wid in present:
            sustained += _window_weight(days.get(wid)) * rank_energy[wid]

        velocities: list[dict[str, Any]] = []
        for i in range(len(chart_order) - 1):
            wa, wb = chart_order[i], chart_order[i + 1]
            if wa in rank_energy and wb in rank_energy:
                velocity = round(rank_energy[wb] - rank_energy[wa], 4)
            else:
                velocity = None
            velocities.append(
                {
                    "from": wa,
                    "to": wb,
                    "from_label": labels.get(wa, wa),
                    "to_label": labels.get(wb, wb),
                    "value": velocity,
                }
            )

        momentum_span = _compute_momentum_span(chart_order, rank_energy)
        event_cluster = _classify_event_cluster(chart_order, rank_energy, momentum_span)
        event_strength = _compute_event_strength(chart_order, rank_energy, event_cluster)

        observed_velocity_values = [v["value"] for v in velocities if v["value"] is not None]
        if observed_velocity_values:
            pos = sum(1 for v in observed_velocity_values if v > 0)
            persistence = round(pos / len(observed_velocity_values), 3)
        else:
            persistence = None

        products.append(
            {
                "key": bucket["key"],
                "product_id": bucket["product_id"],
                "brand": bucket["brand"],
                "product": bucket["product"],
                "url": bucket["url"],
                "image_url": bucket["image_url"],
                "is_sold_out": bucket["is_sold_out"],
                "ranks": dict(ranks),
                "rank_energy": dict(rank_energy),
                "present_windows": present,
                "velocities": velocities,
                "momentum_span": momentum_span,
                "event_cluster": event_cluster,
                "event_strength": event_strength,
                "sustained_rank_energy": round(sustained, 4),
                "persistence": persistence,
            }
        )

    # ------------------------------------------------------------------
    # Pattern classification — needs sustained top-quintile threshold.
    # ------------------------------------------------------------------
    sustained_values = sorted(
        (p["sustained_rank_energy"] for p in products),
        reverse=True,
    )
    if sustained_values:
        cut = max(1, len(sustained_values) // 5)
        sustained_top_threshold = sustained_values[cut - 1]
    else:
        sustained_top_threshold = 0.0

    for product in products:
        product["pattern"] = _classify_pattern(
            product["present_windows"],
            chart_order,
            product["rank_energy"],
            product["velocities"],
            product["momentum_span"],
            product["sustained_rank_energy"],
            sustained_top_threshold,
        )

    # ------------------------------------------------------------------
    # Aggregate views.
    # ------------------------------------------------------------------
    pattern_counts: dict[str, int] = {}
    for product in products:
        pattern_counts[product["pattern"]] = pattern_counts.get(product["pattern"], 0) + 1

    momentum_chart = _rank_line_chart(products, chart_order, labels, days, limits)
    momentum_distribution = _momentum_distribution(products, chart_order, labels)
    event_distribution = _event_distribution(products, chart_order, labels)
    insights = _insight_cards(products, chart_order, labels)
    span_first, span_last = chart_order[0], chart_order[-1]
    momentum_extrema = _momentum_span_extrema(
        products,
        MOMENTUM_EXTREMA_COUNT,
        chart_order,
        labels,
        span_first,
        span_last,
    )

    legacy_pair_summaries = _pair_summaries(by_key, chart_order, labels)
    span_deltas: list[tuple[int, str, int, int]] = []
    for key, row in by_key.items():
        ranks = row["ranks"]
        if span_first in ranks and span_last in ranks:
            ra, rb = ranks[span_first], ranks[span_last]
            span_deltas.append((abs(rb - ra), key, ra, rb))
    span_deltas.sort(key=lambda row: row[0], reverse=True)
    span_rows = [
        _cross_row(
            by_key[key],
            ra,
            rb,
            labels.get(span_first, span_first),
            labels.get(span_last, span_last),
        )
        for _, key, ra, rb in span_deltas[:15]
    ]

    return {
        "has_data": True,
        "disclaimer": disclaimer,
        "window_ids": window_ids,
        "chart_window_order": list(chart_order),
        "axis_labels": {wid: labels.get(wid, wid) for wid in chart_order},
        "axis_days": {wid: days.get(wid) for wid in chart_order},
        "limits_by_window": limits,
        "products": products,
        "momentum_chart": momentum_chart,
        "momentum_distribution": momentum_distribution,
        "event_distribution": event_distribution,
        "pattern_counts": pattern_counts,
        "insights": insights,
        "momentum_extrema": momentum_extrema,
        "join_size": len(by_key),
        "weighting": {
            "energy": "rank_energy(w) = (limit+1-rank)/limit",
            "window_weight": "sqrt(days)",
            "momentum_span": f"rank_energy({span_last}) - rank_energy({span_first})",
            "no_weight_for_velocity": True,
        },
        "legacy": {
            "pair_summaries": legacy_pair_summaries,
            "momentum_span_rows": span_rows,
            "momentum_span_endpoints": {
                "from_window": span_first,
                "to_window": span_last,
                "from_label": labels.get(span_first, span_first),
                "to_label": labels.get(span_last, span_last),
            },
        },
    }


def _product_items(items: list[RankingItem]) -> list[RankingItem]:
    return [item for item in items if item.sub_pan not in _SUB_PAN_BRANDS]


def category_report_filename(category_code: str, *, primary_category_code: str | None = None) -> str:
    """Filename for a category report.

    The primary category is rendered into the top-level `report.html`, so its
    nav entry must point there as well. Other categories use the namespaced
    `report_<code>.html` filename.
    """
    if primary_category_code and normalize_category_code(category_code) == normalize_category_code(primary_category_code):
        return MAIN_REPORT_FILENAME
    return f"report_{normalize_category_code(category_code)}.html"


def brand_portfolio_report_filename() -> str:
    return BRAND_PORTFOLIO_REPORT_FILENAME


def customer_signals_report_filename() -> str:
    """The customer signals page is a single report shared by all categories."""
    return CUSTOMER_SIGNALS_REPORT_FILENAME


def _product_collections(collections: list[RawCollection]) -> list[RawCollection]:
    return [collection for collection in collections if collection.target.sub_pan not in _SUB_PAN_BRANDS]


def _category_key_from_item(item: RankingItem) -> str:
    return normalize_category_code(item.category_code)


def _category_key_from_collection(collection: RawCollection) -> str:
    return normalize_category_code(collection.target.category.code)


def _category_keys(dataset: NormalizedDataset) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for collection in _product_collections(dataset.collections):
        key = _category_key_from_collection(collection)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    for item in _product_items(dataset.items):
        key = _category_key_from_item(item)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _category_nav_entries(
    reports: dict[str, dict[str, Any]],
    active_key: str,
    *,
    primary_key: str | None = None,
    include_customer_signals: bool = True,
    include_brand_tab: bool = True,
) -> list[dict[str, Any]]:
    """Return a flat nav: category tabs, then 고객 신호, then 브랜드.

    Each entry has a `kind` field (`category`, `customer_signals`,
    `brand_portfolio`) so the template can render them uniformly without
    inline conditional logic. The primary category links back to the
    top-level `report.html` so there is no duplicate file.
    """
    entries: list[dict[str, Any]] = []
    for key, report in reports.items():
        entries.append(
            {
                "code": key,
                "kind": "category",
                "label": report["headline"]["category"],
                "href": category_report_filename(key, primary_category_code=primary_key),
                "active": key == active_key,
                "item_count": report["headline"]["item_count"],
                "limit_target": report["headline"].get("limit_target"),
            }
        )
    if include_customer_signals:
        entries.append(
            {
                "code": CUSTOMER_SIGNALS_NAV_KEY,
                "kind": "customer_signals",
                "label": "고객 신호",
                "href": customer_signals_report_filename(),
                "active": active_key == CUSTOMER_SIGNALS_NAV_KEY,
                "item_count": None,
                "limit_target": None,
            }
        )
    if include_brand_tab and reports:
        entries.append(
            {
                "code": BRAND_PORTFOLIO_NAV_KEY,
                "kind": "brand_portfolio",
                "label": "브랜드",
                "href": brand_portfolio_report_filename(),
                "active": active_key == BRAND_PORTFOLIO_NAV_KEY,
                "item_count": None,
                "limit_target": None,
            }
        )
    return entries


def _truncate_display_label(text: str, max_len: int = 34) -> str:
    label = (text or "").strip()
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _target_brand_products_in_items(
    items: list[RankingItem],
    spec: BcaveBrandSpec,
) -> list[dict[str, Any]]:
    matches = [
        item
        for item in items
        if (item.brand or "").strip() in spec.musinsa_names and item.rank is not None
    ]
    matches.sort(key=lambda item: item.rank)
    return [
        {
            "rank": item.rank,
            "rank_label": f"#{item.rank}",
            "product": _truncate_display_label(item.product_clean or item.product or "-"),
            "url": item.product_url,
        }
        for item in matches
    ]


def _target_presence_in_items(items: list[RankingItem]) -> list[dict[str, Any]]:
    """Per BCave target: product-row count and share within a category slice."""
    branded = [item for item in items if item.brand]
    total = len(branded)
    rows: list[dict[str, Any]] = []
    for spec in BCAVE_PORTFOLIO:
        count = sum(
            1
            for item in branded
            if (item.brand or "").strip() in spec.musinsa_names
        )
        rows.append(
            {
                "id": spec.id,
                "label_ko": spec.label_ko,
                "in_list": count > 0,
                "product_count": count,
                "share_pct": round(count / total * 100, 1) if total else 0.0,
            }
        )
    return rows


def _brand_top_products(
    brand_items: list[RankingItem],
    *,
    limit: int = 3,
    show_category: bool = False,
    detailed: bool = False,
    validation_scores: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        [item for item in brand_items if item.rank is not None],
        key=lambda item: item.rank,
    )[:limit]
    scores = validation_scores or {}
    products: list[dict[str, Any]] = []
    for item in ranked:
        if detailed:
            entry = _top10_card(item, scores.get(id(item), 0.0))
            entry["product"] = _truncate_display_label(entry["product"])
        else:
            entry = {
                "rank": item.rank,
                "rank_label": f"#{item.rank}",
                "product": _truncate_display_label(item.product_clean or item.product or "-"),
                "url": item.product_url,
            }
        if show_category:
            entry["category_label"] = item.category_label or "-"
        products.append(entry)
    return products


def _brand_distribution_rows(
    items: list[RankingItem],
    *,
    scope_key: str,
    show_category_on_products: bool = False,
    detailed_products: bool = False,
    validation_scores: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Per-brand SKU count and best rank within one category ranking slice."""
    by_brand: dict[str, list[RankingItem]] = defaultdict(list)
    for item in items:
        brand = (item.brand or "").strip()
        if not brand or item.rank is None:
            continue
        by_brand[brand].append(item)

    total = sum(len(group) for group in by_brand.values())
    rows: list[dict[str, Any]] = []
    for brand, brand_items in by_brand.items():
        count = len(brand_items)
        best_rank = min(item.rank for item in brand_items if item.rank is not None)
        spec = match_bcave_brand(brand)
        rows.append(
            {
                "brand": brand,
                "product_count": count,
                "share_pct": round(count / total * 100, 1) if total else 0.0,
                "best_rank": best_rank,
                "best_rank_label": f"#{best_rank}",
                "is_target": spec is not None,
                "target_label": spec.label_ko if spec else None,
                "top_products": _brand_top_products(
                    brand_items,
                    show_category=show_category_on_products,
                    detailed=detailed_products,
                    validation_scores=validation_scores,
                ),
            }
        )

    rows.sort(key=lambda row: (-row["product_count"], row["best_rank"], row["brand"]))
    for index, row in enumerate(rows, start=1):
        row["display_order"] = index
        row["row_key"] = f"{scope_key}-{index}"
        if row["top_products"]:
            row["lead_product"] = row["top_products"][0]
    return rows


def _primary_window_product_items(
    category_reports: dict[str, dict[str, Any]],
    product_items: list[RankingItem],
) -> list[RankingItem]:
    slice_items: list[RankingItem] = []
    for key, report in category_reports.items():
        primary_window_id = report.get("meta", {}).get("primary_window_id", "1d")
        slice_items.extend(
            item
            for item in product_items
            if _category_key_from_item(item) == key
            and item.ranking_window_id == primary_window_id
            and item.gender_filter == PERIOD_GENDER_CODE
            and item.age_band == PERIOD_AGE_BAND_CODE
        )
    return slice_items


def _overall_brand_distribution_table(
    category_reports: dict[str, dict[str, Any]],
    product_items: list[RankingItem],
    *,
    top_n: int = 10,
) -> dict[str, Any] | None:
    if not category_reports:
        return None
    slice_items = _primary_window_product_items(category_reports, product_items)
    validation_scores = {id(item): _validation_score(item) for item in slice_items}
    rows = _brand_distribution_rows(
        slice_items,
        scope_key="overall",
        show_category_on_products=True,
        detailed_products=True,
        validation_scores=validation_scores,
    )
    return {
        "category_code": "overall",
        "category_label": "전체 상품",
        "is_overall": True,
        "item_count": len([item for item in slice_items if item.rank is not None]),
        "brand_count": len(rows),
        "top_n": top_n,
        "rows": rows[:top_n],
    }


def _category_brand_distribution_tables(
    category_reports: dict[str, dict[str, Any]],
    product_items: list[RankingItem],
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for key, report in category_reports.items():
        headline = report.get("headline", {})
        primary_window_id = report.get("meta", {}).get("primary_window_id", "1d")
        slice_items = [
            item
            for item in product_items
            if _category_key_from_item(item) == key
            and item.ranking_window_id == primary_window_id
            and item.gender_filter == PERIOD_GENDER_CODE
            and item.age_band == PERIOD_AGE_BAND_CODE
        ]
        rows = _brand_distribution_rows(slice_items, scope_key=key)
        tables.append(
            {
                "category_code": key,
                "category_label": headline.get("category", key),
                "item_count": len([item for item in slice_items if item.rank is not None]),
                "brand_count": len(rows),
                "rows": rows,
            }
        )
    return tables


def _category_share_rows(
    category_reports: dict[str, dict[str, Any]],
    product_items: list[RankingItem],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, report in category_reports.items():
        headline = report.get("headline", {})
        kpis = report.get("kpis", {})
        bc = report.get("brand_concentration", {})
        slice_items = [
            item
            for item in product_items
            if _category_key_from_item(item) == key
            and item.gender_filter == PERIOD_GENDER_CODE
            and item.age_band == PERIOD_AGE_BAND_CODE
        ]
        targets = _target_presence_in_items(slice_items)
        top_brands: list[dict[str, Any]] = []
        for bar in bc.get("top_bars", [])[:5]:
            spec = match_bcave_brand(bar["brand"])
            top_brands.append(
                {
                    "brand": bar["brand"],
                    "share_pct": bar["share_pct"],
                    "count": bar["count"],
                    "is_target": spec is not None,
                    "target_label": spec.label_ko if spec else None,
                }
            )
        max_share = top_brands[0]["share_pct"] if top_brands else 0.0
        for bar in top_brands:
            bar["bar_scale_pct"] = (
                round(bar["share_pct"] / max_share * 100, 1) if max_share else 0.0
            )
        rows.append(
            {
                "category_code": key,
                "category_label": headline.get("category", key),
                "item_count": headline.get("item_count", 0),
                "brand_count": kpis.get("brand_count", 0),
                "hhi": bc.get("hhi"),
                "hhi_level_label": bc.get("level_label"),
                "top_brands": top_brands,
                "targets": targets,
                "has_targets": any(t["in_list"] for t in targets),
            }
        )
    return rows


def _target_brand_matrix(
    category_reports: dict[str, dict[str, Any]],
    bcave: dict[str, Any],
    product_items: list[RankingItem],
) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in bcave.get("brands", [])}
    matrix: list[dict[str, Any]] = []
    for spec in BCAVE_PORTFOLIO:
        bcave_row = by_id.get(spec.id, {})
        categories: list[dict[str, Any]] = []
        for key, report in category_reports.items():
            primary_window_id = report.get("meta", {}).get("primary_window_id", "1d")
            slice_items = [
                item
                for item in product_items
                if _category_key_from_item(item) == key
                and item.ranking_window_id == primary_window_id
                and item.gender_filter == PERIOD_GENDER_CODE
                and item.age_band == PERIOD_AGE_BAND_CODE
            ]
            targets = _target_presence_in_items(slice_items)
            target_cell = next((t for t in targets if t["id"] == spec.id), None)
            products = _target_brand_products_in_items(slice_items, spec)
            categories.append(
                {
                    "category_code": key,
                    "category_label": report["headline"]["category"],
                    "in_list": bool(target_cell and target_cell["in_list"]),
                    "share_pct": target_cell["share_pct"] if target_cell else 0.0,
                    "product_count": target_cell["product_count"] if target_cell else 0,
                    "products": products,
                }
            )
        matrix.append(
            {
                "id": spec.id,
                "label_ko": spec.label_ko,
                "style_lane_label": spec.style_lane_label,
                "overall": bcave_row.get("overall", {}),
                "style_lane": bcave_row.get("style_lane", {}),
                "categories": categories,
            }
        )
    return matrix


def _brand_portfolio_payload(
    dataset: NormalizedDataset,
    category_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bcave = _bcave_brand_tracker(dataset)
    product_items = _product_items(dataset.items)
    category_share_rows = _category_share_rows(category_reports, product_items)
    category_brand_tables = _category_brand_distribution_tables(
        category_reports, product_items
    )
    overall_brand_table = _overall_brand_distribution_table(
        category_reports, product_items
    )
    target_matrix = _target_brand_matrix(category_reports, bcave, product_items)

    collected_candidates = [item.collected_at for item in product_items if item.collected_at]
    collected_at = max(collected_candidates) if collected_candidates else utc_timestamp()

    primary_window_label = ""
    if bcave.get("has_data"):
        primary_window_label = bcave.get("ranking_window_label", "")
    elif category_reports:
        first = next(iter(category_reports.values()))
        primary_window_label = first.get("headline", {}).get("primary_window_label", "")

    multiple_windows = False
    if category_reports:
        multiple_windows = any(
            report.get("meta", {}).get("multiple_windows") for report in category_reports.values()
        )

    return {
        "has_data": bool(bcave.get("has_data") or category_share_rows),
        "bcave": bcave,
        "category_share_rows": category_share_rows,
        "category_brand_tables": category_brand_tables,
        "overall_brand_table": overall_brand_table,
        "target_brand_matrix": target_matrix,
        "target_count": len(BCAVE_PORTFOLIO),
        "category_count": len(category_share_rows),
        "collected_at_utc": collected_at,
        "collected_at_kst_pretty": _format_kst(collected_at),
        "primary_window_label": primary_window_label,
        "multiple_windows": multiple_windows,
        "disclaimer": (
            "브랜드 탭 순위는 무신사 subPan=brand 기준(전체/전체 + 스타일 레인)입니다. "
            "카테고리 점유율은 각 카테고리 상품 TOP 100 안 SKU 개수 비율이며 매출·GMV가 아닙니다."
        ),
    }


def _build_brand_portfolio_report(
    dataset: NormalizedDataset,
    category_reports: dict[str, dict[str, Any]],
    *,
    primary_key: str | None = None,
) -> dict[str, Any]:
    portfolio = _brand_portfolio_payload(dataset, category_reports)
    primary_window_id = "1d"
    if category_reports:
        first = next(iter(category_reports.values()))
        primary_window_id = first.get("meta", {}).get("primary_window_id", primary_window_id)

    return {
        "generated_at": utc_timestamp(),
        "headline": {
            "category": "브랜드 포트폴리오",
            "collected_at_utc": portfolio["collected_at_utc"],
            "collected_at_kst_pretty": portfolio["collected_at_kst_pretty"],
            "item_count": portfolio["category_count"],
            "total_item_count": portfolio["category_count"],
            "limit_target": portfolio["target_count"],
            "primary_window_id": primary_window_id,
            "primary_window_label": portfolio["primary_window_label"],
        },
        "meta": {
            "report_kind": "brand_portfolio",
            "multiple_categories": len(category_reports) > 1,
            "multiple_windows": portfolio["multiple_windows"],
            "primary_window_id": primary_window_id,
            "window_ids": [],
        },
        "kpis": {
            "brand_count": portfolio["target_count"],
            "median_price": None,
            "avg_discount_rate": None,
            "validated_count": 0,
            "validated_pct": 0.0,
            "sold_out_count": 0,
            "item_count": portfolio["category_count"],
        },
        "brand_portfolio": portfolio,
        "category_nav": _category_nav_entries(
            category_reports,
            BRAND_PORTFOLIO_NAV_KEY,
            primary_key=primary_key,
            include_customer_signals=_dataset_has_customer_signals(dataset),
            include_brand_tab=True,
        ),
        "bcave_tracker": {"has_data": False, "brands": [], "sections": []},
        "brand_concentration": {"has_data": False},
        "cross_window": {"has_data": False},
        "scatter": {"has_data": False},
        "outliers": [],
        "windows": {},
        "price_dot_strip": {"has_data": False},
        "price_age_heatmap": {"has_data": False},
        "quality": {
            "ok": True,
            "sentence": f"카테고리 {portfolio['category_count']}종 · 타겟 브랜드 {portfolio['target_count']}개",
        },
    }


def _dataset_has_customer_signals(dataset: NormalizedDataset) -> bool:
    cs = dataset.customer_signals
    if cs is None:
        return False
    if cs.contents:
        return True
    return any(board.rows for board in cs.keywords_by_gender)


def _product_rank_maps(
    items: list[RankingItem],
    *,
    primary_window_id: str,
    realtime_window_id: str,
) -> tuple[dict[str, int], dict[str, int]]:
    period_rank_by_id: dict[str, int] = {}
    for item in _period_baseline_items(items):
        if item.ranking_window_id != primary_window_id:
            continue
        if not item.product_id or item.rank is None:
            continue
        product_id = str(item.product_id)
        current = period_rank_by_id.get(product_id)
        if current is None or item.rank < current:
            period_rank_by_id[product_id] = item.rank

    realtime_rank_by_id: dict[str, int] = {}
    for item in items:
        if item.ranking_window_id != realtime_window_id:
            continue
        if item.gender_filter != PERIOD_GENDER_CODE:
            continue
        if not item.product_id or item.rank is None:
            continue
        product_id = str(item.product_id)
        current = realtime_rank_by_id.get(product_id)
        if current is None or item.rank < current:
            realtime_rank_by_id[product_id] = item.rank
    return period_rank_by_id, realtime_rank_by_id


def _build_customer_signals_payload(
    items: list[RankingItem],
    customer_signals: CustomerSignalDataset | None,
    *,
    demographics_window_label: str,
    realtime_window_label: str,
    primary_window_id: str,
    realtime_window_id: str,
) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "has_data": False,
        "summary": {
            "content_count": 0,
            "keyword_gender_count": 0,
            "demographics_window_label": demographics_window_label,
            "realtime_window_label": realtime_window_label,
        },
        "contents": [],
        "keywords": {
            "default_gender": "all",
            "genders": [],
        },
        "errors": [],
    }
    if customer_signals is None:
        return empty

    period_rank_by_id, realtime_rank_by_id = _product_rank_maps(
        items,
        primary_window_id=primary_window_id,
        realtime_window_id=realtime_window_id,
    )
    validation_by_item_id = {id(item): _validation_score(item) for item in items}
    product_lookup: dict[str, RankingItem] = {}
    for item in _period_baseline_items(items):
        if item.ranking_window_id == primary_window_id and item.product_id:
            product_lookup.setdefault(str(item.product_id), item)
    for item in items:
        if item.product_id:
            product_lookup.setdefault(str(item.product_id), item)

    unmatched_product_ids: list[str] = []
    for content in customer_signals.contents:
        for product_id in content.product_ids:
            pid = str(product_id).strip()
            if pid and pid not in product_lookup and pid not in unmatched_product_ids:
                unmatched_product_ids.append(pid)

    goods_lookup: dict[str, dict[str, Any]] = {}
    if unmatched_product_ids:
        try:
            goods_lookup = fetch_goods_catalog(unmatched_product_ids)
        except Exception:
            goods_lookup = {}

    content_rows: list[dict[str, Any]] = []
    for content in customer_signals.contents:
        products = join_content_products(
            content,
            product_lookup=product_lookup,
            goods_lookup=goods_lookup,
            period_rank_by_id=period_rank_by_id,
            realtime_rank_by_id=realtime_rank_by_id,
            validation_by_item_id=validation_by_item_id,
        )
        matched_count = sum(1 for product in products if product.matched)
        content_rows.append(
            {
                "rank": content.rank,
                "content_id": content.content_id,
                "title": content.title,
                "brand": content.brand,
                "content_type": content.content_type,
                "view_count_text": content.view_count_text,
                "comment_count_text": content.comment_count_text,
                "popularity_score_text": content.popularity_score_text,
                "url": content.url,
                "product_count": len(content.product_ids),
                "matched_product_count": matched_count,
                "products": [product.to_dict() for product in products],
            }
        )

    keyword_genders: list[dict[str, Any]] = []
    default_gender = "all"
    for board in customer_signals.keywords_by_gender:
        gender_id = gender_key(board.gender_code)
        keyword_genders.append(
            {
                "id": gender_id,
                "code": board.gender_code,
                "label": board.gender_label or gender_label_for_code(board.gender_code),
                "has_data": bool(board.rows),
                "rows": [row.to_dict() for row in board.rows],
            }
        )
    if keyword_genders and not any(entry["has_data"] for entry in keyword_genders):
        default_gender = keyword_genders[0]["id"]
    else:
        for entry in keyword_genders:
            if entry["has_data"]:
                default_gender = entry["id"]
                break

    has_data = bool(content_rows) or any(entry["has_data"] for entry in keyword_genders)
    return {
        "has_data": has_data,
        "summary": {
            "content_count": len(content_rows),
            "keyword_gender_count": sum(1 for entry in keyword_genders if entry["has_data"]),
            "demographics_window_label": demographics_window_label,
            "realtime_window_label": realtime_window_label,
        },
        "contents": content_rows,
        "keywords": {
            "default_gender": default_gender,
            "genders": keyword_genders,
        },
        "errors": list(customer_signals.errors),
    }


def _analyze_product_scope(
    items: list[RankingItem],
    collections: list[RawCollection],
    customer_signals: CustomerSignalDataset | None = None,
) -> dict[str, Any]:
    period_items = _period_baseline_items(items)
    momentum_items = _period_momentum_items(items)
    period_collections = [
        c for c in collections
        if c.target.gender_filter == PERIOD_GENDER_CODE
        and c.target.age_band == PERIOD_AGE_BAND_CODE
    ]
    demo_window_id, demo_window_label = _demographics_window_from_collections(collections)
    heatmap_items = [item for item in items if item.ranking_window_id == demo_window_id]

    window_source = {item.ranking_window_id for item in period_items} or {
        collection.target.ranking_window.id for collection in period_collections
    }

    window_ids = (
        _ordered_window_ids_for_report(window_source)
        if window_source
        else ["default"]
    )
    momentum_window_ids = _period_momentum_window_ids(items, period_collections)

    windows: dict[str, Any] = {}
    for window_id in window_ids:
        slice_items = [item for item in period_items if item.ranking_window_id == window_id]
        successful, failed = _collections_for_window(period_collections, window_id)
        inner = _analyze_slice(slice_items, successful, failed)
        window_label = slice_items[0].ranking_window_label if slice_items else window_id
        windows[window_id] = {
            "headline": {
                "ranking_window_id": window_id,
                "ranking_window_label": window_label,
                "item_count": len(slice_items),
            },
            **inner,
        }

    primary_window_id = window_ids[0]
    headline = _aggregate_headline(period_items, primary_window_id, period_collections)
    effective_primary = headline["primary_window_id"]
    primary_slice = windows[effective_primary]
    heatmap_payload = dict(_price_age_heatmap(heatmap_items))
    if heatmap_payload.get("has_data"):
        heatmap_payload["ranking_window_id"] = demo_window_id
        heatmap_payload["ranking_window_label"] = (
            heatmap_items[0].ranking_window_label if heatmap_items else demo_window_label
        )

    meta = {
        "multiple_windows": len(window_ids) > 1,
        "window_ids": list(window_ids),
        "primary_window_id": effective_primary,
        "demographics_window_id": demo_window_id,
        "demographics_window_label": demo_window_label,
    }
    result: dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "headline": headline,
        "meta": meta,
        "windows": windows,
        "cross_window": _cross_window_analysis(
            momentum_items,
            momentum_window_ids if len(momentum_window_ids) >= 2 else list(window_ids),
            period_collections,
        ),
        "category_nav": [],
    }
    for key in (
        "kpis",
        "heat_top",
        "scatter",
        "brand_concentration",
        "outliers",
        "price_dot_strip",
        "quality",
    ):
        result[key] = primary_slice[key]
    result["price_age_heatmap"] = heatmap_payload
    rt_window_id, rt_window_label = _resolve_realtime_window_id(items, collections)
    result["age_rankings"] = _age_ranking_tables(
        items,
        window_id=rt_window_id,
        window_label=rt_window_label,
        top_n=30,
    )
    result["customer_signals"] = _build_customer_signals_payload(
        items,
        customer_signals,
        demographics_window_label=demo_window_label,
        realtime_window_label=rt_window_label,
        primary_window_id=effective_primary,
        realtime_window_id=rt_window_id,
    )
    result["bcave_tracker"] = {"has_data": False, "brands": [], "sections": []}
    return result


def analyze_dataset(dataset: NormalizedDataset) -> dict[str, Any]:
    product_items = _product_items(dataset.items)
    product_collections = _product_collections(dataset.collections)
    keys = _category_keys(dataset)
    has_customer_signals = _dataset_has_customer_signals(dataset)

    reports: dict[str, dict[str, Any]] = {}
    for key in keys:
        scoped_items = [item for item in product_items if _category_key_from_item(item) == key]
        scoped_collections = [
            collection
            for collection in product_collections
            if _category_key_from_collection(collection) == key
        ]
        reports[key] = _analyze_product_scope(
            scoped_items,
            scoped_collections,
            customer_signals=dataset.customer_signals,
        )

    if reports:
        primary_key = keys[0]
        brand_portfolio_report = _build_brand_portfolio_report(
            dataset, reports, primary_key=primary_key
        )
        for key, report in reports.items():
            report["meta"]["multiple_categories"] = len(reports) > 1
            report["meta"]["category_code"] = key
            report["meta"]["report_kind"] = "category"
            report["category_nav"] = _category_nav_entries(
                reports,
                key,
                primary_key=primary_key,
                include_customer_signals=has_customer_signals,
            )

        customer_signals_report = (
            _build_customer_signals_report(dataset, reports, primary_key)
            if has_customer_signals
            else None
        )

        result = dict(reports[primary_key])
        result["meta"] = dict(result["meta"])
        result["meta"]["primary_category_code"] = primary_key
        result["meta"]["report_kind"] = "category"
        result["category_reports"] = reports
        result["customer_signals_report"] = customer_signals_report
        result["brand_portfolio"] = brand_portfolio_report["brand_portfolio"]
        result["brand_portfolio_report"] = brand_portfolio_report
        return result

    brand_only = _build_brand_portfolio_report(dataset, {}, primary_key=None)
    result = _analyze_product_scope([], product_collections, customer_signals=dataset.customer_signals)
    result["category_reports"] = {}
    result["customer_signals_report"] = None
    result["brand_portfolio"] = brand_only["brand_portfolio"]
    result["brand_portfolio_report"] = brand_only
    result["meta"]["report_kind"] = "category"
    return result


def _build_customer_signals_report(
    dataset: NormalizedDataset,
    category_reports: dict[str, dict[str, Any]],
    primary_key: str,
) -> dict[str, Any]:
    """Single customer-signals page: per-category heatmap panels + aggregated
    age rankings + shared content/keyword data.

    The page exposes one heatmap panel per category (client-side toggle in the
    template). Age rankings, content, and keyword sections do not depend on
    category, so they are computed once across the whole dataset.
    """

    primary = category_reports[primary_key]
    primary_customer_signals = primary.get("customer_signals", {
        "has_data": False,
        "summary": {},
        "contents": [],
        "keywords": {"default_gender": "all", "genders": []},
        "errors": [],
    })

    panels: list[dict[str, Any]] = []
    for code, report in category_reports.items():
        heatmap = dict(report.get("price_age_heatmap", {"has_data": False}))
        panels.append(
            {
                "code": code,
                "label": report["headline"]["category"],
                "heatmap": heatmap,
                "has_data": bool(heatmap.get("has_data")),
            }
        )
    default_panel = next((p for p in panels if p["has_data"]), panels[0] if panels else None)
    default_code = default_panel["code"] if default_panel else ""

    product_items = _product_items(dataset.items)
    product_collections = _product_collections(dataset.collections)
    rt_window_id, rt_window_label = _resolve_realtime_window_id(
        product_items, product_collections
    )
    aggregated_age_rankings = _aggregated_age_ranking_tables(
        product_items,
        window_id=rt_window_id,
        window_label=rt_window_label,
        top_n=30,
    )

    headline = dict(primary["headline"])
    headline["category"] = "고객 신호"

    return {
        "generated_at": utc_timestamp(),
        "headline": headline,
        "meta": {
            **dict(primary["meta"]),
            "report_kind": "customer_signals",
            "multiple_categories": len(category_reports) > 1,
            "primary_category_code": primary_key,
        },
        "category_nav": _category_nav_entries(
            category_reports,
            CUSTOMER_SIGNALS_NAV_KEY,
            primary_key=primary_key,
            include_customer_signals=True,
            include_brand_tab=True,
        ),
        "customer_signals": primary_customer_signals,
        "price_age_heatmap_panels": {
            "default_code": default_code,
            "categories": panels,
            "has_data": any(p["has_data"] for p in panels),
        },
        "age_rankings": aggregated_age_rankings,
        # Template guards still reference these fields; stub them out so the
        # category-report-only sections stay hidden on the CS page.
        "kpis": {
            "brand_count": 0,
            "median_price": None,
            "avg_discount_rate": None,
            "validated_count": 0,
            "validated_pct": 0.0,
            "sold_out_count": 0,
            "item_count": 0,
        },
        "scatter": {"has_data": False},
        "cross_window": {"has_data": False},
        "outliers": [],
        "windows": {},
        "price_dot_strip": {"has_data": False},
        "price_age_heatmap": {"has_data": False},
        "brand_concentration": {"has_data": False},
        "quality": {"ok": True, "sentence": "고객 신호 통합 페이지"},
        "bcave_tracker": {"has_data": False, "brands": [], "sections": []},
    }


def _aggregated_age_ranking_tables(
    items: list[RankingItem],
    *,
    window_id: str | None,
    window_label: str | None = None,
    top_n: int = 30,
) -> dict[str, Any]:
    """Same shape as :func:`_age_ranking_tables` but unioned across categories.

    For each age band we keep the lowest rank seen for a given product across
    every category so the table reads as a global "전체 카테고리 × 연령" view.
    """
    baseline = DEFAULT_AGE_BANDS[0]
    empty: dict[str, Any] = {
        "has_data": False,
        "window_id": window_id,
        "window_label": window_label or "실시간",
        "default_age": baseline.id,
        "top_n": top_n,
        "ages": [],
        "baseline_age_code": PERIOD_AGE_BAND_CODE,
        "empty_reason": (
            "실시간 연령별 랭킹 데이터가 없습니다. "
            "periodic-multag.json 의 age_rankings_window(실시간) 수집이 포함되어 있는지 확인하세요."
        ),
    }
    if not window_id:
        return empty

    scoped = [
        item
        for item in items
        if item.gender_filter == PERIOD_GENDER_CODE
        and item.ranking_window_id == window_id
        and item.rank is not None
    ]
    if not scoped:
        return empty

    validation_scores = {id(item): _validation_score(item) for item in scoped}

    ages_payload: list[dict[str, Any]] = []
    any_data = False
    for spec in DEFAULT_AGE_BANDS:
        age_items = [item for item in scoped if item.age_band == spec.code]
        # Same product can appear in multiple categories; surface the best rank.
        best_by_key: dict[str, RankingItem] = {}
        for item in age_items:
            key = item.product_id or _product_join_key(item) or f"obj:{id(item)}"
            current = best_by_key.get(key)
            if current is None or (item.rank or 9999) < (current.rank or 9999):
                best_by_key[key] = item
        unique_items = list(best_by_key.values())
        rows = _top10_for_window(unique_items, validation_scores, top_n=top_n)
        has_rows = bool(rows)
        if has_rows:
            any_data = True
        ages_payload.append({
            "id": spec.id,
            "code": spec.code,
            "label": spec.label,
            "has_data": has_rows,
            "rows": rows,
            "item_count": len(unique_items),
        })

    if not any_data:
        return empty

    default_age = next(
        (entry["id"] for entry in ages_payload if entry["has_data"]),
        baseline.id,
    )

    return {
        "has_data": True,
        "window_id": window_id,
        "window_label": window_label or window_id,
        "default_age": default_age,
        "top_n": top_n,
        "ages": ages_payload,
        "baseline_age_code": PERIOD_AGE_BAND_CODE,
    }


def _brand_rank_board(items: list[RankingItem]) -> dict[str, int]:
    board: dict[str, int] = {}
    for item in items:
        brand = (item.brand or "").strip()
        if not brand or item.rank is None:
            continue
        if brand not in board or item.rank < board[brand]:
            board[brand] = item.rank
    return board


def _lookup_bcave_rank(board: dict[str, int], spec: BcaveBrandSpec) -> int | None:
    for name in spec.musinsa_names:
        if name in board:
            return board[name]
    return None


def _bcave_rank_cell(rank: int | None, limit: int) -> dict[str, Any]:
    if rank is None:
        return {
            "rank": None,
            "rank_label": "미진입",
            "in_list": False,
            "rank_energy": None,
        }
    return {
        "rank": rank,
        "rank_label": f"#{rank}",
        "in_list": True,
        "rank_energy": round((limit + 1 - rank) / limit, 4) if limit > 0 else None,
    }


def _bcave_specs_for_pane(pane: BcaveBrandPane) -> list[BcaveBrandSpec]:
    if pane.is_overall:
        return list(BCAVE_PORTFOLIO)
    return [spec for spec in BCAVE_PORTFOLIO if spec.style_section_id == pane.section.section_id]


def _bcave_section_table_rows(
    board: dict[str, int],
    target_specs: list[BcaveBrandSpec],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Build table rows for one brand-tab pane.

    Always render the full top *k* leaderboard so readers see the head of
    the ranking even when targets sit lower. If any tracked target is below
    *k* (or missing), append an ellipsis row followed by each outside target
    so the gap to their actual position is communicated explicitly.
    """
    sorted_board = sorted(board.items(), key=lambda pair: pair[1])
    top_slice = sorted_board[:top_k]

    target_meta: list[dict[str, Any]] = []
    for spec in target_specs:
        rank = _lookup_bcave_rank(board, spec)
        target_meta.append(
            {
                "spec": spec,
                "rank": rank,
                "brand": spec.musinsa_names[0],
                "in_top_k": rank is not None and rank <= top_k,
            }
        )

    def _brand_row(brand: str, rank: int | None, *, is_target: bool, target_label: str | None) -> dict[str, Any]:
        cell = _bcave_rank_cell(rank, top_k)
        return {
            "kind": "brand",
            "rank": rank,
            "rank_label": cell["rank_label"],
            "brand": brand,
            "is_target": is_target,
            "target_label": target_label,
            "highlight": is_target,
            "missing": rank is None,
        }

    rows: list[dict[str, Any]] = []
    shown: set[str] = set()
    for brand, rank in top_slice:
        shown.add(brand)
        spec = match_bcave_brand(brand)
        rows.append(
            _brand_row(
                brand,
                rank,
                is_target=spec is not None,
                target_label=spec.label_ko if spec else None,
            )
        )

    outside = [entry for entry in target_meta if entry["brand"] not in shown]
    if outside:
        rows.append({"kind": "ellipsis"})
        for entry in sorted(outside, key=lambda item: (item["rank"] is None, item["rank"] or 9999)):
            rows.append(
                _brand_row(
                    entry["brand"],
                    entry["rank"],
                    is_target=True,
                    target_label=entry["spec"].label_ko,
                )
            )

    return rows


def _bcave_limit_for_window(collections: list[RawCollection], window_id: str) -> int:
    limits = [
        collection.target.limit
        for collection in collections
        if collection.target.ranking_window.id == window_id
        and collection.target.sub_pan in _SUB_PAN_BRANDS
    ]
    return max(limits) if limits else 100


def _bcave_brand_tracker(dataset: NormalizedDataset) -> dict[str, Any]:
    brand_items = [item for item in dataset.items if item.sub_pan in _SUB_PAN_BRANDS]
    if not brand_items:
        return {"has_data": False, "brands": [], "sections": []}

    window_ids = list(dict.fromkeys(item.ranking_window_id for item in brand_items))
    primary_window_id = window_ids[0]
    primary_items = [item for item in brand_items if item.ranking_window_id == primary_window_id]
    if not primary_items:
        return {"has_data": False, "brands": [], "sections": []}

    limit = _bcave_limit_for_window(dataset.collections, primary_window_id)
    boards_by_section: dict[str, dict[str, int]] = {
        pane.section.section_id: _brand_rank_board(
            [item for item in primary_items if item.section_id == pane.section.section_id]
        )
        for pane in BCAVE_BRAND_PANES
    }

    brand_rows: list[dict[str, Any]] = []
    for spec in BCAVE_PORTFOLIO:
        overall_rank = _lookup_bcave_rank(boards_by_section.get(STYLE_SECTION_OVERALL, {}), spec)
        lane_rank = _lookup_bcave_rank(boards_by_section.get(spec.style_section_id, {}), spec)
        brand_rows.append(
            {
                "id": spec.id,
                "label_ko": spec.label_ko,
                "style_lane_label": spec.style_lane_label,
                "overall": _bcave_rank_cell(overall_rank, limit),
                "style_lane": _bcave_rank_cell(lane_rank, limit),
            }
        )

    section_summaries = []
    for pane in BCAVE_BRAND_PANES:
        board = boards_by_section.get(pane.section.section_id, {})
        target_specs = _bcave_specs_for_pane(pane)
        table_rows = _bcave_section_table_rows(board, target_specs, top_k=10)
        leader = None
        if board:
            top_brand, top_rank = min(board.items(), key=lambda pair: pair[1])
            leader = {"brand": top_brand, "rank": top_rank}
        section_summaries.append(
            {
                "label": pane.label,
                "is_overall": pane.is_overall,
                "leader": leader,
                "target_count": len(target_specs),
                "table_rows": table_rows,
                "uses_compressed_view": any(row.get("kind") == "ellipsis" for row in table_rows),
            }
        )

    return {
        "has_data": True,
        "ranking_window_label": primary_items[0].ranking_window_label,
        "limit": limit,
        "top_k": 10,
        "brands": brand_rows,
        "sections": section_summaries,
        "note": (
            "무신사 랭킹 페이지 **브랜드 탭** 응답(subPan=brand)을 그대로 사용합니다. "
            "전체/전체는 sectionId=1054 + categoryCode 전체, 스타일 레인은 "
            "영캐주얼·여성캐주얼·스트릿캐주얼 sectionId별 호출입니다."
        ),
    }


def _kpis(
    items: list[RankingItem],
    prices: list[int],
    discounts: list[float],
    sold_out_count: int,
) -> dict[str, Any]:
    brand_count = len({item.brand for item in items if item.brand})
    sales_labeled = sum(1 for item in items if _sales_label_score(item.labels) > 0)
    validated_pct = round(sales_labeled / len(items) * 100, 1) if items else 0.0
    discounted_count = sum(1 for item in items if _item_has_discount(item))
    discount_application_pct = (
        round(discounted_count / len(items) * 100, 1) if items else 0.0
    )
    return {
        "brand_count": brand_count,
        "median_price": int(median(prices)) if prices else None,
        "avg_discount_rate": _avg(discounts),
        "discounted_count": discounted_count,
        "discount_application_pct": discount_application_pct,
        "validated_count": sales_labeled,
        "validated_pct": validated_pct,
        "sold_out_count": sold_out_count,
        "item_count": len(items),
    }


def _heat_top(items: list[RankingItem]) -> list[dict[str, Any]]:
    ranked = sorted(
        ((item, _heat(item)) for item in items),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top = [(item, score) for item, score in ranked if score > 0][:5]
    max_score = top[0][1] if top else 1
    return [
        {
            **_item_card(item),
            "heat_score": score,
            "heat_bar_pct": round(score / max_score * 100, 1) if max_score else 0,
            "viewers_now": item.viewers_now or 0,
            "buyers_now": item.buyers_now or 0,
        }
        for item, score in top
    ]


def _heat(item: RankingItem) -> int:
    return (item.viewers_now or 0) + (item.buyers_now or 0) * HEAT_WEIGHT_BUYERS


def _validation_score(item: RankingItem) -> float:
    """Plataform-resistant composite of "real user activity" signals.

    The dimensions chosen here are intentionally NOT the platform-controlled
    `review_score`. They are signals that cost the platform either money
    (real-time buyer counter), engineering integrity (cumulative reviews), or
    public credibility (sales-volume labels) to fake — so we trust them more
    than any single rating average.
    """
    review_term = math.log10((item.review_count or 0) + 1) * VALIDATION_WEIGHT_REVIEW
    buyers_term = math.log10((item.buyers_now or 0) + 1) * VALIDATION_WEIGHT_BUYERS
    sales_term = _sales_label_score(item.labels) * VALIDATION_WEIGHT_SALES_LABEL
    return round(review_term + buyers_term + sales_term, 4)


def _sales_label_score(labels: list[str] | None) -> float:
    """Convert sales-volume labels into a log10(units / 1만) score.

    Examples (each is its own best-of):
      "판매 5,000개"           -> 0.0  (no '만' = ignored)
      "판매 1.1만개"           -> log10(1.1)  ~= 0.04
      "판매 10만개"            -> log10(10)   = 1.0
      "누적 판매 50만 돌파"     -> log10(50)   ~= 1.7
      "BEST" / "베스트"        -> 0.5 (qualitative bump)
    """
    if not labels:
        return 0.0
    best = 0.0
    for raw in labels:
        if not raw:
            continue
        label = str(raw)
        # 판매 N만개 (decimal possible)
        m = _SALES_QUANTITY_RE.search(label)
        if m:
            major = int(m.group(1).replace(",", ""))
            minor = m.group(2)
            value_man = major + (int(minor) / 10 ** len(minor) if minor else 0)
            if value_man > 0:
                best = max(best, round(math.log10(value_man) + 0, 3))
            continue
        # 누적 판매 N만 돌파
        m = _SALES_MILESTONE_RE.search(label)
        if m:
            value_man = int(m.group(1).replace(",", ""))
            if value_man > 0:
                best = max(best, round(math.log10(value_man) + 0, 3))
            continue
        upper = label.upper()
        if any(hint in upper for hint in _BEST_HINT):
            best = max(best, 0.5)
    return best


def _ranking_validation_scatter(
    items: list[RankingItem],
    validation_scores: dict[int, float],
) -> dict[str, Any]:
    """Scatter: x=rank, y=Validation Score, with a least-squares trend line.

    The trend line answers "what validation score does a product at rank N
    *typically* have?". The signed residual (point - trend) is the outlier
    signal: positive = under-ranked given how validated it is; negative =
    over-ranked given how little user activity it has (push/new-product
    suspect).
    """
    eligible = [item for item in items if item.rank is not None]
    if not eligible:
        return {
            "has_data": False,
            "points": [],
            "labels": [],
            "canvas": _scatter_canvas(),
            "missing_count": len(items),
            "summary": {"above": 0, "below": 0},
        }

    xs = [float(item.rank) for item in eligible]
    ys = [validation_scores[id(item)] for item in eligible]

    slope, intercept = _linear_regression(xs, ys)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Make the y range include the trend line endpoints so the line never
    # extends outside the drawing area.
    fit_y_at_min_x = slope * min_x + intercept
    fit_y_at_max_x = slope * max_x + intercept
    min_y = min(min_y, fit_y_at_min_x, fit_y_at_max_x)
    max_y = max(max_y, fit_y_at_min_x, fit_y_at_max_x)
    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1

    canvas = _scatter_canvas()
    inner_w = canvas["inner_w"]
    inner_h = canvas["inner_h"]
    left = canvas["pad_left"]
    top = canvas["pad_top"]

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        px = left + (x - min_x) / (max_x - min_x) * inner_w
        py = top + (1 - (y - min_y) / (max_y - min_y)) * inner_h
        return px, py

    points = []
    above = 0
    below = 0
    for item, x_val, y_val in zip(eligible, xs, ys):
        join_key = _product_join_key(item)
        residual = y_val - (slope * x_val + intercept)
        px, py = to_pixel(x_val, y_val)
        side = "above" if residual >= 0 else "below"
        if side == "above":
            above += 1
        else:
            below += 1
        rank = int(item.rank) if item.rank is not None else 0
        if rank <= 20:
            rank_band = "top20"
        elif rank <= 50:
            rank_band = "mid"
        else:
            rank_band = "low"
        points.append({
            "key": join_key or f"rank:{rank}:{item.brand}",
            "x": round(px, 1),
            "y": round(py, 1),
            "r": _validation_radius(item),
            "rank": rank,
            "rank_band": rank_band,
            "review_count": item.review_count or 0,
            "buyers_now": item.buyers_now or 0,
            "sales_label_score": _sales_label_score(item.labels),
            "validation": round(y_val, 3),
            "residual": round(residual, 3),
            "side": side,
            "brand": item.brand,
            "product": item.product_clean or item.product,
            "url": item.product_url,
            "is_outlier": False,
        })

    # Trend-line endpoints in pixel space.
    trend_x1, trend_y1 = to_pixel(min_x, fit_y_at_min_x)
    trend_x2, trend_y2 = to_pixel(max_x, fit_y_at_max_x)

    # Label the 3 strongest positive residuals (검증된 강자) and 3 strongest
    # negative residuals (푸시/신상 의심). Keep label text short.
    sorted_points = sorted(points, key=lambda p: p["residual"], reverse=True)
    outlier_keys = {
        p["key"] for p in sorted(points, key=lambda pt: abs(pt["residual"]), reverse=True)[:SCATTER_OUTLIER_N]
    }
    for point in points:
        point["is_outlier"] = point["key"] in outlier_keys

    labels = []
    for p in sorted_points[:3]:
        labels.append(_residual_label(p, kind="above"))
    for p in sorted_points[-3:]:
        if p["residual"] < 0:
            labels.append(_residual_label(p, kind="below"))

    # Y-axis hint ticks: 0, 1, 2, 3 in validation space, only those inside range.
    y_ticks = []
    for v in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if v < min_y - 0.05 or v > max_y + 0.05:
            continue
        _, py = to_pixel(min_x, v)
        y_ticks.append({"y": round(py, 1), "label": f"{v:g}"})

    scatter_presets = {
        "top20": {
            "label": "TOP 20",
            "default": SCATTER_DEFAULT_PRESET == "top20",
        },
        "mid": {
            "label": "21–50위",
            "default": SCATTER_DEFAULT_PRESET == "mid",
        },
        "low": {
            "label": "51위+",
            "default": SCATTER_DEFAULT_PRESET == "low",
        },
        "outliers": {
            "label": f"이상치 {SCATTER_OUTLIER_N}",
            "default": SCATTER_DEFAULT_PRESET == "outliers",
        },
        "all": {
            "label": "전체",
            "default": SCATTER_DEFAULT_PRESET == "all",
        },
    }

    return {
        "has_data": True,
        "points": points,
        "labels": labels,
        "canvas": canvas,
        "missing_count": len(items) - len(eligible),
        "trend": {
            "x1": round(trend_x1, 1),
            "y1": round(trend_y1, 1),
            "x2": round(trend_x2, 1),
            "y2": round(trend_y2, 1),
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
        },
        "y_ticks": y_ticks,
        "presets": scatter_presets,
        "default_preset": SCATTER_DEFAULT_PRESET,
        "summary": {"above": above, "below": below},
    }


def _residual_label(point: dict[str, Any], kind: str) -> dict[str, Any]:
    """Compute a label placement for a single outlier point, nudged to not
    overlap the point itself."""
    dy = -10 if kind == "above" else 18
    return {
        "key": point.get("key"),
        "x": point["x"],
        "y": round(point["y"] + dy, 1),
        "kind": kind,
        "brand": point["brand"],
        "product": point["product"],
        "rank": point["rank"],
        "residual": point["residual"],
    }


def _validation_radius(item: RankingItem) -> float:
    """Map validation magnitude to point radius (3.0 ~ 8.0 px)."""
    score = _validation_score(item)
    # Practical range of validation score: 0 (no activity) to ~3 (top product).
    capped = max(0.0, min(score, 3.0))
    return round(3.0 + (capped / 3.0) * 5.0, 1)


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope and intercept for y = slope * x + intercept. NumPy-free."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, mean_y
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _scatter_canvas() -> dict[str, Any]:
    pad = SCATTER_PAD
    return {
        "w": SCATTER_W,
        "h": SCATTER_H,
        "pad_top": pad["top"],
        "pad_right": pad["right"],
        "pad_bottom": pad["bottom"],
        "pad_left": pad["left"],
        "inner_w": SCATTER_W - pad["left"] - pad["right"],
        "inner_h": SCATTER_H - pad["top"] - pad["bottom"],
    }


def _brand_concentration(items: list[RankingItem]) -> dict[str, Any]:
    brands = [item.brand for item in items if item.brand]
    if not brands:
        return {"has_data": False, "hhi": 0, "level": "none", "level_label": "데이터 없음"}

    counter = Counter(brands)
    total = sum(counter.values())
    shares_pct = [(brand, count, count / total * 100) for brand, count in counter.most_common()]
    hhi = sum(pct * pct for _, _, pct in shares_pct)

    if hhi < HHI_DIFFUSE_MAX:
        level = "diffuse"
        level_label = "분산"
    elif hhi < HHI_CONCENTRATED_MIN:
        level = "moderate"
        level_label = "보통"
    else:
        level = "concentrated"
        level_label = "집중"

    top8 = shares_pct[:8]
    other_share = sum(pct for _, _, pct in shares_pct[8:])
    other_count = sum(count for _, count, _ in shares_pct[8:])
    cumulative_pct: list[float] = []
    running = 0.0
    for _, _, pct in top8:
        running += pct
        cumulative_pct.append(round(running, 1))

    bars = [
        {
            "brand": brand,
            "count": count,
            "share_pct": round(pct, 1),
            "cumulative_pct": cumulative_pct[i],
        }
        for i, (brand, count, pct) in enumerate(top8)
    ]

    # Bullet chart: position the HHI value along a 0~10000 scale, but cap at 4000 visually
    # so realistic clothing values (~500-3000) remain readable.
    scale_max = 4000
    hhi_capped = min(hhi, scale_max)
    bullet = {
        "value": round(hhi, 1),
        "pct": round(hhi_capped / scale_max * 100, 2),
        "diffuse_pct": round(HHI_DIFFUSE_MAX / scale_max * 100, 2),
        "concentrated_pct": round(HHI_CONCENTRATED_MIN / scale_max * 100, 2),
        "scale_max": scale_max,
    }

    return {
        "has_data": True,
        "hhi": round(hhi, 1),
        "level": level,
        "level_label": level_label,
        "brand_total": len(counter),
        "top_bars": bars,
        "other_share_pct": round(other_share, 1),
        "other_count": other_count,
        "bullet": bullet,
    }


def _outlier_lists(
    items: list[RankingItem],
    validation_scores: dict[int, float],
) -> list[dict[str, Any]]:
    """Five lenses on the same 100 items. Each lens picks at most 5 cards.

    The lists are derived from the platform-resistant signals (validation
    residual, sales labels, real-time buyers, discount strategy) instead of
    the inflated review_score axis from v2.0.
    """
    eligible = [item for item in items if item.rank is not None]
    if not eligible:
        return []

    xs = [float(item.rank) for item in eligible]
    ys = [validation_scores[id(item)] for item in eligible]
    slope, intercept = _linear_regression(xs, ys)
    residuals = {id(item): ys[i] - (slope * xs[i] + intercept) for i, item in enumerate(eligible)}

    by_heat = sorted(items, key=_heat, reverse=True)
    hot = [item for item in by_heat if _heat(item) > 0][:5]

    # "누적 판매 강자": platform has tagged it with a sales-volume label, and
    # the product sits outside the top 50 (so it's not just "popular today").
    sales_strong = [
        item for item in items
        if _sales_label_score(item.labels) >= 1.0 and (item.rank or 9999) > 50
    ]
    sales_strong.sort(key=lambda item: -_sales_label_score(item.labels))

    # "푸시/신상 의심": the trend-line residual is strongly negative — the
    # product sits in a rank that products with much more user activity
    # usually occupy. We require rank <= 30 to keep the alert near the top.
    push_suspect = [
        item for item in eligible
        if residuals[id(item)] < -0.3 and (item.rank or 9999) <= 30
    ]
    push_suspect.sort(key=lambda item: residuals[id(item)])

    # "검증된 숨은 강자": positive residual outside TOP 30. Strong validation
    # signal that the platform isn't surfacing prominently.
    hidden_validated = [
        item for item in eligible
        if residuals[id(item)] > 0.3 and (item.rank or 9999) > 30
    ]
    hidden_validated.sort(key=lambda item: -residuals[id(item)])

    # "할인 의존": deep discount inside TOP 30 — platform may be using price
    # cuts to lift the rank.
    discount_dependent = [
        item for item in items
        if (item.discount_rate or 0) >= 40 and (item.rank or 9999) <= 30
    ]
    discount_dependent.sort(key=lambda item: (item.rank or 9999, -(item.discount_rate or 0)))

    return [
        {
            "key": "hot",
            "title": "지금 가장 뜨거운",
            "subtitle": "동시 시청 + 구매 (* 2) 합계",
            "cards": [_outlier_card(item, extra={"heat": _heat(item)}) for item in hot[:5]],
        },
        {
            "key": "sales_strong",
            "title": "누적 판매 강자",
            "subtitle": "판매 1만개+ 라벨 · 순위 50위 밖",
            "cards": [_outlier_card(item) for item in sales_strong[:5]],
        },
        {
            "key": "hidden_validated",
            "title": "검증된 숨은 강자",
            "subtitle": "리뷰·실구매은 많은데 순위는 30위 밖",
            "cards": [
                _outlier_card(item, extra={"residual": round(residuals[id(item)], 2)})
                for item in hidden_validated[:5]
            ],
        },
        {
            "key": "push_suspect",
            "title": "푸시/신상 의심",
            "subtitle": "TOP 30 · 검증 신호 빈약 (추세선보다 한참 아래)",
            "cards": [
                _outlier_card(item, extra={"residual": round(residuals[id(item)], 2)})
                for item in push_suspect[:5]
            ],
        },
        {
            "key": "discount_dependent",
            "title": "할인 의존",
            "subtitle": "TOP 30 · 40% 이상 할인",
            "cards": [_outlier_card(item) for item in discount_dependent[:5]],
        },
    ]


def _outlier_card(item: RankingItem, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    card = _item_card(item)
    card.update({
        "review_count": item.review_count,
        "buyers_now": item.buyers_now,
        "sales_label_text": _first_sales_label(item.labels),
    })
    if extra:
        card.update(extra)
    return card


def _first_sales_label(labels: list[str] | None) -> str:
    """Return the human-friendly sales label if present, else empty string."""
    if not labels:
        return ""
    for raw in labels:
        text = str(raw)
        if _SALES_QUANTITY_RE.search(text) or _SALES_MILESTONE_RE.search(text):
            return text
    return ""


def _item_card(item: RankingItem) -> dict[str, Any]:
    return {
        "rank": item.rank,
        "brand": item.brand or "-",
        "product": item.product_clean or item.product or "-",
        "price": item.price,
        "original_price": item.original_price,
        "discount_rate": item.discount_rate,
        "url": item.product_url,
        "is_sold_out": item.is_sold_out,
    }


def _price_bucket_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": f"b{index}",
            "label": label,
            "min": low,
            "max": high,
        }
        for index, (label, low, high) in enumerate(PRICE_BUCKETS)
    ]


def _is_realtime_window(
    window_id: str,
    label: str | None = None,
    query_params: dict[str, str] | None = None,
) -> bool:
    if query_params and str(query_params.get("period", "")).upper() == "REALTIME":
        return True
    if window_id.lower() in _REALTIME_WINDOW_IDS:
        return True
    return bool(label and "실시간" in label)


def _resolve_realtime_window_id(
    items: list[RankingItem],
    collections: list[RawCollection],
) -> tuple[str | None, str | None]:
    for collection in collections:
        window = collection.target.ranking_window
        if _is_realtime_window(window.id, window.label, window.query_params):
            return window.id, window.label
    for item in items:
        wid = item.ranking_window_id
        if wid and _is_realtime_window(wid, item.ranking_window_label):
            return wid, item.ranking_window_label
    return None, None


def _age_ranking_tables(
    items: list[RankingItem],
    *,
    window_id: str | None,
    window_label: str | None = None,
    top_n: int = 30,
) -> dict[str, Any]:
    baseline = DEFAULT_AGE_BANDS[0]
    empty: dict[str, Any] = {
        "has_data": False,
        "window_id": window_id,
        "window_label": window_label or "실시간",
        "default_age": baseline.id,
        "top_n": top_n,
        "ages": [],
        "baseline_age_code": PERIOD_AGE_BAND_CODE,
        "empty_reason": (
            "실시간 연령별 랭킹 데이터가 없습니다. "
            "periodic-multag.json 의 age_rankings_window(실시간) 수집이 포함되어 있는지 확인하세요."
        ),
    }
    if not window_id:
        return empty

    scoped = [
        item
        for item in items
        if item.gender_filter == PERIOD_GENDER_CODE
        and item.ranking_window_id == window_id
        and item.rank is not None
    ]
    validation_scores = {id(item): _validation_score(item) for item in scoped}

    ages_payload: list[dict[str, Any]] = []
    any_data = False
    for spec in DEFAULT_AGE_BANDS:
        age_items = [item for item in scoped if item.age_band == spec.code]
        rows = _top10_for_window(age_items, validation_scores, top_n=top_n)
        has_rows = bool(rows)
        if has_rows:
            any_data = True
        ages_payload.append({
            "id": spec.id,
            "code": spec.code,
            "label": spec.label,
            "has_data": has_rows,
            "rows": rows,
            "item_count": len(age_items),
        })

    if not any_data:
        return empty

    default_age = next(
        (entry["id"] for entry in ages_payload if entry["has_data"]),
        baseline.id,
    )

    return {
        "has_data": True,
        "window_id": window_id,
        "window_label": window_label or window_id,
        "default_age": default_age,
        "top_n": top_n,
        "ages": ages_payload,
        "baseline_age_code": PERIOD_AGE_BAND_CODE,
    }


def _price_dot_strip(items: list[RankingItem]) -> dict[str, Any]:
    pad_left = 20
    pad_right = 20
    presets = _price_bucket_presets()
    base_axis = {"pad_left": pad_left, "pad_right": pad_right, "lo_log": 0.0, "hi_log": 1.0}

    prices = [(item, item.price) for item in items if item.price is not None and item.price > 0]
    if not prices:
        return {
            "has_data": False,
            "dots": [],
            "w": DOT_STRIP_W,
            "h": DOT_STRIP_H,
            "ticks": [],
            "axis": base_axis,
            "presets": presets,
        }

    log_values = [math.log10(p) for _, p in prices]
    lo, hi = min(log_values), max(log_values)
    if hi == lo:
        hi = lo + 0.1
    inner_w = DOT_STRIP_W - pad_left - pad_right
    center_y = DOT_STRIP_H / 2

    dots: list[dict[str, Any]] = []
    for (item, price), lv in zip(prices, log_values):
        x_ratio = (lv - lo) / (hi - lo)
        x = pad_left + x_ratio * inner_w
        # Deterministic jitter from product_id so re-runs don't shuffle the layout.
        seed_source = item.product_id or item.product or str(item.rank)
        jitter = (sum(ord(c) for c in seed_source) % 21 - 10)
        y = center_y + jitter * 1.6
        dots.append({
            "x": round(x, 1),
            "y": round(y, 1),
            "r": 3.5,
            "price": price,
            "brand": item.brand,
            "product": item.product_clean or item.product,
            "rank": item.rank,
            "is_sold_out": item.is_sold_out,
            "key": _product_join_key(item),
            "url": item.product_url,
        })

    # Axis ticks at natural break points (~3만, 5만, 10만, 30만, 100만).
    tick_values = [10_000, 30_000, 50_000, 100_000, 300_000, 1_000_000]
    ticks = []
    for v in tick_values:
        lv = math.log10(v)
        if lv < lo or lv > hi:
            continue
        x = pad_left + (lv - lo) / (hi - lo) * inner_w
        ticks.append({"x": round(x, 1), "label": _format_price_short(v), "value": v})

    return {
        "has_data": True,
        "dots": dots,
        "ticks": ticks,
        "w": DOT_STRIP_W,
        "h": DOT_STRIP_H,
        "axis": {"pad_left": pad_left, "pad_right": pad_right, "lo_log": lo, "hi_log": hi},
        "presets": presets,
    }


def _format_price_short(value: int) -> str:
    if value >= 1_000_000:
        return f"{value // 10_000}만"
    if value >= 10_000:
        return f"{value // 10_000}만"
    return f"{value:,}"


# Non-zero cells: min–max normalize within the gender grid, then gamma > 1 to widen contrast.
_HEATMAP_ALPHA_GAMMA = 1.55
_HEATMAP_NONZERO_T_FLOOR = 0.1


def _heatmap_cell_alpha(count: int, min_count: int, max_count: int) -> float:
    if count <= 0 or max_count <= 0:
        return 0.0
    if max_count <= min_count:
        return 1.0
    span = max_count - min_count
    t = (count - min_count) / span
    t = max(_HEATMAP_NONZERO_T_FLOOR, min(1.0, t))
    return round(t**_HEATMAP_ALPHA_GAMMA, 3)


def _price_age_heatmap(items: list[RankingItem]) -> dict[str, Any]:
    """Count products per price bucket × age band, split by gender tab."""

    age_headers = [spec.label for spec in DEFAULT_AGE_BANDS]
    age_codes = [spec.code for spec in DEFAULT_AGE_BANDS]
    gender_specs = list(DEFAULT_GENDER_FILTERS)

    scoped: dict[str, list[RankingItem]] = {spec.id: [] for spec in gender_specs}
    for item in items:
        if item.price is None or item.rank is None:
            continue
        key = gender_key(item.gender_filter)
        if key not in scoped:
            scoped[key] = []
        scoped[key].append(item)

    genders_payload: list[dict[str, Any]] = []
    any_data = False

    for spec in gender_specs:
        subset = scoped.get(spec.id, [])
        grid: dict[tuple[int, int], int] = defaultdict(int)
        for item in subset:
            price_idx = _price_bucket_index(item.price)
            age_idx = age_band_index(item.age_band)
            if price_idx is None or age_idx is None:
                continue
            grid[(price_idx, age_idx)] += 1

        if not grid:
            genders_payload.append({
                "key": spec.id,
                "label": spec.label,
                "code": spec.code,
                "has_data": False,
                "max_count": 0,
                "age_headers": age_headers,
                "rows": [],
                "column_totals": [0] * len(age_headers),
            })
            continue

        any_data = True
        nonzero_counts = [v for v in grid.values() if v > 0]
        min_count = min(nonzero_counts)
        max_count = max(nonzero_counts)
        column_totals = [0] * len(age_headers)
        rows: list[dict[str, Any]] = []

        for price_idx, (price_label, _, _) in enumerate(PRICE_BUCKETS):
            row_cells = []
            row_total = 0
            for age_idx, age_label in enumerate(age_headers):
                count = grid.get((price_idx, age_idx), 0)
                row_total += count
                column_totals[age_idx] += count
                row_cells.append({
                    "count": count,
                    "alpha": _heatmap_cell_alpha(count, min_count, max_count),
                    "density": round(count / max_count, 3) if max_count else 0,
                    "age_label": age_label,
                    "age_code": age_codes[age_idx],
                    "price_label": price_label,
                })
            rows.append({
                "price_label": price_label,
                "row_total": row_total,
                "cells": row_cells,
            })

        genders_payload.append({
            "key": spec.id,
            "label": spec.label,
            "code": spec.code,
            "has_data": True,
            "max_count": max_count,
            "age_headers": age_headers,
            "rows": rows,
            "column_totals": column_totals,
        })

    default_gender = next(
        (g["key"] for g in genders_payload if g.get("has_data")),
        gender_specs[0].id,
    )
    active = next(
        (g for g in genders_payload if g["key"] == default_gender and g.get("has_data")),
        next((g for g in genders_payload if g.get("has_data")), None),
    )

    return {
        "has_data": any_data,
        "default_gender": default_gender,
        "age_headers": age_headers,
        "genders": genders_payload,
        "active": active,
    }


def _price_bucket_index(price: int) -> int | None:
    for index, (_, low, high) in enumerate(PRICE_BUCKETS):
        if price >= low and (high is None or price < high):
            return index
    return None


def _rank_band_index(rank: int) -> int | None:
    for index, (lo, hi) in enumerate(RANK_BANDS):
        if lo <= rank <= hi:
            return index
    return None


def _quality_oneliner(
    items: list[RankingItem],
    successful: list[Any],
    failed: list[Any],
) -> dict[str, Any]:
    missing_brand = sum(1 for item in items if not item.brand)
    missing_product = sum(1 for item in items if not item.product)
    missing_price = sum(1 for item in items if item.price is None)
    missing_total = missing_brand + missing_product + missing_price

    return {
        "ok": not failed and missing_total == 0,
        "success_count": len(successful),
        "fail_count": len(failed),
        "item_count": len(items),
        "missing_total": missing_total,
        "missing_brand": missing_brand,
        "missing_product": missing_product,
        "missing_price": missing_price,
        "sentence": _quality_sentence(items, successful, failed, missing_total),
    }


def _quality_sentence(
    items: list[RankingItem],
    successful: list[Any],
    failed: list[Any],
    missing_total: int,
) -> str:
    if not items:
        return "수집된 상품이 없습니다."
    parts = [f"수집 {len(items)}건"]
    if failed:
        parts.append(f"실패 {len(failed)}건")
    parts.append(f"누락 {missing_total}건")
    return " · ".join(parts)


def _format_kst(iso_utc: str) -> str:
    """Convert ISO-8601 UTC ('Z' suffix or +00:00) to '2026-05-14 11:13 KST' for headlines."""
    if not iso_utc:
        return ""
    try:
        text = iso_utc.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        kst = dt.astimezone(KST)
        return kst.strftime("%Y-%m-%d %H:%M KST")
    except ValueError:
        return iso_utc


def _avg(values: list[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 1)
