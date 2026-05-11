#!/usr/bin/env python3
"""
시계열·텍스트 융합 분석 테이블 생성.
태그-성과 상관, 브랜드 파워 인덱스, 트렌드, 제품 프로파일.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _normalize_missing(value: Any, missing: str = "미분류") -> str:
    if pd.isna(value):
        return missing
    text = str(value).strip()
    return text if text and text not in {"nan", "None", "null"} else missing


def _merge_text_features(frame: pd.DataFrame, text_features: Optional[pd.DataFrame]) -> pd.DataFrame:
    if text_features is None or text_features.empty or frame.empty:
        return frame.copy()
    merge_keys = [key for key in ("snapshot_id", "product_id") if key in frame.columns and key in text_features.columns]
    if not merge_keys:
        return frame.copy()
    cols_join = [c for c in text_features.columns if c not in frame.columns or c in merge_keys]
    if not cols_join:
        return frame.copy()
    deduped = text_features[cols_join].drop_duplicates(subset=merge_keys, keep="last")
    return frame.merge(deduped, on=merge_keys, how="left")


def _prepare_category_frame(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
    latest_only: bool = False,
) -> pd.DataFrame:
    if fact_snapshots.empty:
        return pd.DataFrame()
    base = build_product_profile(fact_snapshots, text_features) if latest_only else _merge_text_features(fact_snapshots, text_features)
    if base.empty:
        return base

    frame = base.copy()
    fallback_label = (
        frame["name_item"].map(lambda value: _normalize_missing(value, "")) if "name_item" in frame.columns else pd.Series("", index=frame.index)
    )
    raw_status = (
        frame["category_ingest_status"].map(lambda value: _normalize_missing(value, "skipped"))
        if "category_ingest_status" in frame.columns
        else pd.Series("skipped", index=frame.index)
    )
    raw_source = (
        frame["category_source"].map(lambda value: _normalize_missing(value, "unavailable"))
        if "category_source" in frame.columns
        else pd.Series("unavailable", index=frame.index)
    )
    raw_usable = (
        raw_source.eq("raw_taxonomy")
        & raw_status.isin(["success", "partial"])
        & (
            frame.get("category_l1", pd.Series(index=frame.index)).notna()
            | frame.get("category_l2", pd.Series(index=frame.index)).notna()
            | frame.get("category_l3", pd.Series(index=frame.index)).notna()
            | frame.get("category_code", pd.Series(index=frame.index)).notna()
        )
    )
    can_fallback = raw_status.eq("skipped") & fallback_label.ne("")

    resolved_source = pd.Series("unavailable", index=frame.index, dtype=object)
    resolved_source.loc[raw_usable] = "raw_taxonomy"
    resolved_source.loc[~raw_usable & can_fallback] = "fallback_name_item"

    frame["category_source"] = resolved_source
    frame["category_is_fallback"] = resolved_source.eq("fallback_name_item")
    frame["category_fallback_label"] = fallback_label.where(frame["category_is_fallback"], None)
    frame["category_label_l1"] = frame.get("category_l1", pd.Series(index=frame.index)).map(lambda value: _normalize_missing(value, ""))
    frame["category_label_l2"] = frame.get("category_l2", pd.Series(index=frame.index)).map(lambda value: _normalize_missing(value, ""))
    frame["category_label_l3"] = frame.get("category_l3", pd.Series(index=frame.index)).map(lambda value: _normalize_missing(value, ""))
    for column in ("category_label_l1", "category_label_l2", "category_label_l3"):
        frame.loc[frame["category_source"] == "fallback_name_item", column] = fallback_label.loc[frame["category_source"] == "fallback_name_item"]
        frame[column] = frame[column].replace("", "미분류")
    frame["category_label"] = frame["category_label_l3"]
    frame["category_has_raw_taxonomy"] = raw_usable
    return frame


def build_tag_performance(
    fact_snapshots: pd.DataFrame,
    tag_column: str = "tags_joined",
) -> pd.DataFrame:
    """
    태그별 평균 순위·모멘텀·안정성 집계.
    fact_snapshots에 tags_joined(또는 tag_column)가 쉼표/공백 구분 태그 문자열로 있어야 함.
    """
    if fact_snapshots.empty or tag_column not in fact_snapshots.columns:
        return pd.DataFrame()

    def split_tags(x):
        if pd.isna(x):
            return []
        s = str(x).replace(",", " ").split()
        return [t.strip().strip("#") for t in s if t.strip()]

    exploded = fact_snapshots.copy()
    exploded["_tag"] = exploded[tag_column].map(split_tags)
    exploded = exploded.explode("_tag")
    exploded = exploded[exploded["_tag"].astype(str).str.len() > 0]

    agg = exploded.groupby("_tag", as_index=False).agg(
        record_count=("product_id", "count"),
        avg_rank=("rank", "mean"),
        avg_rank_velocity=("rank_velocity", "mean"),
        avg_stability_score=("stability_score", "mean"),
        avg_momentum_score=("momentum_score", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
    ).rename(columns={"_tag": "tag"})
    agg = agg.sort_values("record_count", ascending=False).reset_index(drop=True)
    return agg


def build_brand_index(fact_snapshots: pd.DataFrame) -> pd.DataFrame:
    """브랜드별 평균 순위·제품 수·안정성·할인율 집계."""
    if fact_snapshots.empty or "brand" not in fact_snapshots.columns:
        return pd.DataFrame()

    agg = fact_snapshots.groupby("brand", as_index=False).agg(
        record_count=("product_id", "count"),
        product_count=("product_id", "nunique"),
        avg_rank=("rank", "mean"),
        avg_stability_score=("stability_score", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_momentum_score=("momentum_score", "mean"),
    )
    agg = agg.sort_values("avg_rank", ascending=True).reset_index(drop=True)
    return agg


def build_trends(
    fact_snapshots: pd.DataFrame,
    date_col: str = "snapshot_date",
    tag_column: str = "tags_joined",
) -> pd.DataFrame:
    """
    시점별 태그·색상 등 분포 변화 (날짜별 집계).
    """
    if fact_snapshots.empty:
        return pd.DataFrame()
    if date_col not in fact_snapshots.columns:
        return pd.DataFrame()

    by_date = fact_snapshots.groupby(date_col, as_index=False).agg(
        record_count=("product_id", "count"),
        product_count=("product_id", "nunique"),
        avg_rank=("rank", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
    )
    return by_date


def build_product_profile(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    제품별 최신 스냅샷 + 텍스트 피처 통합.
    fact_snapshots에서 product_id당 최신 crawl_datetime 1건만 사용.
    """
    if fact_snapshots.empty:
        return pd.DataFrame()

    if "crawl_datetime" in fact_snapshots.columns:
        latest = (
            fact_snapshots.sort_values("crawl_datetime", ascending=True)
            .groupby("product_id", as_index=False)
            .tail(1)
        )
    else:
        latest = fact_snapshots.drop_duplicates("product_id", keep="last")

    if text_features is not None and not text_features.empty:
        tf_latest = text_features.sort_values("snapshot_id").groupby("product_id", as_index=False).tail(1)
        cols_join = [c for c in tf_latest.columns if c not in latest.columns or c == "product_id"]
        if cols_join:
            latest = latest.merge(
                tf_latest[cols_join],
                on="product_id",
                how="left",
            )
    return _prepare_category_frame(latest, latest[["snapshot_id", "product_id", "name_item"]] if "name_item" in latest.columns else None, latest_only=False)


def _category_level_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for level, column in (("l1", "category_label_l1"), ("l2", "category_label_l2"), ("l3", "category_label_l3")):
        if column not in frame.columns:
            continue
        scoped = frame.copy()
        scoped["category_level"] = level
        scoped["category_label"] = scoped[column].map(lambda value: _normalize_missing(value))
        frames.append(scoped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_category_overview(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    latest = _prepare_category_frame(fact_snapshots, text_features, latest_only=True)
    scoped = _category_level_rows(latest)
    if scoped.empty:
        return pd.DataFrame()
    total_products = max(int(scoped["product_id"].nunique()), 1)
    grouped = (
        scoped.groupby(["category_level", "category_label"], as_index=False)
        .agg(
            record_count=("product_id", "count"),
            product_count=("product_id", "nunique"),
            brand_count=("brand", "nunique"),
            avg_rank=("rank", "mean"),
            avg_momentum_score=("momentum_score", "mean"),
            avg_price=("price", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            fallback_count=("category_is_fallback", "sum"),
            raw_taxonomy_count=("category_has_raw_taxonomy", "sum"),
        )
    )
    grouped["share_of_catalog"] = (grouped["product_count"] / total_products) * 100.0
    return grouped.sort_values(["category_level", "product_count", "avg_rank"], ascending=[True, False, True]).reset_index(drop=True)


def build_category_relationships(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    latest = _prepare_category_frame(fact_snapshots, text_features, latest_only=True)
    scoped = _category_level_rows(latest)
    if scoped.empty:
        return pd.DataFrame()

    axis_specs = [
        ("price", "price_band", "price_band_label"),
        ("material", "material_normalized" if "material_normalized" in scoped.columns else "material", "attribute_value"),
        ("color", "color_normalized" if "color_normalized" in scoped.columns else "color", "attribute_value"),
    ]
    outputs: List[pd.DataFrame] = []
    for axis_name, source_column, target_column in axis_specs:
        if source_column not in scoped.columns:
            continue
        relation = scoped.copy()
        relation[target_column] = relation[source_column].map(lambda value: _normalize_missing(value)).astype(str)
        grouped = (
            relation.groupby(["category_level", "category_label", target_column], as_index=False)
            .agg(
                count=("product_id", "count"),
                avg_rank=("rank", "mean"),
                avg_price=("price", "mean"),
                avg_discount_pct=("discount_pct", "mean"),
                avg_momentum_score=("momentum_score", "mean"),
            )
        )
        grouped["relationship_axis"] = axis_name
        outputs.append(grouped)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def build_category_timeseries(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    scoped = _category_level_rows(_prepare_category_frame(fact_snapshots, text_features, latest_only=False))
    if scoped.empty or "snapshot_date" not in scoped.columns:
        return pd.DataFrame()
    grouped = (
        scoped.groupby(["snapshot_date", "category_level", "category_label"], as_index=False)
        .agg(
            record_count=("product_id", "count"),
            product_count=("product_id", "nunique"),
            avg_rank=("rank", "mean"),
            avg_momentum_score=("momentum_score", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            fallback_count=("category_is_fallback", "sum"),
        )
    )
    totals = grouped.groupby("snapshot_date", as_index=False).agg(total_product_count=("product_count", "sum"))
    grouped = grouped.merge(totals, on="snapshot_date", how="left")
    grouped["share_of_catalog"] = (
        grouped["product_count"] / grouped["total_product_count"].replace({0: pd.NA})
    ) * 100.0
    return grouped.sort_values(["category_level", "category_label", "snapshot_date"]).reset_index(drop=True)


def build_category_quality(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    latest = _prepare_category_frame(fact_snapshots, text_features, latest_only=True)
    scoped = _category_level_rows(latest)
    if scoped.empty:
        return pd.DataFrame()
    grouped = (
        scoped.groupby(["category_level", "category_label", "category_ingest_status"], as_index=False)
        .agg(
            product_count=("product_id", "nunique"),
            record_count=("product_id", "count"),
            fallback_count=("category_is_fallback", "sum"),
        )
    )
    totals = grouped.groupby(["category_level", "category_label"], as_index=False).agg(
        total_product_count=("product_count", "sum"),
        total_fallback_count=("fallback_count", "sum"),
    )
    grouped = grouped.merge(totals, on=["category_level", "category_label"], how="left")
    grouped["fallback_rate"] = (
        grouped["total_fallback_count"] / grouped["total_product_count"].replace({0: pd.NA})
    ) * 100.0
    return grouped.sort_values(["category_level", "category_label", "category_ingest_status"]).reset_index(drop=True)


def run_analysis_fusion(
    fact_snapshots: pd.DataFrame,
    text_features: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    fact_snapshots와 (선택) text_features로 분석 테이블 4종 생성.
    Returns:
        tag_performance, brand_index, trends, product_profile, category_overview, category_relationships, category_timeseries, category_quality
    """
    out: Dict[str, pd.DataFrame] = {}
    out["tag_performance"] = build_tag_performance(fact_snapshots)
    out["brand_index"] = build_brand_index(fact_snapshots)
    out["trends"] = build_trends(fact_snapshots)
    out["product_profile"] = build_product_profile(fact_snapshots, text_features)
    out["category_overview"] = build_category_overview(fact_snapshots, text_features)
    out["category_relationships"] = build_category_relationships(fact_snapshots, text_features)
    out["category_timeseries"] = build_category_timeseries(fact_snapshots, text_features)
    out["category_quality"] = build_category_quality(fact_snapshots, text_features)
    return out
