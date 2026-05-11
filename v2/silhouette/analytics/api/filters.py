from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from fastapi import Query


@dataclass
class DashboardFilters:
    dataset: Optional[str] = None
    brands: list[str] = None
    source_datasets: list[str] = None
    platforms: list[str] = None
    schema_versions: list[str] = None
    snapshot_window: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    def __post_init__(self) -> None:
        if self.brands is None:
            self.brands = []
        if self.source_datasets is None:
            self.source_datasets = []
        if self.platforms is None:
            self.platforms = []
        if self.schema_versions is None:
            self.schema_versions = []


def parse_dashboard_filters(
    dataset: str | None = Query(default=None),
    brands: list[str] = Query(default_factory=list),
    source_datasets: list[str] = Query(default_factory=list),
    platforms: list[str] = Query(default_factory=list),
    schema_versions: list[str] = Query(default_factory=list),
    snapshot_window: int | None = Query(default=None, ge=1, le=5000),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> DashboardFilters:
    return DashboardFilters(
        dataset=dataset,
        brands=brands,
        source_datasets=source_datasets,
        platforms=platforms,
        schema_versions=schema_versions,
        snapshot_window=snapshot_window,
        date_from=date_from,
        date_to=date_to,
    )


def apply_fact_filters(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    filtered = df.copy()
    if "crawl_datetime" in filtered.columns:
        filtered["crawl_datetime"] = pd.to_datetime(filtered["crawl_datetime"], errors="coerce")

    if filters.brands and "brand" in filtered.columns:
        filtered = filtered[filtered["brand"].astype(str).isin(filters.brands)]

    if filters.source_datasets and "source_dataset" in filtered.columns:
        filtered = filtered[filtered["source_dataset"].astype(str).isin(filters.source_datasets)]

    if filters.platforms and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"].astype(str).isin(filters.platforms)]

    if filters.schema_versions and "schema_version" in filtered.columns:
        filtered = filtered[filtered["schema_version"].astype(str).isin(filters.schema_versions)]

    if filters.date_from:
        if "snapshot_date" in filtered.columns:
            filtered = filtered[filtered["snapshot_date"].astype(str) >= filters.date_from]
        elif "crawl_datetime" in filtered.columns:
            filtered = filtered[filtered["crawl_datetime"] >= pd.to_datetime(filters.date_from, errors="coerce")]

    if filters.date_to:
        if "snapshot_date" in filtered.columns:
            filtered = filtered[filtered["snapshot_date"].astype(str) <= filters.date_to]
        elif "crawl_datetime" in filtered.columns:
            filtered = filtered[filtered["crawl_datetime"] <= pd.to_datetime(filters.date_to, errors="coerce")]

    if filters.snapshot_window and {"snapshot_id", "crawl_datetime"}.issubset(filtered.columns):
        snapshot_ids = (
            filtered[["snapshot_id", "crawl_datetime"]]
            .drop_duplicates()
            .sort_values("crawl_datetime")
            .tail(filters.snapshot_window)["snapshot_id"]
            .tolist()
        )
        filtered = filtered[filtered["snapshot_id"].isin(snapshot_ids)]

    return filtered.reset_index(drop=True)


def filter_related_table(table_df: pd.DataFrame, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
    if table_df.empty or filtered_fact_df.empty:
        return table_df.iloc[0:0].copy()

    scoped = table_df.copy()
    if {"snapshot_id", "product_id"}.issubset(scoped.columns):
        keys = filtered_fact_df[["snapshot_id", "product_id"]].drop_duplicates()
        by_product = scoped[scoped["product_id"].notna()].merge(keys, on=["snapshot_id", "product_id"], how="inner")

        aggregate_rows = scoped[scoped["product_id"].isna()].copy()
        if not aggregate_rows.empty:
            if "brand" in aggregate_rows.columns and "brand" in filtered_fact_df.columns:
                brand_keys = filtered_fact_df[["snapshot_id", "brand"]].drop_duplicates()
                aggregate_rows = aggregate_rows.merge(brand_keys, on=["snapshot_id", "brand"], how="inner")
            else:
                snapshot_keys = filtered_fact_df[["snapshot_id"]].drop_duplicates()
                aggregate_rows = aggregate_rows.merge(snapshot_keys, on="snapshot_id", how="inner")
        return pd.concat([by_product, aggregate_rows], ignore_index=True)
    if "product_id" in scoped.columns:
        product_ids = filtered_fact_df["product_id"].astype(str).unique().tolist()
        return scoped[scoped["product_id"].astype(str).isin(product_ids)].reset_index(drop=True)
    if "brand" in scoped.columns:
        brands = filtered_fact_df["brand"].dropna().astype(str).unique().tolist()
        return scoped[scoped["brand"].astype(str).isin(brands)].reset_index(drop=True)
    return scoped
