#!/usr/bin/env python3
"""
시각화 전용 파생 산출물 생성 유틸리티.
애니메이션 차트에서 바로 쓸 수 있는 시계열 프레임 데이터를 만든다.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _project_embeddings(vectors: List[List[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 2), dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return np.column_stack([centered[:, 0], np.zeros(centered.shape[0], dtype=np.float32)])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    coords = centered @ basis
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=np.float32)])
    return coords[:, :2]


def _movement_group(rank_velocity: float) -> str:
    if pd.isna(rank_velocity):
        return "no_prior_rank"
    if rank_velocity >= 3:
        return "rank_up_fast"
    if rank_velocity > 0:
        return "rank_up"
    if rank_velocity <= -3:
        return "rank_down_fast"
    if rank_velocity < 0:
        return "rank_down"
    return "rank_unchanged"


def build_embedding_projection(
    fact_snapshots: pd.DataFrame,
    segments_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    임베딩 벡터를 2차원 좌표로 투영해 프레임 기반 스캐터 애니메이션용 데이터를 만든다.
    """
    if fact_snapshots.empty or segments_df.empty or embeddings_df.empty:
        return pd.DataFrame()

    enriched = embeddings_df.copy()
    if "selected_for_embedding" in enriched.columns:
        enriched = enriched[enriched["selected_for_embedding"].fillna(False)]
    if "embedding_status" in enriched.columns:
        enriched = enriched[enriched["embedding_status"] == "ok"]
    if enriched.empty or "embedding" not in enriched.columns:
        return pd.DataFrame()

    segment_metadata_columns = [
        "segment_id",
        "snapshot_id",
        "snapshot_date",
        "snapshot_time",
        "product_id",
        "brand",
        "rank",
        "image_path",
    ]
    segment_join_columns = [
        column
        for column in segment_metadata_columns
        if column == "segment_id" or column not in enriched.columns
    ]
    enriched = enriched.merge(
        segments_df[segment_join_columns],
        on="segment_id",
        how="left",
    )
    metadata = fact_snapshots[
        [
            "snapshot_id",
            "product_id",
            "crawl_datetime",
            "name",
            "rank_velocity",
            "momentum_score",
            "rank_energy",
            "energy_velocity",
            "energy_acceleration",
            "consistency_score",
            "momentum_event_state",
            "momentum_event_label",
            "discount_pct",
            "price",
        ]
    ].drop_duplicates(["snapshot_id", "product_id"])
    enriched = enriched.merge(metadata, on=["snapshot_id", "product_id"], how="left")
    enriched = enriched[enriched["embedding"].map(lambda value: isinstance(value, Iterable) and len(value) > 0)]
    if enriched.empty:
        return pd.DataFrame()

    coords = _project_embeddings(enriched["embedding"].tolist())
    projection = enriched.copy()
    projection["x"] = coords[:, 0]
    projection["y"] = coords[:, 1]
    projection["crawl_datetime"] = pd.to_datetime(projection["crawl_datetime"], errors="coerce")
    projection["cluster_id"] = projection.get("vision_label", pd.Series(index=projection.index, dtype=str)).fillna("unknown")
    projection["movement_group"] = projection["rank_velocity"].map(_movement_group)
    projection["frame_label"] = projection["crawl_datetime"].dt.strftime("%Y-%m-%d %H:%M").fillna(projection["snapshot_id"])
    columns = [
        "snapshot_id",
        "snapshot_date",
        "snapshot_time",
        "crawl_datetime",
        "frame_label",
        "product_id",
        "brand",
        "name",
        "rank",
        "rank_velocity",
        "momentum_score",
        "rank_energy",
        "energy_velocity",
        "energy_acceleration",
        "consistency_score",
        "momentum_event_state",
        "momentum_event_label",
        "discount_pct",
        "price",
        "vision_label",
        "vision_label_score",
        "cluster_id",
        "movement_group",
        "image_path",
        "x",
        "y",
    ]
    existing = [column for column in columns if column in projection.columns]
    return projection[existing].sort_values(["crawl_datetime", "rank", "product_id"]).reset_index(drop=True)


def build_rank_trajectories(fact_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    상품/브랜드 단위 순위 이동 애니메이션용 시계열 데이터를 만든다.
    """
    if fact_snapshots.empty:
        return pd.DataFrame()

    fact = fact_snapshots.copy()
    fact["crawl_datetime"] = pd.to_datetime(fact["crawl_datetime"], errors="coerce")

    product_rows = fact[
        [
            "snapshot_id",
            "crawl_datetime",
            "product_id",
            "brand",
            "name",
            "rank",
            "rank_velocity",
            "momentum_score",
            "rank_energy",
            "energy_velocity",
            "energy_acceleration",
            "consistency_score",
            "momentum_event_state",
            "momentum_event_label",
            "discount_pct",
        ]
    ].copy()
    product_rows["entity_type"] = "product"
    product_rows["entity_id"] = product_rows["product_id"].astype(str)
    product_rows["entity_label"] = product_rows["name"].fillna(product_rows["product_id"].astype(str))
    product_rows["rank_delta"] = product_rows["rank_velocity"]
    product_rows["record_count"] = 1

    brand_rows = (
        fact.groupby(["snapshot_id", "crawl_datetime", "brand"], as_index=False)
        .agg(
            rank=("rank", "mean"),
            rank_delta=("rank_velocity", "mean"),
            momentum_score=("momentum_score", "mean"),
            rank_energy=("rank_energy", "mean"),
            energy_velocity=("energy_velocity", "mean"),
            energy_acceleration=("energy_acceleration", "mean"),
            consistency_score=("consistency_score", "mean"),
            discount_pct=("discount_pct", "mean"),
            record_count=("product_id", "count"),
        )
    )
    brand_rows["entity_type"] = "brand"
    brand_rows["entity_id"] = brand_rows["brand"].astype(str)
    brand_rows["entity_label"] = brand_rows["brand"].astype(str)
    brand_rows["product_id"] = None
    brand_rows["name"] = None

    combined = pd.concat(
        [
            product_rows[
                [
                    "snapshot_id",
                    "crawl_datetime",
                    "entity_type",
                    "entity_id",
                    "entity_label",
                    "brand",
                    "product_id",
                    "name",
                    "rank",
                    "rank_delta",
                    "momentum_score",
                    "rank_energy",
                    "energy_velocity",
                    "energy_acceleration",
                    "consistency_score",
                    "momentum_event_state",
                    "momentum_event_label",
                    "discount_pct",
                    "record_count",
                ]
            ],
            brand_rows[
                [
                    "snapshot_id",
                    "crawl_datetime",
                    "entity_type",
                    "entity_id",
                    "entity_label",
                    "brand",
                    "product_id",
                    "name",
                    "rank",
                    "rank_delta",
                    "momentum_score",
                    "rank_energy",
                    "energy_velocity",
                    "energy_acceleration",
                    "consistency_score",
                    "discount_pct",
                    "record_count",
                ]
            ],
        ],
        ignore_index=True,
    )
    combined["frame_label"] = combined["crawl_datetime"].dt.strftime("%Y-%m-%d %H:%M").fillna(combined["snapshot_id"])
    return combined.sort_values(["entity_type", "crawl_datetime", "rank", "entity_id"]).reset_index(drop=True)


def build_rank_race(fact_snapshots: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    프레임별 상위 N개 순위 경쟁(bar race)용 데이터를 만든다.
    """
    trajectories = build_rank_trajectories(fact_snapshots)
    if trajectories.empty:
        return trajectories

    race_rows: List[pd.DataFrame] = []
    for entity_type in ("product", "brand"):
        scoped = trajectories[trajectories["entity_type"] == entity_type].copy()
        if scoped.empty:
            continue
        race_rows.append(
            scoped.sort_values(["crawl_datetime", "rank", "entity_id"])
            .groupby("snapshot_id", as_index=False)
            .head(top_n)
        )
    if not race_rows:
        return pd.DataFrame()
    return pd.concat(race_rows, ignore_index=True).sort_values(
        ["entity_type", "crawl_datetime", "rank", "entity_id"]
    ).reset_index(drop=True)
