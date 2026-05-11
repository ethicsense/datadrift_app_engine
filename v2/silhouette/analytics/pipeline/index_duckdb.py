#!/usr/bin/env python3
"""
Parquet 산출물을 DuckDB 인덱스로 연결한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def build_duckdb_index(output_dir: Path) -> Dict[str, str]:
    try:
        import duckdb
    except Exception:
        return {"status": "duckdb_unavailable"}

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "analytics.duckdb"
    con = duckdb.connect(str(db_path))

    parquet_map = {
        "fact_snapshots": "fact_snapshots.parquet",
        "raw_snapshot_products": "raw_snapshot_products.parquet",
        "product_latest": "product_latest.parquet",
        "dim_products": "dim_products.parquet",
        "product_snapshot_coverage": "product_snapshot_coverage.parquet",
        "image_manifest": "image_manifest.parquet",
        "image_segments": "image_segments.parquet",
        "image_embeddings": "image_embeddings.parquet",
        "text_features": "text_features.parquet",
        "text_review_facts": "text_review_facts.parquet",
        "text_claim_facts": "text_claim_facts.parquet",
        "text_gap_metrics": "text_gap_metrics.parquet",
        "text_fusion_profile": "text_fusion_profile.parquet",
        "text_trend_keywords": "text_trend_keywords.parquet",
        "brand_style_embedding_agg": "brand_style_embedding_agg.parquet",
        "brand_style_embedding_evidence": "brand_style_embedding_evidence.parquet",
        "analysis_tag_performance": "analysis_tag_performance.parquet",
        "analysis_brand_index": "analysis_brand_index.parquet",
        "analysis_trends": "analysis_trends.parquet",
        "analysis_product_profile": "analysis_product_profile.parquet",
        "analysis_category_overview": "analysis_category_overview.parquet",
        "analysis_category_relationships": "analysis_category_relationships.parquet",
        "analysis_category_timeseries": "analysis_category_timeseries.parquet",
        "analysis_category_quality": "analysis_category_quality.parquet",
        "analysis_embedding_projection": "analysis_embedding_projection.parquet",
        "analysis_rank_trajectories": "analysis_rank_trajectories.parquet",
        "analysis_rank_race": "analysis_rank_race.parquet",
    }

    created = 0
    for table_name, parquet_name in parquet_map.items():
        parquet_path = output_dir / parquet_name
        if not parquet_path.exists():
            continue
        try:
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
            )
            created += 1
        except Exception as e:
            # 빈 Parquet(컬럼 없음 등)는 DuckDB read_parquet에서 실패할 수 있음
            logger.warning("Parquet 로드 스킵: %s (%s)", parquet_name, e)

    con.close()
    return {"status": "ok", "tables": str(created), "db_path": str(db_path)}

