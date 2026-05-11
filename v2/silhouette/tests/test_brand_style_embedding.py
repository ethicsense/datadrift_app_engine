"""브랜드 스타일 임베딩 집계 단위 테스트(스텁 인코더, 네트워크 없음)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
os.environ["SILHOUETTE_BRAND_STYLE_EMBED_STUB"] = "1"

from analytics.pipeline.brand_style_embedding import (
    BrandStyleEmbeddingConfig,
    build_brand_style_embedding_artifacts,
    encode_anchor_prototypes,
    load_style_axes,
    softmax_scores,
)


def test_softmax_sums_to_one() -> None:
    s = np.array([0.1, 0.5, -0.2], dtype=np.float64)
    p = softmax_scores(s, alpha=10.0)
    assert abs(float(np.sum(p)) - 1.0) < 1e-6
    assert p.shape == s.shape


def test_load_axes() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "analytics" / "pipeline" / "brand_style_axes.yaml"
    axes, ver = load_style_axes(yaml_path)
    assert ver
    assert len(axes) >= 8
    assert all("axis_id" in a and a.get("anchor_texts") for a in axes)


def test_prototype_matrix_shape() -> None:
    from analytics.pipeline.brand_style_embedding import build_encode_fn

    cfg = BrandStyleEmbeddingConfig(use_stub_encoder=True)
    encode_fn, dim = build_encode_fn(cfg)
    yaml_path = Path(__file__).resolve().parents[1] / "analytics" / "pipeline" / "brand_style_axes.yaml"
    axes, _ = load_style_axes(yaml_path)
    P = encode_anchor_prototypes(axes, encode_fn, dim)
    assert P.shape == (len(axes), dim)
    norms = np.linalg.norm(P, axis=1)
    for i, n in enumerate(norms):
        if np.any(P[i] != 0):
            assert abs(float(n) - 1.0) < 0.01 or n < 0.01


def test_build_artifacts_stub(tmp_path: Path) -> None:
    claim_df = pd.DataFrame(
        [
            {
                "snapshot_id": "s1",
                "brand": "BrandA",
                "claim_text": "미니멀한 실루엣이 좋다.",
                "confidence": 0.9,
                "product_id": "p1",
            }
        ]
    )
    review_df = pd.DataFrame(
        [
            {
                "snapshot_id": "s1",
                "brand": "BrandA",
                "sentence_text": "스트릿 무드가 난다.",
                "product_id": "p1",
            }
        ]
    )
    cfg = BrandStyleEmbeddingConfig(use_stub_encoder=True, axes_yaml_path=Path(__file__).resolve().parents[1] / "analytics/pipeline/brand_style_axes.yaml")
    out = build_brand_style_embedding_artifacts(claim_df, review_df, tmp_path, cfg)
    assert out["meta"].get("status") == "ok"
    agg = out["agg"]
    assert not agg.empty
    assert set(agg["brand"].unique()) == {"BrandA"}
    assert (tmp_path / "brand_style_embedding_agg.parquet").exists()
    assert (tmp_path / "brand_style_embedding_meta.json").exists()
