#!/usr/bin/env python3
"""
세그먼트 임베딩 생성 및 Qdrant 적재 유틸리티.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "product_image_segments"
    batch_size: int = 16
    quota_product_packshot: int = 1
    quota_detail_closeup: int = 1
    quota_model_wearing: int = 1


def _load_openclip(config: EmbeddingConfig):
    try:
        import open_clip
        import torch
    except Exception:
        return None, None, None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(config.model_name, pretrained=config.pretrained)
        model.eval()
        tokenizer = open_clip.get_tokenizer(config.model_name)
        logger.info("OpenCLIP 모델 로드 완료: %s (%s)", config.model_name, config.pretrained)
        return model, preprocess, tokenizer
    except Exception:
        logger.warning("OpenCLIP 모델 로드 실패")
        return None, None, None


def _crop_segment(segment: Dict) -> Optional[Image.Image]:
    image_path = Path(str(segment["image_path"]))
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            return img.crop((int(segment["x1"]), int(segment["y1"]), int(segment["x2"]), int(segment["y2"]))).copy()
    except Exception:
        return None


def generate_segment_embeddings(segments_df: pd.DataFrame, config: EmbeddingConfig) -> pd.DataFrame:
    if segments_df.empty:
        return pd.DataFrame()

    embedding_target = segments_df.get("embedding_target", pd.Series(True, index=segments_df.index)).fillna(False)
    embed_df = segments_df[embedding_target].copy()
    skip_df = segments_df[~embedding_target]

    def _skip_rows(segment_ids: pd.Series, status: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "segment_id": sid,
                    "embedding_status": status,
                    "embedding_dim": 0,
                    "embedding": None,
                    "vision_label": "unknown",
                    "vision_label_score": 0.0,
                }
                for sid in segment_ids.tolist()
            ]
        )

    logger.info(
        "임베딩 생성 시작: segments=%d (embedding_target=%d, 스킵=%d), unique_products=%d, unique_images=%d",
        len(segments_df),
        len(embed_df),
        len(skip_df),
        int(segments_df["product_id"].nunique()) if "product_id" in segments_df.columns else 0,
        int(segments_df["image_id"].nunique()) if "image_id" in segments_df.columns else 0,
    )

    if embed_df.empty:
        result = _skip_rows(segments_df["segment_id"], "not_target")
        logger.info("임베딩 생성 완료: embedding_target인 세그먼트 없음, rows=%d", len(result))
        return result

    model, preprocess, tokenizer = _load_openclip(config)
    if model is None:
        logger.warning("임베딩 생성 불가: 모델 없음")
        return pd.concat(
            [
                _skip_rows(skip_df["segment_id"], "not_target"),
                pd.DataFrame(
                    [
                        {
                            "segment_id": seg_id,
                            "embedding_status": "model_unavailable",
                            "embedding_dim": 0,
                            "embedding": None,
                            "vision_label": "unknown",
                            "vision_label_score": 0.0,
                        }
                        for seg_id in embed_df["segment_id"].tolist()
                    ]
                ),
            ],
            ignore_index=True,
        )

    import torch

    prompts = [
        "a product worn by a model",
        "a standalone product photo on plain background",
        "a closeup photo showing product material detail texture",
        "a collage lookbook fashion image",
        "a promotional banner with text",
        "a product specification text panel",
    ]
    label_map = {
        0: "model_wearing",
        1: "product_packshot",
        2: "detail_closeup",
        3: "lookbook_collage",
        4: "promo_banner",
        5: "spec_text_panel",
    }

    with torch.no_grad():
        text_tokens = tokenizer(prompts)
        text_feat = model.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    rows: List[Dict] = []
    total = len(embed_df)
    for idx_seg, seg in enumerate(embed_df.to_dict("records"), start=1):
        crop = _crop_segment(seg)
        if crop is None:
            rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "embedding_status": "crop_failed",
                    "embedding_dim": 0,
                    "embedding": None,
                    "vision_label": "unknown",
                    "vision_label_score": 0.0,
                }
            )
            continue

        try:
            image_tensor = preprocess(crop).unsqueeze(0)
            with torch.no_grad():
                img_feat = model.encode_image(image_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sims = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)[0].cpu().numpy()
                vec = img_feat[0].cpu().numpy().astype(np.float32)
        except Exception:
            rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "embedding_status": "encode_failed",
                    "embedding_dim": 0,
                    "embedding": None,
                    "vision_label": "unknown",
                    "vision_label_score": 0.0,
                }
            )
            continue

        label_idx = int(np.argmax(sims))
        rows.append(
            {
                "segment_id": seg["segment_id"],
                "embedding_status": "ok",
                "embedding_dim": int(vec.shape[0]),
                "embedding": vec.tolist(),
                "vision_label": label_map.get(label_idx, "unknown"),
                "vision_label_score": float(sims[label_idx]),
            }
        )
        if idx_seg % 50 == 0:
            logger.info("임베딩 진행: %d/%d", idx_seg, total)

    embed_result = pd.DataFrame(rows)
    skip_result = _skip_rows(skip_df["segment_id"], "not_target")
    result = pd.concat([skip_result, embed_result], ignore_index=True)
    status_count = result["embedding_status"].value_counts(dropna=False).to_dict() if not result.empty else {}
    logger.info("임베딩 생성 완료: rows=%d, status=%s", len(result), status_count)
    return result


def apply_embedding_quota(
    segments_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    config: EmbeddingConfig,
) -> pd.DataFrame:
    """
    제품별 타입 쿼터를 적용해 임베딩 적재 대상을 선별한다.
    """
    if embeddings_df.empty:
        return embeddings_df

    merged = embeddings_df.merge(
        segments_df[["segment_id", "snapshot_id", "product_id"]],
        on="segment_id",
        how="left",
    )
    merged["selected_for_embedding"] = False

    quota_map = {
        "product_packshot": int(config.quota_product_packshot),
        "detail_closeup": int(config.quota_detail_closeup),
        "model_wearing": int(config.quota_model_wearing),
    }

    valid = merged[merged["embedding_status"] == "ok"].copy()
    for label, q in quota_map.items():
        if q <= 0:
            continue
        label_df = valid[valid["vision_label"] == label].copy()
        if label_df.empty:
            continue

        selected_ids = (
            label_df.sort_values(["snapshot_id", "product_id", "vision_label_score"], ascending=[True, True, False])
            .groupby(["snapshot_id", "product_id"], as_index=False)
            .head(q)["segment_id"]
            .tolist()
        )
        merged.loc[merged["segment_id"].isin(selected_ids), "selected_for_embedding"] = True

    selected_cnt = int(merged["selected_for_embedding"].sum())
    logger.info(
        "임베딩 쿼터 적용 완료: selected=%d/%d, quota=%s",
        selected_cnt,
        len(merged),
        quota_map,
    )
    return merged


def upsert_embeddings_to_qdrant(
    merged_segments_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    config: EmbeddingConfig,
) -> Dict[str, str]:
    logger.info("Qdrant upsert 시작: collection=%s", config.qdrant_collection)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams
    except Exception:
        return {"status": "client_unavailable"}

    if embeddings_df.empty:
        return {"status": "no_embeddings"}

    payload_df = merged_segments_df.merge(embeddings_df, on="segment_id", how="inner")
    payload_df = payload_df[payload_df["embedding_status"] == "ok"]
    if "selected_for_embedding" in payload_df.columns:
        payload_df = payload_df[payload_df["selected_for_embedding"] == True]
    if payload_df.empty:
        return {"status": "no_valid_embeddings"}

    dim = int(payload_df.iloc[0]["embedding_dim"])
    client = QdrantClient(url=config.qdrant_url, timeout=60.0)
    try:
        client.get_collection(config.qdrant_collection)
    except Exception:
        client.recreate_collection(
            collection_name=config.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points: List[PointStruct] = []
    for i, row in enumerate(payload_df.to_dict("records"), start=1):
        vector = row.get("embedding")
        if not vector:
            continue
        payload = {
            "segment_id": row.get("segment_id"),
            "image_id": row.get("image_id"),
            "snapshot_id": row.get("snapshot_id"),
            "snapshot_date": row.get("snapshot_date"),
            "product_id": row.get("product_id"),
            "brand": row.get("brand"),
            "rank": int(row.get("rank")) if row.get("rank") is not None else None,
            "segment_type_rule": row.get("segment_type_rule"),
            "vision_label": row.get("vision_label"),
        }
        points.append(PointStruct(id=i, vector=vector, payload=payload))

    if not points:
        return {"status": "no_points"}

    # Qdrant 기본 요청 크기 한도(~33MB) 초과 방지: 청크 단위로 upsert
    upsert_batch_size = 500
    for start in range(0, len(points), upsert_batch_size):
        chunk = points[start : start + upsert_batch_size]
        client.upsert(collection_name=config.qdrant_collection, points=chunk, wait=True)
        logger.debug("Qdrant upsert 청크: %d~%d/%d", start, start + len(chunk), len(points))
    logger.info("Qdrant upsert 완료: points=%d (batch_size=%d)", len(points), upsert_batch_size)
    return {"status": "ok", "points": str(len(points))}

