#!/usr/bin/env python3
"""
브랜드 이미지 스타일 축: 문장 임베딩 + 프로토타입 유사도로 의도(클레임)·지각(리뷰) 집계.
문장 벡터는 스트리밍으로만 사용하고 전수 저장하지 않는다.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
AXES_PATH = Path(__file__).resolve().parent / "brand_style_axes.yaml"


@dataclass
class BrandStyleEmbeddingConfig:
    """파이프라인용 브랜드 스타일 임베딩 설정."""

    model_name: str = DEFAULT_MODEL_NAME
    alpha: float = 12.0
    batch_size: int = 64
    max_snippet_chars: int = 500
    evidence_top_n: int = 5
    axes_yaml_path: Path = field(default_factory=lambda: AXES_PATH)
    # True면 환경변수 SILHOUETTE_BRAND_STYLE_EMBED_STUB=1과 동일한 스텁 인코더 사용(테스트)
    use_stub_encoder: bool = False
    # 클레임/리뷰 루프에서 N행마다 진행 로그(INFO)
    progress_log_interval: int = 1000


def load_style_axes(yaml_path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    version = str(raw.get("version", "1"))
    axes = raw.get("axes") or []
    out: list[dict[str, Any]] = []
    for block in axes:
        aid = str(block.get("axis_id", "")).strip()
        if not aid:
            continue
        anchors = [str(t).strip() for t in (block.get("anchor_texts") or []) if str(t).strip()]
        out.append({
            "axis_id": aid,
            "label_ko": str(block.get("label_ko", aid)),
            "anchor_texts": anchors,
        })
    return out, version


def softmax_scores(scores: np.ndarray, alpha: float) -> np.ndarray:
    """scores: (K,) → softmax(alpha * scores), 수치 안정화."""
    if scores.size == 0:
        return scores
    scaled = alpha * np.asarray(scores, dtype=np.float64)
    scaled = scaled - np.max(scaled)
    ex = np.exp(scaled)
    s = float(np.sum(ex))
    if s <= 0 or not math.isfinite(s):
        n = len(scores)
        return np.ones(n, dtype=np.float64) / max(n, 1)
    return (ex / s).astype(np.float64)


def _stub_encode_factory(dim: int = 64) -> Callable[[list[str]], np.ndarray]:
    """결정적 단위 벡터(테스트·CI용)."""

    def _encode(texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hash((t, i)) % (2**32)
            rng.seed(h)
            v = rng.randn(dim).astype(np.float32)
            n = np.linalg.norm(v)
            if n > 1e-9:
                v = v / n
            out[i] = v
        return out

    return _encode


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers가 필요합니다. pip install sentence-transformers") from e
    logger.info("brand_style_embedding: SentenceTransformer 다운로드/로드 시작 model=%s", model_name)
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    logger.info("brand_style_embedding: SentenceTransformer 로드 완료 model=%s elapsed=%.1fs", model_name, time.perf_counter() - t0)
    return model


def _sentence_transformer_embedding_dim(model: Any) -> int:
    fn = getattr(model, "get_embedding_dimension", None)
    if callable(fn):
        return int(fn())
    return int(model.get_sentence_embedding_dimension())


def build_encode_fn(
    config: BrandStyleEmbeddingConfig,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """(encode_fn, embedding_dim) 반환. normalize_embeddings=True."""
    stub = config.use_stub_encoder or os.environ.get("SILHOUETTE_BRAND_STYLE_EMBED_STUB", "").strip() == "1"
    if stub:
        logger.info("brand_style_embedding: 스텁 인코더 사용(dim=64)")
        fn = _stub_encode_factory(64)
        return fn, 64

    model = _load_sentence_transformer(config.model_name)
    dim = _sentence_transformer_embedding_dim(model)
    logger.info("brand_style_embedding: 인코더 준비 완료 embedding_dim=%d batch_size=%d", dim, config.batch_size)

    def _encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        emb = model.encode(
            texts,
            batch_size=config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    return _encode, dim


def encode_anchor_prototypes(
    axes: list[dict[str, Any]],
    encode: Callable[[list[str]], np.ndarray],
    embedding_dim: int,
) -> np.ndarray:
    """축별 앵커 문장을 인코딩해 프로토타입 행렬 (K, d) 생성."""
    logger.info("brand_style_embedding: 앵커 프로토타입 인코딩 시작 (축 %d개)", len(axes))
    t0 = time.perf_counter()
    protos: list[np.ndarray] = []
    for ax in axes:
        anchors = [t for t in (ax.get("anchor_texts") or []) if str(t).strip()]
        if not anchors:
            protos.append(np.zeros(embedding_dim, dtype=np.float32))
            continue
        aid = str(ax.get("axis_id", ""))
        logger.info("brand_style_embedding:   축 %s 앵커 %d문장 인코딩", aid, len(anchors))
        embs = encode(anchors)
        if embs.size == 0:
            protos.append(np.zeros(embedding_dim, dtype=np.float32))
            continue
        m = np.mean(embs.astype(np.float64), axis=0)
        nrm = np.linalg.norm(m)
        if nrm > 1e-12:
            m = m / nrm
        protos.append(m.astype(np.float32))
    if not protos:
        return np.zeros((0, embedding_dim), dtype=np.float32)
    mat = np.stack(protos, axis=0)
    logger.info(
        "brand_style_embedding: 앵커 프로토타입 완료 shape=%s elapsed=%.2fs",
        mat.shape,
        time.perf_counter() - t0,
    )
    return mat


def _clip_snippet(text: str, max_chars: int) -> str:
    t = text.replace("\u200b", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def build_brand_style_embedding_artifacts(
    claim_df: pd.DataFrame,
    review_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[BrandStyleEmbeddingConfig] = None,
) -> dict[str, Any]:
    """
    클레임·리뷰 DataFrame으로 집계 parquet·메타·근거(선택) 생성.
    반환: {"agg": df, "evidence": df, "meta": dict} — 실패 시 빈 프레임.
    """
    cfg = config or BrandStyleEmbeddingConfig()
    empty_agg = pd.DataFrame(
        columns=[
            "snapshot_id",
            "brand",
            "axis_id",
            "intent_raw",
            "perceived_raw",
        ],
    )
    empty_ev = pd.DataFrame(
        columns=[
            "snapshot_id",
            "brand",
            "axis_id",
            "source",
            "rank",
            "snippet",
            "contrib_score",
            "product_id",
        ],
    )

    if claim_df.empty and review_df.empty:
        return {"agg": empty_agg, "evidence": empty_ev, "meta": {"status": "skipped", "reason": "no_input"}}

    try:
        axes, axis_ver = load_style_axes(cfg.axes_yaml_path)
    except Exception as e:
        logger.warning("brand_style_axes.yaml 로드 실패: %s", e)
        return {"agg": empty_agg, "evidence": empty_ev, "meta": {"status": "error", "reason": str(e)}}

    if not axes:
        return {"agg": empty_agg, "evidence": empty_ev, "meta": {"status": "skipped", "reason": "no_axes"}}

    axis_ids = [a["axis_id"] for a in axes]
    K = len(axis_ids)
    try:
        encode_fn, dim = build_encode_fn(cfg)
    except Exception as e:
        logger.warning("임베딩 인코더 로드 실패: %s", e)
        return {"agg": empty_agg, "evidence": empty_ev, "meta": {"status": "error", "reason": str(e)}}

    P = encode_anchor_prototypes(axes, encode_fn, dim)
    if P.shape[0] != K:
        return {"agg": empty_agg, "evidence": empty_ev, "meta": {"status": "error", "reason": "prototype_shape"}}

    log_iv = max(1, int(cfg.progress_log_interval))

    # (K, d) @ (d,) = (K,) cosine sims if v unit
    def text_to_pi(text: str) -> Optional[np.ndarray]:
        if not text or not str(text).strip():
            return None
        v = encode_fn([_clip_snippet(str(text), cfg.max_snippet_chars)])
        if v.shape[0] != 1:
            return None
        s = (P @ v[0].astype(np.float64)).astype(np.float64)
        return softmax_scores(s, cfg.alpha)

    # intent: (snapshot_id, brand) -> vector length K
    intent_acc: Dict[Tuple[Any, str], np.ndarray] = {}
    perc_acc: Dict[Tuple[Any, str], np.ndarray] = {}

    def _add_vec(store: dict, key: Tuple[Any, str], vec: np.ndarray, w: float) -> None:
        if key not in store:
            store[key] = np.zeros(K, dtype=np.float64)
        store[key] += w * vec

    # evidence heaps: key (snapshot_id, brand, axis_id, source) -> list of (-score, snippet, product_id)
    ev_heaps: Dict[Tuple[Any, str, str, str], list] = {}

    def _push_ev(
        snap: Any,
        brand: str,
        axis_idx: int,
        source: str,
        score: float,
        snippet: str,
        product_id: Any,
    ) -> None:
        if score <= 0 or not math.isfinite(score):
            return
        key = (snap, brand, axis_ids[axis_idx], source)
        h = ev_heaps.setdefault(key, [])
        sn = _clip_snippet(snippet, 200)
        pid = str(product_id or "")
        if len(h) < cfg.evidence_top_n:
            heapq.heappush(h, (score, sn, pid))
        elif score > h[0][0]:
            heapq.heapreplace(h, (score, sn, pid))

    if not claim_df.empty and {"brand", "claim_text", "snapshot_id"}.issubset(claim_df.columns):
        claim_total = int(len(claim_df))
        claim_encoded = 0
        logger.info(
            "brand_style_embedding: 클레임 임베딩 루프 시작 (행 %d, 로그 간격 %d행)",
            claim_total,
            log_iv,
        )
        t_claim = time.perf_counter()
        for idx, row in enumerate(claim_df.itertuples(index=False), start=1):
            if idx == 1 or idx % log_iv == 0:
                logger.info(
                    "brand_style_embedding: 클레임 진행 %d/%d행 (누적 인코딩 %d건)",
                    idx,
                    claim_total,
                    claim_encoded,
                )
            brand = getattr(row, "brand", None)
            if brand is None or (isinstance(brand, float) and math.isnan(brand)):
                continue
            b = str(brand).strip()
            if not b:
                continue
            text = str(getattr(row, "claim_text", "") or "").strip()
            if not text:
                continue
            pi = text_to_pi(text)
            if pi is None:
                continue
            claim_encoded += 1
            conf = float(getattr(row, "confidence", 1.0) or 1.0)
            if not math.isfinite(conf):
                conf = 1.0
            snap = getattr(row, "snapshot_id", None)
            pid = getattr(row, "product_id", None)
            _add_vec(intent_acc, (snap, b), pi, conf)
            for ki in range(K):
                _push_ev(snap, b, ki, "intent", float(conf * pi[ki]), text, pid)
        logger.info(
            "brand_style_embedding: 클레임 루프 완료 행=%d 실제인코딩=%d elapsed=%.1fs",
            claim_total,
            claim_encoded,
            time.perf_counter() - t_claim,
        )

    if not review_df.empty and {"brand", "sentence_text", "snapshot_id"}.issubset(review_df.columns):
        review_total = int(len(review_df))
        review_encoded = 0
        logger.info(
            "brand_style_embedding: 리뷰 문장 임베딩 루프 시작 (행 %d, 로그 간격 %d행)",
            review_total,
            log_iv,
        )
        t_rev = time.perf_counter()
        for idx, row in enumerate(review_df.itertuples(index=False), start=1):
            if idx == 1 or idx % log_iv == 0:
                logger.info(
                    "brand_style_embedding: 리뷰 진행 %d/%d행 (누적 인코딩 %d건)",
                    idx,
                    review_total,
                    review_encoded,
                )
            brand = getattr(row, "brand", None)
            if brand is None or (isinstance(brand, float) and math.isnan(brand)):
                continue
            b = str(brand).strip()
            if not b:
                continue
            text = str(getattr(row, "sentence_text", "") or "").strip()
            if not text:
                continue
            pi = text_to_pi(text)
            if pi is None:
                continue
            review_encoded += 1
            w = 1.0
            snap = getattr(row, "snapshot_id", None)
            pid = getattr(row, "product_id", None)
            _add_vec(perc_acc, (snap, b), pi, w)
            for ki in range(K):
                _push_ev(snap, b, ki, "perceived", float(w * pi[ki]), text, pid)
        logger.info(
            "brand_style_embedding: 리뷰 루프 완료 행=%d 실제인코딩=%d elapsed=%.1fs",
            review_total,
            review_encoded,
            time.perf_counter() - t_rev,
        )

    n_keys = len(set(intent_acc.keys()) | set(perc_acc.keys()))
    logger.info("brand_style_embedding: 집계 행 생성 중 (브랜드×스냅샷 키 %d개)", n_keys)
    all_keys = set(intent_acc.keys()) | set(perc_acc.keys())
    agg_rows: list[dict[str, Any]] = []
    for key in all_keys:
        snap, br = key
        iv = intent_acc.get(key)
        pv = perc_acc.get(key)
        for ki, aid in enumerate(axis_ids):
            agg_rows.append({
                "snapshot_id": snap,
                "brand": br,
                "axis_id": aid,
                "intent_raw": float(iv[ki]) if iv is not None else 0.0,
                "perceived_raw": float(pv[ki]) if pv is not None else 0.0,
            })

    agg_df = pd.DataFrame(agg_rows) if agg_rows else empty_agg

    ev_rows: list[dict[str, Any]] = []
    for (snap, br, aid, src), heap in ev_heaps.items():
        for rank, (sc, snip, pid) in enumerate(sorted(heap, key=lambda x: -x[0]), start=1):
            ev_rows.append({
                "snapshot_id": snap,
                "brand": br,
                "axis_id": aid,
                "source": src,
                "rank": rank,
                "snippet": snip,
                "contrib_score": round(float(sc), 6),
                "product_id": pid or None,
            })
    ev_df = pd.DataFrame(ev_rows) if ev_rows else empty_ev

    meta = {
        "status": "ok",
        "model_name": cfg.model_name,
        "embedding_dim": dim,
        "alpha": cfg.alpha,
        "axis_version": axis_ver,
        "axes_yaml": str(cfg.axes_yaml_path),
        "stub": bool(cfg.use_stub_encoder or os.environ.get("SILHOUETTE_BRAND_STYLE_EMBED_STUB", "").strip() == "1"),
    }

    logger.info("brand_style_embedding: Parquet·메타 저장 중 output_dir=%s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "brand_style_embedding_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not agg_df.empty:
        agg_df.to_parquet(output_dir / "brand_style_embedding_agg.parquet", index=False)
    if not ev_df.empty:
        ev_df.to_parquet(output_dir / "brand_style_embedding_evidence.parquet", index=False)

    logger.info(
        "brand_style_embedding: agg_rows=%d evidence_rows=%d model=%s",
        len(agg_df),
        len(ev_df),
        cfg.model_name,
    )
    return {"agg": agg_df, "evidence": ev_df, "meta": meta}


def label_ko_for_axis(axes: list[dict[str, Any]], axis_id: str) -> str:
    for a in axes:
        if a["axis_id"] == axis_id:
            return str(a.get("label_ko", axis_id))
    return axis_id
