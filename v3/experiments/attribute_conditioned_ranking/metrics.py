"""Coarse cluster metrics and graded color-neighbor metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
GARMENT_DIR = HERE.parent / "garment_segmentation"
if str(GARMENT_DIR) not in sys.path:
    sys.path.insert(0, str(GARMENT_DIR))

from cluster_quality import (  # noqa: E402
    _paired_bootstrap_delta,
    _per_item_purity,
    _r_precision_map,
)
from color_space import delta_e00
from controls import l2_normalize

from protocol import NEIGHBOR_K


def _topk(sim: np.ndarray, k: int) -> np.ndarray:
    n = sim.shape[0]
    kk = min(k, n - 1)
    part = np.argpartition(-sim, kth=kk, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    order = np.argsort(-sim[row, part], axis=1)
    return np.take_along_axis(part, order, axis=1)


def _similarity(x: np.ndarray) -> np.ndarray:
    unit = l2_normalize(x)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def coarse_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = NEIGHBOR_K,
) -> dict[str, Any]:
    y = np.asarray(labels)
    per_item = _per_item_purity(embeddings, y, k)
    purity = float(per_item.mean())
    r_precision, map_at_r = _r_precision_map(l2_normalize(embeddings), y)
    per_class: dict[str, float] = {}
    for label in sorted(set(y.tolist())):
        per_class[str(label)] = float(per_item[y == label].mean())
    return {
        "n": int(len(y)),
        "purity_at_k": purity,
        "r_precision": float(r_precision),
        "map_at_r": float(map_at_r),
        "per_class_purity": per_class,
        "per_item_purity": per_item,
    }


def graded_color_metrics(
    embeddings: np.ndarray,
    lab: np.ndarray,
    de: np.ndarray,
    *,
    k: int,
    tau: float,
    near_de: float,
    far_de: float,
    triplet_pairs: int,
    seed: int,
    fine_labels: np.ndarray,
    min_class_n: int,
) -> dict[str, Any]:
    sim = _similarity(embeddings)
    top = _topk(sim, k)
    row = np.arange(len(sim))[:, None]
    neighbor_de = de[row, top]
    relevance = np.exp(-neighbor_de / tau)
    weighted = relevance.mean(axis=1)
    mean_de = neighbor_de.mean(axis=1)
    ndcg = _ndcg_at_k(sim, de, k, tau)
    triplet = _triplet_accuracy(
        sim,
        de,
        near_de=near_de,
        far_de=far_de,
        pairs=triplet_pairs,
        seed=seed,
    )
    return {
        "n": int(len(sim)),
        "weighted_purity_at_k": weighted,
        "mean_delta_e_at_k": mean_de,
        "ndcg_at_k": ndcg,
        "triplet_accuracy": triplet,
        "centroid_spearman": _centroid_spearman(embeddings, lab, fine_labels, min_class_n),
    }


def _ndcg_at_k(sim: np.ndarray, de: np.ndarray, k: int, tau: float) -> np.ndarray:
    rel = np.exp(-de / tau)
    np.fill_diagonal(rel, 0.0)
    kk = min(k, sim.shape[0] - 1)
    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    top = _topk(sim, kk)
    row = np.arange(len(sim))[:, None]
    dcg = (rel[row, top] * discounts).sum(axis=1)
    ideal_part = np.argpartition(-rel, kth=kk, axis=1)[:, :kk]
    ideal = np.take_along_axis(ideal_part, np.argsort(-rel[row, ideal_part], axis=1), axis=1)
    idcg = (rel[row, ideal] * discounts).sum(axis=1)
    return dcg / np.clip(idcg, 1e-12, None)


def _triplet_accuracy(
    sim: np.ndarray,
    de: np.ndarray,
    *,
    near_de: float,
    far_de: float,
    pairs: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = sim.shape[0]
    scores = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        near = np.flatnonzero((de[i] <= near_de) & (np.arange(n) != i))
        far = np.flatnonzero(de[i] >= far_de)
        if len(near) == 0 or len(far) == 0:
            continue
        take = min(pairs, len(near) * len(far))
        near_idx = rng.choice(near, size=take, replace=True)
        far_idx = rng.choice(far, size=take, replace=True)
        scores[i] = float(np.mean(sim[i, near_idx] > sim[i, far_idx]))
    return scores


def _centroid_spearman(
    embeddings: np.ndarray,
    lab: np.ndarray,
    fine_labels: np.ndarray,
    min_class_n: int,
) -> dict[str, Any]:
    labels = [label for label in sorted(set(fine_labels.tolist())) if label]
    kept = [label for label in labels if int((fine_labels == label).sum()) >= min_class_n]
    if len(kept) < 4:
        return {"n_classes": len(kept), "spearman": None}
    unit = l2_normalize(embeddings)
    centers = []
    colors = []
    for label in kept:
        idx = fine_labels == label
        centers.append(unit[idx].mean(axis=0))
        colors.append(lab[idx].mean(axis=0))
    center = l2_normalize(np.stack(centers))
    color = np.stack(colors)
    emb_dist = 1.0 - (center @ center.T)
    color_dist = delta_e00(color[:, None, :], color[None, :, :])
    iu = np.triu_indices(len(kept), k=1)
    spearman = _spearman(emb_dist[iu], color_dist[iu])
    return {"n_classes": len(kept), "classes": kept, "spearman": spearman}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rank(np.asarray(a, dtype=np.float64))
    rb = _rank(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0:
        return float("nan")
    return float(ra @ rb / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def neighbor_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = _topk(_similarity(a), k)
    top_b = _topk(_similarity(b), k)
    scores = []
    for left, right in zip(top_a, top_b):
        union = len(set(left.tolist()) | set(right.tolist()))
        inter = len(set(left.tolist()) & set(right.tolist()))
        scores.append(inter / union if union else 0.0)
    return float(np.mean(scores))


def similarity_spearman(a: np.ndarray, b: np.ndarray) -> float:
    sim_a = _similarity(a)
    sim_b = _similarity(b)
    # Rank every other item. Diagonal is tied at -inf and is harmless.
    ra = np.argsort(np.argsort(sim_a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(sim_b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
    corr = (ra * rb).sum(axis=1) / np.clip(denom, 1e-12, None)
    return float(corr.mean())


def summarize_scores(scores: np.ndarray) -> dict[str, float] | None:
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return None
    return {"mean": float(valid.mean()), "n": int(len(valid))}


def bootstrap_mean_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float] | None:
    mask = np.isfinite(baseline) & np.isfinite(candidate)
    if int(mask.sum()) < 8:
        return None
    return _paired_bootstrap_delta(baseline[mask], candidate[mask], n_boot=n_boot, seed=seed)
