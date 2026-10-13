"""Compare attribute cluster-quality metrics across preprocess embedding banks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

ATTRS = ("pattern", "sleeve", "neck", "color", "fit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument("--banks", nargs="*", default=["abcd", "sam3"])
    p.add_argument("--arms", nargs="*", default=None, help="Default: all available .npy in bank")
    p.add_argument("--min-class-n", type=int, default=30)
    p.add_argument("--neighbor-k", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_labels(root: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        labels[str(row["item_id"])] = dict(row.get("labels") or {})
    human = root / "human_labels_review50.jsonl"
    if human.is_file():
        for line in human.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row["item_id"])
            labels[iid] = {**labels.get(iid, {}), **dict(row.get("labels") or {})}
    return labels


def _finite_rows(mat: np.ndarray) -> np.ndarray:
    return np.isfinite(mat).all(axis=1)


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def _r_precision_map(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Per-query R-precision and AP@R, macro over queries with R>=1."""
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    n = len(y)
    rps: list[float] = []
    aps: list[float] = []
    for i in range(n):
        same = np.where(y == y[i])[0]
        same = same[same != i]
        r = len(same)
        if r < 1:
            continue
        order = np.argsort(-sim[i])
        top = order[:r]
        hits = np.isin(top, same)
        rps.append(float(hits.mean()))
        # AP@R
        if hits.any():
            ranks = np.where(hits)[0] + 1
            precisions = [(k + 1) / ranks[k] for k in range(len(ranks))]
            aps.append(float(np.mean(precisions)))
        else:
            aps.append(0.0)
    return float(np.mean(rps)) if rps else float("nan"), float(np.mean(aps)) if aps else float("nan")


def _purity_lift(x: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float, float]:
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    n = len(y)
    purities: list[float] = []
    lifts: list[float] = []
    counts = {lab: int((y == lab).sum()) for lab in set(y.tolist())}
    for i in range(n):
        ki = min(k, n - 1, max(1, counts[y[i]] - 1))
        nn = np.argsort(-sim[i])[:ki]
        pur = float((y[nn] == y[i]).mean())
        base = (counts[y[i]] - 1) / max(n - 1, 1)
        purities.append(pur)
        lifts.append(pur / base if base > 0 else float("nan"))
    return float(np.mean(purities)), float(np.nanmean(lifts)), float(np.std(purities))


def _separation_gap(x: np.ndarray, y: np.ndarray) -> float:
    labs = sorted(set(y.tolist()))
    gaps: list[float] = []
    for lab in labs:
        idx = np.where(y == lab)[0]
        if len(idx) < 2:
            continue
        intra = x[idx] @ x[idx].T
        # exclude diagonal
        mask = ~np.eye(len(idx), dtype=bool)
        intra_mean = float(intra[mask].mean())
        other = np.where(y != lab)[0]
        if len(other) == 0:
            continue
        inter = float((x[idx] @ x[other].T).mean())
        gaps.append(intra_mean - inter)
    return float(np.mean(gaps)) if gaps else float("nan")


def _silhouette_macro(x: np.ndarray, y: np.ndarray, min_class_n: int) -> dict[str, Any]:
    labs = sorted(set(y.tolist()))
    if len(labs) < 2:
        return {"macro": float("nan"), "per_class": {}, "used_classes": []}
    samples = silhouette_samples(x, y, metric="cosine")
    per: dict[str, float] = {}
    used: list[str] = []
    for lab in labs:
        idx = np.where(y == lab)[0]
        if len(idx) < min_class_n:
            continue
        per[lab] = float(samples[idx].mean())
        used.append(lab)
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return {"macro": macro, "per_class": per, "used_classes": used, "global_mean": float(samples.mean())}


def _metrics_for_subset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    min_class_n: int,
    seed: int,
) -> dict[str, Any]:
    x = _l2(x.astype(np.float32))
    rp, mapr = _r_precision_map(x, y)
    purity, lift, purity_std = _purity_lift(x, y, k)
    sil = _silhouette_macro(x, y, min_class_n)
    gap = _separation_gap(x, y)
    # sklearn cluster indices prefer euclidean on L2-normalized ≈ cosine
    db = float(davies_bouldin_score(x, y)) if len(set(y)) > 1 else float("nan")
    ch = float(calinski_harabasz_score(x, y)) if len(set(y)) > 1 else float("nan")
    n_clusters = len(set(y.tolist()))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(x)
    nmi = float(normalized_mutual_info_score(y, pred))
    ari = float(adjusted_rand_score(y, pred))
    return {
        "n": int(len(y)),
        "n_classes": n_clusters,
        "class_counts": {str(lab): int((y == lab).sum()) for lab in sorted(set(y.tolist()))},
        "r_precision": rp,
        "map_at_r": mapr,
        "purity_at_k": purity,
        "purity_std": purity_std,
        "lift": lift,
        "silhouette": sil,
        "separation_gap": gap,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "kmeans_nmi": nmi,
        "kmeans_ari": ari,
        "k": k,
    }


def _paired_bootstrap_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = float(scores_b[idx].mean() - scores_a[idx].mean())
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": float(scores_b.mean() - scores_a.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
    }


def _per_item_purity(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    x = _l2(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    n = len(y)
    out = np.zeros(n, dtype=np.float64)
    counts = {lab: int((y == lab).sum()) for lab in set(y.tolist())}
    for i in range(n):
        ki = min(k, n - 1, max(1, counts[y[i]] - 1))
        nn = np.argsort(-sim[i])[:ki]
        out[i] = float((y[nn] == y[i]).mean())
    return out


def main() -> None:
    args = parse_args()
    root = args.cohort_dir
    cohort = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    item_ids = json.loads((root / "embeddings" / "item_ids.json").read_text(encoding="utf-8"))["item_ids"]
    labels = _load_labels(root)
    attr_classes = cohort["attributes"]

    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "min_class_n": args.min_class_n,
        "neighbor_k": args.neighbor_k,
        "banks": {},
    }
    out_dir = root / "metrics"
    out_path = out_dir / "cluster_quality.json"
    # Preserve metrics for banks not recomputed in this invocation.
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for bank_name, payload in (existing.get("banks") or {}).items():
                if bank_name not in args.banks:
                    report["banks"][bank_name] = payload
        except json.JSONDecodeError:
            pass

    for bank in args.banks:
        bank_dir = root / "embeddings" / bank
        if not bank_dir.is_dir():
            print(f"skip missing bank {bank}", flush=True)
            continue
        arms = args.arms or sorted(p.stem for p in bank_dir.glob("*.npy"))
        emb = {arm: np.load(bank_dir / f"{arm}.npy") for arm in arms}
        bank_out: dict[str, Any] = {"arms": {}, "pairwise": {}}
        for arm, mat in emb.items():
            arm_out: dict[str, Any] = {}
            ok = _finite_rows(mat)
            for attr in ATTRS:
                classes = set(attr_classes[attr])
                mask = []
                y_list = []
                for i, iid in enumerate(item_ids):
                    if not ok[i]:
                        mask.append(False)
                        continue
                    lab = (labels.get(iid) or {}).get(attr)
                    keep = bool(lab and lab != "uncertain" and lab in classes)
                    mask.append(keep)
                    if keep:
                        y_list.append(lab)
                mask_arr = np.asarray(mask, dtype=bool)
                if mask_arr.sum() < args.min_class_n * 2:
                    arm_out[attr] = {"skipped": True, "reason": "too_few_labeled", "n": int(mask_arr.sum())}
                    continue
                x = mat[mask_arr]
                y = np.asarray(y_list)
                # drop classes below min for silhouette macro only inside metric helper
                counts = {lab: int((y == lab).sum()) for lab in set(y.tolist())}
                keep_cls = {lab for lab, c in counts.items() if c >= 1}
                # keep all for retrieval metrics; silhouette uses min_class_n
                sel = np.array([lab in keep_cls for lab in y])
                metrics = _metrics_for_subset(
                    x[sel],
                    y[sel],
                    k=args.neighbor_k,
                    min_class_n=args.min_class_n,
                    seed=args.seed,
                )
                arm_out[attr] = metrics
            bank_out["arms"][arm] = arm_out

        # paired A vs other arms on purity
        if "A" in emb:
            for arm in arms:
                if arm == "A":
                    continue
                pair: dict[str, Any] = {}
                ok = _finite_rows(emb["A"]) & _finite_rows(emb[arm])
                for attr in ATTRS:
                    classes = set(attr_classes[attr])
                    idxs = []
                    y_list = []
                    for i, iid in enumerate(item_ids):
                        if not ok[i]:
                            continue
                        lab = (labels.get(iid) or {}).get(attr)
                        if lab and lab != "uncertain" and lab in classes:
                            idxs.append(i)
                            y_list.append(lab)
                    if len(idxs) < args.min_class_n * 2:
                        continue
                    idxs_arr = np.asarray(idxs)
                    y = np.asarray(y_list)
                    pa = _per_item_purity(emb["A"][idxs_arr], y, args.neighbor_k)
                    pb = _per_item_purity(emb[arm][idxs_arr], y, args.neighbor_k)
                    pair[attr] = _paired_bootstrap_delta(pa, pb, n_boot=args.bootstrap, seed=args.seed)
                bank_out["pairwise"][f"A_vs_{arm}"] = pair
        report["banks"][bank] = bank_out
        print(f"bank {bank} arms={arms}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # compact markdown summary
    lines = ["# Cluster quality summary", "", f"created: {report['created_at']}", ""]
    for bank, payload in report["banks"].items():
        lines.append(f"## {bank}")
        for arm, attrs in payload["arms"].items():
            lines.append(f"### arm {arm}")
            for attr, m in attrs.items():
                if m.get("skipped"):
                    lines.append(f"- {attr}: skipped ({m.get('reason')})")
                    continue
                lines.append(
                    f"- {attr}: n={m['n']} R-prec={m['r_precision']:.3f} "
                    f"lift={m['lift']:.3f} sil_macro={m['silhouette']['macro']:.3f} "
                    f"gap={m['separation_gap']:.3f}"
                )
        for pair_name, attrs in payload.get("pairwise", {}).items():
            lines.append(f"### {pair_name} (purity Δ, 95% CI)")
            for attr, d in attrs.items():
                lines.append(
                    f"- {attr}: Δ={d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] n={d['n']}"
                )
        lines.append("")
    (out_dir / "cluster_quality_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "cluster_quality_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
