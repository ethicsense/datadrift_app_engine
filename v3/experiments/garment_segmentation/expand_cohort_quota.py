"""Expand attribute cohort so each class has target_n labeled+embeddable samples.

Uses existing agent/human labels for current embed-ok items, then samples
additional tops from the retail pool via weak/heuristic provisional signals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from freeze import item_id, rank_key
from heuristics import resolve_subtype
from io_utils import load_attr_dataset, read_json, read_jsonl, write_json, write_jsonl
from sample_balanced_cohort import (
    ATTRIBUTE_CLASSES,
    _candidate_score,
    _public_item,
    provisional_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument("--target", type=int, default=90, help="Target count per class (80-100 band)")
    p.add_argument("--pool-target", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new", type=int, default=350)
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


def _embed_ok_ids(root: Path) -> set[str]:
    ids = read_json(root / "embeddings" / "item_ids.json")["item_ids"]
    mat = np.load(root / "embeddings" / "sam3_abcd" / "A.npy")
    ok = np.isfinite(mat).all(axis=1)
    return {ids[i] for i, flag in enumerate(ok) if flag}


def _coverage_from_labels(
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
) -> dict[str, Counter[str]]:
    cov = {attr: Counter() for attr in ATTRIBUTE_CLASSES}
    for iid in item_ids:
        lab = labels.get(iid) or {}
        for attr, classes in ATTRIBUTE_CLASSES.items():
            val = lab.get(attr)
            if val and val != "uncertain" and val in classes:
                cov[attr][val] += 1
    return cov


def _shortfalls(cov: dict[str, Counter[str]], target: int) -> dict[tuple[str, str], int]:
    need: dict[tuple[str, str], int] = {}
    for attr, classes in ATTRIBUTE_CLASSES.items():
        for cls in classes:
            have = cov[attr][cls]
            if have < target:
                need[(attr, cls)] = target - have
    return need


def main() -> None:
    args = parse_args()
    root = args.cohort_dir
    cohort = read_json(root / "cohort.json")
    labels = _load_labels(root)
    ok_ids = _embed_ok_ids(root)
    print(f"embed_ok={len(ok_ids)} labeled={len(labels)} target={args.target}", flush=True)

    cov = _coverage_from_labels(sorted(ok_ids), labels)
    need = _shortfalls(cov, args.target)
    print("shortfalls:", {f"{a}.{c}": n for (a, c), n in sorted(need.items())}, flush=True)

    existing_keys = {(str(r["source"]), str(r["product_id"])) for r in cohort["items"]}
    existing_ids = {str(r["item_id"]) for r in cohort["items"]}

    if not args.config.is_file():
        raise SystemExit(f"missing {args.config}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_root = Path(config["input_root"])
    sources = {str(a): dict(m) for a, m in (config.get("sources") or {}).items()}

    dataset = load_attr_dataset()
    print("loading registry …", flush=True)
    registry = dataset.load_registry(input_root, sources)
    pools, scanned = dataset.collect_candidates(
        input_root,
        registry,
        sources=sources,
        families=("tops",),
        target_per_family=args.pool_target,
    )
    print(f"scanned_days={len(scanned)} tops={len(pools['tops'])}", flush=True)

    pool_rows: list[tuple[dict[str, Any], dict[str, tuple[str, str]]]] = []
    seen_digest: set[str] = set()
    for item in sorted(pools["tops"].values(), key=lambda row: rank_key(row, args.seed)):
        key = (str(item.get("source")), str(item.get("product_id")))
        if key in existing_keys:
            continue
        digest = str(item.get("content_sha256") or "")
        if digest and digest in seen_digest:
            continue
        path = dataset._image_path(input_root, item, sources=sources)
        if path is None:
            continue
        product = str(item.get("product") or "")
        subtype = resolve_subtype(product, str(item.get("category_label") or ""))
        weak = dataset.weak_labels(product)
        enriched = {
            **item,
            "family": "tops",
            "image_path": str(path.resolve()),
            "subtype": subtype,
            "weak_labels": weak,
            "item_id": item_id({**item, "content_sha256": digest}),
            "split": "supplement",
            "heuristic": {},
        }
        if enriched["item_id"] in existing_ids:
            continue
        if digest:
            seen_digest.add(digest)
        provisional = provisional_labels(product, subtype=subtype, weak=weak)
        pool_rows.append((enriched, provisional))
        if len(pool_rows) >= args.pool_target:
            break
    print(f"pool_candidates={len(pool_rows)}", flush=True)

    # Working coverage: start from embed-ok gold labels. New items contribute provisional until labeled.
    working_ids = set(ok_ids)
    # Fake provisional into label space for scoring only
    provisional_as_labels: dict[str, dict[str, str]] = {
        iid: {a: v for a, (v, _) in (labels.get(iid) and {})}
        for iid in []
    }
    # rebuild cov from gold only for need; when adding, subtract provisional fills optimistically
    selected_new: list[dict[str, Any]] = []
    while need and len(selected_new) < args.max_new:
        best_i = -1
        best_score = 0.0
        for i, (item, provisional) in enumerate(pool_rows):
            if item["item_id"] in working_ids:
                continue
            score = _candidate_score(provisional, need)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0 or best_score <= 0:
            break
        item, provisional = pool_rows[best_i]
        public = _public_item(item, origin="pool_supplement", provisional=provisional)
        selected_new.append(public)
        working_ids.add(public["item_id"])
        # optimistic fill
        for attr, (cls, _conf) in provisional.items():
            key = (attr, cls)
            if key in need:
                need[key] -= 1
                if need[key] <= 0:
                    del need[key]

    print(f"selected_new={len(selected_new)} remaining_shortfalls={ {f'{a}.{c}':n for (a,c),n in sorted(need.items())} }", flush=True)

    # Keep prior cohort items, but mark embed failures for replacement (drop from active list)
    kept = [r for r in cohort["items"] if r["item_id"] in ok_ids]
    dropped = [r for r in cohort["items"] if r["item_id"] not in ok_ids]
    expanded = kept + selected_new

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "plan": "attribute_cluster_quality_20261002",
        "quota_per_class": args.target,
        "min_quota_per_class": 80,
        "attributes": {attr: list(classes) for attr, classes in ATTRIBUTE_CLASSES.items()},
        "counts": {
            "kept_embed_ok": len(kept),
            "dropped_sam3_fail": len(dropped),
            "new_supplements": len(selected_new),
            "total_unique": len(expanded),
        },
        "coverage_before_expand": {attr: dict(c) for attr, c in cov.items()},
        "optimistic_remaining_shortfalls": {f"{a}.{c}": n for (a, c), n in sorted(need.items())},
        "dropped_item_ids": [r["item_id"] for r in dropped],
        "new_item_ids": [r["item_id"] for r in selected_new],
        "items": expanded,
        "notes": [
            f"expand to ~{args.target}/class; agent labels required for new items before metrics",
            "SAM3-fail supplements dropped and replaced from pool",
        ],
    }
    write_json(root / "cohort.json", payload)
    write_jsonl(root / "expansion_new.jsonl", selected_new)
    write_json(
        root / "expansion_summary.json",
        {
            "created_at": payload["created_at"],
            "counts": payload["counts"],
            "coverage_before_expand": payload["coverage_before_expand"],
            "optimistic_remaining_shortfalls": payload["optimistic_remaining_shortfalls"],
            "new_item_ids": payload["new_item_ids"],
            "dropped_item_ids": payload["dropped_item_ids"],
        },
    )
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
