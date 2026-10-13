"""Sample N new tops (excluding existing cohort IDs) with provisional class balancing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from freeze import item_id, rank_key
from heuristics import resolve_subtype
from io_utils import load_attr_dataset, read_json, write_json, write_jsonl
from sample_balanced_cohort import (
    ATTRIBUTE_CLASSES,
    _candidate_score,
    _public_item,
    provisional_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cohort-dir", type=Path, required=True)
    p.add_argument("--out-cohort-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument("--max-new", type=int, default=2000)
    p.add_argument("--pool-target", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--origin", default="pool_add2k")
    return p.parse_args()


def _load_labels(root: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    path = root / "labels.jsonl"
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        labels[str(row["item_id"])] = dict(row.get("labels") or {})
    return labels


def main() -> None:
    args = parse_args()
    src = args.source_cohort_dir
    out = args.out_cohort_dir
    out.mkdir(parents=True, exist_ok=True)

    src_cohort = read_json(src / "cohort.json")
    labels = _load_labels(src)
    existing_keys = {(str(r["source"]), str(r["product_id"])) for r in src_cohort["items"]}
    existing_ids = {str(r["item_id"]) for r in src_cohort["items"]}

    # Working counts from gold labels (prefer) else provisional later.
    working = {attr: Counter() for attr in ATTRIBUTE_CLASSES}
    for iid in existing_ids:
        lab = labels.get(iid) or {}
        for attr, classes in ATTRIBUTE_CLASSES.items():
            val = lab.get(attr)
            if val and val != "uncertain" and val in classes:
                working[attr][val] += 1

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
            "split": "add2k",
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

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def need_from_working() -> dict[tuple[str, str], int]:
        # Soft target: raise each class toward median*1.5 of current attr, min gap 1.
        need: dict[tuple[str, str], int] = {}
        for attr, classes in ATTRIBUTE_CLASSES.items():
            counts = [working[attr][c] for c in classes]
            target = max(int(sorted(counts)[len(counts) // 2] * 1.35) + 50, max(counts) + 20)
            for c in classes:
                gap = target - working[attr][c]
                if gap > 0:
                    need[(attr, c)] = gap
        return need

    need = need_from_working()
    # Phase 1: fill soft shortfalls
    while len(selected) < args.max_new and need:
        best_i, best_score = -1, 0.0
        for i, (item, provisional) in enumerate(pool_rows):
            if item["item_id"] in selected_ids or item["item_id"] in existing_ids:
                continue
            score = _candidate_score(provisional, need)
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0 or best_score <= 0:
            break
        item, provisional = pool_rows[best_i]
        public = _public_item(item, origin=args.origin, provisional=provisional)
        selected.append(public)
        selected_ids.add(public["item_id"])
        for attr, (cls, _) in provisional.items():
            working[attr][cls] += 1
            key = (attr, cls)
            if key in need:
                need[key] -= 1
                if need[key] <= 0:
                    del need[key]
        if len(selected) % 200 == 0:
            need = need_from_working()
            print(f"selected={len(selected)} soft_need_keys={len(need)}", flush=True)

    # Phase 2: top up to max-new with diversified random-ish rank order
    if len(selected) < args.max_new:
        for item, provisional in pool_rows:
            if len(selected) >= args.max_new:
                break
            if item["item_id"] in selected_ids or item["item_id"] in existing_ids:
                continue
            public = _public_item(item, origin=args.origin, provisional=provisional)
            selected.append(public)
            selected_ids.add(public["item_id"])
            for attr, (cls, _) in provisional.items():
                working[attr][cls] += 1

    print(f"selected_new={len(selected)}", flush=True)

    # Dataset freeze-style items.jsonl for SAM/preprocess
    cohort_run = out / "pipeline" / "cohorts" / "cohort-add2k-20261007"
    (cohort_run / "dataset").mkdir(parents=True, exist_ok=True)
    write_jsonl(
        cohort_run / "dataset" / "items.jsonl",
        [
            {
                "item_id": r["item_id"],
                "source": r["source"],
                "product_id": r["product_id"],
                "product": r.get("product"),
                "brand": r.get("brand"),
                "family": "tops",
                "subtype": r.get("subtype"),
                "content_sha256": r.get("content_sha256"),
                "image_path": r["image_path"],
                "split": "add2k",
                "shoot_type": r.get("shoot_type"),
            }
            for r in selected
        ],
    )
    write_json(
        cohort_run / "dataset" / "inventory.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "n": len(selected),
            "origin": args.origin,
            "dataset_fingerprint": f"add2k-{len(selected)}-{args.seed}",
        },
    )
    write_jsonl(out / "add2k_new.jsonl", selected)
    write_json(
        out / "add2k_summary.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "n_new": len(selected),
            "n_source_items": len(src_cohort["items"]),
            "working_counts_after_provisional": {a: dict(c) for a, c in working.items()},
            "new_item_ids": [r["item_id"] for r in selected],
            "cohort_run": str(cohort_run.relative_to(out)) if cohort_run.is_relative_to(out) else str(cohort_run),
        },
    )
    print(f"wrote {cohort_run} n={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
