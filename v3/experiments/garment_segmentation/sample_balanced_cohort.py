"""Build a class-balanced tops cohort for attribute cluster-quality eval.

Uses tops-450 as seed, then supplements shortfalls from the retail tops pool
via weak / heuristic product-name signals. Does not assign gold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
from io_utils import load_attr_dataset, read_jsonl, write_json, write_jsonl

# Priority-1 (+ fit priority-2). Class ids are stable keys for later labeling.
ATTRIBUTE_CLASSES: dict[str, tuple[str, ...]] = {
    "pattern": ("solid", "stripe", "check", "graphic_logo", "other"),
    "sleeve": ("sleeveless", "short", "long"),
    "neck": ("round", "vneck", "collar", "hood", "turtleneck"),
    "color": (
        "white_ivory",
        "black",
        "gray",
        "navy_blue",
        "beige_brown",
        "chromatic",
    ),
    "fit": ("slim", "regular", "oversized"),
}

PRIORITY_ATTRIBUTES = ("pattern", "sleeve", "neck", "color", "fit")

SLEEVE_ALIASES: dict[str, tuple[str, ...]] = {
    "sleeveless": ("민소매", "나시", "슬리브리스", "sleeveless", "tank top", "tanktop", "캡소매"),
    "short": ("반팔", "숏슬리브", "short sleeve", "shortsleeve", "반소매", "반팔티"),
    "long": ("긴팔", "롱슬리브", "long sleeve", "longsleeve", "긴팔티"),
}

NECK_ALIASES: dict[str, tuple[str, ...]] = {
    "round": ("라운드넥", "라운드", "크루넥", "crew neck", "crewneck"),
    "vneck": ("브이넥", "v넥", "v-neck", "vneck", "브이 넥"),
    "collar": ("셔츠칼라", "클래식 칼라", "옥스포드", "oxford", "카라셔츠"),
    "hood": ("후드", "hoodie", "후디"),
    "turtleneck": (
        "터틀넥",
        "터틀",
        "폴라넥",
        "목폴라",
        "하이넥",
        "turtleneck",
        "mock neck",
        "mockneck",
        "폴라",
    ),
}

FIT_ALIASES: dict[str, tuple[str, ...]] = {
    "slim": ("slim", "슬림"),
    "regular": ("레귤러", "regular", "스탠다드", "standard", "클래식핏", "classic fit"),
    "oversized": ("oversized", "oversize", "오버핏", "오버사이즈", "baggy", "배기", "벌룬"),
}

COLOR_MERGE: dict[str, str] = {
    "white": "white_ivory",
    "black": "black",
    "gray": "gray",
    "navy": "navy_blue",
    "blue": "navy_blue",
    "brown": "beige_brown",
    "red": "chromatic",
    "green": "chromatic",
    "yellow": "chromatic",
    "pink": "chromatic",
    "purple": "chromatic",
    "orange": "chromatic",
}

PATTERN_MERGE: dict[str, str] = {
    "solid": "solid",
    "stripe": "stripe",
    "check": "check",
    "graphic": "graphic_logo",
    "logo": "graphic_logo",
    "floral": "other",
    "dot": "other",
    "animal": "other",
    "camouflage": "other",
}


def _text(product: str) -> str:
    return f" {product.lower()} "


def _first_alias_hit(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    for label, tokens in aliases.items():
        if any(token.lower() in text for token in tokens):
            return label
    return None


def provisional_labels(
    product: str,
    *,
    subtype: str,
    weak: dict[str, list[str]] | None,
) -> dict[str, tuple[str, str]]:
    """Return attr -> (class_id, confidence) where confidence is strong|weak."""
    weak = weak or {}
    text = _text(product)
    out: dict[str, tuple[str, str]] = {}

    colors = [COLOR_MERGE[c] for c in weak.get("color") or [] if c in COLOR_MERGE]
    if colors:
        # Prefer non-chromatic if mixed; else first
        preferred = next((c for c in colors if c != "chromatic"), colors[0])
        out["color"] = (preferred, "strong")

    patterns = [PATTERN_MERGE[p] for p in weak.get("pattern") or [] if p in PATTERN_MERGE]
    if patterns:
        preferred = next(
            (p for p in ("stripe", "check", "graphic_logo", "solid", "other") if p in patterns),
            patterns[0],
        )
        out["pattern"] = (preferred, "strong")
    elif not weak.get("pattern"):
        # Many solid items never say "solid" in the title.
        out["pattern"] = ("solid", "weak")

    if subtype == "sleeveless":
        out["sleeve"] = ("sleeveless", "strong")
    else:
        sleeve = _first_alias_hit(text, SLEEVE_ALIASES)
        if sleeve:
            out["sleeve"] = (sleeve, "strong")

    if subtype == "hoodie":
        out["neck"] = ("hood", "strong")
    else:
        neck = _first_alias_hit(text, NECK_ALIASES)
        if neck:
            out["neck"] = (neck, "strong")
        elif subtype == "shirt":
            out["neck"] = ("collar", "weak")

    fit = _first_alias_hit(text, FIT_ALIASES)
    if fit:
        out["fit"] = (fit, "strong")
    else:
        shapes = set(weak.get("shape") or [])
        if "slim" in shapes:
            out["fit"] = ("slim", "strong")
        elif shapes & {"oversized", "baggy"}:
            out["fit"] = ("oversized", "strong")
        else:
            out["fit"] = ("regular", "weak")

    return out


def _public_item(row: dict[str, Any], *, origin: str, provisional: dict[str, tuple[str, str]]) -> dict[str, Any]:
    heuristic = row.get("heuristic") or {}
    return {
        "item_id": row["item_id"],
        "origin": origin,
        "source": row.get("source"),
        "product_id": str(row.get("product_id")),
        "product": row.get("product"),
        "brand": row.get("brand"),
        "family": row.get("family") or "tops",
        "subtype": row.get("subtype"),
        "content_sha256": row.get("content_sha256"),
        "image_path": row.get("image_path"),
        "split": row.get("split"),
        "shoot_type": heuristic.get("shoot_type") if isinstance(heuristic, dict) else None,
        "weak_labels": row.get("weak_labels") or {},
        "provisional": {attr: {"class": cls, "confidence": conf} for attr, (cls, conf) in provisional.items()},
    }


def _coverage(
    rows: list[dict[str, Any]],
    *,
    confidence: str | None = None,
) -> dict[str, Counter[str]]:
    cov: dict[str, Counter[str]] = {attr: Counter() for attr in ATTRIBUTE_CLASSES}
    for row in rows:
        for attr, payload in (row.get("provisional") or {}).items():
            if attr not in cov:
                continue
            if confidence is not None and payload.get("confidence") != confidence:
                continue
            cls = payload.get("class")
            if cls in ATTRIBUTE_CLASSES[attr]:
                cov[attr][cls] += 1
    return cov


def _shortfalls(cov: dict[str, Counter[str]], quota: int) -> dict[tuple[str, str], int]:
    need: dict[tuple[str, str], int] = {}
    for attr, classes in ATTRIBUTE_CLASSES.items():
        for cls in classes:
            have = cov[attr][cls]
            if have < quota:
                need[(attr, cls)] = quota - have
    return need


def _candidate_score(
    provisional: dict[str, tuple[str, str]],
    need: dict[tuple[str, str], int],
) -> float:
    score = 0.0
    for attr, (cls, conf) in provisional.items():
        key = (attr, cls)
        if key not in need:
            continue
        weight = 1.0 if conf == "strong" else 0.35
        # Prefer filling larger remaining gaps.
        score += weight * (1.0 + 0.02 * need[key])
        if attr == "fit" and conf == "weak" and cls == "regular":
            # regular-via-absence is abundant; do not let it dominate.
            score -= 0.25
        if attr == "pattern" and conf == "weak" and cls == "solid":
            score -= 0.15
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-dataset",
        type=Path,
        default=REPO_ROOT
        / "artifacts/garment_segmentation/attribute_cluster_eval-20261002/pipeline/cohorts/tops-450-20260918/dataset/items.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.yaml",
        help="Local garment_segmentation config with input_root/sources",
    )
    parser.add_argument("--quota", type=int, default=50)
    parser.add_argument("--min-quota", type=int, default=30)
    parser.add_argument("--pool-target", type=int, default=12000, help="Tops metadata candidates to scan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    parser.add_argument(
        "--max-supplements",
        type=int,
        default=400,
        help="Cap on newly added products (keeps labeling volume bounded)",
    )
    return parser.parse_args()


def _load_sources(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw = config.get("sources") or {}
    return {str(alias): dict(meta) for alias, meta in raw.items()}


def _scan_pool(
    *,
    input_root: Path,
    sources: dict[str, dict[str, str]],
    pool_target: int,
    seed: int,
    exclude_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    dataset = load_attr_dataset()
    print(f"loading registry from {input_root} …", flush=True)
    registry = dataset.load_registry(input_root, sources)
    print(f"registry size={len(registry)}; collecting tops candidates …", flush=True)
    pools, scanned_days = dataset.collect_candidates(
        input_root,
        registry,
        sources=sources,
        families=("tops",),
        target_per_family=pool_target,
    )
    tops = list(pools["tops"].values())
    print(f"scanned_days={len(scanned_days)} tops_candidates={len(tops)}", flush=True)

    enriched: list[dict[str, Any]] = []
    seen_digest: set[str] = set()
    for item in sorted(tops, key=lambda row: rank_key(row, seed)):
        key = (str(item.get("source")), str(item.get("product_id")))
        if key in exclude_keys:
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
        row = {
            **item,
            "family": "tops",
            "image_path": str(path.resolve()),
            "subtype": subtype,
            "weak_labels": weak,
            "item_id": item_id({**item, "content_sha256": digest}),
            "split": "supplement",
            "heuristic": {},
        }
        if digest:
            seen_digest.add(digest)
        enriched.append(row)
        if len(enriched) >= pool_target:
            break
    print(f"pool with images={len(enriched)}", flush=True)
    return enriched


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise SystemExit(f"missing config: {args.config} (local paths; not committed)")
    if not args.seed_dataset.is_file():
        raise SystemExit(f"missing seed dataset: {args.seed_dataset}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_root = Path(config["input_root"])
    sources = _load_sources(config)

    seed_raw = read_jsonl(args.seed_dataset)
    seed_rows: list[dict[str, Any]] = []
    exclude_keys: set[tuple[str, str]] = set()
    for item in seed_raw:
        product = str(item.get("product") or "")
        subtype = str(item.get("subtype") or resolve_subtype(product, str(item.get("category_label") or "")))
        weak = item.get("weak_labels") or {}
        provisional = provisional_labels(product, subtype=subtype, weak=weak)
        row = dict(item)
        if not row.get("item_id"):
            row["item_id"] = item_id(row)
        seed_rows.append(_public_item(row, origin="seed_450", provisional=provisional))
        exclude_keys.add((str(item.get("source")), str(item.get("product_id"))))

    cov = _coverage(seed_rows)
    need = _shortfalls(cov, args.quota)
    print("seed shortfalls:", {f"{a}.{c}": n for (a, c), n in sorted(need.items())}, flush=True)

    pool = _scan_pool(
        input_root=input_root,
        sources=sources,
        pool_target=args.pool_target,
        seed=args.seed,
        exclude_keys=exclude_keys,
    )

    selected = list(seed_rows)
    selected_ids = {row["item_id"] for row in selected}
    supplements: list[dict[str, Any]] = []

    # Precompute provisional for pool.
    pool_prov: list[tuple[dict[str, Any], dict[str, tuple[str, str]]]] = []
    for item in pool:
        provisional = provisional_labels(
            str(item.get("product") or ""),
            subtype=str(item.get("subtype") or "other"),
            weak=item.get("weak_labels") or {},
        )
        pool_prov.append((item, provisional))

    while need and len(supplements) < args.max_supplements:
        best_i = -1
        best_score = 0.0
        for i, (item, provisional) in enumerate(pool_prov):
            if item["item_id"] in selected_ids:
                continue
            score = _candidate_score(provisional, need)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0 or best_score <= 0:
            break
        item, provisional = pool_prov[best_i]
        public = _public_item(item, origin="pool_supplement", provisional=provisional)
        selected.append(public)
        supplements.append(public)
        selected_ids.add(public["item_id"])
        cov = _coverage(selected)
        need = _shortfalls(cov, args.quota)

    final_cov = _coverage(selected)
    strong_cov = _coverage(selected, confidence="strong")
    weak_cov = _coverage(selected, confidence="weak")
    remaining = _shortfalls(final_cov, args.quota)
    below_min = {
        f"{attr}.{cls}": final_cov[attr][cls]
        for attr, classes in ATTRIBUTE_CLASSES.items()
        for cls in classes
        if final_cov[attr][cls] < args.min_quota
    }
    strong_below_quota = {
        f"{attr}.{cls}": strong_cov[attr][cls]
        for attr, classes in ATTRIBUTE_CLASSES.items()
        for cls in classes
        if strong_cov[attr][cls] < args.quota
    }

    # Review sample: stratified 50 item_ids for dual labeling later.
    review_ids: list[str] = []
    by_attr = defaultdict(list)
    for row in selected:
        for attr, payload in (row.get("provisional") or {}).items():
            by_attr[(attr, payload["class"])].append(row["item_id"])
    rng_order = sorted(
        {(attr, cls) for attr, classes in ATTRIBUTE_CLASSES.items() for cls in classes},
        key=lambda key: hashlib.sha1(f"{args.seed}|{key[0]}|{key[1]}".encode()).hexdigest(),
    )
    seen_review: set[str] = set()
    for key in rng_order:
        for item_id_value in sorted(
            by_attr.get(key, []),
            key=lambda x: hashlib.sha1(f"{args.seed}|review|{x}".encode()).hexdigest(),
        ):
            if item_id_value in seen_review:
                continue
            review_ids.append(item_id_value)
            seen_review.add(item_id_value)
            if len(review_ids) >= 50:
                break
        if len(review_ids) >= 50:
            break

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "plan": "attribute_cluster_quality_20261002",
        "seed_dataset": str(args.seed_dataset),
        "quota_per_class": args.quota,
        "min_quota_per_class": args.min_quota,
        "attributes": {attr: list(classes) for attr, classes in ATTRIBUTE_CLASSES.items()},
        "priority_attributes": list(PRIORITY_ATTRIBUTES),
        "counts": {
            "seed": len(seed_rows),
            "supplements": len(supplements),
            "total_unique": len(selected),
            "pool_scanned": len(pool),
        },
        "coverage": {attr: dict(counter) for attr, counter in final_cov.items()},
        "coverage_strong": {attr: dict(counter) for attr, counter in strong_cov.items()},
        "coverage_weak": {attr: dict(counter) for attr, counter in weak_cov.items()},
        "shortfall_after_quota": {f"{a}.{c}": n for (a, c), n in sorted(remaining.items())},
        "below_min_quota": below_min,
        "strong_below_quota": strong_below_quota,
        "review_item_ids": review_ids,
        "items": selected,
        "notes": [
            "provisional labels are weak/heuristic only; gold labels come in step 2",
            "pattern.solid and fit.regular often use weak confidence (title absence)",
            "quota fill uses any confidence; strong_below_quota shows title-signal gaps",
            "absolute image_path is local-only under artifacts/",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output_dir / "cohort.json"
    write_json(cohort_path, payload)
    write_jsonl(args.output_dir / "supplements.jsonl", supplements)
    write_json(
        args.output_dir / "cohort_summary.json",
        {
            "created_at": payload["created_at"],
            "counts": payload["counts"],
            "coverage": payload["coverage"],
            "coverage_strong": payload["coverage_strong"],
            "shortfall_after_quota": payload["shortfall_after_quota"],
            "below_min_quota": payload["below_min_quota"],
            "strong_below_quota": payload["strong_below_quota"],
            "review_item_ids": review_ids,
        },
    )
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print("coverage:", json.dumps(payload["coverage"], ensure_ascii=False, indent=2))
    print("coverage_strong:", json.dumps(payload["coverage_strong"], ensure_ascii=False, indent=2))
    print("shortfall:", payload["shortfall_after_quota"])
    print("below_min:", below_min)
    print(f"wrote {cohort_path}")


if __name__ == "__main__":
    main()
