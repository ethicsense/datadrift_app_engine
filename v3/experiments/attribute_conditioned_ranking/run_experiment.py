"""Build attribute-controlled embeddings and evaluate them against the frozen protocol."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GARMENT_DIR = HERE.parent / "garment_segmentation"
if str(GARMENT_DIR) not in sys.path:
    sys.path.insert(0, str(GARMENT_DIR))

from cluster_quality import _load_labels  # noqa: E402
from color_reference import build_color_reference  # noqa: E402
from color_space import pairwise_delta_e00  # noqa: E402
from controls import (  # noqa: E402
    data_centroid_basis,
    fingerprint,
    l2_normalize,
    mix,
    project,
    remove_subspace,
    subspace_basis,
)
from metrics import (  # noqa: E402
    bootstrap_mean_delta,
    coarse_metrics,
    graded_color_metrics,
    neighbor_jaccard,
    similarity_spearman,
    summarize_scores,
)
from protocol import (  # noqa: E402
    ALPHAS,
    BOOTSTRAP,
    COARSE_ATTRS,
    COLOR_NON_TARGETS,
    COLOR_PROMPTS,
    DATE,
    DEV_FRAC,
    FAR_DE,
    MODEL_ID,
    NECK_PROMPTS,
    NEAR_DE,
    NEIGHBOR_K,
    NON_TARGET_DROP_LIMIT,
    PATTERN_NON_TARGETS,
    PATTERN_PROMPTS,
    PROJECT_ID,
    PROMPT_GROUPS,
    SEED,
    SLEEVE_PROMPTS,
    SOURCE_COHORT,
    SOURCE_EMBEDDING,
    SOURCE_PREPROCESS,
    SVD_RANKS,
    TAU_DE,
    TRAIN_FRAC,
    as_dict,
    decision_non_targets,
    project_id,
    removal_prompt_names,
)
from scatter import build_scatter  # noqa: E402
from text_encode import encode_prompt_groups  # noqa: E402

sys.path.insert(0, str(HERE))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported value: {type(value)!r}")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def _assign_splits(
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    seed: int,
    stratify: str = "color",
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for iid in item_ids:
        key = (labels.get(iid) or {}).get(stratify) or "missing"
        groups.setdefault(key, []).append(iid)
    splits: dict[str, str] = {}
    for ids in groups.values():
        order = np.array(ids)
        rng.shuffle(order)
        n = len(order)
        n_train = int(round(n * TRAIN_FRAC))
        n_dev = int(round(n * DEV_FRAC))
        if n >= 5:
            n_train = min(max(n_train, 1), n - 2)
            n_dev = min(max(n_dev, 1), n - n_train - 1)
        splits.update({iid: "train" for iid in order[:n_train]})
        splits.update({iid: "dev" for iid in order[n_train : n_train + n_dev]})
        splits.update({iid: "test" for iid in order[n_train + n_dev :]})
    return splits


def _stack_groups(groups: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.concatenate([groups[name] for name in names], axis=0)


def _encode_all(cache: Path) -> dict[str, np.ndarray]:
    if cache.is_file():
        loaded = np.load(cache)
        return {key: loaded[key] for key in loaded.files}
    prompts = {
        **{f"color:{key}": value for key, value in COLOR_PROMPTS.items()},
        **{f"pattern:{key}": value for key, value in PATTERN_PROMPTS.items()},
        **{f"sleeve:{key}": value for key, value in SLEEVE_PROMPTS.items()},
        **{f"neck:{key}": value for key, value in NECK_PROMPTS.items()},
    }
    encoded = encode_prompt_groups(prompts, model_id=MODEL_ID)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **encoded)
    return encoded


def _color_arms(
    images: np.ndarray,
    encoded: dict[str, np.ndarray],
    labels: dict[str, dict[str, str]],
    item_ids: list[str],
    splits: dict[str, str],
    allowed_colors: list[str],
    finite: np.ndarray,
) -> dict[str, np.ndarray]:
    color_names = list(COLOR_PROMPTS)
    color_groups = [encoded[f"color:{name}"] for name in color_names]
    color_matrix = _stack_groups(encoded, [f"color:{name}" for name in color_names])
    nontarget_names = (
        [f"pattern:{name}" for name in PATTERN_PROMPTS]
        + [f"sleeve:{name}" for name in SLEEVE_PROMPTS]
        + [f"neck:{name}" for name in NECK_PROMPTS]
    )
    nontarget = _stack_groups(encoded, nontarget_names)
    arms: dict[str, np.ndarray] = {
        "baseline": l2_normalize(images),
        "fingerprint": fingerprint(images, color_groups),
    }
    train = np.array([splits[iid] == "train" for iid in item_ids]) & finite
    color_labels = np.array(
        [(labels.get(iid) or {}).get("color") or "" for iid in item_ids]
    )
    arms["data_centroid"] = project(
        images,
        data_centroid_basis(images, color_labels, train, allowed_colors, max(SVD_RANKS)),
    )
    for rank in SVD_RANKS:
        basis = subspace_basis(color_matrix, rank)
        projected = project(images, basis)
        arms[f"text_r{rank:02d}"] = projected
        removed = remove_subspace(images, subspace_basis(nontarget, rank))
        arms[f"remove_r{rank:02d}"] = project(removed, basis)
        for alpha in ALPHAS:
            name = f"mix_r{rank:02d}_a{int(round(alpha * 100)):03d}"
            arms[name] = mix(images, projected, alpha)
    return arms


def _pattern_arms(
    images: np.ndarray,
    encoded: dict[str, np.ndarray],
    labels: dict[str, dict[str, str]],
    item_ids: list[str],
    splits: dict[str, str],
    allowed_patterns: list[str],
    rank: int,
    alpha: float,
    finite: np.ndarray,
) -> dict[str, np.ndarray]:
    pattern_names = list(PATTERN_PROMPTS)
    groups = [encoded[f"pattern:{name}"] for name in pattern_names]
    matrix = _stack_groups(encoded, [f"pattern:{name}" for name in pattern_names])
    nontarget_names = (
        [f"color:{name}" for name in COLOR_PROMPTS]
        + [f"sleeve:{name}" for name in SLEEVE_PROMPTS]
        + [f"neck:{name}" for name in NECK_PROMPTS]
    )
    basis = subspace_basis(matrix, rank)
    projected = project(images, basis)
    train = np.array([splits[iid] == "train" for iid in item_ids]) & finite
    pattern_labels = np.array(
        [(labels.get(iid) or {}).get("pattern") or "" for iid in item_ids]
    )
    removed = remove_subspace(images, subspace_basis(_stack_groups(encoded, nontarget_names), rank))
    return {
        "baseline": l2_normalize(images),
        "fingerprint": fingerprint(images, groups),
        f"text_r{rank:02d}": projected,
        f"mix_r{rank:02d}_a{int(round(alpha * 100)):03d}": mix(images, projected, alpha),
        f"remove_r{rank:02d}": project(removed, basis),
        "data_centroid": project(
            images,
            data_centroid_basis(
                images, pattern_labels, train, allowed_patterns, rank
            ),
        ),
    }


def _indices(
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    splits: dict[str, str],
    finite: np.ndarray,
    attr: str,
    allowed: set[str],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    idxs: list[int] = []
    values: list[str] = []
    for i, iid in enumerate(item_ids):
        if not finite[i]:
            continue
        if split_name != "full" and splits[iid] != split_name:
            continue
        label = (labels.get(iid) or {}).get(attr)
        if label and label != "uncertain" and label in allowed:
            idxs.append(i)
            values.append(label)
    return np.asarray(idxs, dtype=int), np.asarray(values)


def _public_coarse(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "per_item_purity"
    }


def _mean_non_target(arm_metrics: dict[str, dict[str, Any]], attrs: tuple[str, ...]) -> float:
    return float(np.mean([arm_metrics[attr]["purity_at_k"] for attr in attrs]))


def _select(dev: dict[str, dict[str, dict[str, Any]]], graded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_color = dev["baseline"]["color"]["purity_at_k"]
    baseline_other = _mean_non_target(dev["baseline"], COLOR_NON_TARGETS)
    ranked: list[dict[str, Any]] = []
    for arm, attrs in dev.items():
        if arm in {"baseline", "data_centroid"}:
            continue
        other = _mean_non_target(attrs, COLOR_NON_TARGETS)
        weighted = graded.get(arm, {}).get("weighted_purity_at_k", {}).get("mean")
        ranked.append(
            {
                "arm": arm,
                "color_purity": attrs["color"]["purity_at_k"],
                "color_delta": attrs["color"]["purity_at_k"] - baseline_color,
                "non_target_mean": other,
                "non_target_drop": baseline_other - other,
                "weighted_purity": weighted,
                "eligible": (baseline_other - other) <= NON_TARGET_DROP_LIMIT,
            }
        )
    eligible = [row for row in ranked if row["eligible"]]
    pool = eligible or ranked
    rule = "dev color purity among arms whose non-target drop is within the frozen limit"
    if not eligible:
        rule = "no arm met the non-target limit; selected the highest dev color purity"
    pool.sort(
        key=lambda row: (
            row["color_purity"],
            row["weighted_purity"] if row["weighted_purity"] is not None else -1,
        ),
        reverse=True,
    )
    return {"rule": rule, "selected": pool[0]["arm"], "candidates": ranked}


def _recipe(arm: str) -> tuple[str, int, float]:
    if arm == "fingerprint":
        return "fingerprint", SVD_RANKS[0], 0.5
    if arm == "data_centroid":
        return "data_centroid", SVD_RANKS[0], 0.5
    if arm.startswith("text_r"):
        return "text", int(arm.split("_r")[1]), 0.5
    if arm.startswith("remove_r"):
        return "remove", int(arm.split("_r")[1]), 0.5
    rank = int(arm.split("_r")[1].split("_")[0])
    alpha = int(arm.split("_a")[1]) / 100
    return "mix", rank, alpha


def _transferred_name(kind: str, rank: int, alpha: float) -> str:
    if kind == "fingerprint":
        return "fingerprint"
    if kind == "text":
        return f"text_r{rank:02d}"
    if kind == "remove":
        return f"remove_r{rank:02d}"
    return f"mix_r{rank:02d}_a{int(round(alpha * 100)):03d}"


def _evaluate_coarse(
    arms: dict[str, np.ndarray],
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    splits: dict[str, str],
    finite: np.ndarray,
    allowed: dict[str, set[str]],
    split_name: str,
    attrs: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm, matrix in arms.items():
        arm_out: dict[str, Any] = {}
        for attr in attrs:
            idx, y = _indices(item_ids, labels, splits, finite, attr, allowed[attr], split_name)
            metrics = coarse_metrics(matrix[idx], y, k=NEIGHBOR_K)
            arm_out[attr] = _public_coarse(metrics)
            arm_out[attr]["_scores"] = metrics["per_item_purity"]
        out[arm] = arm_out
        print(f"coarse {split_name} {arm}", flush=True)
    return out


def _pairwise(evaluated: dict[str, dict[str, Any]], attrs: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    base = evaluated["baseline"]
    pairs: dict[str, dict[str, Any]] = {}
    for arm, attrs_metrics in evaluated.items():
        if arm == "baseline":
            continue
        pairs[arm] = {}
        for attr in attrs:
            delta = bootstrap_mean_delta(
                base[attr]["_scores"],
                attrs_metrics[attr]["_scores"],
                n_boot=BOOTSTRAP,
                seed=SEED,
            )
            pairs[arm][attr] = delta
    return pairs


def _strip_scores(evaluated: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        arm: {
            attr: {key: value for key, value in metrics.items() if key != "_scores"}
            for attr, metrics in attrs.items()
        }
        for arm, attrs in evaluated.items()
    }


def _graded_subset(
    arms: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    item_pos: dict[str, int],
    split_name: str,
) -> dict[str, Any]:
    chosen = [
        row
        for row in rows
        if row.get("status") == "ok" and (split_name == "full" or row["split"] == split_name)
    ]
    if len(chosen) < 20:
        return {"n": len(chosen)}
    idx = np.array([item_pos[row["item_id"]] for row in chosen])
    lab = np.array([row["lab"] for row in chosen], dtype=np.float64)
    fine = np.array([row["fine_color"] for row in chosen])
    de = pairwise_delta_e00(lab)
    result: dict[str, Any] = {"n": len(chosen), "arms": {}}
    base_pack = None
    packs: dict[str, dict[str, np.ndarray]] = {}
    for arm, matrix in arms.items():
        pack = graded_color_metrics(
            matrix[idx],
            lab,
            de,
        k=NEIGHBOR_K,
        tau=TAU_DE,
        near_de=NEAR_DE,
        far_de=FAR_DE,
            triplet_pairs=16,
            seed=SEED,
            fine_labels=fine,
            min_class_n=12,
        )
        packs[arm] = pack
        if arm == "baseline":
            base_pack = pack
    assert base_pack is not None
    for arm, pack in packs.items():
        public = {
            "weighted_purity_at_k": summarize_scores(pack["weighted_purity_at_k"]),
            "mean_delta_e_at_k": summarize_scores(pack["mean_delta_e_at_k"]),
            "ndcg_at_k": summarize_scores(pack["ndcg_at_k"]),
            "triplet_accuracy": summarize_scores(pack["triplet_accuracy"]),
            "centroid_spearman": pack["centroid_spearman"],
        }
        if arm != "baseline":
            public["delta"] = {
                "weighted_purity_at_k": bootstrap_mean_delta(
                    base_pack["weighted_purity_at_k"], pack["weighted_purity_at_k"], n_boot=BOOTSTRAP, seed=SEED
                ),
                "mean_delta_e_at_k": bootstrap_mean_delta(
                    base_pack["mean_delta_e_at_k"], pack["mean_delta_e_at_k"], n_boot=BOOTSTRAP, seed=SEED
                ),
                "ndcg_at_k": bootstrap_mean_delta(
                    base_pack["ndcg_at_k"], pack["ndcg_at_k"], n_boot=BOOTSTRAP, seed=SEED
                ),
                "triplet_accuracy": bootstrap_mean_delta(
                    base_pack["triplet_accuracy"], pack["triplet_accuracy"], n_boot=BOOTSTRAP, seed=SEED
                ),
            }
        result["arms"][arm] = public
        print(f"graded {split_name} {arm}", flush=True)
    return result


def _gallery(
    project: Path,
    cohort_dir: Path,
    arms: dict[str, np.ndarray],
    selected: str,
    rows: list[dict[str, Any]],
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    splits: dict[str, str],
    finite: np.ndarray,
    allowed_colors: set[str],
    raw_scores: np.ndarray,
    score_names: list[str],
) -> list[dict[str, Any]]:
    idx, _y = _indices(item_ids, labels, splits, finite, "color", allowed_colors, "test")
    by_id = {row["item_id"]: row for row in rows if row.get("status") == "ok" and row["split"] == "test"}
    candidates = [item_ids[i] for i in idx if item_ids[i] in by_id and by_id[item_ids[i]]["delta_e_to_anchor"] <= 18]
    candidates.sort(key=lambda iid: by_id[iid]["delta_e_to_anchor"])
    picked: list[str] = []
    seen: set[str] = set()
    for iid in candidates:
        color = by_id[iid]["fine_color"]
        if color in seen:
            continue
        picked.append(iid)
        seen.add(color)
        if len(picked) == 6:
            break
    pos = {iid: i for i, iid in enumerate(item_ids)}
    cohort = json.loads((cohort_dir / "cohort.json").read_text(encoding="utf-8"))
    image_paths = {
        str(row["item_id"]): Path(row["image_path"])
        for row in cohort["items"]
        if row.get("image_path")
    }
    local_ids = [item_ids[i] for i in idx]
    local_pos = {iid: i for i, iid in enumerate(local_ids)}
    galleries = []
    thumb_root = cohort_dir / "thumbs"
    out_root = project / "previews"
    for iid in picked:
        query_local = local_pos[iid]
        entry: dict[str, Any] = {
            "item_id": iid,
            "fine_color": by_id[iid]["fine_color"],
            "coarse_color": by_id[iid]["coarse_color"],
            "lab": by_id[iid]["lab"],
            "prompt_scores": _top_scores(raw_scores[pos[iid]], score_names),
            "arms": {},
        }
        _copy_thumb(
            thumb_root / f"{iid}.jpg",
            out_root / "query" / f"{iid}.jpg",
            image_paths.get(iid),
        )
        for arm in ("baseline", selected):
            sim = l2_normalize(arms[arm][idx])
            scores = sim @ sim[query_local]
            scores[query_local] = -np.inf
            order = np.argsort(-scores)[:8]
            neighbors = []
            for rank, local in enumerate(order, 1):
                nid = local_ids[int(local)]
                _copy_thumb(
                    thumb_root / f"{nid}.jpg",
                    out_root / arm / f"{nid}.jpg",
                    image_paths.get(nid),
                )
                ref = by_id.get(nid)
                neighbors.append(
                    {
                        "rank": rank,
                        "item_id": nid,
                        "coarse_color": (labels.get(nid) or {}).get("color"),
                        "fine_color": None if ref is None else ref.get("fine_color"),
                        "image": f"previews/{arm}/{nid}.jpg",
                    }
                )
            entry["arms"][arm] = neighbors
        entry["image"] = f"previews/query/{iid}.jpg"
        galleries.append(entry)
    return galleries


def _top_scores(scores: np.ndarray, names: list[str]) -> list[dict[str, float | str]]:
    order = np.argsort(-scores)[:3]
    return [{"name": names[int(i)], "score": round(float(scores[int(i)]), 4)} for i in order]


def _copy_thumb(src: Path, dest: Path, fallback_image: Path | None = None) -> None:
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        dest.write_bytes(src.read_bytes())
        return
    if fallback_image is None or not fallback_image.is_file():
        return
    from PIL import Image

    with Image.open(fallback_image) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((384, 384))
        rgb.save(dest, format="JPEG", quality=85)


def _success(test_pairs: dict[str, Any], graded: dict[str, Any], selected: str, test_coarse: dict[str, Any]) -> dict[str, Any]:
    color = test_pairs[selected]["color"]
    graded_arm = graded["arms"][selected]
    base_other = _mean_non_target(test_coarse["baseline"], COLOR_NON_TARGETS)
    arm_other = _mean_non_target(test_coarse[selected], COLOR_NON_TARGETS)
    checks = {
        "coarse_color_ci_positive": bool(color and color["ci_low"] > 0),
        "weighted_purity_ci_positive": _ci_positive(graded_arm["delta"]["weighted_purity_at_k"]),
        "ndcg_ci_positive": _ci_positive(graded_arm["delta"]["ndcg_at_k"]),
        "mean_delta_e_ci_negative": _ci_negative(graded_arm["delta"]["mean_delta_e_at_k"]),
        "non_target_drop_within_limit": (base_other - arm_other) <= NON_TARGET_DROP_LIMIT,
    }
    return {"passed": all(checks.values()), "checks": checks, "non_target_drop": base_other - arm_other}


def _ci_positive(delta: dict[str, float] | None) -> bool:
    return bool(delta and delta["ci_low"] > 0)


def _ci_negative(delta: dict[str, float] | None) -> bool:
    return bool(delta and delta["ci_high"] < 0)


def main() -> None:
    cohort_dir = REPO / SOURCE_COHORT
    project = REPO / "artifacts" / "attribute_conditioned_ranking" / PROJECT_ID
    project.mkdir(parents=True, exist_ok=True)
    _write_json(project / "protocol.json", as_dict())
    item_ids = json.loads((cohort_dir / "embeddings" / "item_ids.json").read_text())["item_ids"]
    labels = _load_labels(cohort_dir)
    cohort = json.loads((cohort_dir / "cohort.json").read_text())
    allowed = {attr: set(cohort["attributes"][attr]) for attr in COARSE_ATTRS}
    embedding_path = REPO / SOURCE_EMBEDDING
    images = np.load(embedding_path).astype(np.float32)
    if len(images) != len(item_ids):
        raise ValueError(f"{embedding_path} rows {len(images)} != item_ids {len(item_ids)}")
    finite = np.isfinite(images).all(axis=1)
    images = np.where(finite[:, None], images, 0.0).astype(np.float32)
    split_path = project / "splits.json"
    if split_path.is_file():
        splits = json.loads(split_path.read_text())
    else:
        splits = _assign_splits(item_ids, labels, SEED)
        _write_json(split_path, splits)
    ref_path = project / "color_reference.jsonl"
    if ref_path.is_file():
        reference = [json.loads(line) for line in ref_path.read_text().splitlines() if line.strip()]
    else:
        reference = build_color_reference(
            cohort_dir=cohort_dir,
            item_ids=item_ids,
            labels=labels,
            splits=splits,
            seed=SEED,
        )
        ref_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reference),
            encoding="utf-8",
        )
    encoded = _encode_all(project / "text_prompts.npz")
    color_arms = _color_arms(
        images, encoded, labels, item_ids, splits, cohort["attributes"]["color"], finite
    )
    emb_dir = project / "embeddings" / "color"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in color_arms.items():
        np.save(emb_dir / f"{name}.npy", matrix.astype(np.float32))
    raw_names = list(COLOR_PROMPTS)
    raw_text = np.stack(
        [l2_normalize(encoded[f"color:{name}"].mean(axis=0)) for name in raw_names],
        axis=0,
    )
    raw_scores = images @ raw_text.T
    np.save(emb_dir / "fingerprint_raw.npy", raw_scores.astype(np.float32))

    dev = _evaluate_coarse(
        color_arms, item_ids, labels, splits, finite, allowed, "dev", ("color",) + COLOR_NON_TARGETS
    )
    item_pos = {iid: i for i, iid in enumerate(item_ids)}
    graded_dev = _graded_subset(color_arms, reference, item_pos, "dev")
    selection = _select(
        {arm: {attr: metrics for attr, metrics in attrs.items()} for arm, attrs in dev.items()},
        graded_dev.get("arms") or {},
    )
    test = _evaluate_coarse(
        color_arms, item_ids, labels, splits, finite, allowed, "test", COARSE_ATTRS
    )
    full = _evaluate_coarse(
        color_arms, item_ids, labels, splits, finite, allowed, "full", COARSE_ATTRS
    )
    graded_test = _graded_subset(color_arms, reference, item_pos, "test")
    pairs_test = _pairwise(test, COARSE_ATTRS)
    structure = {}
    color_idx, _y = _indices(item_ids, labels, splits, finite, "color", allowed["color"], "test")
    for arm, matrix in color_arms.items():
        if arm == "baseline":
            continue
        structure[arm] = {
            "jaccard_at_10": neighbor_jaccard(color_arms["baseline"][color_idx], matrix[color_idx], NEIGHBOR_K),
            "similarity_spearman": similarity_spearman(color_arms["baseline"][color_idx], matrix[color_idx]),
        }
        print(f"structure {arm}", flush=True)
    kind, rank, alpha = _recipe(selection["selected"])
    pattern_arms = _pattern_arms(
        images,
        encoded,
        labels,
        item_ids,
        splits,
        cohort["attributes"]["pattern"],
        rank,
        alpha,
        finite,
    )
    pattern_dir = project / "embeddings" / "pattern"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in pattern_arms.items():
        np.save(pattern_dir / f"{name}.npy", matrix.astype(np.float32))
    pattern_test = _evaluate_coarse(
        pattern_arms, item_ids, labels, splits, finite, allowed, "test", COARSE_ATTRS
    )
    pattern_pairs = _pairwise(pattern_test, COARSE_ATTRS)
    transferred = _transferred_name(kind, rank, alpha)
    build_scatter(
        project=project,
        cohort_dir=cohort_dir,
        cohort=cohort,
        item_ids=item_ids,
        labels=labels,
        finite=finite,
        reference=reference,
        arms={
            "baseline": color_arms["baseline"],
            "color_control": color_arms[selection["selected"]],
            "pattern_control": pattern_arms[transferred],
        },
        panels=[
            ("baseline", f"{SOURCE_PREPROCESS} · 색상으로 표시", "color"),
            ("color_control", f"색상 제어 · {selection['selected']}", "color"),
            ("baseline", f"{SOURCE_PREPROCESS} · 무늬로 표시", "pattern"),
            ("pattern_control", f"무늬 제어 · {transferred}", "pattern"),
        ],
        seed=SEED,
    )
    gallery = _gallery(
        project,
        cohort_dir,
        color_arms,
        selection["selected"],
        reference,
        item_ids,
        labels,
        splits,
        finite,
        allowed["color"],
        raw_scores,
        raw_names,
    )
    payload = {
        "date": DATE,
        "preprocess": SOURCE_PREPROCESS,
        "n": len(item_ids),
        "finite_n": int(finite.sum()),
        "split_counts": {name: int(sum(v == name for v in splits.values())) for name in ("train", "dev", "test")},
        "color_reference": {
            "n_solid_ok": int(sum(row.get("status") == "ok" for row in reference)),
            "n_solid_rows": len(reference),
        },
        "selection": selection,
        "transferred_pattern_arm": transferred,
        "pattern_recipe": {"kind": kind, "rank": rank, "alpha": alpha},
        "color_test": _strip_scores(test),
        "color_test_pairs": pairs_test,
        "color_full": _strip_scores(full),
        "color_dev": _strip_scores(dev),
        "graded_dev": graded_dev,
        "graded_test": graded_test,
        "structure_test": structure,
        "pattern_test": _strip_scores(pattern_test),
        "pattern_test_pairs": pattern_pairs,
        "success": _success(pairs_test, graded_test, selection["selected"], test),
        "gallery": gallery,
    }
    _write_json(project / "metrics" / "evaluation.json", payload)
    from build_report import build

    print(build(project))


def _attribute_arms(
    images: np.ndarray,
    encoded: dict[str, np.ndarray],
    labels: dict[str, dict[str, str]],
    item_ids: list[str],
    splits: dict[str, str],
    target: str,
    allowed_labels: list[str],
    finite: np.ndarray,
) -> dict[str, np.ndarray]:
    groups = PROMPT_GROUPS[target]
    target_groups = [encoded[f"{target}:{name}"] for name in groups]
    target_matrix = _stack_groups(encoded, [f"{target}:{name}" for name in groups])
    other_names = removal_prompt_names(target)
    nontarget = _stack_groups(encoded, other_names)
    arms: dict[str, np.ndarray] = {
        "baseline": l2_normalize(images),
        "fingerprint": fingerprint(images, target_groups),
    }
    train = np.array([splits[iid] == "train" for iid in item_ids]) & finite
    target_labels = np.array([(labels.get(iid) or {}).get(target) or "" for iid in item_ids])
    arms["data_centroid"] = project(
        images,
        data_centroid_basis(images, target_labels, train, allowed_labels, max(SVD_RANKS)),
    )
    for rank in SVD_RANKS:
        basis = subspace_basis(target_matrix, rank)
        projected = project(images, basis)
        arms[f"text_r{rank:02d}"] = projected
        removed = remove_subspace(images, subspace_basis(nontarget, rank))
        arms[f"remove_r{rank:02d}"] = project(removed, basis)
        for alpha in ALPHAS:
            name = f"mix_r{rank:02d}_a{int(round(alpha * 100)):03d}"
            arms[name] = mix(images, projected, alpha)
    return arms


def _select_target(
    dev: dict[str, dict[str, dict[str, Any]]],
    target: str,
    non_targets: tuple[str, ...],
) -> dict[str, Any]:
    baseline_target = dev["baseline"][target]["purity_at_k"]
    baseline_other = _mean_non_target(dev["baseline"], non_targets)
    ranked: list[dict[str, Any]] = []
    for arm, attrs in dev.items():
        if arm in {"baseline", "data_centroid"}:
            continue
        other = _mean_non_target(attrs, non_targets)
        ranked.append(
            {
                "arm": arm,
                "target_purity": attrs[target]["purity_at_k"],
                "target_delta": attrs[target]["purity_at_k"] - baseline_target,
                "non_target_mean": other,
                "non_target_drop": baseline_other - other,
                "eligible": (baseline_other - other) <= NON_TARGET_DROP_LIMIT,
            }
        )
    eligible = [row for row in ranked if row["eligible"]]
    pool = eligible or ranked
    rule = "dev target purity among arms whose non-target drop is within the frozen limit"
    if not eligible:
        rule = "no arm met the non-target limit; selected the highest dev target purity"
    pool.sort(key=lambda row: row["target_purity"], reverse=True)
    return {"rule": rule, "selected": pool[0]["arm"], "candidates": ranked}


def _attribute_gallery(
    project: Path,
    cohort_dir: Path,
    arms: dict[str, np.ndarray],
    selected: str,
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    splits: dict[str, str],
    finite: np.ndarray,
    allowed: set[str],
    target: str,
    raw_scores: np.ndarray,
    score_names: list[str],
) -> list[dict[str, Any]]:
    idx, values = _indices(item_ids, labels, splits, finite, target, allowed, "test")
    candidates = list(zip([item_ids[i] for i in idx], values.tolist(), strict=True))
    picked: list[tuple[str, str]] = []
    seen: set[str] = set()
    for iid, label in candidates:
        if label in seen:
            continue
        picked.append((iid, label))
        seen.add(label)
        if len(picked) == 6:
            break
    cohort = json.loads((cohort_dir / "cohort.json").read_text(encoding="utf-8"))
    image_paths = {
        str(row["item_id"]): Path(row["image_path"])
        for row in cohort["items"]
        if row.get("image_path")
    }
    pos = {iid: i for i, iid in enumerate(item_ids)}
    local_ids = [item_ids[i] for i in idx]
    local_pos = {iid: i for i, iid in enumerate(local_ids)}
    label_of = {iid: label for iid, label in candidates}
    thumb_root = cohort_dir / "thumbs"
    out_root = project / "previews"
    galleries = []
    for iid, label in picked:
        entry: dict[str, Any] = {
            "item_id": iid,
            "label": label,
            "prompt_scores": _top_scores(raw_scores[pos[iid]], score_names),
            "arms": {},
        }
        _copy_thumb(thumb_root / f"{iid}.jpg", out_root / "query" / f"{iid}.jpg", image_paths.get(iid))
        for arm in ("baseline", selected):
            sim = l2_normalize(arms[arm][idx])
            scores = sim @ sim[local_pos[iid]]
            scores[local_pos[iid]] = -np.inf
            order = np.argsort(-scores)[:8]
            neighbors = []
            for rank, local in enumerate(order, 1):
                nid = local_ids[int(local)]
                _copy_thumb(
                    thumb_root / f"{nid}.jpg",
                    out_root / arm / f"{nid}.jpg",
                    image_paths.get(nid),
                )
                neighbors.append(
                    {
                        "rank": rank,
                        "item_id": nid,
                        "label": label_of.get(nid) or (labels.get(nid) or {}).get(target),
                        "image": f"previews/{arm}/{nid}.jpg",
                    }
                )
            entry["arms"][arm] = neighbors
        entry["image"] = f"previews/query/{iid}.jpg"
        galleries.append(entry)
    return galleries


def _success_target(
    test_pairs: dict[str, Any],
    selected: str,
    test_coarse: dict[str, Any],
    target: str,
    non_targets: tuple[str, ...],
) -> dict[str, Any]:
    target_delta = test_pairs[selected][target]
    base_other = _mean_non_target(test_coarse["baseline"], non_targets)
    arm_other = _mean_non_target(test_coarse[selected], non_targets)
    checks = {
        f"test {target} purity 95% CI가 0보다 큼": bool(target_delta and target_delta["ci_low"] > 0),
        "비타깃 평균 하락이 2%p 이내": (base_other - arm_other) <= NON_TARGET_DROP_LIMIT,
    }
    return {"passed": all(checks.values()), "checks": checks, "non_target_drop": base_other - arm_other}


def _encode_prompts(cache: Path) -> dict[str, np.ndarray]:
    if cache.is_file():
        loaded = np.load(cache)
        return {key: loaded[key] for key in loaded.files}
    prompts = {
        f"{attr}:{key}": phrase
        for attr, groups in PROMPT_GROUPS.items()
        for key, phrase in groups.items()
    }
    encoded = encode_prompt_groups(prompts, model_id=MODEL_ID)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **encoded)
    return encoded


def run_attribute(target: str, encoded: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    if target not in PROMPT_GROUPS:
        raise ValueError(f"unknown target {target}")
    cohort_dir = REPO / SOURCE_COHORT
    project = REPO / "artifacts" / "attribute_conditioned_ranking" / project_id(target)
    project.mkdir(parents=True, exist_ok=True)
    protocol = as_dict()
    protocol["project_id"] = project.name
    protocol["target"] = target
    _write_json(project / "protocol.json", protocol)
    item_ids = json.loads((cohort_dir / "embeddings" / "item_ids.json").read_text())["item_ids"]
    labels = _load_labels(cohort_dir)
    cohort = json.loads((cohort_dir / "cohort.json").read_text())
    allowed = {attr: set(cohort["attributes"][attr]) for attr in COARSE_ATTRS}
    images = np.load(REPO / SOURCE_EMBEDDING).astype(np.float32)
    if len(images) != len(item_ids):
        raise ValueError(f"rows {len(images)} != item_ids {len(item_ids)}")
    finite = np.isfinite(images).all(axis=1)
    images = np.where(finite[:, None], images, 0.0).astype(np.float32)
    split_path = project / "splits.json"
    if split_path.is_file():
        splits = json.loads(split_path.read_text())
    else:
        splits = _assign_splits(item_ids, labels, SEED, stratify=target)
        _write_json(split_path, splits)
    shared_cache = REPO / "artifacts" / "attribute_conditioned_ranking" / "text_prompts_with_fit.npz"
    if encoded is None:
        encoded = _encode_prompts(shared_cache)
    project_cache = project / "text_prompts.npz"
    if not project_cache.is_file():
        np.savez(project_cache, **encoded)
    non_targets = decision_non_targets(target)
    arms = _attribute_arms(
        images,
        encoded,
        labels,
        item_ids,
        splits,
        target,
        cohort["attributes"][target],
        finite,
    )
    emb_dir = project / "embeddings" / target
    emb_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in arms.items():
        np.save(emb_dir / f"{name}.npy", matrix.astype(np.float32))
    score_names = list(PROMPT_GROUPS[target])
    raw_text = np.stack(
        [l2_normalize(encoded[f"{target}:{name}"].mean(axis=0)) for name in score_names],
        axis=0,
    )
    raw_scores = images @ raw_text.T
    np.save(emb_dir / "fingerprint_raw.npy", raw_scores.astype(np.float32))
    dev = _evaluate_coarse(
        arms, item_ids, labels, splits, finite, allowed, "dev", (target,) + non_targets
    )
    selection = _select_target(dev, target, non_targets)
    test = _evaluate_coarse(arms, item_ids, labels, splits, finite, allowed, "test", COARSE_ATTRS)
    full = _evaluate_coarse(arms, item_ids, labels, splits, finite, allowed, "full", COARSE_ATTRS)
    pairs_test = _pairwise(test, COARSE_ATTRS)
    structure = {}
    target_idx, target_values = _indices(
        item_ids, labels, splits, finite, target, allowed[target], "test"
    )
    for arm, matrix in arms.items():
        if arm == "baseline":
            continue
        structure[arm] = {
            "jaccard_at_10": neighbor_jaccard(arms["baseline"][target_idx], matrix[target_idx], NEIGHBOR_K),
            "similarity_spearman": similarity_spearman(arms["baseline"][target_idx], matrix[target_idx]),
        }
        print(f"structure {target} {arm}", flush=True)
    contrast = "pattern" if target == "color" else "color"
    selected = selection["selected"]
    attr_ko = {"pattern": "무늬", "sleeve": "소매", "neck": "넥라인", "color": "색상", "fit": "핏"}

    def _ro(word: str) -> str:
        code = ord(word[-1]) - 0xAC00
        jong = code % 28 if 0 <= code <= 11172 else 0
        return word + ("으로" if jong else "로")

    target_ko = attr_ko[target]
    contrast_ko = attr_ko[contrast]
    build_scatter(
        project=project,
        cohort_dir=cohort_dir,
        cohort=cohort,
        item_ids=item_ids,
        labels=labels,
        finite=finite,
        reference=[],
        arms={"baseline": arms["baseline"], "control": arms[selected]},
        panels=[
            ("baseline", f"{SOURCE_PREPROCESS} · {_ro(target_ko)} 표시", target),
            ("control", f"{target_ko} 제어 · {selected}", target),
            ("baseline", f"{SOURCE_PREPROCESS} · {_ro(contrast_ko)} 표시", contrast),
            ("control", f"{target_ko} 제어 · {selected} · {_ro(contrast_ko)} 표시", contrast),
        ],
        seed=SEED,
    )
    gallery = _attribute_gallery(
        project,
        cohort_dir,
        arms,
        selected,
        item_ids,
        labels,
        splits,
        finite,
        allowed[target],
        target,
        raw_scores,
        score_names,
    )
    class_counts: dict[str, int] = {}
    for label in target_values.tolist():
        class_counts[label] = class_counts.get(label, 0) + 1
    removal_attributes = [attr for attr in PROMPT_GROUPS if attr != target]
    payload = {
        "date": DATE,
        "target": target,
        "preprocess": SOURCE_PREPROCESS,
        "n": len(item_ids),
        "finite_n": int(finite.sum()),
        "split_counts": {name: int(sum(value == name for value in splits.values())) for name in ("train", "dev", "test")},
        "non_targets": list(non_targets),
        "removal_attributes": removal_attributes,
        "class_counts": class_counts,
        "selection": selection,
        "test": _strip_scores(test),
        "test_pairs": pairs_test,
        "full": _strip_scores(full),
        "dev": _strip_scores(dev),
        "structure_test": structure,
        "success": _success_target(pairs_test, selected, test, target, non_targets),
        "gallery": gallery,
    }
    _write_json(project / "metrics" / "evaluation.json", payload)
    from build_report import build

    print(build(project), flush=True)
    return encoded


if __name__ == "__main__":
    parser = ArgumentParser(description="Run one attribute-conditioned ranking project per target.")
    parser.add_argument("--target", nargs="+", required=True, choices=tuple(PROMPT_GROUPS))
    args = parser.parse_args()
    encoded: dict[str, np.ndarray] | None = None
    for target in args.target:
        encoded = run_attribute(target, encoded)
