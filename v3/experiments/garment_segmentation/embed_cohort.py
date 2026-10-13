"""Assemble cohort embeddings: reuse seed runs, encode supplements with SCHP arms."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapters.base import AdapterContext
from adapters.schp import SchpSegmentor
from io_utils import read_json, read_jsonl, restore_mask, write_json, write_jsonl
from preprocess import (
    ALL_ARMS,
    DEFAULT_GARMENT_LABELS,
    build_arm_images,
    pad_box,
    resolve_mask_in_box,
)


def _load_attr_repr():
    import importlib.util

    attr_dir = HERE.parent / "attribute_embedding"
    if str(attr_dir) not in sys.path:
        sys.path.insert(0, str(attr_dir))
    path = attr_dir / "representations.py"
    spec = importlib.util.spec_from_file_location("attr_emb_repr", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument(
        "--sam3-run",
        type=Path,
        default=REPO_ROOT
        / "artifacts/garment_segmentation/_archive/superseded/preprocess-ac-sam3-450",
    )
    p.add_argument(
        "--abcd-run",
        type=Path,
        default=REPO_ROOT
        / "artifacts/garment_segmentation/_archive/superseded/preprocess-abcd-450",
    )
    p.add_argument("--config", type=Path, default=HERE / "config.preprocess.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--skip-schp", action="store_true", help="Only assemble A for supplements")
    p.add_argument("--limit-supp", type=int, default=None)
    return p.parse_args()


def _seed_index(records_path: Path) -> dict[str, int]:
    rows = read_jsonl(records_path)
    return {str(row["item_id"]): i for i, row in enumerate(rows)}


def _embed_images(images: list[Image.Image], *, model, preprocess, device: str, batch_size: int) -> np.ndarray:
    import torch

    repr_mod = _load_attr_repr()
    vectors: list[np.ndarray] = []
    total = len(images)
    for start in range(0, total, batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(image.convert("RGB")) for image in batch]).to(device)
        with torch.inference_mode():
            encoded = model.encode_image(tensors, normalize=True)
        vectors.append(encoded.detach().float().cpu().numpy())
        print(f"  embed {min(start + batch_size, total)}/{total}", flush=True)
    return repr_mod.l2_normalize(np.concatenate(vectors, axis=0).astype(np.float32))


def _pick_top_proposal(proposals: list[dict[str, Any]]) -> dict[str, Any] | None:
    tops = [
        p
        for p in proposals
        if str(p.get("label") or "").lower().replace("_", "-")
        in {"upper-clothes", "dress", "top", "shirt", "blouse", "hoodie", "sweater", "t-shirt", "tshirt"}
    ]
    pool = tops or proposals
    if not pool:
        return None
    return max(pool, key=lambda p: float(p.get("score") or 0.0))


def _box_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return [0.0, 0.0, float(w), float(h)]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def main() -> None:
    args = parse_args()
    root = args.cohort_dir
    cohort = read_json(root / "cohort.json")
    items = cohort["items"]
    item_ids = [str(row["item_id"]) for row in items]
    n = len(item_ids)
    id_to_pos = {iid: i for i, iid in enumerate(item_ids)}

    sam3_idx = _seed_index(args.sam3_run / "dataset" / "records.jsonl")
    abcd_idx = _seed_index(args.abcd_run / "dataset" / "records.jsonl")
    print(f"cohort={n} sam3_seed={len(sam3_idx)} abcd_seed={len(abcd_idx)}", flush=True)

    out = root / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "item_ids.json", {"item_ids": item_ids, "n": n})

    # --- SAM3 A/C: copy seed ---
    sam3_dir = out / "sam3"
    sam3_dir.mkdir(exist_ok=True)
    sam3_A = np.full((n, 768), np.nan, dtype=np.float32)
    sam3_C = np.full((n, 768), np.nan, dtype=np.float32)
    src_A = np.load(args.sam3_run / "embeddings" / "A.npy")
    src_C = np.load(args.sam3_run / "embeddings" / "C.npy")
    for iid, j in sam3_idx.items():
        if iid in id_to_pos:
            i = id_to_pos[iid]
            sam3_A[i] = src_A[j]
            sam3_C[i] = src_C[j]

    # --- ABCD: copy seed ---
    abcd_dir = out / "abcd"
    abcd_dir.mkdir(exist_ok=True)
    abcd_emb = {arm: np.full((n, 768), np.nan, dtype=np.float32) for arm in ALL_ARMS}
    for arm in ALL_ARMS:
        src = np.load(args.abcd_run / "embeddings" / f"{arm}.npy")
        for iid, j in abcd_idx.items():
            if iid in id_to_pos:
                abcd_emb[arm][id_to_pos[iid]] = src[j]

    supplements = [row for row in items if row.get("origin") == "pool_supplement"]
    if args.limit_supp:
        supplements = supplements[: args.limit_supp]
    print(f"supplements_to_encode={len(supplements)}", flush=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    preprocess_cfg = dict(config.get("preprocess") or {})
    pad_ratio = float(preprocess_cfg.get("pad_ratio") or 0.12)
    dilate_px = int(preprocess_cfg.get("dilate_px") or 7)
    alpha_sigma = float(preprocess_cfg.get("alpha_sigma") or 5.0)
    bg_blur = float(preprocess_cfg.get("background_blur_sigma") or 21.0)
    min_fill = float(preprocess_cfg.get("min_mask_fill") or 0.08)
    garment_labels = tuple(preprocess_cfg.get("garment_labels") or DEFAULT_GARMENT_LABELS)

    repr_mod = _load_attr_repr()
    device = args.device or config.get("device") or "auto"
    encoder_id = str(config.get("encoder_id") or "Marqo/marqo-fashionSigLIP")
    model, preprocess, _tok, device = repr_mod.load_model(encoder_id, device)
    print(f"encoder device={device}", flush=True)

    # Always encode arm A for supplements (also fill sam3_A / abcd A)
    a_images: list[Image.Image] = []
    a_positions: list[int] = []
    for row in supplements:
        path = Path(row["image_path"])
        img = Image.open(path).convert("RGB")
        i = id_to_pos[row["item_id"]]
        a_images.append(img)
        a_positions.append(i)
    if a_images:
        print("embedding supplement arm A …", flush=True)
        vecs = _embed_images(
            a_images, model=model, preprocess=preprocess, device=device, batch_size=args.batch_size
        )
        for pos, vec in zip(a_positions, vecs):
            sam3_A[pos] = vec
            abcd_emb["A"][pos] = vec

    mask_meta: list[dict[str, Any]] = []
    if not args.skip_schp and supplements:
        print("running SCHP for supplement B/C/D …", flush=True)
        schp = SchpSegmentor(requested_device=device, download_weights=False)
        if not schp.available():
            raise RuntimeError("SCHP/segformer unavailable")
        schp.load()
        arm_images: dict[str, list[Image.Image]] = {arm: [] for arm in ("B", "C", "D")}
        arm_positions: list[int] = []
        ctx = AdapterContext(prompts=["top"], box_threshold=0.25, text_threshold=0.25, max_proposals=12)
        for row in supplements:
            path = Path(row["image_path"])
            rgb = Image.open(path).convert("RGB")
            w, h = rgb.size
            proposals = schp.infer(rgb, ctx)
            selected = _pick_top_proposal(proposals)
            if selected is None:
                box = [0.0, 0.0, float(w), float(h)]
                mask = np.ones((h, w), dtype=bool)
                source = "full_image_fallback"
            else:
                mask = np.asarray(restore_mask(selected, h, w), dtype=bool)
                box = _box_from_mask(mask)
                source = "schp_top"
            padded = pad_box(box, w, h, pad_ratio=pad_ratio)
            resolved = resolve_mask_in_box(
                height=h,
                width=w,
                box_xyxy=padded,
                proposals=proposals,
                combined_selected=selected,
                garment_labels=garment_labels,
                min_mask_fill=min_fill,
            )
            arms = build_arm_images(
                rgb,
                box_xyxy=box,
                mask=resolved["mask"],
                pad_ratio=pad_ratio,
                dilate_px=dilate_px,
                alpha_sigma=alpha_sigma,
                background_blur_sigma=bg_blur,
                arms=("A", "B", "C", "D"),
            )
            i = id_to_pos[row["item_id"]]
            arm_positions.append(i)
            for arm in ("B", "C", "D"):
                arm_images[arm].append(arms[arm])
            mask_meta.append(
                {
                    "item_id": row["item_id"],
                    "mask_source": source,
                    "resolve_source": resolved.get("source"),
                    "fill_ratio": resolved.get("fill_ratio"),
                    "box_xyxy": padded,
                }
            )
            print(f"  schp {len(mask_meta)}/{len(supplements)} {row['item_id']} {source}", flush=True)

        for arm in ("B", "C", "D"):
            print(f"embedding supplement arm {arm} …", flush=True)
            vecs = _embed_images(
                arm_images[arm],
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=args.batch_size,
            )
            for pos, vec in zip(arm_positions, vecs):
                abcd_emb[arm][pos] = vec
                if arm == "C":
                    # provisional C for sam3 bank until true SAM3 masks exist
                    sam3_C[pos] = vec

    np.save(sam3_dir / "A.npy", sam3_A)
    np.save(sam3_dir / "C.npy", sam3_C)
    for arm, mat in abcd_emb.items():
        np.save(abcd_dir / f"{arm}.npy", mat)
    if mask_meta:
        write_jsonl(out / "supplement_mask_meta.jsonl", mask_meta)

    def _coverage(mat: np.ndarray) -> int:
        return int(np.isfinite(mat).all(axis=1).sum())

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "n": n,
        "supplements_encoded": len(supplements),
        "sam3": {"A": _coverage(sam3_A), "C": _coverage(sam3_C), "C_note": "seed=sam3; supplements=SCHP soft C until SAM3 run"},
        "abcd": {arm: _coverage(mat) for arm, mat in abcd_emb.items()},
        "encoder_id": encoder_id,
        "device": device,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
