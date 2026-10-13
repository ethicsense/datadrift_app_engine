"""Embed soft-mask pad/blur sweep variants (C_1..C_3) for attribute cluster eval."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_utils import read_json, read_jsonl, write_json
from preprocess import (
    DEFAULT_GARMENT_LABELS,
    compose_soft_variant,
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
    p.add_argument("--config", type=Path, default=HERE / "config.preprocess.soft_sweep.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Encode only first N eligible items (smoke)")
    p.add_argument("--skip-encode", action="store_true", help="Only assemble reused A/B/C_0 bank")
    p.add_argument("--variants", nargs="*", default=None, help="Subset of variants to encode")
    p.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing variant .npy without copying to embeddings/.../archive/",
    )
    return p.parse_args()


def _index_jsonl(path: Path, key: str = "item_id") -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path) if row.get(key)}


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


def _save_preview(image: Image.Image, path: Path, size: int = 384) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if max(rgb.size) > size:
        rgb = ImageOps.contain(rgb, (size, size))
    rgb.save(path, format="JPEG", quality=85, optimize=True)


def _load_sam3_index(
    sam3_root: Path,
    run_names: list[str],
    *,
    box_model: str,
    box_resolver: str,
    mask_model: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in run_names:
        run_dir = sam3_root / name
        box_path = run_dir / "selected" / box_model / box_resolver / "assignments.jsonl"
        prop_path = run_dir / "proposals" / mask_model / "proposals.jsonl"
        if not box_path.is_file() or not prop_path.is_file():
            print(f"skip sam3 run missing files: {name}", flush=True)
            continue
        boxes = _index_jsonl(box_path)
        props = {
            str(row["item_id"]): row.get("proposals") or []
            for row in read_jsonl(prop_path)
            if row.get("item_id")
        }
        n_added = 0
        for item_id, box_row in boxes.items():
            selected = box_row.get("selected") or {}
            if not selected.get("box_xyxy"):
                continue
            out[item_id] = {
                "box_xyxy": selected["box_xyxy"],
                "selected": selected,
                "proposals": props.get(item_id) or [],
                "run": name,
            }
            n_added += 1
        print(f"indexed {name}: {n_added} selected (bank size now {len(out)})", flush=True)
    return out


def _render_variant(
    rgb: Image.Image,
    *,
    box_xyxy: list[float],
    proposals: list[dict[str, Any]],
    selected: dict[str, Any],
    pad_ratio: float,
    dilate_px: int,
    alpha_sigma: float,
    background_blur_sigma: float,
    garment_labels: tuple[str, ...],
    min_fill: float,
) -> tuple[Image.Image, dict[str, Any]]:
    width, height = rgb.size
    padded = pad_box(box_xyxy, width, height, pad_ratio=pad_ratio)
    resolved = resolve_mask_in_box(
        height=height,
        width=width,
        box_xyxy=padded,
        proposals=proposals,
        combined_selected=selected,
        garment_labels=garment_labels,
        min_mask_fill=min_fill,
    )
    img = compose_soft_variant(
        rgb,
        box_xyxy=box_xyxy,
        mask=resolved["mask"],
        pad_ratio=pad_ratio,
        dilate_px=dilate_px,
        alpha_sigma=alpha_sigma,
        background_blur_sigma=background_blur_sigma,
    )
    return img, {
        "mask_source": resolved["source"],
        "fill_ratio": resolved["fill_ratio"],
        "pad_ratio": pad_ratio,
        "background_blur_sigma": background_blur_sigma,
    }


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_file():
        example = HERE / "config.preprocess.soft_sweep.example.yaml"
        raise FileNotFoundError(f"missing {cfg_path}; copy from {example}")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cohort_dir = (REPO_ROOT / config["cohort_dir"]).resolve()
    source_bank = str(config.get("source_bank") or "sam3_abcd")
    output_bank = str(config.get("output_bank") or "sam3_soft_sweep")
    src_dir = cohort_dir / "embeddings" / source_bank
    out_dir = cohort_dir / "embeddings" / output_bank
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids = read_json(cohort_dir / "embeddings" / "item_ids.json")["item_ids"]
    cohort = read_json(cohort_dir / "cohort.json")
    items_by_id = {str(row["item_id"]): row for row in cohort["items"]}
    n = len(item_ids)
    pos = {iid: i for i, iid in enumerate(item_ids)}

    src_a = np.load(src_dir / "A.npy")
    src_b = np.load(src_dir / "B.npy")
    src_c = np.load(src_dir / "C.npy")
    if src_a.shape[0] != n:
        raise ValueError(f"source bank length {src_a.shape[0]} != item_ids {n}")

    np.save(out_dir / "A.npy", src_a)
    np.save(out_dir / "B.npy", src_b)
    np.save(out_dir / "C_0.npy", src_c)

    preprocess_cfg = dict(config.get("preprocess") or {})
    dilate_px = int(preprocess_cfg.get("dilate_px") or 7)
    alpha_sigma = float(preprocess_cfg.get("alpha_sigma") or 5.0)
    min_fill = float(preprocess_cfg.get("min_mask_fill") or 0.08)
    garment_labels = tuple(preprocess_cfg.get("garment_labels") or DEFAULT_GARMENT_LABELS)
    variants_cfg = dict(config.get("variants") or {})
    encode_names = [
        name
        for name, spec in variants_cfg.items()
        if not spec.get("reuse_from") and (args.variants is None or name in args.variants)
    ]

    finite_a = np.isfinite(src_a).all(axis=1)
    eligible_ids = [iid for iid, ok in zip(item_ids, finite_a) if ok]
    if args.limit is not None:
        eligible_ids = eligible_ids[: args.limit]
    print(f"eligible finite A: {int(finite_a.sum())}; encoding up to {len(eligible_ids)}", flush=True)

    sam3_root = cohort_dir / "pipeline" / "sam3"
    sam3_index = _load_sam3_index(
        sam3_root,
        list(config.get("sam3_runs") or []),
        box_model=str(config.get("box_model") or "sam3"),
        box_resolver=str(config.get("box_resolver") or "combined"),
        mask_model=str(config.get("mask_model") or "sam3"),
    )

    preview_ids = set(str(x) for x in (config.get("preview_item_ids") or []))
    preview_root = cohort_dir / "previews" / "soft_sweep"
    for iid in preview_ids:
        src_prev = (
            cohort_dir
            / "pipeline"
            / "preprocess"
            / "preprocess-abcd-sam3-450"
            / "previews"
            / "C"
            / f"{iid}.jpg"
        )
        if src_prev.is_file():
            dest = preview_root / "C_0" / f"{iid}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_prev, dest)

    mats = {name: np.full((n, 768), np.nan, dtype=np.float32) for name in encode_names}
    encode_meta: list[dict[str, Any]] = []
    skipped = 0

    if args.skip_encode or not encode_names:
        print("skip encode", flush=True)
    else:
        repr_mod = _load_attr_repr()
        device = args.device or config.get("device") or "auto"
        encoder_id = str(config.get("encoder_id") or "Marqo/marqo-fashionSigLIP")
        batch_size = int(args.batch_size or config.get("batch_size") or 8)
        model, preprocess, _tok, device = repr_mod.load_model(encoder_id, device)
        print(f"encoder device={device} variants={encode_names}", flush=True)

        # Encode one variant at a time to bound memory; flush every flush_every items.
        flush_every = max(batch_size * 4, 32)
        for name in encode_names:
            spec = variants_cfg[name]
            pad_ratio = float(spec["pad_ratio"]) if spec.get("pad_ratio") is not None else 0.12
            bg_blur = float(spec.get("background_blur_sigma") or 21.0)
            variant_dilate = int(spec["dilate_px"]) if spec.get("dilate_px") is not None else dilate_px
            print(
                f"=== {name} pad={pad_ratio} dilate={variant_dilate} blur={bg_blur} ===",
                flush=True,
            )
            pending_imgs: list[Image.Image] = []
            pending_pos: list[int] = []
            done = 0

            def flush() -> None:
                nonlocal pending_imgs, pending_pos
                if not pending_imgs:
                    return
                vecs = _embed_images(
                    pending_imgs,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    batch_size=batch_size,
                )
                for i, vec in zip(pending_pos, vecs):
                    mats[name][i] = vec
                pending_imgs = []
                pending_pos = []

            for iid in eligible_ids:
                row = items_by_id.get(iid)
                src = sam3_index.get(iid)
                if row is None or src is None:
                    skipped += 1
                    continue
                image_path = Path(row["image_path"])
                if not image_path.is_file():
                    skipped += 1
                    continue
                with Image.open(image_path) as im:
                    rgb = im.convert("RGB")
                    img, meta = _render_variant(
                        rgb,
                        box_xyxy=src["box_xyxy"],
                        proposals=src["proposals"],
                        selected=src["selected"],
                        pad_ratio=pad_ratio,
                        dilate_px=variant_dilate,
                        alpha_sigma=alpha_sigma,
                        background_blur_sigma=bg_blur,
                        garment_labels=garment_labels,
                        min_fill=min_fill,
                    )
                encode_meta.append(
                    {
                        "item_id": iid,
                        "variant": name,
                        "sam3_run": src["run"],
                        "dilate_px": variant_dilate,
                        **meta,
                    }
                )
                if iid in preview_ids:
                    _save_preview(img, preview_root / name / f"{iid}.jpg")
                pending_imgs.append(img)
                pending_pos.append(pos[iid])
                done += 1
                if len(pending_imgs) >= flush_every:
                    flush()
                    print(f"{name} flushed {done}/{len(eligible_ids)}", flush=True)
            flush()
            out_npy = out_dir / f"{name}.npy"
            if out_npy.is_file() and not args.force_overwrite:
                archive_dir = out_dir / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                backup = archive_dir / f"{name}_{stamp}.npy"
                shutil.copy2(out_npy, backup)
                print(f"backed up existing {out_npy.name} -> {backup}", flush=True)
            np.save(out_npy, mats[name])
            cov = int(np.isfinite(mats[name]).all(axis=1).sum())
            print(f"wrote {out_npy} coverage={cov}", flush=True)

    coverage = {
        "A": int(np.isfinite(src_a).all(axis=1).sum()),
        "B": int(np.isfinite(src_b).all(axis=1).sum()),
        "C_0": int(np.isfinite(src_c).all(axis=1).sum()),
    }
    for path in sorted(out_dir.glob("C_*.npy")):
        coverage[path.stem] = int(np.isfinite(np.load(path)).all(axis=1).sum())

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "n": n,
        "coverage": coverage,
        "source_bank": source_bank,
        "output_bank": output_bank,
        "variants": variants_cfg,
        "sam3_runs": list(config.get("sam3_runs") or []),
        "encode_meta_n": len(encode_meta),
        "skipped": skipped,
        "note": "A/B reused; C_0=source C; C_1..C_3 re-encoded soft-mask pad/blur sweep",
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "encode_meta.json", {"items": encode_meta})
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    print(coverage, flush=True)


if __name__ == "__main__":
    main()
