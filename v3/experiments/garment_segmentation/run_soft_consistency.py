"""Create soft_consistency_pad_blur project: pad_ratio & blur ladders vs A."""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    path = attr_dir / "representations.py"
    if str(attr_dir) not in sys.path:
        sys.path.insert(0, str(attr_dir))
    spec = importlib.util.spec_from_file_location("attr_emb_repr", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "config.preprocess.soft_consistency.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-encode", action="store_true")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--skip-report", action="store_true")
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
        print(f"indexed {name}: bank={len(out)}", flush=True)
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
) -> Image.Image:
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
    return compose_soft_variant(
        rgb,
        box_xyxy=box_xyxy,
        mask=resolved["mask"],
        pad_ratio=pad_ratio,
        dilate_px=dilate_px,
        alpha_sigma=alpha_sigma,
        background_blur_sigma=background_blur_sigma,
    )


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _unique_variants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in config.get("variant_grid") or []:
        name = str(row["name"])
        if name not in out:
            out[name] = dict(row)
    for ladder in ("pad_ladder", "blur_ladder"):
        for row in config.get(ladder) or []:
            name = str(row["name"])
            if name in out:
                continue
            out[name] = dict(row)
    return out


def _variant_has_reuse(spec: dict[str, Any]) -> bool:
    return bool(spec.get("reuse_from") or spec.get("reuse_npy"))


def _resolve_reuse_npy(spec: dict[str, Any], reuse_bank: Path) -> Path:
    if spec.get("reuse_npy"):
        path = Path(str(spec["reuse_npy"]))
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    reuse = spec.get("reuse_from")
    if not reuse:
        raise ValueError("variant has no reuse source")
    return (reuse_bank / f"{reuse}.npy").resolve()


def _setup_project(parent: Path, project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    for name in (
        "cohort.json",
        "labels.jsonl",
        "label_schema.json",
        "kappa_review50.json",
        "human_labels_review50.jsonl",
    ):
        src = parent / name
        if src.is_file():
            _symlink(src, project / name)
    emb = project / "embeddings"
    emb.mkdir(exist_ok=True)
    _symlink(parent / "embeddings" / "item_ids.json", emb / "item_ids.json")
    (project / "metrics").mkdir(exist_ok=True)
    (project / "previews").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_file():
        example = HERE / "config.preprocess.soft_consistency.example.yaml"
        shutil.copy2(example, cfg_path)
        print(f"created {cfg_path} from example", flush=True)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    source = (REPO_ROOT / config["source_cohort_dir"]).resolve()
    output_root = (REPO_ROOT / config.get("output_root", "artifacts/garment_segmentation")).resolve()
    project = output_root / str(config["project_id"])
    _setup_project(source, project)

    bank_name = str(config.get("bank") or "soft_consistency")
    bank_dir = project / "embeddings" / bank_name
    bank_dir.mkdir(parents=True, exist_ok=True)

    item_ids = read_json(source / "embeddings" / "item_ids.json")["item_ids"]
    cohort = read_json(source / "cohort.json")
    items_by_id = {str(row["item_id"]): row for row in cohort["items"]}
    n = len(item_ids)
    pos = {iid: i for i, iid in enumerate(item_ids)}

    src_bank = source / "embeddings" / str(config.get("source_bank") or "sam3_abcd")
    reuse_bank = source / "embeddings" / str(config.get("reuse_bank") or "sam3_soft_sweep")
    src_a = np.load(src_bank / "A.npy")
    np.save(bank_dir / "A.npy", src_a)

    variants = _unique_variants(config)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in set(args.variants)}

    preprocess_cfg = dict(config.get("preprocess") or {})
    alpha_sigma = float(preprocess_cfg.get("alpha_sigma") or 5.0)
    min_fill = float(preprocess_cfg.get("min_mask_fill") or 0.08)
    garment_labels = tuple(preprocess_cfg.get("garment_labels") or DEFAULT_GARMENT_LABELS)

    finite_a = np.isfinite(src_a).all(axis=1)
    eligible_ids = [iid for iid, ok in zip(item_ids, finite_a) if ok]
    if args.limit is not None:
        eligible_ids = eligible_ids[: args.limit]

    # Reuse matching prior arms.
    for name, spec in variants.items():
        if not _variant_has_reuse(spec):
            continue
        src = _resolve_reuse_npy(spec, reuse_bank)
        if not src.is_file():
            raise FileNotFoundError(f"reuse missing: {src}")
        shutil.copy2(src, bank_dir / f"{name}.npy")
        print(f"reused {src.name} -> {name}", flush=True)

    encode_names = [n for n, s in variants.items() if not _variant_has_reuse(s)]
    if args.skip_encode:
        encode_names = []

    preview_ids = set(str(x) for x in (config.get("preview_item_ids") or []))
    preview_root = project / "previews"
    # A/B previews from parent seed run for report context
    for arm in ("A", "B"):
        for iid in preview_ids:
            src = (
                source
                / "pipeline"
                / "preprocess"
                / "preprocess-abcd-sam3-450"
                / "previews"
                / arm
                / f"{iid}.jpg"
            )
            if src.is_file():
                dest = preview_root / arm / f"{iid}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

    if encode_names:
        sam3_index = _load_sam3_index(
            source / "pipeline" / "sam3",
            list(config.get("sam3_runs") or []),
            box_model=str(config.get("box_model") or "sam3"),
            box_resolver=str(config.get("box_resolver") or "combined"),
            mask_model=str(config.get("mask_model") or "sam3"),
        )
        repr_mod = _load_attr_repr()
        device = args.device or config.get("device") or "auto"
        encoder_id = str(config.get("encoder_id") or "Marqo/marqo-fashionSigLIP")
        batch_size = int(args.batch_size or config.get("batch_size") or 8)
        model, preprocess, _tok, device = repr_mod.load_model(encoder_id, device)
        print(f"encoder device={device} encode={encode_names}", flush=True)
        flush_every = max(batch_size * 4, 32)

        for name in encode_names:
            spec = variants[name]
            pad_ratio = float(spec["pad_ratio"])
            dilate_px = int(spec.get("dilate_px") if spec.get("dilate_px") is not None else 7)
            bg_blur = float(spec["background_blur_sigma"])
            print(f"=== {name} pad={pad_ratio} dilate={dilate_px} blur={bg_blur} ===", flush=True)
            mat = np.full((n, 768), np.nan, dtype=np.float32)
            pending_imgs: list[Image.Image] = []
            pending_pos: list[int] = []
            done = 0
            skipped = 0

            def flush() -> None:
                nonlocal pending_imgs, pending_pos
                if not pending_imgs:
                    return
                vecs = _embed_images(
                    pending_imgs, model=model, preprocess=preprocess, device=device, batch_size=batch_size
                )
                for i, vec in zip(pending_pos, vecs):
                    mat[i] = vec
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
                    img = _render_variant(
                        rgb,
                        box_xyxy=src["box_xyxy"],
                        proposals=src["proposals"],
                        selected=src["selected"],
                        pad_ratio=pad_ratio,
                        dilate_px=dilate_px,
                        alpha_sigma=alpha_sigma,
                        background_blur_sigma=bg_blur,
                        garment_labels=garment_labels,
                        min_fill=min_fill,
                    )
                if iid in preview_ids:
                    _save_preview(img, preview_root / name / f"{iid}.jpg")
                pending_imgs.append(img)
                pending_pos.append(pos[iid])
                done += 1
                if len(pending_imgs) >= flush_every:
                    flush()
                    print(f"{name} flushed {done}/{len(eligible_ids)} skipped={skipped}", flush=True)
            flush()
            out_npy = bank_dir / f"{name}.npy"
            if out_npy.is_file():
                archive = bank_dir / "archive"
                archive.mkdir(exist_ok=True)
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                shutil.copy2(out_npy, archive / f"{name}_{stamp}.npy")
            np.save(out_npy, mat)
            print(f"wrote {out_npy} coverage={int(np.isfinite(mat).all(1).sum())}", flush=True)

    # soft_ref previews: copy from parent C if needed
    for iid in preview_ids:
        src = (
            source
            / "pipeline"
            / "preprocess"
            / "preprocess-abcd-sam3-450"
            / "previews"
            / "C"
            / f"{iid}.jpg"
        )
        if src.is_file():
            dest = preview_root / "soft_ref" / f"{iid}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    coverage = {
        p.stem: int(np.isfinite(np.load(p)).all(1).sum())
        for p in sorted(bank_dir.glob("*.npy"))
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "n": n,
        "coverage": coverage,
        "source_cohort": str(source),
        "project": str(project),
        "bank": bank_name,
        "pad_ladder": config.get("pad_ladder"),
        "blur_ladder": config.get("blur_ladder"),
        "note": "pad_ratio ladder + blur σ ladder; baseline A; soft_ref=pad0.12/dilate7/blur21",
    }
    write_json(bank_dir / "summary.json", summary)
    write_json(
        project / "README_project.json",
        {
            "project_id": config["project_id"],
            "date": "2026-10-07",
            "output_root": str(output_root),
            "source_cohort": str(source),
            "question": "Does cluster quality change consistently across pad_ratio and blur σ ladders?",
            "baseline": "A",
            "bank": bank_name,
        },
    )
    print(coverage, flush=True)

    if not args.skip_metrics:
        cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(HERE / "cluster_quality.py"),
            "--cohort-dir",
            str(project),
            "--banks",
            bank_name,
            "--bootstrap",
            "2000",
        ]
        print("running", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)

    if not args.skip_report:
        report_script = str(config.get("report_script") or "build_soft_consistency_report.py")
        cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(HERE / report_script),
            "--project-dir",
            str(project),
            "--bank",
            bank_name,
        ]
        print("running", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
