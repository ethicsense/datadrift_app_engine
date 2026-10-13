"""Compare worthwhile background-suppression methods in the optimal pad range."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_utils import read_json, write_json
from preprocess import crop_array, crop_mask, pad_box, resolve_mask_in_box
from run_soft_consistency import (
    _embed_images,
    _load_attr_repr,
    _load_sam3_index,
    _save_preview,
    _setup_project,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.preprocess.background_methods_n2k.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def _soft_alpha(mask: np.ndarray, dilate_px: int, sigma: float) -> np.ndarray:
    keep = np.asarray(mask, dtype=bool)
    if dilate_px > 0:
        keep = ndimage.binary_dilation(keep, iterations=dilate_px)
    if sigma > 0:
        alpha = ndimage.gaussian_filter(keep.astype(np.float32), sigma=sigma)
    else:
        alpha = keep.astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)[..., None]


def _adaptive_mean_background(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    outside = rgb[~mask]
    pixels = outside if len(outside) >= 32 else rgb.reshape(-1, 3)
    # Median is robust to a few foreground leaks in an imperfect SAM mask.
    color = np.median(pixels, axis=0).astype(np.float32)
    return np.broadcast_to(color, rgb.shape).copy()


def _pixelated(rgb: np.ndarray, target_long_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = max(width, height) / max(int(target_long_side), 1)
    small = (max(1, round(width / scale)), max(1, round(height / scale)))
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize(small, Image.Resampling.BOX)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def render_variant(
    rgb: Image.Image,
    *,
    box_xyxy: list[float],
    proposals: list[dict[str, Any]],
    selected: dict[str, Any],
    method: str,
    pad_ratio: float,
    strength: float | None,
    dilate_px: int,
    alpha_sigma: float,
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
    crop = crop_array(rgb, padded).astype(np.float32)
    mask = crop_mask(resolved["mask"], padded)
    if mask.shape[:2] != crop.shape[:2]:
        mask = np.ones(crop.shape[:2], dtype=bool)
    alpha = _soft_alpha(mask, dilate_px, alpha_sigma)

    if method == "gaussian":
        sigma = float(8.0 if strength is None else strength)
        background = np.stack(
            [
                ndimage.gaussian_filter(crop[:, :, channel], sigma=sigma)
                for channel in range(3)
            ],
            axis=-1,
        )
    elif method == "mean_fill":
        background = _adaptive_mean_background(crop, mask)
    elif method == "desaturate":
        saturation = float(0.15 if strength is None else strength)
        luma = (
            crop[:, :, 0] * 0.2126
            + crop[:, :, 1] * 0.7152
            + crop[:, :, 2] * 0.0722
        )[..., None]
        background = luma + saturation * (crop - luma)
    elif method == "dim":
        factor = float(0.65 if strength is None else strength)
        background = crop * factor
    elif method == "pixelate":
        background = _pixelated(crop, int(24 if strength is None else strength))
    else:
        raise ValueError(f"unsupported method: {method}")

    composed = crop * alpha + background * (1.0 - alpha)
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = (REPO_ROOT / config["source_cohort_dir"]).resolve()
    output_root = (REPO_ROOT / config["output_root"]).resolve()
    project = output_root / str(config["project_id"])
    _setup_project(source, project)

    bank_name = str(config.get("bank") or "background_methods")
    bank_dir = project / "embeddings" / bank_name
    bank_dir.mkdir(parents=True, exist_ok=True)

    item_ids = read_json(source / "embeddings" / "item_ids.json")["item_ids"]
    cohort = read_json(source / "cohort.json")
    items_by_id = {str(row["item_id"]): row for row in cohort["items"]}
    pos = {iid: i for i, iid in enumerate(item_ids)}
    n = len(item_ids)

    source_a = np.load(source / "embeddings" / str(config["source_bank"]) / "A.npy")
    np.save(bank_dir / "A.npy", source_a)
    eligible_ids = [
        iid for iid, ok in zip(item_ids, np.isfinite(source_a).all(axis=1)) if ok
    ]

    variants = dict(config.get("variants") or {})
    if args.variants:
        wanted = set(args.variants)
        variants = {name: spec for name, spec in variants.items() if name in wanted}

    encode: dict[str, dict[str, Any]] = {}
    preview_only: dict[str, dict[str, Any]] = {}
    for name, spec in variants.items():
        reuse = spec.get("reuse_npy")
        if reuse:
            src = Path(str(reuse))
            if not src.is_absolute():
                src = REPO_ROOT / src
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, bank_dir / f"{name}.npy")
            print(f"reused {src.name} -> {name}", flush=True)
            preview_only[name] = spec
        elif (bank_dir / f"{name}.npy").is_file():
            print(f"resume skip existing {name}", flush=True)
            preview_only[name] = spec
        else:
            encode[name] = spec

    preprocess_cfg = dict(config.get("preprocess") or {})
    dilate_px = int(preprocess_cfg.get("dilate_px", 7))
    alpha_sigma = float(preprocess_cfg.get("alpha_sigma", 5.0))
    min_fill = float(preprocess_cfg.get("min_mask_fill", 0.08))
    garment_labels = tuple(preprocess_cfg.get("garment_labels") or [])
    preview_ids = set(str(x) for x in config.get("preview_item_ids") or [])

    need_sam = bool(encode) or bool(preview_ids and preview_only)
    sam3_index: dict[str, Any] = {}
    if need_sam:
        sam3_index = _load_sam3_index(
            source / "pipeline" / "sam3",
            list(config.get("sam3_runs") or []),
            box_model=str(config.get("box_model") or "sam3"),
            box_resolver=str(config.get("box_resolver") or "combined"),
            mask_model=str(config.get("mask_model") or "sam3"),
        )

    if encode:
        repr_mod = _load_attr_repr()
        device = args.device or config.get("device") or "auto"
        batch_size = int(args.batch_size or config.get("batch_size") or 8)
        model, image_preprocess, _tok, device = repr_mod.load_model(
            str(config["encoder_id"]), device
        )
        print(f"encoder device={device} variants={list(encode)}", flush=True)
        flush_every = max(batch_size * 4, 32)

        for name, spec in encode.items():
            method = str(spec["method"])
            pad_ratio = float(spec["pad_ratio"])
            strength = spec.get("strength")
            print(
                f"=== {name} method={method} pad={pad_ratio} strength={strength} ===",
                flush=True,
            )
            matrix = np.full((n, 768), np.nan, dtype=np.float32)
            pending_images: list[Image.Image] = []
            pending_positions: list[int] = []
            done = 0
            skipped = 0

            def flush() -> None:
                nonlocal pending_images, pending_positions
                if not pending_images:
                    return
                vectors = _embed_images(
                    pending_images,
                    model=model,
                    preprocess=image_preprocess,
                    device=device,
                    batch_size=batch_size,
                )
                for index, vector in zip(pending_positions, vectors):
                    matrix[index] = vector
                pending_images = []
                pending_positions = []

            for iid in eligible_ids:
                row = items_by_id.get(iid)
                sam = sam3_index.get(iid)
                if row is None or sam is None or not Path(row["image_path"]).is_file():
                    skipped += 1
                    continue
                with Image.open(row["image_path"]) as image:
                    rendered = render_variant(
                        image.convert("RGB"),
                        box_xyxy=sam["box_xyxy"],
                        proposals=sam["proposals"],
                        selected=sam["selected"],
                        method=method,
                        pad_ratio=pad_ratio,
                        strength=None if strength is None else float(strength),
                        dilate_px=dilate_px,
                        alpha_sigma=alpha_sigma,
                        garment_labels=garment_labels,
                        min_fill=min_fill,
                    )
                if iid in preview_ids:
                    _save_preview(
                        rendered, project / "previews" / name / f"{iid}.jpg"
                    )
                pending_images.append(rendered)
                pending_positions.append(pos[iid])
                done += 1
                if len(pending_images) >= flush_every:
                    flush()
                    print(
                        f"{name} flushed {done}/{len(eligible_ids)} skipped={skipped}",
                        flush=True,
                    )
            flush()
            np.save(bank_dir / f"{name}.npy", matrix)
            coverage = int(np.isfinite(matrix).all(axis=1).sum())
            print(f"wrote {name}.npy coverage={coverage}", flush=True)

    # reuse/resume variants still need report previews
    if preview_ids and preview_only:
        for name, spec in preview_only.items():
            missing = [
                iid
                for iid in preview_ids
                if not (project / "previews" / name / f"{iid}.jpg").is_file()
            ]
            if not missing:
                continue
            print(f"preview-only {name} n={len(missing)}", flush=True)
            for iid in missing:
                row = items_by_id.get(iid)
                sam = sam3_index.get(iid)
                if row is None or sam is None or not Path(row["image_path"]).is_file():
                    continue
                with Image.open(row["image_path"]) as image:
                    rendered = render_variant(
                        image.convert("RGB"),
                        box_xyxy=sam["box_xyxy"],
                        proposals=sam["proposals"],
                        selected=sam["selected"],
                        method=str(spec["method"]),
                        pad_ratio=float(spec["pad_ratio"]),
                        strength=(
                            None
                            if spec.get("strength") is None
                            else float(spec["strength"])
                        ),
                        dilate_px=dilate_px,
                        alpha_sigma=alpha_sigma,
                        garment_labels=garment_labels,
                        min_fill=min_fill,
                    )
                _save_preview(rendered, project / "previews" / name / f"{iid}.jpg")

    coverage = {
        path.stem: int(np.isfinite(np.load(path)).all(axis=1).sum())
        for path in sorted(bank_dir.glob("*.npy"))
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "n": n,
        "coverage": coverage,
        "bank": bank_name,
        "source_cohort": str(source),
        "variants": variants,
        "selection_basis": {
            "pad_range": [0.0, 0.06, 0.12],
            "mask_dilate_px": dilate_px,
            "alpha_sigma": alpha_sigma,
            "reference_gaussian_sigma": 8.0,
            "reason": (
                "n2k grid: best balanced mean at pad 6%, sigma 8; "
                "top range pad 0–12%, sigma 4–12; dilate0 harmed sleeve/neck/fit"
            ),
        },
    }
    write_json(bank_dir / "summary.json", summary)
    write_json(
        project / "README_project.json",
        {
            "project_id": config["project_id"],
            "date": config["date"],
            "bank": bank_name,
            "baseline": "A",
            "question": "Which worthwhile background suppression method works best in the optimal pad/mask range?",
        },
    )

    python = str(REPO_ROOT / ".venv/bin/python")
    if not args.skip_metrics:
        subprocess.check_call(
            [
                python,
                str(HERE / "cluster_quality.py"),
                "--cohort-dir",
                str(project),
                "--banks",
                bank_name,
                "--bootstrap",
                "2000",
            ]
        )
    if not args.skip_report:
        subprocess.check_call(
            [
                python,
                str(HERE / "build_background_methods_report.py"),
                "--project-dir",
                str(project),
                "--bank",
                bank_name,
            ]
        )


if __name__ == "__main__":
    main()
