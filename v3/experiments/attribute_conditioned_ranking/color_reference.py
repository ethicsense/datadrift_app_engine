"""Evaluation-only garment Lab reference for solid items."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from color_space import delta_e00, rgb_u8_to_lab
from protocol import PALETTE_RGB, SAM3_RUNS

HERE = Path(__file__).resolve().parent
GARMENT_DIR = HERE.parent / "garment_segmentation"
if str(GARMENT_DIR) not in sys.path:
    sys.path.insert(0, str(GARMENT_DIR))

from preprocess import resolve_mask_in_box  # noqa: E402
from run_soft_consistency import _load_sam3_index  # noqa: E402


def palette_lab() -> dict[str, np.ndarray]:
    return {
        name: rgb_u8_to_lab(np.array(rgb, dtype=np.float64))
        for name, rgb in PALETTE_RGB.items()
    }


def _median_rgb(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    pixels = rgb[mask]
    if len(pixels) < 16:
        return None
    if len(pixels) > 4000:
        pixels = pixels[rng.choice(len(pixels), 4000, replace=False)]
    return np.median(pixels, axis=0)


def _assign(lab: np.ndarray, palette: dict[str, np.ndarray]) -> tuple[str, float]:
    names = list(palette)
    stack = np.stack([palette[name] for name in names], axis=0)
    distances = delta_e00(lab, stack)
    index = int(np.argmin(distances))
    return names[index], float(distances[index])


def build_color_reference(
    *,
    cohort_dir: Path,
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    splits: dict[str, str],
    seed: int,
) -> list[dict[str, Any]]:
    """Median Lab inside the garment mask for solid items, then nearest palette color."""
    cohort = json.loads((cohort_dir / "cohort.json").read_text(encoding="utf-8"))
    items = {str(row["item_id"]): row for row in cohort["items"]}
    sam3 = _load_sam3_index(
        cohort_dir / "pipeline" / "sam3",
        list(SAM3_RUNS),
        box_model="sam3",
        box_resolver="combined",
        mask_model="sam3",
    )
    palette = palette_lab()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    solid_ids = [
        iid
        for iid in item_ids
        if (labels.get(iid) or {}).get("pattern") == "solid"
    ]
    for done, iid in enumerate(solid_ids, 1):
        row = items.get(iid)
        image_path = Path(row["image_path"]) if row and row.get("image_path") else None
        record: dict[str, Any] = {
            "item_id": iid,
            "split": splits[iid],
            "coarse_color": (labels.get(iid) or {}).get("color"),
            "status": "ok",
        }
        sam = sam3.get(iid)
        if image_path is None or not image_path.is_file() or sam is None:
            record["status"] = "missing_image_or_mask"
            rows.append(record)
            continue
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            resolved = resolve_mask_in_box(
                height=height,
                width=width,
                box_xyxy=sam["box_xyxy"],
                proposals=sam["proposals"],
                combined_selected=sam["selected"],
                min_mask_fill=0.08,
            )
            rgb = np.asarray(rgb_image)
        median = _median_rgb(rgb, resolved["mask"], rng)
        if median is None:
            record["status"] = "empty_mask"
            rows.append(record)
            continue
        lab = rgb_u8_to_lab(median)
        fine, distance = _assign(lab, palette)
        record.update(
            {
                "lab": [round(float(v), 4) for v in lab],
                "rgb": [int(round(float(v))) for v in median],
                "fine_color": fine,
                "delta_e_to_anchor": round(distance, 4),
                "mask_source": resolved["source"],
            }
        )
        rows.append(record)
        if done % 100 == 0 or done == len(solid_ids):
            print(f"color reference {done}/{len(solid_ids)}", flush=True)
    return rows
