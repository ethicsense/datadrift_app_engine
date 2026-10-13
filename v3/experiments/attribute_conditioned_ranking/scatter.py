"""2D maps of the pixelate baseline and attribute-controlled embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from controls import l2_normalize

COLOR_HEX = {
    "white_ivory": "#d8d2c2",
    "black": "#222222",
    "gray": "#8a8a8a",
    "navy_blue": "#2a4184",
    "beige_brown": "#a78462",
    "chromatic": "#c44b78",
    "uncertain": "#c5c5c5",
    "": "#c5c5c5",
}
PATTERN_HEX = {
    "solid": "#3d7a62",
    "stripe": "#3d6ea8",
    "check": "#c47b32",
    "graphic_logo": "#8a4ea3",
    "other": "#888888",
    "uncertain": "#c5c5c5",
    "": "#c5c5c5",
}


def fit_umap(embeddings: np.ndarray, *, seed: int) -> np.ndarray:
    import umap

    matrix = l2_normalize(embeddings)
    components = min(32, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=components, random_state=seed).fit_transform(matrix)
    coords = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(30, matrix.shape[0] - 1),
        min_dist=0.08,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(reduced)
    return coords.astype(np.float32)


def _thumb(project: Path, cohort_dir: Path, item_id: str, image_paths: dict[str, Path]) -> str:
    rel = f"previews/scatter/{item_id}.jpg"
    dest = project / rel
    if dest.is_file():
        return rel
    src = cohort_dir / "thumbs" / f"{item_id}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        dest.write_bytes(src.read_bytes())
        return rel
    image = image_paths.get(item_id)
    if image and image.is_file():
        from PIL import Image

        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
            rgb.thumbnail((192, 192))
            rgb.save(dest, format="JPEG", quality=80)
        return rel
    return ""


def _product_catalog(cohort: dict, cohort_dir: Path) -> dict[str, dict]:
    catalog = {str(row["item_id"]): dict(row) for row in cohort["items"]}
    for name in ("add1k_new.jsonl", "add2k_new.jsonl"):
        path = cohort_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            item = catalog.get(str(row.get("item_id")))
            if item is None:
                continue
            if not item.get("product") and row.get("product"):
                item["product"] = row["product"]
            if not item.get("product_id") and row.get("product_id"):
                item["product_id"] = row["product_id"]
    return catalog


def build_scatter(
    *,
    project: Path,
    cohort_dir: Path,
    cohort: dict[str, Any],
    item_ids: list[str],
    labels: dict[str, dict[str, str]],
    finite: np.ndarray,
    reference: list[dict[str, Any]],
    arms: dict[str, np.ndarray],
    panels: list[tuple[str, str, str]],
    seed: int,
) -> dict[str, Any]:
    """panels: (arm_key, title, color_by) where color_by is color or pattern."""
    image_paths = {
        str(row["item_id"]): Path(row["image_path"])
        for row in cohort["items"]
        if row.get("image_path")
    }
    fine = {
        row["item_id"]: row.get("fine_color")
        for row in reference
        if row.get("status") == "ok"
    }
    index = np.flatnonzero(finite)
    plotted_ids = [item_ids[i] for i in index]
    thumbs = [_thumb(project, cohort_dir, iid, image_paths) for iid in plotted_ids]
    coords: dict[str, list[list[float]]] = {}
    seen: set[str] = set()
    for arm, _title, _color_by in panels:
        if arm in seen:
            continue
        print(f"umap {arm}", flush=True)
        xy = fit_umap(arms[arm][index], seed=seed)
        coords[arm] = [[round(float(x), 4), round(float(y), 4)] for x, y in xy]
        seen.add(arm)
    catalog = _product_catalog(cohort, cohort_dir)
    points = []
    for local, iid in enumerate(plotted_ids):
        lab = labels.get(iid) or {}
        item = catalog.get(iid) or {}
        points.append(
            {
                "id": iid,
                "product": str(item.get("product") or ""),
                "product_id": str(item.get("product_id") or ""),
                "color": lab.get("color") or "",
                "pattern": lab.get("pattern") or "",
                "sleeve": lab.get("sleeve") or "",
                "neck": lab.get("neck") or "",
                "fit": lab.get("fit") or "",
                "fine": fine.get(iid) or "",
                "thumb": thumbs[local],
            }
        )
    payload = {
        "note": "각 산점도는 해당 임베딩만으로 따로 맞춘 UMAP입니다. 패널 사이 축 위치는 비교하지 않고, 같은 색 라벨이 모이는지만 봅니다.",
        "points": points,
        "panels": [
            {
                "arm": arm,
                "title": title,
                "color_by": color_by,
                "coords": coords[arm],
            }
            for arm, title, color_by in panels
        ],
    }
    path = project / "metrics" / "scatter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload
