#!/usr/bin/env python3
"""
이미지 인벤토리 및 세그먼트 생성 유틸리티.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# 있으면 우선 사용하고, 없으면 전략적으로 대표 이미지를 고른다.
EXPLICIT_MAIN_IMAGE_FILENAMES = (
    "main_image.jpg",
    "main_image.jpeg",
    "main_image.png",
    "main_image.webp",
    "main_images.jpg",
    "main_images.jpeg",
    "main_images.png",
    "main_images.webp",
)
logger = logging.getLogger(__name__)


@dataclass
class SegmentConfig:
    max_segments_per_product: int = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _list_detail_images(product_dir: Path) -> List[Path]:
    images_dir = product_dir / "detail_images"
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and p.is_file()])


MainImageStrategy = Literal["first", "largest_file"]


def _pick_main_images(
    image_paths: List[Path],
    max_images: int,
    strategy: MainImageStrategy = "first",
) -> List[Path]:
    """
    (snapshot, product)당 사용할 이미지 수를 제한할 때, 어떤 이미지를 쓸지 선택.
    - first: 파일명 정렬 순 첫 장 (무신사 등은 detail_01이 대표인 경우 많음). I/O 없음.
    - largest_file: 파일 크기 최대 순 (해상도/품질 좋은 걸 대표로). stat만 사용.
    """
    if not image_paths or max_images <= 0:
        return []
    if len(image_paths) <= max_images:
        return image_paths
    if strategy == "largest_file":
        with_size = [(p, p.stat().st_size if p.exists() else 0) for p in image_paths]
        with_size.sort(key=lambda x: -x[1])
        return [p for p, _ in with_size[:max_images]]
    # first: 이미 파일명 정렬된 리스트이므로 앞에서 max_images개
    return image_paths[:max_images]


def build_image_manifest(
    base_df: pd.DataFrame,
    data_dir: Path,
    main_image_strategy: MainImageStrategy = "first",
) -> pd.DataFrame:
    """
    product-snapshot 단위에서 detail_images 전체를 인벤토리한다.
    임베딩 파이프라인은 is_main_image=True인 1장만 사용한다.
    """
    if base_df.empty:
        return pd.DataFrame()

    records: List[Dict] = []
    base_columns = ["snapshot_id", "snapshot_date", "snapshot_time", "product_id", "brand", "rank"]
    if "product_path" in base_df.columns:
        base_columns.append("product_path")
    unique_keys = base_df[base_columns].drop_duplicates().to_dict("records")
    unique_keys.sort(key=lambda r: (str(r["snapshot_date"]), str(r["snapshot_time"])))

    total = len(unique_keys)
    logger.info(
        "이미지 인벤토리 생성 시작: 대상 snapshot-product pairs(records)=%d, main_strategy=%s",
        total,
        main_image_strategy,
    )
    for idx_item, item in enumerate(unique_keys, start=1):
        product_dir = (
            Path(str(item["product_path"]))
            if item.get("product_path")
            else (
                data_dir
                / str(item["snapshot_date"])
                / str(item["snapshot_time"])
                / "products"
                / str(item["product_id"])
            )
        )
        image_paths = _list_detail_images(product_dir)
        main_path = None
        main_image_source = None
        explicit_main_names = {name.lower() for name in EXPLICIT_MAIN_IMAGE_FILENAMES}
        for p in image_paths:
            if p.name.lower() in explicit_main_names:
                main_path = p
                main_image_source = "main_image_filename"
                break
        if main_path is None and image_paths:
            main_path = _pick_main_images(image_paths, 1, main_image_strategy)[0]
            main_image_source = f"strategy:{main_image_strategy}"
        main_paths_set = {main_path} if main_path is not None else set()

        if not image_paths:
            records.append(
                {
                    **item,
                    "product_path": str(product_dir),
                    "image_id": None,
                    "image_path": None,
                    "image_name": None,
                    "image_ext": None,
                    "image_exists": False,
                    "file_size_bytes": 0,
                    "sha256": None,
                    "width": None,
                    "height": None,
                    "aspect_ratio_hw": None,
                    "is_long_stitched": False,
                    "is_main_image": False,
                    "main_image_source": None,
                }
            )
            continue

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None

            ratio = (height / width) if width and height else None
            is_main = image_path in main_paths_set
            records.append(
                {
                    **item,
                    "product_path": str(product_dir),
                    "image_id": f"{item['snapshot_id']}_{item['product_id']}_{idx:03d}",
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "image_ext": image_path.suffix.lower(),
                    "image_exists": True,
                    "file_size_bytes": image_path.stat().st_size if image_path.exists() else 0,
                    "sha256": sha256_file(image_path) if image_path.exists() else None,
                    "width": width,
                    "height": height,
                    "aspect_ratio_hw": ratio,
                    "is_long_stitched": bool(ratio and ratio >= 3.5),
                    "is_main_image": is_main,
                    "main_image_source": main_image_source if is_main else None,
                }
            )

        if idx_item % 500 == 0:
            logger.info("이미지 인벤토리 진행: %d/%d", idx_item, total)

    manifest_df = pd.DataFrame(records)
    n_images = len(manifest_df)
    n_exists = int(manifest_df["image_exists"].fillna(False).sum()) if not manifest_df.empty else 0
    logger.info(
        "이미지 인벤토리 완료: 이미지(manifest)=%d, 파일존재=%d (위 records=%d개 제품 스냅샷에서 수집)",
        n_images,
        n_exists,
        total,
    )
    return manifest_df


def build_image_segments(manifest_df: pd.DataFrame, config: SegmentConfig) -> pd.DataFrame:
    """
    임베딩용 메인 이미지만 세그먼트로 변환한다.
    각 product-snapshot 당 대표 이미지 1장을 전체 크롭 1개로 유지한다.
    """
    if manifest_df.empty:
        return pd.DataFrame()

    segment_rows: List[Dict] = []
    image_rows = manifest_df[
        (manifest_df["image_exists"] == True) & (manifest_df["is_main_image"] == True)
    ].copy()
    if config.max_segments_per_product > 0:
        image_rows = (
            image_rows.sort_values(["snapshot_id", "product_id", "image_id"])
            .groupby(["snapshot_id", "product_id"], as_index=False)
            .head(config.max_segments_per_product)
        )

    total = len(image_rows)
    logger.info("이미지 세그먼트 생성 시작: 대표 이미지=%d", total)
    for idx_row, row in enumerate(image_rows.to_dict("records"), start=1):
        image_path = Path(str(row["image_path"]))
        width = row.get("width")
        height = row.get("height")
        if not width or not height or not image_path.exists():
            continue

        segment_rows.append(
            {
                "segment_id": f"{row['image_id']}_seg_001",
                "image_id": row["image_id"],
                "snapshot_id": row["snapshot_id"],
                "snapshot_date": row["snapshot_date"],
                "snapshot_time": row["snapshot_time"],
                "product_id": row["product_id"],
                "brand": row.get("brand"),
                "rank": row.get("rank"),
                "image_path": row["image_path"],
                "x1": 0,
                "y1": 0,
                "x2": int(width),
                "y2": int(height),
                "segment_width": int(width),
                "segment_height": int(height),
                "segment_type_rule": "main_image",
                "embedding_target": True,
                "sha256": row.get("sha256"),
            }
        )
        if idx_row % 500 == 0:
            logger.info("이미지 세그먼트 진행: %d/%d 이미지", idx_row, total)

    result = pd.DataFrame(segment_rows)
    logger.info("이미지 세그먼트 생성 완료: 세그먼트=%d", len(result))
    return result

