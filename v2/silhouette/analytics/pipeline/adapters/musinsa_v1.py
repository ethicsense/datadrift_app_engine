from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pandas as pd

from analytics.pipeline.adapters.base import ChannelAdapter


PRODUCT_INFO_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "material": ("소재", "제품 소재"),
    "color": ("색상",),
    "manufacturer": ("제조사",),
    "origin_country": ("제조국",),
    "shipping_fee": ("배송비",),
}


class MusinsaV1Adapter(ChannelAdapter):
    adapter_id = "musinsa_v1"
    adapter_version = "1.0.0"

    def __init__(
        self,
        *,
        parse_int: Callable[[Any], int | None],
        normalize_text_block: Callable[[Any], str | None],
        hash_text: Callable[[str | None], str | None],
        extract_business_address_parts: Callable[[Any], Dict[str, Any]],
        resolve_first_present: Callable[..., tuple[Any, Any, str | None, bool]],
    ) -> None:
        self._parse_int = parse_int
        self._normalize_text_block = normalize_text_block
        self._hash_text = hash_text
        self._extract_business_address_parts = extract_business_address_parts
        self._resolve_first_present = resolve_first_present

    def detect(self, summary_payload: Dict[str, Any]) -> bool:
        # 현재는 기본 채널로 사용한다.
        return isinstance(summary_payload, dict)

    def infer_schema_version(self, summary_payload: Dict[str, Any]) -> str:
        if not isinstance(summary_payload, dict):
            return "unknown"
        if "ocr_total_images" in summary_payload:
            return "extended_summary_v2"
        products = summary_payload.get("products")
        if not isinstance(products, list) or not products:
            return "legacy_summary_v1"
        field_union: set[str] = set()
        for product in products[:10]:
            if isinstance(product, dict):
                field_union.update(str(key) for key in product.keys())
        if field_union.intersection(
            {"ocr_images", "original_price", "discount_amount", "discount_rate", "product_meta", "ranking_badge"}
        ):
            return "extended_summary_v2"
        return "legacy_summary_v1"

    def _load_ocr_detail(self, product_dir: Path) -> Dict[str, Any]:
        ocr_path = product_dir / "ocr_data.json"
        if not ocr_path.exists():
            return {
                "ocr_source_exists": False,
                "ocr_has_data": False,
                "ocr_slot_count": 0,
                "ocr_slot_keys_json": None,
                "ocr_entries_json": None,
                "ocr_text_joined": None,
                "ocr_text_length": 0,
                "ocr_text_hash": None,
            }
        entries: Dict[str, str] = {}
        try:
            payload = json.loads(ocr_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for slot_key in sorted(payload.keys()):
                    value = payload.get(slot_key)
                    normalized = self._normalize_text_block(value.get("full_text")) if isinstance(value, dict) else self._normalize_text_block(value)
                    if normalized:
                        entries[str(slot_key)] = normalized
        except Exception:
            pass
        joined = "\n\n".join(entries.values()) if entries else None
        return {
            "ocr_source_exists": True,
            "ocr_has_data": bool(entries),
            "ocr_slot_count": len(entries),
            "ocr_slot_keys_json": json.dumps(list(entries.keys()), ensure_ascii=False) if entries else None,
            "ocr_entries_json": json.dumps(entries, ensure_ascii=False, sort_keys=True) if entries else None,
            "ocr_text_joined": joined,
            "ocr_text_length": len(joined) if joined else 0,
            "ocr_text_hash": self._hash_text(joined),
        }

    def load_product_detail(self, session_dir: Path, product_id: str) -> Dict[str, Any]:
        product_dir = session_dir / "products" / product_id
        tags: list[str] = []
        info_map: Dict[str, Any] = {}
        product_dir_exists = product_dir.exists()
        tags_exists = False
        info_exists = False
        size_table_exists = (product_dir / "size_table.csv").exists()
        detail_images_dir = product_dir / "detail_images"
        detail_images = [p for p in detail_images_dir.iterdir() if p.is_file()] if detail_images_dir.exists() else []

        tags_path = product_dir / "tags.csv"
        if tags_path.exists():
            tags_exists = True
            try:
                tags_df = pd.read_csv(tags_path, encoding="utf-8-sig")
                if "태그" in tags_df.columns:
                    tags = [t for t in tags_df["태그"].dropna().astype(str).tolist() if t.strip()]
            except Exception:
                pass

        info_path = product_dir / "product_info.csv"
        if info_path.exists():
            info_exists = True
            try:
                info_df = pd.read_csv(info_path, encoding="utf-8-sig")
                if {"항목", "내용"}.issubset(set(info_df.columns)):
                    info_map = {
                        str(k).strip(): str(v).strip()
                        for k, v in zip(info_df["항목"], info_df["내용"])
                        if pd.notna(k) and pd.notna(v)
                    }
            except Exception:
                pass

        ocr_detail = self._load_ocr_detail(product_dir)
        business_address_detail = self._extract_business_address_parts(info_map.get("영업소재지"))
        normalized_fields: Dict[str, Any] = {
            "product_dir_exists": product_dir_exists,
            "tags_source_exists": tags_exists,
            "product_info_exists": info_exists,
            "size_table_exists": size_table_exists,
            "detail_images_exists": detail_images_dir.exists(),
            "detail_image_count_actual": len(detail_images),
            "tags": tags,
            "tags_joined": ", ".join(tags),
            "tag_count_actual": len(tags),
            **ocr_detail,
            **business_address_detail,
        }
        source_fields: Dict[str, Any] = {}
        for target, aliases in PRODUCT_INFO_FIELD_ALIASES.items():
            parsed_value, raw_value, source_key, raw_present = self._resolve_first_present(info_map, aliases)
            normalized_fields[target] = parsed_value
            normalized_fields[f"{target}_raw"] = raw_value
            normalized_fields[f"{target}_raw_present"] = raw_present
            normalized_fields[f"{target}_source"] = f"product_info.{source_key}" if source_key else None
            source_fields[target] = source_key
        return {
            "normalized_fields": normalized_fields,
            "raw_info_map": info_map,
            "source_fields": source_fields,
        }

    def build_extension_payload(self, *, summary_payload: Dict[str, Any], product_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary_meta": copy.deepcopy({key: value for key, value in summary_payload.items() if key != "products"}),
            "product": copy.deepcopy(product_payload),
        }
