#!/usr/bin/env python3
"""
무신사 크롤링 누적 데이터를 분석용 데이터셋으로 변환한다.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.pipeline.embedding_search import (
    EmbeddingConfig,
    apply_embedding_quota,
    generate_segment_embeddings,
    upsert_embeddings_to_qdrant,
)
from analytics.pipeline.adapters.musinsa_v1 import MusinsaV1Adapter
from analytics.pipeline.adapters.registry import AdapterRegistry
from analytics.pipeline.image_processing import SegmentConfig, build_image_manifest, build_image_segments
from analytics.pipeline.index_duckdb import build_duckdb_index
from analytics.pipeline.text_enrichment import build_text_features
from analytics.pipeline.brand_style_embedding import BrandStyleEmbeddingConfig, build_brand_style_embedding_artifacts
from analytics.pipeline.text_integrated_analysis import build_text_integrated_artifacts
from analytics.pipeline.analysis_fusion import run_analysis_fusion
from analytics.pipeline.visualization_assets import (
    build_embedding_projection,
    build_rank_race,
    build_rank_trajectories,
)


PRICE_RE = re.compile(r"[^\d]")
DISCOUNT_RE = re.compile(r"[^\d]")
WHITESPACE_RE = re.compile(r"\s+")
ADDRESS_TOKEN_RE = re.compile(r"[\s,()]+")
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    latest_snapshot_limit: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_sources: Optional[list[str]] = None
    exclude_sources: Optional[list[str]] = None
    enable_multimodal: bool = True
    enable_embeddings: bool = True
    enable_qdrant_upsert: bool = True
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "product_image_segments"
    quota_product_packshot: int = 1
    quota_detail_closeup: int = 1
    quota_model_wearing: int = 1
    main_image_strategy: str = "first"
    enable_brand_style_embedding: bool = False
    quality_mode: str = "warn"
    quality_missing_threshold: float = 0.15
    adapter_dual_run: bool = True


@dataclass
class ProductDetailPayload:
    normalized_fields: Dict[str, Any]
    raw_info_map: Dict[str, Any]
    source_fields: Dict[str, Any]


@dataclass
class ProductCategoryPayload:
    normalized_fields: Dict[str, Any]
    raw_payload: Dict[str, Any]


@dataclass(frozen=True)
class SnapshotSummaryRef:
    summary_path: Path
    source_dataset: str
    platform: str
    snapshot_date: str
    snapshot_time: str
    source_path: str
    schema_version_guess: str
    schema_version_legacy: str = "unknown"


ADAPTER_REGISTRY: Optional[AdapterRegistry] = None


def parse_price(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    cleaned = PRICE_RE.sub("", str(raw))
    return int(cleaned) if cleaned else None


def parse_discount(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    cleaned = DISCOUNT_RE.sub("", str(raw))
    return float(cleaned) if cleaned else None


def parse_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    cleaned = PRICE_RE.sub("", str(raw))
    return int(cleaned) if cleaned else None


def parse_date_str(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s)


def normalize_text_block(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).replace("\u200b", " ").strip()
    if not text:
        return None

    normalized_lines: List[str] = []
    seen = set()
    for line in text.splitlines():
        cleaned = WHITESPACE_RE.sub(" ", line).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized_lines.append(cleaned)

    if normalized_lines:
        return "\n".join(normalized_lines)

    cleaned = WHITESPACE_RE.sub(" ", text).strip()
    return cleaned or None


def hash_text(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


RAW_SNAPSHOT_PRODUCTS_FILENAME = "raw_snapshot_products.parquet"
SCHEMA_RAW_FILENAME = "schema_raw.json"
SCHEMA_NORMALIZED_FILENAME = "schema_normalized.json"
SCHEMA_DIFF_FILENAME = "schema_diff.json"
DEPRECATED_NORMALIZED_FIELDS = {"taxonomy_gap_candidate"}
DEPRECATED_RAW_FIELDS = {"product.taxonomy_gap_candidate"}

PRODUCT_INFO_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "material": ("소재", "제품 소재"),
    "color": ("색상",),
    "manufacturer": ("제조사",),
    "origin_country": ("제조국",),
    "shipping_fee": ("배송비",),
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sample_value(value: Any, limit: int = 160) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list, tuple, set)):
        text = _json_dumps(value)
    else:
        text = str(value)
    if not text:
        return None
    return text[:limit]


def _field_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _inferred_type(counter: Counter[str]) -> str:
    scoped = [(field_type, count) for field_type, count in counter.items() if field_type != "null"]
    if not scoped:
        return "null"
    scoped.sort(key=lambda item: (-item[1], item[0]))
    return scoped[0][0]


def _safe_json_loads(payload: Any) -> Any:
    if payload is None or payload == "":
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(str(payload))
    except Exception:
        return None


def _is_deprecated_schema_field(field_name: str, scope: str) -> bool:
    if scope == "raw":
        return field_name in DEPRECATED_RAW_FIELDS
    return field_name in DEPRECATED_NORMALIZED_FIELDS


def _resolve_first_present(
    payload: Dict[str, Any],
    aliases: Tuple[str, ...],
    parser: Any = None,
) -> Tuple[Any, Any, Optional[str], bool]:
    for alias in aliases:
        if alias not in payload:
            continue
        raw_value = payload.get(alias)
        if raw_value is None:
            return None, raw_value, alias, True
        text_value = str(raw_value).strip() if isinstance(raw_value, str) else raw_value
        if text_value in {"", "-", "nan", "None", "null"}:
            return None, raw_value, alias, True
        parsed_value = parser(raw_value) if parser else raw_value
        return parsed_value, raw_value, alias, True
    return None, None, None, False


def _looks_like_date_segment(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except Exception:
        return False
    return True


def _looks_like_time_segment(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}-\d{2}", value))


def _infer_platform(source_dataset: str) -> str:
    if not source_dataset:
        return "unknown"
    first = source_dataset.split("/")[0]
    return first.split("_")[0] if "_" in first else first


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text not in {"nan", "None", "null"} else None


def _normalize_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if pd.notna(numeric) else None


def _normalize_review_reasons(payload: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for source in (payload.get("review_reasons"), evidence.get("review_reasons")):
        if isinstance(source, list):
            values.extend(_normalize_optional_text(item) for item in source)
    deduped: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_category_status(
    payload_exists: bool,
    raw_status: Optional[str],
    category_code: Optional[str],
    confidence: Optional[float],
    decision_source: Optional[str],
    review_reasons: List[str],
    crawl_status: Optional[str],
    skip_reason: Optional[str],
) -> Tuple[str, str]:
    if not payload_exists:
        return "skipped", "none"
    if raw_status == "taxonomy_gap" or category_code in {None, "", "unknown"}:
        return "failure", "none"
    if raw_status != "ok":
        return "partial", "low"
    if crawl_status and str(crawl_status) != "success":
        return "skipped", "none"
    confidence_ok = confidence is None or confidence >= 0.9
    if review_reasons or skip_reason:
        return "partial", "low"
    if decision_source == "fused" and confidence_ok:
        return "success", "high"
    if decision_source in {"image", "metadata"} and confidence_ok:
        return "success", "medium"
    return "partial", "low"


def _load_category_summary_map(session_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    summary_path = session_dir / "category_summary.jsonl"
    if not summary_path.exists():
        return {}, False
    category_map: Dict[str, Dict[str, Any]] = {}
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = _safe_json_loads(line)
                if not isinstance(payload, dict):
                    continue
                product_id = _normalize_optional_text(payload.get("product_id"))
                if not product_id:
                    continue
                category_map[product_id] = payload
    except Exception:
        logger.warning("카테고리 summary 로드 실패: %s", summary_path)
        return {}, True
    return category_map, True


def load_product_category(
    session_dir: Path,
    product_id: str,
    crawl_status: Optional[str],
    summary_category_map: Optional[Dict[str, Dict[str, Any]]] = None,
    summary_exists: bool = False,
) -> ProductCategoryPayload:
    product_dir = session_dir / "products" / product_id
    result_path = product_dir / "category_result.json"
    result_exists = result_path.exists()
    summary_category_map = summary_category_map or {}
    raw_payload = summary_category_map.get(product_id)
    payload_source = "summary_jsonl" if raw_payload else None
    if raw_payload is None and result_exists:
        try:
            raw_payload = _safe_json_loads(result_path.read_text(encoding="utf-8"))
            payload_source = "result_json" if isinstance(raw_payload, dict) else None
        except Exception:
            logger.warning("카테고리 result 로드 실패: %s", result_path)
            raw_payload = None
    if not isinstance(raw_payload, dict):
        raw_payload = {}
        payload_source = None

    evidence = raw_payload.get("evidence") if isinstance(raw_payload.get("evidence"), dict) else {}
    review_reasons = _normalize_review_reasons(raw_payload, evidence)
    raw_status = _normalize_optional_text(raw_payload.get("status"))
    category_code = _normalize_optional_text(raw_payload.get("category_code"))
    confidence = _normalize_optional_float(raw_payload.get("confidence"))
    decision_source = _normalize_optional_text(raw_payload.get("decision_source")) or _normalize_optional_text(
        evidence.get("decision_source")
    )
    payload_exists = bool(raw_payload)
    skip_reason = None
    if not payload_exists:
        if crawl_status and str(crawl_status) != "success":
            skip_reason = "crawl_failed"
        elif summary_exists or result_exists:
            skip_reason = "missing_category_payload"
        else:
            skip_reason = "missing_category_files"
    ingest_status, quality_tier = _normalize_category_status(
        payload_exists=payload_exists,
        raw_status=raw_status,
        category_code=category_code,
        confidence=confidence,
        decision_source=decision_source,
        review_reasons=review_reasons,
        crawl_status=crawl_status,
        skip_reason=skip_reason,
    )
    normalized_fields = {
        "category_summary_exists": summary_exists,
        "category_result_exists": result_exists,
        "category_payload_source": payload_source,
        "category_raw_status": raw_status,
        "category_ingest_status": ingest_status,
        "category_quality_tier": quality_tier,
        "category_source": "raw_taxonomy" if payload_exists else "unavailable",
        "category_is_fallback": False,
        "category_fallback_label": None,
        "category_skip_reason": skip_reason,
        "category_l1": _normalize_optional_text(raw_payload.get("category_l1")),
        "category_l2": _normalize_optional_text(raw_payload.get("category_l2")),
        "category_l3": _normalize_optional_text(raw_payload.get("category_l3")),
        "category_code": category_code,
        "category_primary_color": _normalize_optional_text(raw_payload.get("primary_color")),
        "category_color_candidates_json": _json_dumps(raw_payload.get("color_candidates")) if raw_payload.get("color_candidates") is not None else None,
        "category_confidence": confidence,
        "category_decision_source": decision_source,
        "category_review_reasons_json": _json_dumps(review_reasons) if review_reasons else None,
        "category_vlm_raw_label": _normalize_optional_text(raw_payload.get("vlm_raw_label")) or _normalize_optional_text(
            evidence.get("vlm_raw_label")
        ),
        "category_taxonomy_gap_candidate": _normalize_optional_text(raw_payload.get("taxonomy_gap_candidate"))
        or _normalize_optional_text(evidence.get("taxonomy_gap_candidate")),
        "category_model_version": _normalize_optional_text(raw_payload.get("model_version")),
        "category_prompt_version": _normalize_optional_text(raw_payload.get("prompt_version")),
        "category_taxonomy_version": _normalize_optional_text(raw_payload.get("taxonomy_version")),
        "category_classified_at": _normalize_optional_text(raw_payload.get("classified_at")),
        "category_evidence_reason": normalize_text_block(evidence.get("reason")),
        "category_image_path": _normalize_optional_text(raw_payload.get("image_path")),
        "category_selected_image_path": _normalize_optional_text(evidence.get("selected_image_path")),
        "category_candidate_image_count": parse_int(evidence.get("candidate_image_count")),
        "category_image_quality_score": _normalize_optional_float(evidence.get("image_quality_score")),
    }
    return ProductCategoryPayload(
        normalized_fields=normalized_fields,
        raw_payload=copy.deepcopy(raw_payload),
    )


def infer_schema_version_guess(summary: Dict[str, Any]) -> str:
    adapter = _get_adapter_registry().select(summary)
    return adapter.infer_schema_version(summary)


def _legacy_infer_schema_version_guess(summary: Dict[str, Any]) -> str:
    if not isinstance(summary, dict):
        return "unknown"
    if "ocr_total_images" in summary:
        return "extended_summary_v2"
    products = summary.get("products")
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


PROVINCE_ALIASES: Dict[str, str] = {
    "서울": "서울특별시",
    "서울시": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}


def normalize_business_address(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = WHITESPACE_RE.sub(" ", str(raw)).strip()
    return text or None


def _normalize_province_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    cleaned = str(token).strip()
    if not cleaned:
        return None
    return PROVINCE_ALIASES.get(cleaned, cleaned)


def extract_business_address_parts(raw: Any) -> Dict[str, Optional[str]]:
    address = normalize_business_address(raw)
    if not address:
        return {
            "business_address": None,
            "business_province": None,
            "business_district": None,
            "business_dong": None,
            "business_location_label": None,
        }

    tokens = [token for token in ADDRESS_TOKEN_RE.split(address) if token]
    province = _normalize_province_token(tokens[0] if tokens else None)
    district = next((token for token in tokens[1:4] if re.search(r"(시|군|구)$", token)), None)
    dong = next((token for token in tokens[1:6] if re.search(r"(읍|면|동|가|리)$", token)), None)

    location_bits = [bit for bit in [province, district, dong] if bit]
    return {
        "business_address": address,
        "business_province": province,
        "business_district": district,
        "business_dong": dong,
        "business_location_label": " ".join(location_bits) if location_bits else address,
    }


def load_ocr_detail(product_dir: Path) -> Dict[str, Any]:
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
                if isinstance(value, dict):
                    normalized = normalize_text_block(value.get("full_text"))
                else:
                    normalized = normalize_text_block(value)
                if normalized:
                    entries[str(slot_key)] = normalized
    except Exception:
        logger.warning("OCR 로드 실패: %s", ocr_path)

    joined = "\n\n".join(entries.values()) if entries else None
    return {
        "ocr_source_exists": True,
        "ocr_has_data": bool(entries),
        "ocr_slot_count": len(entries),
        "ocr_slot_keys_json": json.dumps(list(entries.keys()), ensure_ascii=False) if entries else None,
        "ocr_entries_json": json.dumps(entries, ensure_ascii=False, sort_keys=True) if entries else None,
        "ocr_text_joined": joined,
        "ocr_text_length": len(joined) if joined else 0,
        "ocr_text_hash": hash_text(joined),
    }


def _get_adapter_registry() -> AdapterRegistry:
    global ADAPTER_REGISTRY
    if ADAPTER_REGISTRY is None:
        ADAPTER_REGISTRY = AdapterRegistry(
            [
                MusinsaV1Adapter(
                    parse_int=parse_int,
                    normalize_text_block=normalize_text_block,
                    hash_text=hash_text,
                    extract_business_address_parts=extract_business_address_parts,
                    resolve_first_present=_resolve_first_present,
                )
            ]
        )
    return ADAPTER_REGISTRY


def discover_summaries(
    data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[SnapshotSummaryRef]:
    summaries = sorted(data_dir.rglob("ranking_summary.json"))
    start_obj = parse_date_str(start_date)
    end_obj = parse_date_str(end_date)

    filtered: List[SnapshotSummaryRef] = []
    for summary in summaries:
        relative = summary.relative_to(data_dir)
        parts = relative.parts
        if len(parts) < 3:
            continue
        date_str = parts[-3]
        time_str = parts[-2]
        if not _looks_like_date_segment(date_str) or not _looks_like_time_segment(time_str):
            continue
        source_parts = parts[:-3]
        source_dataset = "/".join(source_parts) if source_parts else data_dir.name
        try:
            d = date.fromisoformat(date_str)
        except Exception:
            continue
        if start_obj and d < start_obj:
            continue
        if end_obj and d > end_obj:
            continue
        try:
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("요약 파일 로드 실패로 스킵: %s", summary)
            continue
        filtered.append(
            SnapshotSummaryRef(
                summary_path=summary,
                source_dataset=source_dataset,
                platform=_infer_platform(source_dataset),
                snapshot_date=date_str,
                snapshot_time=time_str,
                source_path=str(summary.parent),
                schema_version_guess=infer_schema_version_guess(summary_payload),
                schema_version_legacy=_legacy_infer_schema_version_guess(summary_payload),
            )
        )
    return filtered


def load_product_detail(session_dir: Path, product_id: str) -> ProductDetailPayload:
    adapter = _get_adapter_registry().select({})
    detail_payload = adapter.load_product_detail(session_dir, product_id)
    return ProductDetailPayload(
        normalized_fields=detail_payload.get("normalized_fields", {}),
        raw_info_map=detail_payload.get("raw_info_map", {}),
        source_fields=detail_payload.get("source_fields", {}),
    )


def build_snapshot_records(snapshot_ref: SnapshotSummaryRef) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_path = snapshot_ref.summary_path
    session_dir = summary_path.parent
    date_str = snapshot_ref.snapshot_date
    time_str = snapshot_ref.snapshot_time
    source_snapshot_id = f"{date_str}_{time_str}"
    snapshot_id = f"{snapshot_ref.source_dataset}:{source_snapshot_id}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    adapter = _get_adapter_registry().select(summary)
    adapter_id = getattr(adapter, "adapter_id", "unknown")
    adapter_version = getattr(adapter, "adapter_version", "unknown")
    summary_meta = {key: value for key, value in summary.items() if key != "products"}
    summary_category_map, summary_category_exists = _load_category_summary_map(session_dir)

    crawl_datetime = summary_meta.get("crawl_datetime")
    records: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    seen_product_ids = set()
    for product in summary.get("products", []):
        raw_product_id = str(product.get("product_id", ""))
        product_id = raw_product_id
        if product_id in seen_product_ids:
            logger.warning("동일 스냅샷 내 중복 product_id 감지: snapshot=%s product_id=%s", snapshot_id, product_id)
            continue
        seen_product_ids.add(product_id)
        detail_payload = adapter.load_product_detail(session_dir, raw_product_id)
        detail = ProductDetailPayload(
            normalized_fields=detail_payload.get("normalized_fields", {}),
            raw_info_map=detail_payload.get("raw_info_map", {}),
            source_fields=detail_payload.get("source_fields", {}),
        )
        category = load_product_category(
            session_dir,
            raw_product_id,
            crawl_status=product.get("crawl_status"),
            summary_category_map=summary_category_map,
            summary_exists=summary_category_exists,
        )
        product_dir = session_dir / "products" / raw_product_id
        discount_pct, discount_raw, discount_source_key, discount_raw_present = _resolve_first_present(
            product,
            ("discount", "discount_rate"),
            parse_discount,
        )
        original_price, original_price_raw, original_price_source_key, original_price_raw_present = _resolve_first_present(
            product,
            ("original_price",),
            parse_price,
        )
        discount_amount, discount_amount_raw, discount_amount_source_key, discount_amount_raw_present = _resolve_first_present(
            product,
            ("discount_amount",),
            parse_price,
        )
        ocr_images_count, ocr_images_raw, ocr_images_source_key, ocr_images_raw_present = _resolve_first_present(
            product,
            ("ocr_images",),
            parse_int,
        )
        records.append(
            {
                "snapshot_id": snapshot_id,
                "snapshot_date": date_str,
                "snapshot_time": time_str,
                "crawl_datetime": crawl_datetime,
                "platform": snapshot_ref.platform,
                "source_dataset": snapshot_ref.source_dataset,
                "source_path": snapshot_ref.source_path,
                "schema_version": snapshot_ref.schema_version_guess,
                "source_snapshot_id": source_snapshot_id,
                "entity_id": f"{snapshot_ref.platform}:{raw_product_id}",
                "rank": product.get("rank"),
                "product_id": product_id,
                "product_id_raw": raw_product_id,
                "product_url": product.get("product_url"),
                "brand": product.get("brand"),
                "name": product.get("name"),
                "product_path": str(product_dir),
                "price_raw": product.get("price"),
                "price": parse_price(product.get("price")),
                "discount_raw": discount_raw,
                "discount_pct": discount_pct,
                "discount_pct_source": f"product.{discount_source_key}" if discount_source_key else None,
                "discount_pct_raw_present": discount_raw_present,
                "crawl_status": product.get("crawl_status"),
                "crawl_error": product.get("error"),
                "tags_count_summary": product.get("tags_count"),
                "info_count_summary": product.get("info_count"),
                "images_count_summary": product.get("images_count"),
                "original_price_raw": original_price_raw,
                "original_price": original_price,
                "original_price_source": f"product.{original_price_source_key}" if original_price_source_key else None,
                "original_price_raw_present": original_price_raw_present,
                "discount_amount_raw": discount_amount_raw,
                "discount_amount": discount_amount,
                "discount_amount_source": f"product.{discount_amount_source_key}" if discount_amount_source_key else None,
                "discount_amount_raw_present": discount_amount_raw_present,
                "buying_count_text": product.get("buying_count_text"),
                "watching_count_text": product.get("watching_count_text"),
                "ranking_badge": product.get("ranking_badge"),
                "ocr_images_count_summary": ocr_images_count,
                "ocr_images_count_summary_raw": ocr_images_raw,
                "ocr_images_count_summary_source": f"product.{ocr_images_source_key}" if ocr_images_source_key else None,
                "ocr_images_count_summary_raw_present": ocr_images_raw_present,
                "product_meta_json": _json_dumps(product.get("product_meta")) if product.get("product_meta") is not None else None,
                "ocr_total_images_summary": parse_int(summary_meta.get("ocr_total_images")),
                "ocr_total_images_summary_raw_present": "ocr_total_images" in summary_meta,
                **detail.normalized_fields,
                **category.normalized_fields,
            }
        )
        raw_rows.append(
            {
                "snapshot_id": snapshot_id,
                "snapshot_date": date_str,
                "snapshot_time": time_str,
                "crawl_datetime": crawl_datetime,
                "platform": snapshot_ref.platform,
                "source_dataset": snapshot_ref.source_dataset,
                "source_path": snapshot_ref.source_path,
                "schema_version": snapshot_ref.schema_version_guess,
                "source_snapshot_id": source_snapshot_id,
                "product_id": product_id,
                "product_id_raw": raw_product_id,
                "brand": product.get("brand"),
                "name": product.get("name"),
                "channel_id": snapshot_ref.source_dataset,
                "adapter_id": adapter_id,
                "adapter_version": adapter_version,
                "ingested_at": pd.Timestamp.utcnow().isoformat(),
                "raw_schema_fingerprint": hashlib.sha256(
                    _json_dumps(
                        adapter.build_extension_payload(summary_payload=summary, product_payload=product)
                    ).encode("utf-8")
                ).hexdigest(),
                "raw_summary": copy.deepcopy(summary_meta),
                "raw_product": copy.deepcopy(product),
                "raw_info_map": copy.deepcopy(detail.raw_info_map),
                "raw_category": copy.deepcopy(category.raw_payload),
            }
        )
    return records, raw_rows


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    featured = df.copy()
    # metrics-refresh 모드에서는 이미 계산된 파생 컬럼이 들어올 수 있으므로
    # 재계산 전에 제거해 merge suffix 충돌(rank_velocity_x/y 등)을 방지한다.
    derived_columns_to_reset = [
        "rank_prev",
        "rank_prev_observed",
        "rank_prev_filled",
        "rank_filled",
        "score",
        "rank_energy",
        "rank_energy_prev",
        "energy_velocity",
        "energy_acceleration",
        "entry_score",
        "exit_score",
        "is_first_seen",
        "is_continuous_observation",
        "momentum_event_state",
        "momentum_event_label",
        "rank_velocity",
        "rank_acceleration",
        "rank_volatility_6",
        "stability_score",
        "discount_prev",
        "discount_velocity",
        "discount_efficiency",
        "rank_velocity_capped",
        "rank_acceleration_capped",
        "rank_velocity_ema3",
        "consistency_score",
        "consistency_scaled",
        "trend_score",
        "acceleration_score",
        "momentum_core",
        "momentum_score",
        "breakout_score",
        "is_reentry",
        "is_dropout",
        "presence_ratio_5",
        "gap_steps",
        "rank_tier_weight",
        "velocity_signal",
        "continuity_weight",
        "event_raw",
        "event_bonus",
        "event_bonus_cap",
        "momentum_scale_k",
    ]
    drop_existing = [column for column in derived_columns_to_reset if column in featured.columns]
    if drop_existing:
        featured = featured.drop(columns=drop_existing)
    featured["crawl_datetime"] = pd.to_datetime(featured["crawl_datetime"], errors="coerce")
    entity_key = "entity_id" if "entity_id" in featured.columns else "product_id"
    featured = featured.sort_values([entity_key, "crawl_datetime"]).reset_index(drop=True)

    scope_cols = [col for col in ("source_dataset", "platform", "schema_version") if col in featured.columns]
    calc_keys = scope_cols + [entity_key, "snapshot_id"]
    rank_base = featured[[col for col in calc_keys + ["crawl_datetime", "rank"] if col in featured.columns]].copy()
    rank_features_frames: list[pd.DataFrame] = []
    if "snapshot_id" in rank_base.columns:
        grouped_items = rank_base.groupby(scope_cols, dropna=False) if scope_cols else [((), rank_base)]
        for scope_key, scope_df in grouped_items:
            snapshots = (
                scope_df[["snapshot_id", "crawl_datetime"]]
                .drop_duplicates(subset=["snapshot_id"])
                .sort_values(["crawl_datetime", "snapshot_id"], na_position="last")
            )
            if snapshots.empty:
                continue
            products = scope_df[[entity_key]].drop_duplicates()
            panel = products.assign(_key=1).merge(snapshots.assign(_key=1), on="_key", how="inner").drop(columns="_key")
            merged = panel.merge(
                scope_df[[entity_key, "snapshot_id", "rank"]],
                on=[entity_key, "snapshot_id"],
                how="left",
            )
            merged["rank_observed"] = merged["rank"].notna()
            merged["rank_numeric"] = pd.to_numeric(merged["rank"], errors="coerce")
            merged["rank_filled"] = merged["rank_numeric"].fillna(51.0)
            merged["score"] = 51.0 - merged["rank_filled"]
            standard_ratio = (merged["score"] / 50.0).clip(lower=0.0, upper=1.0)
            merged["rank_energy"] = merged["score"] * (1.0 + (2.0 * standard_ratio.pow(2)))
            merged = merged.sort_values([entity_key, "crawl_datetime", "snapshot_id"], na_position="last")
            by_product = merged.groupby(entity_key, group_keys=False)
            merged["rank_prev_observed"] = by_product["rank_numeric"].transform(lambda series: series.ffill().shift(1))
            merged["rank_prev"] = merged["rank_prev_observed"]
            merged["rank_prev_filled"] = by_product["rank_filled"].shift(1).fillna(51.0)
            merged["rank_energy_prev"] = by_product["rank_energy"].shift(1).fillna(0.0)
            merged["prev_observed"] = by_product["rank_observed"].shift(1).fillna(False).astype(bool)
            merged["prior_observed_count"] = by_product["rank_observed"].transform(
                lambda series: series.astype(int).cumsum().shift(1).fillna(0)
            )
            merged["is_first_seen"] = merged["rank_observed"] & merged["prior_observed_count"].eq(0)
            merged["is_reentry"] = (~merged["prev_observed"]) & merged["rank_observed"] & merged["prior_observed_count"].gt(0)
            merged["is_dropout"] = merged["prev_observed"] & (~merged["rank_observed"])
            merged["is_continuous_observation"] = merged["prev_observed"] & merged["rank_observed"]
            merged["energy_velocity"] = np.where(
                merged["is_continuous_observation"],
                merged["rank_energy"] - merged["rank_energy_prev"],
                np.nan,
            )
            merged["energy_acceleration"] = np.where(
                merged["is_continuous_observation"],
                by_product["energy_velocity"].diff(1),
                np.nan,
            )
            merged["rank_velocity"] = np.where(
                merged["is_continuous_observation"],
                merged["rank_prev_observed"] - merged["rank_numeric"],
                np.nan,
            )
            merged["rank_acceleration"] = by_product["rank_velocity"].diff(1)
            merged["entry_score"] = np.where(
                merged["rank_observed"] & (~merged["prev_observed"]),
                merged["rank_energy"],
                np.nan,
            )
            merged["exit_score"] = np.where(merged["is_dropout"], merged["rank_energy_prev"], np.nan)
            merged["_snapshot_ord"] = merged["snapshot_id"].astype("category").cat.codes
            merged["prev_snapshot_ord"] = by_product["_snapshot_ord"].shift(1)
            merged["gap_steps"] = (
                (merged["_snapshot_ord"] - merged["prev_snapshot_ord"] - 1)
                .clip(lower=0)
                .fillna(0)
                .astype(int)
            )
            merged["presence_ratio_5"] = by_product["rank_observed"].transform(
                lambda series: series.astype(float).rolling(window=5, min_periods=2).mean()
            ).fillna(1.0)
            if scope_cols:
                if not isinstance(scope_key, tuple):
                    scope_key = (scope_key,)
                for idx, col in enumerate(scope_cols):
                    merged[col] = scope_key[idx] if idx < len(scope_key) else None
            rank_features_frames.append(
                merged[
                    calc_keys
                    + [
                        "rank_prev",
                        "rank_prev_observed",
                        "rank_prev_filled",
                        "rank_filled",
                        "score",
                        "rank_energy",
                        "rank_energy_prev",
                        "energy_velocity",
                        "energy_acceleration",
                        "entry_score",
                        "exit_score",
                        "is_first_seen",
                        "is_continuous_observation",
                        "rank_velocity",
                        "rank_acceleration",
                        "is_reentry",
                        "is_dropout",
                        "gap_steps",
                        "presence_ratio_5",
                    ]
                ]
            )
    if rank_features_frames:
        rank_features = pd.concat(rank_features_frames, ignore_index=True)
        rank_features = rank_features.drop_duplicates(subset=calc_keys, keep="last")
        featured = featured.merge(rank_features, on=calc_keys, how="left")
    else:
        grouped = featured.groupby(entity_key, group_keys=False)
        featured["rank_prev"] = grouped["rank"].shift(1)
        featured["rank_prev_observed"] = featured["rank_prev"]
        featured["rank_prev_filled"] = featured["rank_prev"].fillna(51.0)
        featured["rank_filled"] = pd.to_numeric(featured["rank"], errors="coerce").fillna(51.0)
        featured["score"] = 51.0 - featured["rank_filled"]
        standard_ratio = (featured["score"] / 50.0).clip(lower=0.0, upper=1.0)
        featured["rank_energy"] = featured["score"] * (1.0 + (2.0 * standard_ratio.pow(2)))
        grouped = featured.groupby(entity_key, group_keys=False)
        featured["rank_energy_prev"] = grouped["rank_energy"].shift(1).fillna(0.0)
        featured["is_first_seen"] = featured["rank_prev"].isna()
        featured["is_continuous_observation"] = featured["rank_prev"].notna() & featured["rank"].notna()
        featured["energy_velocity"] = np.where(
            featured["is_continuous_observation"],
            featured["rank_energy"] - featured["rank_energy_prev"],
            np.nan,
        )
        featured["energy_acceleration"] = grouped["energy_velocity"].diff(1)
        featured["rank_velocity"] = np.where(
            featured["is_continuous_observation"],
            featured["rank_prev"] - featured["rank"],
            np.nan,
        )
        featured["rank_acceleration"] = grouped["rank_velocity"].diff(1)
        featured["entry_score"] = np.where(featured["is_first_seen"], featured["rank_energy"], np.nan)
        featured["exit_score"] = np.nan
        featured["is_reentry"] = False
        featured["is_dropout"] = False
        featured["gap_steps"] = 0
        featured["presence_ratio_5"] = 1.0

    grouped = featured.groupby(entity_key, group_keys=False)
    rolling_std = grouped["rank"].rolling(window=6, min_periods=2).std().reset_index(level=0, drop=True)
    featured["rank_volatility_6"] = rolling_std
    featured["stability_score"] = (1 / (1 + featured["rank_volatility_6"])).fillna(1.0)

    featured["discount_prev"] = grouped["discount_pct"].shift(1)
    featured["discount_velocity"] = featured["discount_pct"] - featured["discount_prev"]

    featured["discount_efficiency"] = featured["rank_velocity"] / featured["discount_pct"]
    featured.loc[featured["discount_pct"].isna() | (featured["discount_pct"] <= 0), "discount_efficiency"] = pd.NA

    # 실무형 모멘텀: 표준화 없이 순위 에너지의 절대 변화량을 보존한다.
    featured["rank_velocity_capped"] = featured["rank_velocity"]
    featured["rank_acceleration_capped"] = featured["rank_acceleration"]
    featured["rank_velocity_ema3"] = grouped["rank_velocity"].transform(
        lambda series: series.fillna(0).ewm(span=3, adjust=False).mean()
    )
    featured["consistency_score"] = grouped["energy_velocity"].transform(
        lambda series: series.gt(0).rolling(window=5, min_periods=2).mean()
    ).fillna(0.5)
    featured["consistency_scaled"] = (featured["consistency_score"] - 0.5) * 20.0
    featured["trend_score"] = featured["energy_velocity"].fillna(0)
    featured["acceleration_score"] = featured["energy_acceleration"].fillna(0)
    featured["presence_ratio_5"] = pd.to_numeric(featured.get("presence_ratio_5"), errors="coerce").fillna(1.0).clip(0, 1)
    energy_ratio = (pd.to_numeric(featured["score"], errors="coerce").fillna(0) / 50.0).clip(lower=0.0, upper=1.0)
    featured["rank_tier_weight"] = 1.0 + (2.0 * energy_ratio.pow(2))
    featured["velocity_signal"] = featured["energy_velocity"].fillna(0)
    featured["continuity_weight"] = featured["consistency_score"].fillna(0.5)
    featured["momentum_scale_k"] = 1.0
    featured["momentum_core"] = featured["energy_velocity"]
    featured["event_raw"] = featured["entry_score"].fillna(0)
    featured["event_bonus_cap"] = featured["event_raw"]
    featured["event_bonus"] = featured["event_raw"]
    featured["breakout_score"] = featured["entry_score"]
    featured["momentum_score"] = featured["momentum_core"]
    velocity = pd.to_numeric(featured["energy_velocity"], errors="coerce").fillna(0.0)
    acceleration = pd.to_numeric(featured["energy_acceleration"], errors="coerce").fillna(0.0)
    persistence = pd.to_numeric(featured["consistency_score"], errors="coerce").fillna(0.5)
    is_first_seen = featured["is_first_seen"].fillna(False).astype(bool)
    is_reentry = featured["is_reentry"].fillna(False).astype(bool)
    is_continuous = featured["is_continuous_observation"].fillna(False).astype(bool)
    featured["momentum_event_state"] = np.select(
        [
            is_first_seen,
            is_reentry,
            is_continuous & velocity.gt(0) & acceleration.gt(0.5),
            is_continuous & velocity.gt(0) & persistence.ge(0.6),
            is_continuous & velocity.gt(0) & acceleration.lt(-0.5),
            is_continuous & velocity.lt(0) & acceleration.lt(0),
        ],
        ["first_seen", "chart_in_spike", "breakout", "sustained_growth", "cooling", "reversal"],
        default="steady",
    )
    featured["momentum_event_label"] = featured["momentum_event_state"].map(
        {
            "chart_in_spike": "순위권 진입",
            "first_seen": "첫 관측",
            "chart_out_drop": "순위권 탈락",
            "out_of_chart": "순위권 밖",
            "breakout": "가속 상승",
            "sustained_growth": "지속 상승",
            "cooling": "상승 둔화",
            "reversal": "하락 전환",
            "steady": "정체",
        }
    )
    featured["price_band"] = pd.cut(
        featured["price"],
        bins=[0, 30000, 70000, 120000, 200000, 500000, float("inf")],
        labels=["~3만", "3-7만", "7-12만", "12-20만", "20-50만", "50만+"],
        include_lowest=True,
    )
    return featured


def merge_multimodal_metrics(featured_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    if featured_df.empty:
        return featured_df

    result = featured_df.copy()
    if not segments_df.empty:
        seg_stats = (
            segments_df.groupby(["snapshot_id", "product_id"], as_index=False)
            .agg(
                segment_count=("segment_id", "count"),
                main_image_segment_count=("embedding_target", "sum"),
            )
        )
        result = result.merge(seg_stats, on=["snapshot_id", "product_id"], how="left")

    return result


def build_snapshot_asset_coverage(featured_df: pd.DataFrame) -> pd.DataFrame:
    if featured_df.empty:
        return pd.DataFrame()
    columns = [
        "snapshot_id",
        "snapshot_date",
        "snapshot_time",
        "platform",
        "source_dataset",
        "schema_version",
        "product_id",
        "crawl_datetime",
        "crawl_status",
        "product_dir_exists",
        "tags_source_exists",
        "product_info_exists",
        "size_table_exists",
        "detail_images_exists",
        "detail_image_count_actual",
        "ocr_source_exists",
        "ocr_has_data",
        "ocr_slot_count",
        "ocr_text_hash",
    ]
    existing = [c for c in columns if c in featured_df.columns]
    return featured_df[existing].copy()


def build_product_dimension(featured_df: pd.DataFrame, manifest_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if featured_df.empty:
        return pd.DataFrame()

    working = featured_df.copy()
    entity_key = _entity_column_name(working)
    working["crawl_datetime"] = pd.to_datetime(working["crawl_datetime"], errors="coerce")
    working["ocr_slot_count"] = pd.to_numeric(working.get("ocr_slot_count"), errors="coerce").fillna(0)
    working["ocr_text_length"] = pd.to_numeric(working.get("ocr_text_length"), errors="coerce").fillna(0)
    working["_ocr_rank"] = working["ocr_has_data"].fillna(False).astype(int)
    working = working.sort_values(
        ["_ocr_rank", "crawl_datetime", "ocr_slot_count", "ocr_text_length"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    dim = working.groupby(entity_key, as_index=False).head(1).copy()

    if manifest_df is not None and not manifest_df.empty:
        manifest = manifest_df[(manifest_df["image_exists"] == True) & (manifest_df["is_main_image"] == True)].copy()
        if not manifest.empty:
            manifest = manifest.sort_values(
                ["product_id", "snapshot_date", "snapshot_time", "file_size_bytes"],
                ascending=[True, False, False, False],
                na_position="last",
            )
            rep_images = manifest.groupby("product_id", as_index=False).head(1)[
                ["product_id", "image_path", "sha256", "width", "height"]
            ].rename(
                columns={
                    "image_path": "representative_image_path",
                    "sha256": "representative_image_sha256",
                    "width": "representative_image_width",
                    "height": "representative_image_height",
                }
            )
            dim = dim.merge(rep_images, on="product_id", how="left")

    snapshot_counts = (
        featured_df.groupby(entity_key, as_index=False)
        .agg(
            observed_snapshot_count=("snapshot_id", "nunique"),
            observed_date_count=("snapshot_date", "nunique"),
            ocr_snapshot_count=("ocr_has_data", "sum"),
        )
    )
    dim = dim.merge(snapshot_counts, on=entity_key, how="left")
    dim["ocr_snapshot_count"] = dim["ocr_snapshot_count"].fillna(0).astype(int)
    dim["is_repeated_product"] = dim["observed_snapshot_count"].fillna(0).astype(int) > 1

    keep_columns = [
        "platform",
        "source_dataset",
        "schema_version",
        "entity_id",
        "product_id",
        "product_url",
        "brand",
        "name",
        "tags",
        "tags_joined",
        "tag_count_actual",
        "material",
        "color",
        "manufacturer",
        "origin_country",
        "shipping_fee",
        "category_l1",
        "category_l2",
        "category_l3",
        "category_code",
        "category_primary_color",
        "category_payload_source",
        "category_raw_status",
        "category_ingest_status",
        "category_quality_tier",
        "category_source",
        "category_is_fallback",
        "category_fallback_label",
        "category_skip_reason",
        "category_confidence",
        "category_decision_source",
        "category_review_reasons_json",
        "category_vlm_raw_label",
        "category_taxonomy_gap_candidate",
        "category_model_version",
        "category_prompt_version",
        "category_taxonomy_version",
        "category_classified_at",
        "category_evidence_reason",
        "business_address",
        "business_province",
        "business_district",
        "business_dong",
        "business_location_label",
        "product_dir_exists",
        "tags_source_exists",
        "product_info_exists",
        "size_table_exists",
        "detail_images_exists",
        "detail_image_count_actual",
        "ocr_source_exists",
        "ocr_has_data",
        "ocr_slot_count",
        "ocr_slot_keys_json",
        "ocr_entries_json",
        "ocr_text_joined",
        "ocr_text_length",
        "ocr_text_hash",
        "snapshot_id",
        "snapshot_date",
        "snapshot_time",
        "crawl_datetime",
        "observed_snapshot_count",
        "observed_date_count",
        "ocr_snapshot_count",
        "is_repeated_product",
        "representative_image_path",
        "representative_image_sha256",
        "representative_image_width",
        "representative_image_height",
    ]
    existing = [c for c in keep_columns if c in dim.columns]
    dim = dim[existing].rename(
        columns={
            "snapshot_id": "representative_snapshot_id",
            "snapshot_date": "representative_snapshot_date",
            "snapshot_time": "representative_snapshot_time",
            "crawl_datetime": "representative_crawl_datetime",
        }
    )
    return dim.sort_values(["product_id"]).reset_index(drop=True)


def build_raw_snapshot_products_table(raw_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame()
    serialized_rows = []
    for row in raw_rows:
        serialized_rows.append(
            {
                "snapshot_id": row["snapshot_id"],
                "source_snapshot_id": row.get("source_snapshot_id"),
                "snapshot_date": row["snapshot_date"],
                "snapshot_time": row["snapshot_time"],
                "crawl_datetime": row.get("crawl_datetime"),
                "platform": row.get("platform"),
                "source_dataset": row.get("source_dataset"),
                "schema_version": row.get("schema_version"),
                "product_id": row["product_id"],
                "product_id_raw": row.get("product_id_raw"),
                "brand": row.get("brand"),
                "name": row.get("name"),
                "raw_summary_json": _json_dumps(row.get("raw_summary", {})),
                "raw_product_json": _json_dumps(row.get("raw_product", {})),
                "raw_info_map_json": _json_dumps(row.get("raw_info_map", {})),
                "raw_category_json": _json_dumps(row.get("raw_category", {})),
                "raw_summary_keys_json": _json_dumps(sorted((row.get("raw_summary") or {}).keys())),
                "raw_product_keys_json": _json_dumps(sorted((row.get("raw_product") or {}).keys())),
                "raw_info_keys_json": _json_dumps(sorted((row.get("raw_info_map") or {}).keys())),
                "raw_category_keys_json": _json_dumps(sorted((row.get("raw_category") or {}).keys())),
            }
        )
    return pd.DataFrame(serialized_rows)


def build_raw_schema_artifact(raw_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    field_stats: Dict[str, Dict[str, Any]] = {}
    total_rows = len(raw_rows)
    source_counts = Counter()
    for row in raw_rows:
        source_counts[str(row.get("source_dataset") or "unknown")] += 1
        for source_name, prefix in (
            ("raw_summary", "summary"),
            ("raw_product", "product"),
            ("raw_info_map", "product_info"),
            ("raw_category", "category"),
        ):
            payload = row.get(source_name)
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                field_name = f"{prefix}.{key}"
                if _is_deprecated_schema_field(field_name, "raw"):
                    continue
                entry = field_stats.setdefault(
                    field_name,
                    {
                        "field": field_name,
                        "scope": "raw",
                        "sourcePath": field_name,
                        "observedCount": 0,
                        "missingValueCount": 0,
                        "typeCounter": Counter(),
                        "sampleValue": None,
                    },
                )
                entry["observedCount"] += 1
                value_type = _field_type(value)
                entry["typeCounter"][value_type] += 1
                if value is None:
                    entry["missingValueCount"] += 1
                if entry["sampleValue"] is None:
                    entry["sampleValue"] = _sample_value(value)
    fields = []
    for field_name in sorted(field_stats.keys()):
        entry = field_stats[field_name]
        type_counter = entry.pop("typeCounter")
        observed_count = int(entry["observedCount"])
        missing_value_count = int(entry["missingValueCount"])
        fields.append(
            {
                **entry,
                "observedCount": observed_count,
                "observedRatePct": round((observed_count / total_rows) * 100.0, 2) if total_rows else 0.0,
                "nonNullCount": observed_count - missing_value_count,
                "nonNullRatePct": round(((observed_count - missing_value_count) / total_rows) * 100.0, 2) if total_rows else 0.0,
                "inferredType": _inferred_type(type_counter),
                "types": dict(type_counter),
            }
        )
    return {
        "scope": "raw",
        "rowCount": total_rows,
        "fieldCount": len(fields),
        "sourceDatasets": dict(source_counts),
        "fields": fields,
    }


def build_normalized_schema_artifact(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"scope": "normalized", "rowCount": 0, "fieldCount": 0, "fields": []}
    fields = []
    total_rows = len(df)
    source_counts = df["source_dataset"].fillna("unknown").astype(str).value_counts().to_dict() if "source_dataset" in df.columns else {}
    for column in sorted(df.columns):
        if _is_deprecated_schema_field(column, "normalized"):
            continue
        series = df[column]
        non_null = series.dropna()
        type_counter: Counter[str] = Counter(_field_type(value) for value in non_null.head(500).tolist())
        if not type_counter and total_rows:
            type_counter["null"] = total_rows
        sample_value = None
        for value in non_null.head(20).tolist():
            sample_value = _sample_value(value)
            if sample_value is not None:
                break
        non_null_count = int(non_null.shape[0])
        fields.append(
            {
                "field": column,
                "scope": "normalized",
                "sourcePath": column,
                "observedCount": total_rows,
                "observedRatePct": 100.0,
                "nonNullCount": non_null_count,
                "nonNullRatePct": round((non_null_count / total_rows) * 100.0, 2) if total_rows else 0.0,
                "inferredType": _inferred_type(type_counter),
                "types": dict(type_counter),
                "sampleValue": sample_value,
            }
        )
    return {
        "scope": "normalized",
        "rowCount": total_rows,
        "fieldCount": len(fields),
        "sourceDatasets": {str(key): int(value) for key, value in source_counts.items()},
        "fields": fields,
    }


def build_schema_diff_artifact(raw_schema: Dict[str, Any], normalized_schema: Dict[str, Any], normalized_df: pd.DataFrame) -> Dict[str, Any]:
    raw_fields = {str(field["field"]): field for field in raw_schema.get("fields", [])}
    normalized_fields = {str(field["field"]): field for field in normalized_schema.get("fields", [])}
    raw_only = [raw_fields[name] for name in sorted(set(raw_fields) - set(normalized_fields))]
    normalized_only = [normalized_fields[name] for name in sorted(set(normalized_fields) - set(raw_fields))]
    source_mapping_rows = []
    if not normalized_df.empty:
        for column in sorted([name for name in normalized_df.columns if name.endswith("_source")]):
            base_field = column[:-7]
            if base_field not in normalized_df.columns:
                continue
            sources = sorted({str(value) for value in normalized_df[column].dropna().astype(str).tolist() if str(value).strip()})
            raw_present_column = f"{base_field}_raw_present"
            source_mapping_rows.append(
                {
                    "normalizedField": base_field,
                    "observedSources": sources,
                    "filledCount": int(normalized_df[base_field].notna().sum()),
                    "rawPresentCount": int(normalized_df[raw_present_column].fillna(False).astype(bool).sum())
                    if raw_present_column in normalized_df.columns
                    else None,
                }
            )
    return {
        "summary": {
            "rawFieldCount": len(raw_fields),
            "normalizedFieldCount": len(normalized_fields),
            "rawOnlyFieldCount": len(raw_only),
            "normalizedOnlyFieldCount": len(normalized_only),
            "sourceMappedFieldCount": len(source_mapping_rows),
        },
        "rawOnlyFields": raw_only,
        "normalizedOnlyFields": normalized_only,
        "sourceMappings": source_mapping_rows,
    }


CORE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "snapshot_id",
    "crawl_datetime",
    "product_id",
    "name",
    "source_dataset",
    "platform",
    "schema_version",
)


def _normalize_source_dataset_key(value: Any) -> str:
    """source_dataset에 리스트·dict 등이 섞여도 groupby/unique가 실패하지 않도록 문자열 키로 만든다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value).strip()


def _profile_value_fingerprint(value: Any) -> str:
    """프로파일 distinct_count용: list/dict 등 비해시 값을 안정적인 문자열로 만든다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "__null__"
    if isinstance(value, (list, dict, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _safe_distinct_count_nonnull(series: pd.Series) -> int:
    if series.empty:
        return 0
    return len({_profile_value_fingerprint(value) for value in series})


def build_extra_metadata(
    raw_df: pd.DataFrame,
    fact_df: pd.DataFrame,
    dim_products_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    dedup_dropped_rows: int,
    raw_products_df: pd.DataFrame,
    raw_schema: Dict[str, Any],
    normalized_schema: Dict[str, Any],
    schema_diff: Dict[str, Any],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(extra_metadata or {})
    metadata["dedup"] = {
        "dropped_snapshot_product_duplicates": int(dedup_dropped_rows),
    }
    metadata["coverage"] = {
        "product_dir_missing_rows": int((~fact_df["product_dir_exists"].fillna(False)).sum()) if "product_dir_exists" in fact_df.columns else 0,
        "ocr_rows": int(fact_df["ocr_has_data"].fillna(False).sum()) if "ocr_has_data" in fact_df.columns else 0,
        "ocr_slot_total": int(fact_df["ocr_slot_count"].fillna(0).sum()) if "ocr_slot_count" in fact_df.columns else 0,
        "detail_image_rows": int(fact_df["detail_images_exists"].fillna(False).sum()) if "detail_images_exists" in fact_df.columns else 0,
    }
    repeated_distribution: Dict[str, int] = {}
    entity_column = _entity_column_name(fact_df)
    if not raw_df.empty and entity_column in raw_df.columns:
        snapshot_counts = raw_df.groupby(entity_column)["snapshot_id"].nunique()
        repeated_distribution = {
            "1": int((snapshot_counts == 1).sum()),
            "2_4": int(((snapshot_counts >= 2) & (snapshot_counts <= 4)).sum()),
            "5_9": int(((snapshot_counts >= 5) & (snapshot_counts <= 9)).sum()),
            "10_plus": int((snapshot_counts >= 10).sum()),
        }
    metadata["product_observation_distribution"] = repeated_distribution
    metadata["outputs"] = {
        "dim_product_rows": int(len(dim_products_df)),
        "coverage_rows": int(len(coverage_df)),
        "raw_snapshot_product_rows": int(len(raw_products_df)),
    }
    if "source_dataset" in fact_df.columns:
        fact_work = fact_df.copy()
        fact_work["__source_dataset_key__"] = fact_work["source_dataset"].map(_normalize_source_dataset_key)
        source_summary = (
            fact_work.groupby("__source_dataset_key__", as_index=False)
            .agg(
                snapshot_count=("snapshot_id", "nunique"),
                record_count=(entity_column, "count"),
                product_count=(entity_column, "nunique"),
            )
            .rename(columns={"__source_dataset_key__": "source_dataset"})
            .sort_values("source_dataset")
        )
        metadata["sourceDatasets"] = source_summary.to_dict("records")
    metadata["schema"] = {
        "raw": {
            "field_count": int(raw_schema.get("fieldCount", 0)),
            "row_count": int(raw_schema.get("rowCount", 0)),
        },
        "normalized": {
            "field_count": int(normalized_schema.get("fieldCount", 0)),
            "row_count": int(normalized_schema.get("rowCount", 0)),
        },
        "diff_summary": schema_diff.get("summary", {}),
    }
    raw_field_names = {str(field["field"]) for field in raw_schema.get("fields", [])}
    warnings: list[dict[str, Any]] = []
    if "product.discount_rate" in raw_field_names and "discount_pct" in fact_df.columns:
        missing_with_discount_rate = 0
        for payload in raw_products_df.get("raw_product_json", pd.Series(dtype=object)).tolist():
            product_payload = _safe_json_loads(payload)
            if isinstance(product_payload, dict) and product_payload.get("discount_rate") not in {None, ""}:
                missing_with_discount_rate += 1
        unresolved = int(
            fact_df[(fact_df["discount_pct"].isna()) & (fact_df["discount_pct_raw_present"].fillna(False).astype(bool))].shape[0]
        )
        warnings.append(
            {
                "kind": "discount_fallback",
                "message": "discount_rate가 존재하지만 normalized discount_pct가 비어 있는 레코드를 점검하세요.",
                "rawDiscountRateRows": missing_with_discount_rate,
                "unresolvedRows": unresolved,
            }
        )
    if "product_info.제품 소재" in raw_field_names and "material" in fact_df.columns:
        warnings.append(
            {
                "kind": "product_info_alias",
                "message": "product_info의 `제품 소재`가 `material`로 승격되는 fallback 경로를 사용합니다.",
                "fallbackRows": int(
                    fact_df[
                        (fact_df["material_source"] == "product_info.제품 소재")
                        & fact_df["material"].notna()
                    ].shape[0]
                ),
            }
        )
    metadata["qualityWarnings"] = warnings
    return metadata


def build_channel_profiles(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty or "source_dataset" not in df.columns:
        return {}
    work = df.copy()
    work["__source_dataset_key__"] = work["source_dataset"].map(_normalize_source_dataset_key)
    profiles: Dict[str, Dict[str, Any]] = {}
    for channel, channel_df in work.groupby("__source_dataset_key__", dropna=False):
        channel_name = str(channel).strip() if channel is not None and not (isinstance(channel, float) and pd.isna(channel)) else ""
        if not channel_name:
            channel_name = "unknown"
        row_count = int(len(channel_df))
        core_missing_rates: Dict[str, float] = {}
        for field in CORE_REQUIRED_FIELDS:
            if field not in channel_df.columns:
                core_missing_rates[field] = 1.0
                continue
            missing = channel_df[field].isna() | channel_df[field].astype(str).str.strip().eq("")
            core_missing_rates[field] = float(missing.mean()) if row_count else 0.0
        field_profile: Dict[str, Dict[str, Any]] = {}
        for col in [column for column in channel_df.columns if not column.startswith("__")]:
            non_null = channel_df[col].dropna()
            field_profile[col] = {
                "null_ratio": float(1 - (len(non_null) / row_count)) if row_count else 0.0,
                "distinct_count": _safe_distinct_count_nonnull(non_null),
            }
        profiles[channel_name] = {
            "row_count": row_count,
            "core_missing_rates": core_missing_rates,
            "fields": field_profile,
        }
    return profiles


def write_channel_profiles(output_dir: Path, profiles: Dict[str, Dict[str, Any]]) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d")
    for channel, profile in profiles.items():
        channel_key = channel.replace("/", "__")
        target_dir = output_dir / "channel_profiles" / channel_key
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{ts}.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def enforce_quality_gate(
    profiles: Dict[str, Dict[str, Any]],
    *,
    mode: str,
    threshold: float,
) -> None:
    violations: List[str] = []
    for channel, profile in profiles.items():
        missing_rates = profile.get("core_missing_rates", {})
        bad_fields = [name for name, value in missing_rates.items() if float(value) > threshold]
        if bad_fields:
            violations.append(f"{channel}: {', '.join(sorted(bad_fields))}")
    if not violations:
        return
    joined = " | ".join(violations)
    if mode == "fail":
        raise ValueError(f"Core 필수 필드 누락률 임계치 초과: {joined}")
    logger.warning("Core 필수 필드 누락률 경고(threshold=%.2f): %s", threshold, joined)


def prepare_fact_snapshot_output(featured_df: pd.DataFrame) -> pd.DataFrame:
    if featured_df.empty:
        return featured_df.copy()
    fact = featured_df.copy()
    bulky_columns = ["ocr_entries_json", "ocr_text_joined"]
    deprecated_columns = [column for column in DEPRECATED_NORMALIZED_FIELDS if column in fact.columns]
    existing = [c for c in bulky_columns if c in fact.columns] + deprecated_columns
    if existing:
        fact = fact.drop(columns=existing)
    return fact


def _entity_column_name(df: pd.DataFrame) -> str:
    return "entity_id" if "entity_id" in df.columns else "product_id"


def save_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    config: PipelineConfig,
    raw_products_df: Optional[pd.DataFrame] = None,
    dim_products_df: Optional[pd.DataFrame] = None,
    coverage_df: Optional[pd.DataFrame] = None,
    raw_schema: Optional[Dict[str, Any]] = None,
    normalized_schema: Optional[Dict[str, Any]] = None,
    schema_diff: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "fact_snapshots.parquet"
    csv_path = output_dir / "fact_snapshots.csv"
    raw_products_path = output_dir / RAW_SNAPSHOT_PRODUCTS_FILENAME
    latest_path = output_dir / "product_latest.parquet"
    dim_products_path = output_dir / "dim_products.parquet"
    coverage_path = output_dir / "product_snapshot_coverage.parquet"
    kpi_path = output_dir / "kpi_summary.json"
    schema_raw_path = output_dir / SCHEMA_RAW_FILENAME
    schema_normalized_path = output_dir / SCHEMA_NORMALIZED_FILENAME
    schema_diff_path = output_dir / SCHEMA_DIFF_FILENAME

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if raw_products_df is not None and not raw_products_df.empty:
        raw_products_df.to_parquet(raw_products_path, index=False)
    if dim_products_df is not None and not dim_products_df.empty:
        dim_products_df.to_parquet(dim_products_path, index=False)
    if coverage_df is not None and not coverage_df.empty:
        coverage_df.to_parquet(coverage_path, index=False)
    if raw_schema is not None:
        schema_raw_path.write_text(json.dumps(raw_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    if normalized_schema is not None:
        schema_normalized_path.write_text(json.dumps(normalized_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    if schema_diff is not None:
        schema_diff_path.write_text(json.dumps(schema_diff, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_df = (
        df.sort_values("crawl_datetime")
        .groupby(_entity_column_name(df), as_index=False)
        .tail(1)
        .sort_values(["rank", "momentum_score"], ascending=[True, False])
        if not df.empty
        else df.copy()
    )
    latest_df.to_parquet(latest_path, index=False)

    kpi = {
        "snapshot_count": int(df["snapshot_id"].nunique()) if not df.empty else 0,
        "record_count": int(len(df)),
        "unique_product_count": int(df[_entity_column_name(df)].nunique()) if not df.empty else 0,
        "brand_count": int(df["brand"].nunique()) if not df.empty else 0,
        "ocr_product_count": int(dim_products_df["ocr_has_data"].fillna(False).sum()) if dim_products_df is not None and not dim_products_df.empty and "ocr_has_data" in dim_products_df.columns else 0,
        "ocr_slot_total": int(dim_products_df["ocr_slot_count"].fillna(0).sum()) if dim_products_df is not None and not dim_products_df.empty and "ocr_slot_count" in dim_products_df.columns else 0,
        "date_range": {
            "min": str(df["snapshot_date"].min()) if not df.empty else None,
            "max": str(df["snapshot_date"].max()) if not df.empty else None,
        },
        "config": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "latest_snapshot_limit": config.latest_snapshot_limit,
            "enable_multimodal": config.enable_multimodal,
            "enable_embeddings": config.enable_embeddings,
            "enable_qdrant_upsert": config.enable_qdrant_upsert,
            "quota_product_packshot": config.quota_product_packshot,
            "quota_detail_closeup": config.quota_detail_closeup,
            "quota_model_wearing": config.quota_model_wearing,
            "main_image_strategy": config.main_image_strategy,
        },
    }
    if extra_metadata:
        kpi["extra"] = extra_metadata
    kpi_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_multimodal(
    featured_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    start_t = time.perf_counter()
    logger.info("[1/4] 이미지 인벤토리 시작")
    manifest_df = build_image_manifest(
        featured_df,
        config.data_dir,
        main_image_strategy=config.main_image_strategy,
    )
    logger.info("[1/4] 이미지 인벤토리 완료: 이미지=%d, elapsed=%.1fs", len(manifest_df), time.perf_counter() - start_t)

    start_t = time.perf_counter()
    logger.info("[2/4] 임베딩 세그먼트 생성 시작")
    segments_df = build_image_segments(manifest_df, SegmentConfig())
    logger.info("[2/4] 임베딩 세그먼트 생성 완료: 세그먼트=%d, elapsed=%.1fs", len(segments_df), time.perf_counter() - start_t)

    start_t = time.perf_counter()
    logger.info("[3/4] 임베딩 시작: enabled=%s", config.enable_embeddings)
    if not segments_df.empty:
        type_dist = (
            segments_df["segment_type_rule"].value_counts(dropna=False).head(10).to_dict()
            if "segment_type_rule" in segments_df.columns
            else {}
        )
        logger.info(
            "임베딩 대상 점검: total_segments=%d, unique_products=%d, unique_images=%d, type_dist=%s",
            len(segments_df),
            int(segments_df["product_id"].nunique()) if "product_id" in segments_df.columns else 0,
            int(segments_df["image_id"].nunique()) if "image_id" in segments_df.columns else 0,
            type_dist,
        )
    if config.enable_embeddings:
        embeddings_raw_df = generate_segment_embeddings(
            segments_df,
            EmbeddingConfig(
                qdrant_url=config.qdrant_url,
                qdrant_collection=config.qdrant_collection,
                quota_product_packshot=config.quota_product_packshot,
                quota_detail_closeup=config.quota_detail_closeup,
                quota_model_wearing=config.quota_model_wearing,
            ),
        )
        embeddings_df = apply_embedding_quota(
            segments_df,
            embeddings_raw_df,
            EmbeddingConfig(
                qdrant_url=config.qdrant_url,
                qdrant_collection=config.qdrant_collection,
                quota_product_packshot=config.quota_product_packshot,
                quota_detail_closeup=config.quota_detail_closeup,
                quota_model_wearing=config.quota_model_wearing,
            ),
        )
    else:
        embeddings_df = pd.DataFrame(
            [
                {
                    "segment_id": sid,
                    "embedding_status": "disabled",
                    "embedding_dim": 0,
                    "embedding": None,
                    "vision_label": "unknown",
                    "vision_label_score": 0.0,
                    "selected_for_embedding": False,
                }
                for sid in segments_df.get("segment_id", [])
            ]
        )
    selected_cnt = int(embeddings_df["selected_for_embedding"].sum()) if "selected_for_embedding" in embeddings_df.columns else 0
    logger.info(
        "[3/4] 임베딩 완료: rows=%d, selected=%d, elapsed=%.1fs",
        len(embeddings_df),
        selected_cnt,
        time.perf_counter() - start_t,
    )

    start_t = time.perf_counter()
    logger.info("[4/4] Qdrant 적재 시작: enabled=%s", config.enable_qdrant_upsert and config.enable_embeddings)
    qdrant_result: Dict[str, Any] = {"status": "skipped"}
    if config.enable_qdrant_upsert and config.enable_embeddings and not embeddings_df.empty:
        try:
            qdrant_result = upsert_embeddings_to_qdrant(
                segments_df,
                embeddings_df,
                EmbeddingConfig(
                    qdrant_url=config.qdrant_url,
                    qdrant_collection=config.qdrant_collection,
                    quota_product_packshot=config.quota_product_packshot,
                    quota_detail_closeup=config.quota_detail_closeup,
                    quota_model_wearing=config.quota_model_wearing,
                ),
            )
        except Exception as exc:
            logger.warning("Qdrant 적재 실패(파이프라인은 계속 진행): %s", exc)
            qdrant_result = {"status": "error", "reason": str(exc)}
    logger.info("[4/4] Qdrant 적재 완료: result=%s, elapsed=%.1fs", qdrant_result, time.perf_counter() - start_t)

    return manifest_df, segments_df, embeddings_df, {
        "qdrant": qdrant_result,
        "embedding_rows": int(len(embeddings_df)),
        "embedding_selected_rows": selected_cnt,
    }


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    pipeline_start = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("파이프라인 시작: data_dir=%s output_dir=%s", config.data_dir, config.output_dir)
    logger.info(
        "옵션: start_date=%s end_date=%s latest_limit=%s include_sources=%s exclude_sources=%s multimodal=%s embeddings=%s qdrant_upsert=%s main_image_strategy=%s",
        config.start_date,
        config.end_date,
        config.latest_snapshot_limit,
        config.include_sources,
        config.exclude_sources,
        config.enable_multimodal,
        config.enable_embeddings,
        config.enable_qdrant_upsert,
        config.main_image_strategy,
    )
    summaries = discover_summaries(config.data_dir, config.start_date, config.end_date)
    if config.include_sources:
        include_set = {item.strip() for item in config.include_sources if item.strip()}
        summaries = [summary for summary in summaries if summary.source_dataset in include_set]
    if config.exclude_sources:
        exclude_set = {item.strip() for item in config.exclude_sources if item.strip()}
        summaries = [summary for summary in summaries if summary.source_dataset not in exclude_set]
    if config.latest_snapshot_limit is not None:
        summaries = summaries[-config.latest_snapshot_limit :]
    logger.info(
        "대상 스냅샷 수: %d, source_datasets=%s",
        len(summaries),
        sorted({summary.source_dataset for summary in summaries}),
    )
    if config.adapter_dual_run:
        mismatches = [
            {
                "summary_path": str(summary.summary_path),
                "source_dataset": summary.source_dataset,
                "adapter_schema_version": summary.schema_version_guess,
                "legacy_schema_version": summary.schema_version_legacy,
            }
            for summary in summaries
            if summary.schema_version_guess != summary.schema_version_legacy
        ]
        (config.output_dir / "adapter_dual_run_diff.json").write_text(
            json.dumps(
                {
                    "checked_snapshot_count": len(summaries),
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    records: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    total = len(summaries)
    for idx, summary in enumerate(summaries, start=1):
        snapshot_records, snapshot_raw_rows = build_snapshot_records(summary)
        records.extend(snapshot_records)
        raw_rows.extend(snapshot_raw_rows)
        if idx % 10 == 0 or idx == total:
            logger.info("스냅샷 로딩 진행: %d/%d (records=%d)", idx, total, len(records))
    base_df = pd.DataFrame(records)
    dedup_dropped_rows = 0
    if not base_df.empty:
        before = len(base_df)
        base_df = base_df.drop_duplicates(subset=["snapshot_id", "product_id"], keep="first").reset_index(drop=True)
        dedup_dropped_rows = before - len(base_df)
        if dedup_dropped_rows:
            logger.warning("snapshot_id+product_id 중복 제거: dropped=%d", dedup_dropped_rows)
    n_records = len(base_df)
    n_unique_products = int(base_df["product_id"].nunique()) if not base_df.empty and "product_id" in base_df.columns else 0
    logger.info(
        "기초 레코드 생성 완료: rows=%d (스냅샷×제품 레코드), unique_products=%d",
        n_records,
        n_unique_products,
    )
    featured = add_features(base_df)
    logger.info("피처 생성 완료: rows=%d", len(featured))

    extra: Dict[str, Any] = {}
    manifest_df = pd.DataFrame()
    segments_df = pd.DataFrame()
    if config.enable_multimodal and not featured.empty:
        manifest_df, segments_df, embeddings_df, extra = _prepare_multimodal(featured, config)
        output_dir = config.output_dir
        manifest_df.to_parquet(output_dir / "image_manifest.parquet", index=False)
        segments_df.to_parquet(output_dir / "image_segments.parquet", index=False)
        embeddings_df.to_parquet(output_dir / "image_embeddings.parquet", index=False)
        embedding_projection_df = build_embedding_projection(featured, segments_df, embeddings_df)
        if not embedding_projection_df.empty:
            embedding_projection_df.to_parquet(output_dir / "analysis_embedding_projection.parquet", index=False)
        featured = merge_multimodal_metrics(featured, segments_df)
        logger.info("멀티모달 메트릭 병합 완료: rows=%d", len(featured))

    text_features_df = build_text_features(featured)
    if not text_features_df.empty:
        text_features_df.to_parquet(config.output_dir / "text_features.parquet", index=False)
        logger.info("텍스트 피처 저장: rows=%d", len(text_features_df))
    text_integrated = build_text_integrated_artifacts(featured, reviews_root=config.data_dir / "reviews")
    text_artifact_sizes: Dict[str, int] = {}
    for artifact_name, artifact_df in text_integrated.items():
        text_artifact_sizes[artifact_name] = int(len(artifact_df))
        if artifact_df.empty:
            continue
        artifact_path = config.output_dir / f"{artifact_name}.parquet"
        artifact_df.to_parquet(artifact_path, index=False)
        logger.info("텍스트 통합 산출물 저장: %s rows=%d", artifact_path.name, len(artifact_df))
    if text_artifact_sizes:
        extra["text_integrated"] = text_artifact_sizes

    if config.enable_brand_style_embedding:
        claim_df = text_integrated.get("text_claim_facts", pd.DataFrame())
        review_df = text_integrated.get("text_review_facts", pd.DataFrame())
        try:
            be_cfg = BrandStyleEmbeddingConfig()
            be_out = build_brand_style_embedding_artifacts(
                claim_df if isinstance(claim_df, pd.DataFrame) else pd.DataFrame(),
                review_df if isinstance(review_df, pd.DataFrame) else pd.DataFrame(),
                config.output_dir,
                be_cfg,
            )
            extra["brand_style_embedding"] = {
                "status": be_out.get("meta", {}).get("status"),
                "agg_rows": int(len(be_out.get("agg", pd.DataFrame()))),
                "evidence_rows": int(len(be_out.get("evidence", pd.DataFrame()))),
            }
        except Exception as exc:
            logger.warning("brand_style_embedding 실패(키워드 경로만 사용 가능): %s", exc)
            extra["brand_style_embedding"] = {"status": "error", "reason": str(exc)}

    fusion = run_analysis_fusion(featured, text_features_df if not text_features_df.empty else None)
    for name, fusion_df in fusion.items():
        if not fusion_df.empty:
            path = config.output_dir / f"analysis_{name}.parquet"
            fusion_df.to_parquet(path, index=False)
            logger.info("분석 테이블 저장: %s rows=%d", path.name, len(fusion_df))

    rank_trajectories_df = build_rank_trajectories(featured)
    if not rank_trajectories_df.empty:
        path = config.output_dir / "analysis_rank_trajectories.parquet"
        rank_trajectories_df.to_parquet(path, index=False)
        logger.info("분석 테이블 저장: %s rows=%d", path.name, len(rank_trajectories_df))

    rank_race_df = build_rank_race(featured)
    if not rank_race_df.empty:
        path = config.output_dir / "analysis_rank_race.parquet"
        rank_race_df.to_parquet(path, index=False)
        logger.info("분석 테이블 저장: %s rows=%d", path.name, len(rank_race_df))

    dim_products_df = build_product_dimension(featured, manifest_df if not manifest_df.empty else None)
    coverage_df = build_snapshot_asset_coverage(featured)
    fact_output_df = prepare_fact_snapshot_output(featured)
    raw_products_df = build_raw_snapshot_products_table(raw_rows)
    raw_schema = build_raw_schema_artifact(raw_rows)
    normalized_schema = build_normalized_schema_artifact(fact_output_df)
    schema_diff = build_schema_diff_artifact(raw_schema, normalized_schema, fact_output_df)
    channel_profiles = build_channel_profiles(fact_output_df)
    write_channel_profiles(config.output_dir, channel_profiles)
    enforce_quality_gate(
        channel_profiles,
        mode=(config.quality_mode or "warn").strip().lower(),
        threshold=float(config.quality_missing_threshold),
    )
    extra = build_extra_metadata(
        base_df,
        fact_output_df,
        dim_products_df,
        coverage_df,
        dedup_dropped_rows,
        raw_products_df,
        raw_schema,
        normalized_schema,
        schema_diff,
        extra,
    )
    extra["channel_profiles"] = {"channel_count": len(channel_profiles), "quality_mode": config.quality_mode}

    save_outputs(
        fact_output_df,
        config.output_dir,
        config,
        raw_products_df=raw_products_df,
        dim_products_df=dim_products_df,
        coverage_df=coverage_df,
        raw_schema=raw_schema,
        normalized_schema=normalized_schema,
        schema_diff=schema_diff,
        extra_metadata=extra,
    )
    logger.info("산출물 저장 완료: %s", config.output_dir)
    duckdb_result = build_duckdb_index(config.output_dir)
    kpi_path = config.output_dir / "kpi_summary.json"
    if kpi_path.exists():
        kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
        kpi.setdefault("extra", {})
        kpi["extra"]["duckdb"] = duckdb_result
        kpi_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("DuckDB 인덱스 상태: %s", duckdb_result)
    logger.info("파이프라인 완료: elapsed=%.1fs", time.perf_counter() - pipeline_start)
    return featured


def run_metrics_refresh_pipeline(config: PipelineConfig) -> pd.DataFrame:
    pipeline_start = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fact_path = config.output_dir / "fact_snapshots.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(
            f"재분석 입력 파일이 없습니다: {fact_path}. 먼저 전체 파이프라인으로 fact_snapshots를 생성하세요."
        )
    logger.info("재분석(빠른 모드) 시작: fact=%s", fact_path)
    base_df = pd.read_parquet(fact_path)
    if base_df.empty:
        logger.warning("입력 fact_snapshots가 비어 있어 재분석을 건너뜁니다.")
        return base_df

    featured = add_features(base_df)
    fact_output_df = prepare_fact_snapshot_output(featured)
    fact_output_df.to_parquet(fact_path, index=False)
    fact_output_df.to_csv(config.output_dir / "fact_snapshots.csv", index=False, encoding="utf-8-sig")

    entity_col = _entity_column_name(fact_output_df)
    latest_df = (
        fact_output_df.sort_values("crawl_datetime")
        .groupby(entity_col, as_index=False)
        .tail(1)
        .sort_values(["rank", "momentum_score"], ascending=[True, False])
    )
    latest_df.to_parquet(config.output_dir / "product_latest.parquet", index=False)

    text_features_df = build_text_features(featured)
    if not text_features_df.empty:
        text_features_df.to_parquet(config.output_dir / "text_features.parquet", index=False)
        logger.info("재분석: text_features 갱신 rows=%d", len(text_features_df))

    fusion = run_analysis_fusion(featured, text_features_df if not text_features_df.empty else None)
    for name, fusion_df in fusion.items():
        if fusion_df.empty:
            continue
        path = config.output_dir / f"analysis_{name}.parquet"
        fusion_df.to_parquet(path, index=False)
        logger.info("재분석: %s 갱신 rows=%d", path.name, len(fusion_df))

    rank_trajectories_df = build_rank_trajectories(featured)
    if not rank_trajectories_df.empty:
        path = config.output_dir / "analysis_rank_trajectories.parquet"
        rank_trajectories_df.to_parquet(path, index=False)
        logger.info("재분석: %s 갱신 rows=%d", path.name, len(rank_trajectories_df))
    rank_race_df = build_rank_race(featured)
    if not rank_race_df.empty:
        path = config.output_dir / "analysis_rank_race.parquet"
        rank_race_df.to_parquet(path, index=False)
        logger.info("재분석: %s 갱신 rows=%d", path.name, len(rank_race_df))

    kpi_path = config.output_dir / "kpi_summary.json"
    if kpi_path.exists():
        kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
    else:
        kpi = {}
    kpi.setdefault("extra", {})
    kpi["extra"]["metrics_refresh"] = {
        "ran_at": pd.Timestamp.utcnow().isoformat(),
        "record_count": int(len(fact_output_df)),
        "elapsed_sec": round(float(time.perf_counter() - pipeline_start), 2),
    }
    kpi_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")

    duckdb_result = build_duckdb_index(config.output_dir)
    if kpi_path.exists():
        kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
        kpi.setdefault("extra", {})
        kpi["extra"]["duckdb"] = duckdb_result
        kpi_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("재분석(빠른 모드) 완료: elapsed=%.1fs", time.perf_counter() - pipeline_start)
    return featured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="무신사 분석 데이터셋 빌드")
    parser.add_argument("--data-dir", type=str, default="data", help="원천 데이터 디렉토리")
    parser.add_argument("--output-dir", type=str, default="output/analytics", help="결과 출력 디렉토리")
    parser.add_argument("--start-date", type=str, default=None, help="분석 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="분석 종료일 (YYYY-MM-DD)")
    parser.add_argument(
        "--include-source",
        action="append",
        default=None,
        help="특정 source_dataset만 포함 (복수 사용 가능)",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=None,
        help="특정 source_dataset 제외 (복수 사용 가능)",
    )
    parser.add_argument(
        "--latest-snapshot-limit",
        type=int,
        default=None,
        help="최근 N개 스냅샷만 처리 (개발/테스트용)",
    )
    parser.add_argument("--disable-multimodal", action="store_true", help="이미지/임베딩 파이프라인 비활성화")
    parser.add_argument("--disable-embeddings", action="store_true", help="임베딩 단계 비활성화")
    parser.add_argument("--disable-qdrant-upsert", action="store_true", help="Qdrant 적재 비활성화")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--qdrant-collection", type=str, default="product_image_segments", help="Qdrant collection")
    parser.add_argument("--quota-product-packshot", type=int, default=1, help="제품 단독컷 임베딩 쿼터")
    parser.add_argument("--quota-detail-closeup", type=int, default=1, help="디테일 클로즈업 임베딩 쿼터")
    parser.add_argument("--quota-model-wearing", type=int, default=1, help="모델 착용샷 임베딩 쿼터")
    parser.add_argument(
        "--main-image-strategy",
        type=str,
        default="first",
        choices=["first", "largest_file"],
        help="메인 이미지 선택: first=파일명 첫 장, largest_file=파일 크기 최대 1장",
    )
    parser.add_argument(
        "--enable-brand-style-embedding",
        action="store_true",
        help="텍스트 임베딩 기반 브랜드 스타일 집계(brand_style_embedding_agg.parquet) 생성",
    )
    parser.add_argument(
        "--metrics-refresh-only",
        action="store_true",
        help="기존 output의 fact_snapshots를 기반으로 지표/분석 테이블만 빠르게 재계산 (임베딩/이미지/OCR 재처리 없음)",
    )
    parser.add_argument(
        "--quality-mode",
        type=str,
        default="warn",
        choices=["warn", "fail"],
        help="Core 필수 필드 누락률 임계치 초과 시 동작 모드",
    )
    parser.add_argument(
        "--quality-missing-threshold",
        type=float,
        default=0.15,
        help="Core 필수 필드 누락률 임계치(0~1)",
    )
    parser.add_argument(
        "--disable-adapter-dual-run",
        action="store_true",
        help="어댑터/레거시 schema 추론 비교 리포트를 비활성화",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        latest_snapshot_limit=args.latest_snapshot_limit,
        start_date=args.start_date,
        end_date=args.end_date,
        include_sources=args.include_source,
        exclude_sources=args.exclude_source,
        enable_multimodal=not args.disable_multimodal,
        enable_embeddings=not args.disable_embeddings,
        enable_qdrant_upsert=not args.disable_qdrant_upsert,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        quota_product_packshot=args.quota_product_packshot,
        quota_detail_closeup=args.quota_detail_closeup,
        quota_model_wearing=args.quota_model_wearing,
        main_image_strategy=args.main_image_strategy,
        enable_brand_style_embedding=bool(getattr(args, "enable_brand_style_embedding", False)),
        quality_mode=str(getattr(args, "quality_mode", "warn")),
        quality_missing_threshold=float(getattr(args, "quality_missing_threshold", 0.15)),
        adapter_dual_run=not bool(getattr(args, "disable_adapter_dual_run", False)),
    )
    if bool(getattr(args, "metrics_refresh_only", False)):
        df = run_metrics_refresh_pipeline(config)
    else:
        df = run_pipeline(config)
    print(f"완료: {len(df)}건 처리, 출력 경로={config.output_dir}")


if __name__ == "__main__":
    main()

