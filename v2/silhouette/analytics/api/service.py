from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

from analytics.api.filters import DashboardFilters, apply_fact_filters, filter_related_table
from analytics.api.repository import AnalyticsRepository
from analytics.pipeline.analysis_fusion import _prepare_category_frame, build_product_profile, run_analysis_fusion
from analytics.pipeline.image_processing import EXPLICIT_MAIN_IMAGE_FILENAMES
from analytics.pipeline.brand_style_embedding import AXES_PATH, load_style_axes
from analytics.pipeline.text_integrated_analysis import (
    BRAND_IMAGE_STYLE_AXIS_ORDER,
    brand_image_style_label_ko,
    detect_brand_image_styles,
)

PRICE_BAND_BINS = [0, 30000, 70000, 120000, 200000, 500000, float("inf")]
PRICE_BAND_LABELS = ["~3만", "3-7만", "7-12만", "12-20만", "20-50만", "50만+"]
GEOCODE_CACHE_FILENAME = "business_location_geocode_cache.json"
SCHEMA_RAW_FILENAME = "schema_raw.json"
SCHEMA_NORMALIZED_FILENAME = "schema_normalized.json"
SCHEMA_DIFF_FILENAME = "schema_diff.json"
SCHEMA_EXPLORER_RULES_FILENAME = "schema_explorer_rules.yaml"
SCHEMA_FIELD_REPLACEMENTS = {
    "product.taxonomy_gap_candidate": "product.vlm_raw_label",
    "taxonomy_gap_candidate": "vlm_raw_label",
}
DEPRECATED_SCHEMA_FIELDS = set(SCHEMA_FIELD_REPLACEMENTS.keys())
DEFAULT_EXPLORATION_EXCLUDED_RAW_PREFIXES = ("product_info.",)
DEFAULT_EXPLORATION_EXCLUDED_NORMALIZED_FIELDS = {
    "material",
    "color",
    "manufacturer",
    "origin_country",
    "shipping_fee",
    "product_info_exists",
    "business_address",
    "business_province",
    "business_district",
    "business_dong",
    "color_normalized",
    "material_normalized",
}
ATTRIBUTE_JUNK_PATTERNS = (
    re.compile(r"상세\s*페이지\s*참조", re.IGNORECASE),
    re.compile(r"상품\s*tag\s*참고", re.IGNORECASE),
    re.compile(r"제품\s*관련\s*문의", re.IGNORECASE),
    re.compile(r"문의는", re.IGNORECASE),
    re.compile(r"고객센터", re.IGNORECASE),
    re.compile(r"나이키코리아", re.IGNORECASE),
    re.compile(r"참고", re.IGNORECASE),
    re.compile(r"문의", re.IGNORECASE),
)

CORE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "productId": ("product_id",),
    "snapshotId": ("snapshot_id",),
    "sourceDataset": ("source_dataset",),
    "schemaVersion": ("schema_version",),
    "categoryLabel": ("category_label",),
    "mainImage": ("main_image_path", "mainImagePath"),
}


def _normalize_source_dataset_scalar(value: Any) -> str:
    """source_dataset 비스칼라(리스트 등)가 섞여도 집계·unique가 깨지지 않도록 문자열 키로 만든다."""
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


def _sorted_unique_source_datasets(series: pd.Series) -> list[str]:
    if series.empty:
        return []
    keys = {_normalize_source_dataset_scalar(value) for value in series.dropna().tolist()}
    return sorted(k for k in keys if k)


def apply_core_aliases(payload: Any) -> Any:
    if isinstance(payload, list):
        return [apply_core_aliases(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    normalized = {key: apply_core_aliases(value) for key, value in payload.items()}
    for stable_key, legacy_keys in CORE_KEY_ALIASES.items():
        if stable_key in normalized and normalized[stable_key] not in (None, ""):
            continue
        for legacy_key in legacy_keys:
            if legacy_key in normalized and normalized[legacy_key] not in (None, ""):
                normalized[stable_key] = normalized[legacy_key]
                break
    return normalized


def _load_schema_explorer_rules() -> tuple[tuple[str, ...], set[str]]:
    rules_path = Path(__file__).with_name(SCHEMA_EXPLORER_RULES_FILENAME)
    if not rules_path.exists():
        return DEFAULT_EXPLORATION_EXCLUDED_RAW_PREFIXES, DEFAULT_EXPLORATION_EXCLUDED_NORMALIZED_FIELDS
    try:
        payload = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_EXPLORATION_EXCLUDED_RAW_PREFIXES, DEFAULT_EXPLORATION_EXCLUDED_NORMALIZED_FIELDS
    if not isinstance(payload, dict):
        return DEFAULT_EXPLORATION_EXCLUDED_RAW_PREFIXES, DEFAULT_EXPLORATION_EXCLUDED_NORMALIZED_FIELDS

    raw_prefixes = payload.get("exclude_raw_prefixes")
    normalized_fields = payload.get("exclude_normalized_fields")

    parsed_raw_prefixes = tuple(
        value.strip()
        for value in (raw_prefixes or [])
        if isinstance(value, str) and value.strip()
    )
    parsed_normalized_fields = {
        value.strip()
        for value in (normalized_fields or [])
        if isinstance(value, str) and value.strip()
    }

    return (
        parsed_raw_prefixes or DEFAULT_EXPLORATION_EXCLUDED_RAW_PREFIXES,
        parsed_normalized_fields or DEFAULT_EXPLORATION_EXCLUDED_NORMALIZED_FIELDS,
    )


EXPLORATION_EXCLUDED_RAW_PREFIXES, EXPLORATION_EXCLUDED_NORMALIZED_FIELDS = _load_schema_explorer_rules()


def _is_junk_attribute_label(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    normalized = text.lower()
    if normalized in {"-", "기타", "미입력", "미정규화", "미분류", "없음", "none", "null", "nan"}:
        return True
    if any(pattern.search(text) for pattern in ATTRIBUTE_JUNK_PATTERNS):
        return True
    # 색상/소재로 보기 어려운 안내문성 문장은 표 요약에서 제외합니다.
    if len(text) >= 20 and (" " in text or any(char.isdigit() for char in text)):
        return True
    if any(token in normalized for token in ("http", "www.", ".com", ".kr", "tel", "fax")):
        return True
    return False


def _kmeans_assignments(points: np.ndarray, k: int, max_iter: int = 40) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    n_points = points.shape[0]
    k = max(1, min(int(k), n_points))
    rng = np.random.default_rng(42)
    seed_indices = rng.choice(n_points, size=k, replace=False)
    centers = points[seed_indices].copy()
    labels = np.zeros(n_points, dtype=np.int32)
    for _ in range(max_iter):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for idx in range(k):
            members = points[labels == idx]
            if members.size == 0:
                centers[idx] = points[rng.integers(0, n_points)]
            else:
                centers[idx] = members.mean(axis=0)
    return labels


def _normalize_category_series(df: pd.DataFrame, candidate_columns: tuple[str, ...]) -> pd.Series:
    existing = [col for col in candidate_columns if col in df.columns]
    if not existing:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    merged = df[existing].copy().bfill(axis=1).iloc[:, 0]
    merged = merged.fillna("").astype(str).str.strip()
    merged = merged.replace({"": np.nan})
    return merged


def _is_unclassified_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(
        {
            "",
            "unknown",
            "미분류",
            "없음",
            "none",
            "nan",
            "null",
            "-",
            "기타",
            "uncategorized",
        }
    )


def _build_adaptive_embedding_labels(df: pd.DataFrame) -> tuple[pd.Series, str]:
    l1 = _normalize_category_series(df, ("category_label_l1", "category_l1"))
    l2 = _normalize_category_series(df, ("category_label_l2", "category_l2"))
    l3 = _normalize_category_series(df, ("category_label_l3", "category_l3"))
    valid_l1 = ~_is_unclassified_series(l1) & l1.notna()
    valid_l2 = ~_is_unclassified_series(l2) & l2.notna()
    valid_l3 = ~_is_unclassified_series(l3) & l3.notna()
    valid_count = max(int(valid_l1.sum()), 1)
    l2_coverage = float(valid_l2.sum() / valid_count)
    min_support = max(10, int(valid_count * 0.015))
    l2_counts = l2[valid_l2].value_counts()
    strong_l2 = set(l2_counts[l2_counts >= min_support].index.astype(str))
    use_l2 = l2_coverage >= 0.35 and len(strong_l2) >= 4
    labels = l1.fillna("미분류").astype(str)
    strategy_parts = ["L1"]

    if use_l2:
        strong_mask = valid_l2 & l2.astype(str).isin(strong_l2) & valid_l1
        labels.loc[strong_mask] = l1[strong_mask].astype(str) + " > " + l2[strong_mask].astype(str)
        strategy_parts = ["L1>L2(표본 충분 L2만 상세화)"]

    apparel_mask = valid_l1 & valid_l2 & l1.astype(str).eq("패션") & l2.astype(str).eq("의류")
    apparel_count = int(apparel_mask.sum())
    if apparel_count > 0:
        apparel_valid_l3 = apparel_mask & valid_l3
        apparel_l3_coverage = float(apparel_valid_l3.sum() / apparel_count)
        apparel_min_support = max(12, int(apparel_count * 0.025))
        apparel_l3_counts = l3[apparel_valid_l3].value_counts()
        strong_apparel_l3 = set(apparel_l3_counts[apparel_l3_counts >= apparel_min_support].index.astype(str))
        use_apparel_l3 = apparel_l3_coverage >= 0.45 and len(strong_apparel_l3) >= 4
        if use_apparel_l3:
            apparel_mask_strong = apparel_valid_l3 & l3.astype(str).isin(strong_apparel_l3)
            labels.loc[apparel_mask_strong] = (
                l1[apparel_mask_strong].astype(str)
                + " > "
                + l2[apparel_mask_strong].astype(str)
                + " > "
                + l3[apparel_mask_strong].astype(str)
            )
            strategy_parts.append("의류 strong L3 상세화")

    return labels.astype(str), " + ".join(strategy_parts)


def _cluster_count_for_points(n_points: int) -> int:
    if n_points <= 1:
        return 1
    return max(2, min(14, int(np.sqrt(max(n_points, 4) / 2))))


def _stable_cluster_ids(
    centers: dict[int, np.ndarray],
    prev_centers: dict[str, np.ndarray],
    next_cluster_index: int,
) -> tuple[dict[int, str], dict[str, np.ndarray], int]:
    if not centers:
        return {}, {}, next_cluster_index
    if not prev_centers:
        mapping: dict[int, str] = {}
        updated: dict[str, np.ndarray] = {}
        for local_id, center in sorted(centers.items()):
            stable_id = f"cluster_{next_cluster_index}"
            next_cluster_index += 1
            mapping[local_id] = stable_id
            updated[stable_id] = center
        return mapping, updated, next_cluster_index

    prev_ids = list(prev_centers.keys())
    prev_matrix = np.vstack([prev_centers[cluster_id] for cluster_id in prev_ids])
    mapping: dict[int, str] = {}
    used_prev: set[str] = set()
    updated: dict[str, np.ndarray] = {}
    for local_id, center in sorted(centers.items()):
        distances = np.sum((prev_matrix - center[None, :]) ** 2, axis=1)
        order = np.argsort(distances)
        assigned = False
        for idx in order:
            candidate = prev_ids[int(idx)]
            if candidate in used_prev:
                continue
            mapping[local_id] = candidate
            updated[candidate] = center
            used_prev.add(candidate)
            assigned = True
            break
        if not assigned:
            stable_id = f"cluster_{next_cluster_index}"
            next_cluster_index += 1
            mapping[local_id] = stable_id
            updated[stable_id] = center
    return mapping, updated, next_cluster_index


def _frame_cluster_payload(
    frame_df: pd.DataFrame,
    prev_centers: dict[str, np.ndarray],
    next_cluster_index: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, np.ndarray], int]:
    if frame_df.empty:
        return frame_df, [], {}, next_cluster_index

    points = frame_df[["x", "y"]].to_numpy(dtype=np.float64)
    labels = _kmeans_assignments(points, _cluster_count_for_points(len(frame_df)))
    frame = frame_df.copy()
    frame["_local_cluster_id"] = labels

    local_centers: dict[int, np.ndarray] = {}
    for local_id, subset in frame.groupby("_local_cluster_id"):
        center = subset[["x", "y"]].to_numpy(dtype=np.float64).mean(axis=0)
        local_centers[int(local_id)] = center
    mapping, updated_centers, next_cluster_index = _stable_cluster_ids(local_centers, prev_centers, next_cluster_index)
    frame["cluster_id"] = frame["_local_cluster_id"].map(lambda value: mapping.get(int(value), "cluster_unknown"))

    cluster_rows: list[dict[str, Any]] = []
    total = max(len(frame), 1)
    dominant_by_cluster: dict[str, str] = {}
    dominant_share_by_cluster: dict[str, float] = {}
    for cluster_id, subset in frame.groupby("cluster_id"):
        category_counts = subset["category_label"].fillna("미분류").astype(str).value_counts()
        point_count = int(len(subset))
        dominant_category = str(category_counts.index[0]) if not category_counts.empty else "미분류"
        dominant_share = float(category_counts.iloc[0] / point_count) if point_count > 0 and not category_counts.empty else 0.0
        dominant_by_cluster[str(cluster_id)] = dominant_category
        dominant_share_by_cluster[str(cluster_id)] = dominant_share
        cluster_rows.append(
            {
                "clusterId": str(cluster_id),
                "pointCount": point_count,
                "sharePct": float(point_count / total * 100.0),
                "dominantCategory": dominant_category,
                "dominantSharePct": float(dominant_share * 100.0),
            }
        )
    cluster_rows.sort(key=lambda row: row["pointCount"], reverse=True)
    frame["dominant_category"] = frame["cluster_id"].map(lambda cid: dominant_by_cluster.get(str(cid), "미분류"))
    frame["dominant_share_pct"] = frame["cluster_id"].map(lambda cid: float(dominant_share_by_cluster.get(str(cid), 0.0) * 100.0))
    frame = frame.drop(columns=["_local_cluster_id"])
    return frame, cluster_rows, updated_centers, next_cluster_index

COLUMN_METADATA: dict[str, dict[str, str]] = {
    "snapshot_id": {
        "group": "time",
        "grain": "record",
        "meaning": "각 수집 시점을 식별하는 스냅샷 ID",
        "note": "한 상품은 여러 snapshot_id에서 반복 관측될 수 있습니다.",
    },
    "snapshot_date": {
        "group": "time",
        "grain": "record",
        "meaning": "스냅샷 수집 날짜",
        "note": "일 단위 비교 축으로 자주 사용됩니다.",
    },
    "snapshot_time": {
        "group": "time",
        "grain": "record",
        "meaning": "스냅샷 수집 시각",
        "note": "같은 날짜 내 반복 수집이 있을 수 있습니다.",
    },
    "crawl_datetime": {
        "group": "time",
        "grain": "record",
        "meaning": "정렬 가능한 실제 수집 시각",
        "note": "시계열 정렬의 기준 컬럼입니다.",
    },
    "product_id": {
        "group": "identifier",
        "grain": "product",
        "meaning": "상품 고유 ID",
        "note": "상품 단위 집계와 조인의 핵심 키입니다.",
    },
    "product_url": {
        "group": "identifier",
        "grain": "product",
        "meaning": "상품 상세 페이지 URL",
        "note": "식별용이지만 분석 축으로는 거의 사용하지 않습니다.",
    },
    "brand": {
        "group": "attribute",
        "grain": "product",
        "meaning": "브랜드명",
        "note": "브랜드 수준 집계의 핵심 속성입니다.",
    },
    "name": {
        "group": "attribute",
        "grain": "product",
        "meaning": "상품명 원문",
        "note": "name_item, color_normalized 같은 파생 텍스트 속성의 원천입니다.",
    },
    "price": {
        "group": "price",
        "grain": "record",
        "meaning": "해당 시점의 실제 판매가",
        "note": "가격 탭의 추정 정가와 혼동하지 않아야 합니다.",
    },
    "discount_pct": {
        "group": "price",
        "grain": "record",
        "meaning": "해당 시점의 할인율(%)",
        "note": "가격-순위 관계를 해석할 때 핵심 보조축입니다.",
    },
    "price_band": {
        "group": "price",
        "grain": "record",
        "meaning": "실제 판매가 기준 가격대 구간",
        "note": "가격 탭의 추정 정가 기준 밴드와 별개입니다.",
    },
    "rank": {
        "group": "rank",
        "grain": "record",
        "meaning": "해당 시점의 순위",
        "note": "값이 낮을수록 상위 순위입니다.",
    },
    "rank_velocity": {
        "group": "derived",
        "grain": "record",
        "meaning": "직전 시점 대비 순위 변화량",
        "note": "양수일수록 순위가 개선된 것으로 읽습니다.",
    },
    "rank_acceleration": {
        "group": "derived",
        "grain": "record",
        "meaning": "순위 변화량의 변화",
        "note": "모멘텀 계산의 입력값입니다.",
    },
    "rank_energy": {
        "group": "derived",
        "grain": "record",
        "meaning": "현재 순위를 51위 기준 점수로 뒤집은 뒤 상위권일수록 연속 비선형 가중을 곱한 절대 에너지",
        "note": "1위에 가까울수록 값이 커지며, 구간 경계에서 점수가 튀지 않도록 연속 함수로 계산합니다.",
    },
    "energy_velocity": {
        "group": "derived",
        "grain": "record",
        "meaning": "직전 시점 대비 순위 에너지 변화량",
        "note": "현재 모멘텀 점수의 본체이며 양수일수록 절대 성장 에너지가 커진 것입니다.",
    },
    "energy_acceleration": {
        "group": "derived",
        "grain": "record",
        "meaning": "순위 에너지 변화량의 변화",
        "note": "외부 노출·할인·바이럴 등 급격한 원인 추적의 우선 단서로 사용합니다.",
    },
    "entry_score": {
        "group": "derived",
        "grain": "record",
        "meaning": "첫 관측 또는 순위권 재진입 시점의 진입 강도",
        "note": "모멘텀 본체에 더하지 않고 이벤트 신호로 별도 해석합니다.",
    },
    "exit_score": {
        "group": "derived",
        "grain": "record",
        "meaning": "순위권 탈락 첫 시점의 이전 순위 에너지",
        "note": "순위권 밖에 머무는 동안 반복 감점하지 않기 위한 이벤트 신호입니다.",
    },
    "cumulative_rank_energy": {
        "group": "derived",
        "grain": "product",
        "meaning": "선택 기간 동안 상품이 순위권에 관측된 모든 시점의 순위 에너지 합",
        "note": "단발 고순위보다 반복적으로 순위권에 머문 상품을 더 높게 평가하기 위한 제품 단위 점수입니다.",
    },
    "sustained_rank_energy": {
        "group": "derived",
        "grain": "product",
        "meaning": "누적 순위 에너지에 순위권 등장 비율을 함께 반영한 제품 단위 지속 에너지",
        "note": "상위 사례 정렬과 가격대·브랜드 비교에 사용합니다.",
    },
    "momentum_score": {
        "group": "derived",
        "grain": "record",
        "meaning": "표준화하지 않은 순위 에너지 속도",
        "note": "z-score가 아니라 개별 상품의 절대적 움직임을 보존한 계산값입니다.",
    },
    "stability_score": {
        "group": "derived",
        "grain": "record",
        "meaning": "순위 변동성 기반 안정성 점수",
        "note": "높을수록 순위 흐름이 안정적입니다.",
    },
    "tags_joined": {
        "group": "attribute",
        "grain": "product",
        "meaning": "태그 원문 결합 문자열",
        "note": "태그 성과 분석의 원천입니다.",
    },
    "material": {
        "group": "product_info",
        "grain": "product",
        "meaning": "product_info에서 추출한 소재 원문",
        "note": "정규화 소재와 구분해서 봐야 합니다.",
    },
    "color": {
        "group": "product_info",
        "grain": "product",
        "meaning": "product_info에서 추출한 색상 원문",
        "note": "정규화 색상과 구분해서 봐야 합니다.",
    },
    "manufacturer": {
        "group": "product_info",
        "grain": "product",
        "meaning": "제조사 원문",
        "note": "브랜드와 다른 값일 수 있습니다.",
    },
    "origin_country": {
        "group": "product_info",
        "grain": "product",
        "meaning": "제조국 원문",
        "note": "국가별 분포와 가격/순위 차이 탐색에 사용됩니다.",
    },
    "shipping_fee": {
        "group": "product_info",
        "grain": "product",
        "meaning": "배송비 원문 텍스트",
        "note": "현재는 존재 여부와 패턴 수준으로만 해석하는 것이 안전합니다.",
    },
    "business_address": {
        "group": "location",
        "grain": "product",
        "meaning": "영업소재지 원문 주소",
        "note": "지도는 상세 주소가 아니라 구/동 집계 기준으로 해석합니다.",
    },
    "business_province": {
        "group": "location",
        "grain": "product",
        "meaning": "영업소재지에서 파싱한 시도",
        "note": "규칙 기반 파싱 결과입니다.",
    },
    "business_district": {
        "group": "location",
        "grain": "product",
        "meaning": "영업소재지에서 파싱한 구/시군구",
        "note": "공간 분포의 기본 분석 단위입니다.",
    },
    "business_dong": {
        "group": "location",
        "grain": "product",
        "meaning": "영업소재지에서 파싱한 읍면동",
        "note": "주소 품질이 좋을 때만 채워집니다.",
    },
    "product_info_exists": {
        "group": "quality",
        "grain": "record",
        "meaning": "product_info.csv 존재 여부",
        "note": "속성 분석 coverage를 판단하는 핵심 플래그입니다.",
    },
    "ocr_has_data": {
        "group": "quality",
        "grain": "record",
        "meaning": "OCR 텍스트 존재 여부",
        "note": "비전/텍스트 보강 가능 범위를 알려줍니다.",
    },
    "detail_image_count_actual": {
        "group": "quality",
        "grain": "record",
        "meaning": "상세 이미지 수",
        "note": "이미지 기반 분석 가능 범위를 가늠하게 합니다.",
    },
}


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return isinstance(missing, bool) and missing


def _normalize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if _is_missing_value(value):
        return None
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_value(item) for item in sorted(value, key=repr)]
    return value


def _safe_float(value: Any) -> float | None:
    if _is_missing_value(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if _is_missing_value(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _aggregate_keyword_trend_timeseries_by_day(trend_df: pd.DataFrame) -> pd.DataFrame:
    """동일 일자·키워드·유형에 대해 스냅샷별 행을 합산해 시계열이 하루 한 점이 되도록 한다."""
    if trend_df.empty:
        return trend_df
    needed = {"snapshot_date", "keyword", "keyword_type", "mention_count"}
    if not needed.issubset(trend_df.columns):
        return trend_df
    work = trend_df.copy()
    work["mention_count"] = pd.to_numeric(work["mention_count"], errors="coerce").fillna(0)
    if "product_count" in work.columns:
        work["product_count"] = pd.to_numeric(work["product_count"], errors="coerce").fillna(0)
    else:
        work["product_count"] = 0
    keys = ["snapshot_date", "keyword", "keyword_type"]
    return (
        work.groupby(keys, as_index=False)
        .agg(mention_count=("mention_count", "sum"), product_count=("product_count", "sum"))
        .sort_values(["snapshot_date", "mention_count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _optional_fs_path(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    s = str(value).strip()
    return s or None


def _estimated_original_price(price: Any, discount_pct: Any) -> float | None:
    numeric_price = _safe_float(price)
    numeric_discount = _safe_float(discount_pct)
    if numeric_price is None:
        return None
    if numeric_discount is None or numeric_discount <= 0 or numeric_discount >= 95:
        return numeric_price
    denominator = 1 - (numeric_discount / 100.0)
    if denominator <= 0:
        return numeric_price
    return numeric_price / denominator


def _price_band_from_original_price(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=PRICE_BAND_BINS,
        labels=PRICE_BAND_LABELS,
        include_lowest=True,
    )


def _label_or_missing(value: Any, missing_label: str = "미입력") -> str:
    if _is_missing_value(value):
        return missing_label
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return missing_label
    return text


def _latest_by_product(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "product_id" not in df.columns:
        return df.copy()
    scoped = df.copy()
    if "crawl_datetime" in scoped.columns:
        scoped["crawl_datetime"] = pd.to_datetime(scoped["crawl_datetime"], errors="coerce")
        scoped = scoped.sort_values(["crawl_datetime", "snapshot_id"] if "snapshot_id" in scoped.columns else ["crawl_datetime"])
    elif "snapshot_id" in scoped.columns:
        scoped = scoped.sort_values("snapshot_id")
    return scoped.groupby("product_id", as_index=False).tail(1).copy()


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    scoped = df.head(limit) if limit else df
    out: list[dict[str, Any]] = []
    for row in scoped.to_dict("records"):
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            normalized[key] = _normalize_value(value)
        out.append(normalized)
    return out


def _momentum_event_label(state: Any) -> str:
    labels = {
        "first_seen": "첫 관측",
        "chart_in_spike": "순위권 진입",
        "chart_out_drop": "순위권 탈락",
        "out_of_chart": "순위권 밖",
        "breakout": "가속 상승",
        "sustained_growth": "지속 상승",
        "cooling": "상승 둔화",
        "reversal": "하락 전환",
        "steady": "정체",
    }
    return labels.get(str(state), "정체")


def _ensure_rank_energy_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """신규 파생 컬럼이 없는 기존 산출물도 API에서 같은 의미로 읽히게 보정한다."""
    if df.empty or "rank" not in df.columns:
        return df
    required = {
        "rank_energy",
        "rank_energy_prev",
        "energy_velocity",
        "energy_acceleration",
        "entry_score",
        "exit_score",
        "is_first_seen",
        "is_continuous_observation",
        "consistency_score",
        "momentum_event_state",
        "momentum_event_label",
    }
    if required.issubset(set(df.columns)):
        return df

    out = df.copy()
    scope_cols = [col for col in ["source_dataset", "platform", "schema_version"] if col in out.columns]
    time_cols = [col for col in ["crawl_datetime", "snapshot_id"] if col in out.columns]
    sort_cols = scope_cols + [col for col in ["product_id", "entity_id"] if col in out.columns] + time_cols
    if sort_cols:
        out = out.sort_values(sort_cols)
    entity_key = "product_id" if "product_id" in out.columns else "entity_id" if "entity_id" in out.columns else None
    rank_numeric = pd.to_numeric(out["rank"], errors="coerce")
    out["rank_filled"] = pd.to_numeric(out.get("rank_filled", rank_numeric), errors="coerce").fillna(51.0)
    out["score"] = 51.0 - out["rank_filled"]
    standard_ratio = (out["score"] / 50.0).clip(lower=0.0, upper=1.0)
    out["rank_energy"] = pd.to_numeric(
        out.get("rank_energy", out["score"] * (1.0 + (2.0 * standard_ratio.pow(2)))),
        errors="coerce",
    ).fillna(0.0)
    if entity_key:
        if "snapshot_id" in out.columns:
            snapshot_order = (
                out[scope_cols + ["snapshot_id"] + (["crawl_datetime"] if "crawl_datetime" in out.columns else [])]
                .drop_duplicates(subset=scope_cols + ["snapshot_id"])
                .sort_values(scope_cols + (["crawl_datetime", "snapshot_id"] if "crawl_datetime" in out.columns else ["snapshot_id"]))
                .copy()
            )
            snapshot_order["_snapshot_ord"] = snapshot_order.groupby(scope_cols, dropna=False).cumcount() if scope_cols else range(len(snapshot_order))
            out = out.merge(snapshot_order[scope_cols + ["snapshot_id", "_snapshot_ord"]], on=scope_cols + ["snapshot_id"], how="left")
        else:
            out["_snapshot_ord"] = out.groupby(scope_cols + [entity_key], dropna=False).cumcount() if scope_cols else out.groupby(entity_key, dropna=False).cumcount()
        group_cols = scope_cols + [entity_key]
        grouped = out.groupby(group_cols, group_keys=False, dropna=False)
        out["prev_snapshot_ord"] = grouped["_snapshot_ord"].shift(1)
        out["rank_energy_prev"] = pd.to_numeric(grouped["rank_energy"].shift(1), errors="coerce").fillna(0.0)
        out["is_first_seen"] = out["prev_snapshot_ord"].isna()
        out["is_continuous_observation"] = out["prev_snapshot_ord"].notna() & (out["_snapshot_ord"] - out["prev_snapshot_ord"]).eq(1)
        out["is_reentry"] = out["prev_snapshot_ord"].notna() & ~out["is_continuous_observation"]
        out["entry_score"] = np.where(out["is_first_seen"] | out["is_reentry"], out["rank_energy"], np.nan)
        out["exit_score"] = pd.to_numeric(out.get("exit_score", np.nan), errors="coerce")
        out["energy_velocity"] = np.where(
            out["is_continuous_observation"],
            out["rank_energy"] - out["rank_energy_prev"],
            np.nan,
        )
        grouped = out.groupby(group_cols, group_keys=False, dropna=False)
        out["energy_acceleration"] = np.where(
            out["is_continuous_observation"],
            grouped["energy_velocity"].diff(1),
            np.nan,
        )
        out["consistency_score"] = pd.to_numeric(
            grouped["energy_velocity"].transform(
                lambda series: series.gt(0).rolling(window=5, min_periods=2).mean()
            ),
            errors="coerce",
        ).fillna(0.5)
    else:
        out["rank_energy_prev"] = pd.to_numeric(out.get("rank_energy_prev", 0.0), errors="coerce").fillna(0.0)
        out["is_first_seen"] = True
        out["is_reentry"] = False
        out["is_continuous_observation"] = False
        out["entry_score"] = out["rank_energy"]
        out["exit_score"] = np.nan
        out["energy_velocity"] = np.nan
        out["energy_acceleration"] = np.nan
        out["consistency_score"] = pd.to_numeric(out.get("consistency_score", 0.5), errors="coerce").fillna(0.5)

    out["momentum_score"] = out["energy_velocity"]
    if "momentum_event_state" not in out.columns:
        is_reentry = out["is_reentry"].fillna(False).astype(bool) if "is_reentry" in out.columns else pd.Series(False, index=out.index)
        is_first_seen = out["is_first_seen"].fillna(False).astype(bool) if "is_first_seen" in out.columns else pd.Series(False, index=out.index)
        is_continuous = out["is_continuous_observation"].fillna(False).astype(bool) if "is_continuous_observation" in out.columns else pd.Series(False, index=out.index)
        velocity = pd.to_numeric(out["energy_velocity"], errors="coerce").fillna(0.0)
        acceleration = pd.to_numeric(out["energy_acceleration"], errors="coerce").fillna(0.0)
        persistence = pd.to_numeric(out["consistency_score"], errors="coerce").fillna(0.5)
        out["momentum_event_state"] = np.select(
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
    if "momentum_event_label" not in out.columns:
        out["momentum_event_label"] = out["momentum_event_state"].map(_momentum_event_label)
    return out


def _momentum_lifecycle_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    """관측 행에 없는 순위권 탈락/순위권 밖 상태를 패널로 복원해 상태 집계에만 쓴다."""
    if df.empty or not {"snapshot_id", "product_id", "rank_energy"}.issubset(set(df.columns)):
        return pd.DataFrame(columns=["eventLabel", "count", "avg_momentum", "avg_acceleration"])

    scope_cols = [col for col in ["source_dataset", "platform", "schema_version"] if col in df.columns]
    event_frames: list[pd.DataFrame] = []
    grouped_items = df.groupby(scope_cols, dropna=False) if scope_cols else [((), df)]
    for scope_key, scope_df in grouped_items:
        snapshots = (
            scope_df[["snapshot_id", "crawl_datetime"] if "crawl_datetime" in scope_df.columns else ["snapshot_id"]]
            .drop_duplicates(subset=["snapshot_id"])
            .sort_values(["crawl_datetime", "snapshot_id"] if "crawl_datetime" in scope_df.columns else ["snapshot_id"])
            .reset_index(drop=True)
        )
        if snapshots.empty:
            continue
        snapshots["_snapshot_ord"] = range(len(snapshots))
        products = scope_df[["product_id"]].drop_duplicates()
        panel = products.assign(_key=1).merge(snapshots[["snapshot_id", "_snapshot_ord"]].assign(_key=1), on="_key").drop(columns="_key")
        observed = scope_df[["product_id", "snapshot_id", "rank_energy"]].drop_duplicates(["product_id", "snapshot_id"], keep="last")
        panel = panel.merge(observed, on=["product_id", "snapshot_id"], how="left")
        panel["observed"] = panel["rank_energy"].notna()
        panel = panel.sort_values(["product_id", "_snapshot_ord"])
        by_product = panel.groupby("product_id", group_keys=False)
        panel["prev_observed"] = by_product["observed"].shift(1).fillna(False).astype(bool)
        panel["prior_observed_count"] = by_product["observed"].transform(lambda series: series.astype(int).cumsum().shift(1).fillna(0))
        panel["rank_energy_prev"] = by_product["rank_energy"].transform(lambda series: series.ffill().shift(1))
        panel["is_dropout"] = panel["prev_observed"] & (~panel["observed"])
        latest_ord = int(snapshots["_snapshot_ord"].max())
        panel["is_current_out_of_chart"] = (
            panel["_snapshot_ord"].eq(latest_ord)
            & (~panel["observed"])
            & panel["prior_observed_count"].gt(0)
            & (~panel["is_dropout"])
        )
        events = panel[panel["is_dropout"] | panel["is_current_out_of_chart"]].copy()
        if events.empty:
            continue
        events["eventLabel"] = np.where(events["is_dropout"], "순위권 탈락", "순위권 밖")
        events["avg_momentum"] = np.where(events["is_dropout"], -pd.to_numeric(events["rank_energy_prev"], errors="coerce"), np.nan)
        events["avg_acceleration"] = np.nan
        if scope_cols:
            if not isinstance(scope_key, tuple):
                scope_key = (scope_key,)
            for idx, col in enumerate(scope_cols):
                events[col] = scope_key[idx] if idx < len(scope_key) else None
        event_frames.append(events[["eventLabel", "avg_momentum", "avg_acceleration"]])

    if not event_frames:
        return pd.DataFrame(columns=["eventLabel", "count", "avg_momentum", "avg_acceleration"])
    event_df = pd.concat(event_frames, ignore_index=True)
    return (
        event_df.groupby("eventLabel", as_index=False)
        .agg(count=("eventLabel", "count"), avg_momentum=("avg_momentum", "mean"), avg_acceleration=("avg_acceleration", "mean"))
    )


def _resolve_schema_field_name(field_name: str, scope: str) -> str:
    normalized = str(field_name or "").strip()
    if not normalized:
        return normalized
    if scope == "raw" and "." not in normalized:
        normalized = f"product.{normalized}"
    return SCHEMA_FIELD_REPLACEMENTS.get(normalized, normalized)


def _is_deprecated_schema_field(field_name: str, scope: str) -> bool:
    normalized = str(field_name or "").strip()
    if not normalized:
        return False
    if scope == "raw" and "." not in normalized:
        normalized = f"product.{normalized}"
    return normalized in DEPRECATED_SCHEMA_FIELDS


def _filter_schema_fields(fields: list[dict[str, Any]] | None, scope: str) -> list[dict[str, Any]]:
    if not fields:
        return []
    return [
        field
        for field in fields
        if not _is_deprecated_schema_field(str(field.get("field") or ""), scope)
    ]


def _is_exploration_excluded_field(field_name: str, scope: str) -> bool:
    normalized = str(field_name or "").strip()
    if not normalized:
        return False
    if scope == "raw":
        return any(normalized.startswith(prefix) for prefix in EXPLORATION_EXCLUDED_RAW_PREFIXES)
    return normalized.startswith("business_") or normalized in EXPLORATION_EXCLUDED_NORMALIZED_FIELDS


def _filter_exploration_schema_fields(fields: list[dict[str, Any]] | None, scope: str) -> list[dict[str, Any]]:
    if not fields:
        return []
    return [
        field
        for field in fields
        if not _is_exploration_excluded_field(str(field.get("field") or ""), scope)
    ]


def _compose_area_label(province: Any, district: Any, dong: Any) -> str | None:
    bits = [
        _label_or_missing(province, ""),
        _label_or_missing(district, ""),
        _label_or_missing(dong, ""),
    ]
    scoped = [bit for bit in bits if bit]
    return " ".join(scoped) if scoped else None


def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    return "string"


def _value_type(value: Any) -> str:
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


def _dominant_type(counter: Counter[str]) -> str:
    scoped = [(name, count) for name, count in counter.items() if name != "null"]
    if not scoped:
        return "null"
    scoped.sort(key=lambda item: (-item[1], item[0]))
    return scoped[0][0]


def _semantic_type_label(column: str, series: pd.Series) -> str:
    dtype = _dtype_label(series)
    normalized = column.lower()
    if normalized.endswith("_id"):
        return "identifier"
    if "date" in normalized or "time" in normalized:
        return "time"
    if "price" in normalized:
        return "price"
    if "pct" in normalized or "rate" in normalized:
        return "percent"
    if "rank" in normalized:
        return "rank_or_rank_metric"
    if "score" in normalized or "velocity" in normalized or "acceleration" in normalized:
        return "derived_metric"
    if dtype == "boolean":
        return "flag"
    if dtype in {"integer", "number"}:
        return "numeric_measure"
    sample = series.dropna().astype(str).head(3).tolist()
    if any(text.startswith("{") or text.startswith("[") for text in sample):
        return "json_like_text"
    return "categorical_text"


def _infer_column_group(column: str) -> str:
    if column in COLUMN_METADATA:
        return COLUMN_METADATA[column]["group"]
    normalized = column.lower()
    if normalized.endswith("_id") or normalized.endswith("_url"):
        return "identifier"
    if "date" in normalized or "time" in normalized:
        return "time"
    if "price" in normalized or "discount" in normalized:
        return "price"
    if "rank" in normalized:
        return "rank"
    if "score" in normalized or "velocity" in normalized or "acceleration" in normalized or "volatility" in normalized:
        return "derived"
    if normalized.startswith("ocr_") or "image" in normalized or "exists" in normalized or "count" in normalized:
        return "quality"
    if normalized.startswith("business_") or "address" in normalized:
        return "location"
    if normalized in {"material", "color", "manufacturer", "origin_country", "shipping_fee"}:
        return "product_info"
    return "attribute"


def _column_grain(column: str) -> str:
    if column in COLUMN_METADATA:
        return COLUMN_METADATA[column]["grain"]
    normalized = column.lower()
    if normalized.startswith("business_") or normalized in {"brand", "name", "material", "color", "manufacturer", "origin_country"}:
        return "product"
    if normalized.startswith("snapshot_") or normalized in {"rank", "price", "discount_pct", "momentum_score"}:
        return "record"
    return "record"


def _example_value(series: pd.Series) -> str | None:
    for value in series.dropna().tolist():
        normalized = _normalize_value(value)
        if normalized is None:
            continue
        if isinstance(normalized, (dict, list)):
            text = json.dumps(normalized, ensure_ascii=False)
        else:
            text = str(normalized)
        if text:
            return text[:80]
    return None


def _safe_json_loads(value: Any) -> Any:
    if _is_missing_value(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _traverse_object_path(payload: Any, path_bits: list[str]) -> Any:
    current = payload
    for bit in path_bits:
        if isinstance(current, dict):
            current = current.get(bit)
            continue
        return None
    return current


def _top_value_rows(series: pd.Series, limit: int = 12) -> list[dict[str, Any]]:
    if series.empty:
        return []
    scoped = series.dropna().map(lambda value: _json_or_string(value)).astype(str)
    if scoped.empty:
        return []
    counts = scoped.value_counts(dropna=False).head(limit)
    return [
        {
            "value": value,
            "count": _safe_int(count),
            "ratePct": round((count / len(series)) * 100.0, 2) if len(series) else None,
        }
        for value, count in counts.items()
    ]


def _json_or_string(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(_normalize_value(value), ensure_ascii=False, sort_keys=True)
    return str(_normalize_value(value))


def _histogram_rows(series: pd.Series, bins: int = 10) -> list[dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return []
    if numeric.nunique() == 1:
        only = float(numeric.iloc[0])
        return [{"bucket": f"{only:g}", "count": int(len(numeric))}]
    cut = pd.cut(numeric, bins=min(bins, max(2, numeric.nunique())), include_lowest=True, duplicates="drop")
    counts = cut.value_counts(sort=False)
    rows = []
    for interval, count in counts.items():
        rows.append(
            {
                "bucket": str(interval),
                "count": int(count),
            }
        )
    return rows


def _segment_image_path_lookup(repository: AnalyticsRepository, dataset: str | None) -> pd.DataFrame:
    seg = repository.load_table(dataset, "image_segments.parquet", "image_segments")
    if seg.empty or "image_path" not in seg.columns:
        return pd.DataFrame(columns=["snapshot_id", "product_id", "image_path"])
    if not {"snapshot_id", "product_id", "image_path"}.issubset(seg.columns):
        return pd.DataFrame(columns=["snapshot_id", "product_id", "image_path"])
    scoped = seg.dropna(subset=["image_path"])
    if scoped.empty:
        return pd.DataFrame(columns=["snapshot_id", "product_id", "image_path"])
    if "embedding_target" in scoped.columns:
        et = scoped[scoped["embedding_target"].fillna(False)]
        if not et.empty:
            scoped = et
    sort_cols = [c for c in ("snapshot_id", "product_id", "segment_id") if c in scoped.columns]
    if len(sort_cols) < 2:
        sort_cols = ["snapshot_id", "product_id"]
    return (
        scoped.sort_values(sort_cols)
        .drop_duplicates(["snapshot_id", "product_id"], keep="last")[["snapshot_id", "product_id", "image_path"]]
        .copy()
    )


def _merge_embedding_image_paths(projection_df: pd.DataFrame, repository: AnalyticsRepository, dataset: str | None) -> pd.DataFrame:
    if projection_df.empty:
        return projection_df
    if "snapshot_id" not in projection_df.columns or "product_id" not in projection_df.columns:
        return projection_df
    lookup = _segment_image_path_lookup(repository, dataset)
    if lookup.empty:
        return projection_df
    fill = lookup.rename(columns={"image_path": "image_path_fill"})
    out = projection_df.merge(fill, on=["snapshot_id", "product_id"], how="left")
    if "image_path" in projection_df.columns:
        out["image_path"] = out["image_path"].fillna(out["image_path_fill"])
    else:
        out["image_path"] = out["image_path_fill"]
    return out.drop(columns=["image_path_fill"], errors="ignore")


def _manifest_row_allows_embedding_thumbnail(row: dict[str, Any]) -> bool:
    """
    툴팁 썸네일은 명시적 대표 이미지 파일명이 있을 때만 허용한다.
    전략(first/largest)으로 고른 상세 컷은 임베딩에는 쓰이더라도 썸네일로는 쓰지 않는다.
    세로로 긴 스티치/상세형(is_long_stitched)도 제외한다.
    """
    if not row.get("is_main_image") or not row.get("image_exists"):
        return False
    if bool(row.get("is_long_stitched")):
        return False
    src = row.get("main_image_source")
    if src == "main_image_filename":
        return True
    if _is_missing_value(src):
        name = Path(str(row.get("image_path") or "")).name.lower()
        return name in {filename.lower() for filename in EXPLICIT_MAIN_IMAGE_FILENAMES}
    return False


def _thumbnail_path_by_snapshot_product(manifest_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    if manifest_df.empty:
        return {}
    out: dict[tuple[str, str], str] = {}
    for row in manifest_df.to_dict("records"):
        if not _manifest_row_allows_embedding_thumbnail(row):
            continue
        key = (str(row.get("snapshot_id")), str(row.get("product_id")))
        path = str(row.get("image_path") or "").strip()
        if not path:
            continue
        out[key] = path
    return out


def _optional_label(value: Any) -> str | None:
    text = _label_or_missing(value, "")
    return text or None


def _is_meaningful_label(value: Any) -> bool:
    text = _optional_label(value)
    if not text:
        return False
    return text.lower() not in {"unknown", "미분류", "없음", "none", "nan", "null", "-", "기타", "uncategorized"}


def _truncate_text(value: Any, max_length: int = 120) -> str:
    normalized = _normalize_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, (dict, list)):
        text = json.dumps(normalized, ensure_ascii=False)
    else:
        text = str(normalized)
    return text if len(text) <= max_length else f"{text[: max_length - 1]}…"


def _value_shape_kind(value: Any) -> str:
    normalized = _normalize_value(value)
    if normalized is None:
        return "text"
    if isinstance(normalized, dict):
        return "kv"
    if isinstance(normalized, list):
        return "list"
    if isinstance(normalized, (int, float, np.integer, np.floating)) and not isinstance(normalized, bool):
        return "number"
    return "text"


def _split_joined_tags(value: Any) -> list[str]:
    text = _optional_label(value)
    if not text:
        return []
    return [part for part in (bit.strip() for bit in text.split(",")) if part]


def _mapping_rows(payload: Any, limit: int | None = None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        return []
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if _is_missing_value(value):
            continue
        rows.append(
            {
                "key": str(key),
                "label": str(key),
                "value": _truncate_text(value, 220),
                "shape": _value_shape_kind(value),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _snapshot_display_label(snapshot_id: Any, crawl_datetime: Any, snapshot_date: Any, snapshot_time: Any) -> str:
    if isinstance(crawl_datetime, pd.Timestamp) and not pd.isna(crawl_datetime):
        return crawl_datetime.strftime("%Y-%m-%d %H:%M")
    date_text = _optional_label(snapshot_date)
    time_text = _optional_label(snapshot_time)
    if date_text and time_text:
        return f"{date_text} {time_text}"
    if date_text:
        return date_text
    return str(snapshot_id)


def _category_path_from_sources(row: dict[str, Any], raw_category: Any) -> list[str]:
    path: list[str] = []

    def append_if_meaningful(value: Any) -> None:
        text = _optional_label(value)
        if not text or not _is_meaningful_label(text) or text in path:
            return
        path.append(text)

    for candidates in (
        ("category_label_l1", "category_l1"),
        ("category_label_l2", "category_l2"),
        ("category_label_l3", "category_l3"),
    ):
        for key in candidates:
            if key in row:
                append_if_meaningful(row.get(key))
                if path and path[-1] == _optional_label(row.get(key)):
                    break

    if not path and isinstance(raw_category, dict):
        for key in ("category_l1", "category_l2", "category_l3"):
            append_if_meaningful(raw_category.get(key))
    return path


def _thumbnail_shape_legend() -> list[dict[str, Any]]:
    return [
        {
            "key": "snapshot",
            "label": "snapshot",
            "shape": "datetime",
            "grain": "snapshot",
            "meaning": "한 번의 수집 번들을 가리키는 시간 단위입니다.",
        },
        {
            "key": "record",
            "label": "record",
            "shape": "row",
            "grain": "record",
            "meaning": "선택한 snapshot 안에서 상품 1개를 관측한 한 행입니다.",
        },
        {
            "key": "number",
            "label": "숫자형 메타",
            "shape": "number",
            "grain": "record",
            "meaning": "순위, 가격, 할인율처럼 정렬과 비교에 바로 쓰이는 값입니다.",
        },
        {
            "key": "text",
            "label": "텍스트형 메타",
            "shape": "text",
            "grain": "product",
            "meaning": "브랜드, 상품명, 카테고리처럼 의미 해석에 쓰이는 값입니다.",
        },
        {
            "key": "image",
            "label": "이미지 묶음",
            "shape": "image",
            "grain": "product",
            "meaning": "대표 이미지와 상세 이미지 스택을 함께 보여주는 시각 단위입니다.",
        },
        {
            "key": "kv",
            "label": "고시정보",
            "shape": "kv",
            "grain": "product",
            "meaning": "product_info 같은 key-value 필드를 펼쳐볼 수 있는 묶음 값입니다.",
        },
    ]


def _thumbnail_gallery_lookup(manifest_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if manifest_df.empty or not {"snapshot_id", "product_id", "image_path"}.issubset(manifest_df.columns):
        return {}
    scoped = manifest_df.copy()
    if "image_exists" in scoped.columns:
        scoped = scoped[scoped["image_exists"].fillna(False).astype(bool)]
    scoped = scoped.dropna(subset=["image_path"]).copy()
    if scoped.empty:
        return {}

    out: dict[tuple[str, str], dict[str, Any]] = {}
    explicit_main_names = {filename.lower() for filename in EXPLICIT_MAIN_IMAGE_FILENAMES}
    for (snapshot_id, product_id), frame in scoped.groupby(["snapshot_id", "product_id"], dropna=False):
        images: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for raw_row in frame.to_dict("records"):
            path = _optional_fs_path(raw_row.get("image_path"))
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            source = _optional_label(raw_row.get("main_image_source"))
            name = Path(path).name.lower()
            is_explicit_main = source == "main_image_filename" or name in explicit_main_names
            images.append(
                {
                    "path": path,
                    "isMainImage": is_explicit_main,
                    "mainImageSource": source,
                    "isExplicitMainImage": is_explicit_main,
                }
            )
        if not images:
            continue
        images.sort(
            key=lambda item: (
                0 if item["isExplicitMainImage"] else 1,
                str(item["path"]),
            )
        )
        explicit_main_image = next((image for image in images if image["isExplicitMainImage"]), None)
        out[(str(snapshot_id), str(product_id))] = {
            "images": images,
            "imageCount": len(images),
            "mainImagePath": explicit_main_image["path"] if explicit_main_image else None,
            "hasMainImage": explicit_main_image is not None,
            "hasExplicitMainImage": any(image["isExplicitMainImage"] for image in images),
            "mainImageSource": explicit_main_image.get("mainImageSource") if explicit_main_image else None,
        }
    return out


class DashboardService:
    def __init__(self, repository: AnalyticsRepository) -> None:
        self.repository = repository
        self._reviews_meta_root = Path("data/reviews/products")

    def _load_fact(self, filters: DashboardFilters) -> tuple[pd.DataFrame, pd.DataFrame]:
        full_df = self.repository.load_fact_snapshots(filters.dataset)
        filtered_df = apply_fact_filters(full_df, filters)
        return full_df, filtered_df

    def _load_total_reviews_from_meta(self, product_ids: pd.Series) -> pd.Series:
        """reviews/products/<pid>/meta.json 의 total_reviews_reported를 로드한다."""
        if product_ids.empty or not self._reviews_meta_root.exists():
            return pd.Series(dtype="int64")
        totals: dict[str, int] = {}
        for pid in product_ids.astype(str).drop_duplicates():
            meta_path = self._reviews_meta_root / pid / "meta.json"
            if not meta_path.exists():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            value = payload.get("total_reviews_reported")
            if _is_missing_value(value) and isinstance(payload.get("last_collection_meta"), dict):
                value = payload["last_collection_meta"].get("total_reviews_reported")
            if _is_missing_value(value):
                continue
            try:
                parsed = int(float(str(value).replace(",", "").strip()))
            except Exception:
                continue
            totals[pid] = max(parsed, 0)
        return pd.Series(totals, dtype="int64")

    def _build_thumbnail_snapshot_summaries(
        self,
        filtered_fact_df: pd.DataFrame,
        gallery_lookup: dict[tuple[str, str], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if filtered_fact_df.empty or "snapshot_id" not in filtered_fact_df.columns:
            return []

        scoped = filtered_fact_df.copy()
        scoped["snapshot_id"] = scoped["snapshot_id"].astype(str)
        if "crawl_datetime" in scoped.columns:
            scoped["crawl_datetime"] = pd.to_datetime(scoped["crawl_datetime"], errors="coerce")
        if "product_id" in scoped.columns:
            scoped["product_id"] = scoped["product_id"].astype(str)
        if {"snapshot_id", "product_id"}.issubset(scoped.columns):
            scoped = scoped.drop_duplicates(subset=["snapshot_id", "product_id"], keep="last").reset_index(drop=True)

        scoped["thumbnail_category_label"] = _normalize_category_series(
            scoped,
            ("category_label_l3", "category_l3", "category_label_l2", "category_l2", "category_label_l1", "category_l1"),
        )
        scoped["has_category"] = scoped["thumbnail_category_label"].notna() & ~_is_unclassified_series(scoped["thumbnail_category_label"])
        if "product_info_exists" in scoped.columns:
            scoped["has_detail_info"] = scoped["product_info_exists"].fillna(False).astype(bool)
        else:
            scoped["has_detail_info"] = False

        gallery_rows = []
        for (snapshot_id, product_id), bundle in gallery_lookup.items():
            gallery_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "product_id": product_id,
                    "image_count": int(bundle.get("imageCount") or 0),
                    "has_main_image": bool(bundle.get("hasMainImage")),
                    "has_explicit_main_image": bool(bundle.get("hasExplicitMainImage")),
                }
            )
        if gallery_rows:
            scoped = scoped.merge(pd.DataFrame(gallery_rows), on=["snapshot_id", "product_id"], how="left")
        image_count_series = scoped["image_count"] if "image_count" in scoped.columns else pd.Series([0] * len(scoped), index=scoped.index)
        main_image_series = (
            scoped["has_main_image"] if "has_main_image" in scoped.columns else pd.Series([False] * len(scoped), index=scoped.index)
        )
        explicit_main_series = (
            scoped["has_explicit_main_image"]
            if "has_explicit_main_image" in scoped.columns
            else pd.Series([False] * len(scoped), index=scoped.index)
        )
        scoped["image_count"] = pd.to_numeric(image_count_series, errors="coerce").fillna(0).astype(int)
        scoped["has_main_image"] = main_image_series.fillna(False).astype(bool)
        scoped["has_explicit_main_image"] = explicit_main_series.fillna(False).astype(bool)

        snapshots: list[dict[str, Any]] = []
        for snapshot_id, frame in scoped.groupby("snapshot_id", dropna=False):
            valid_categories = frame.loc[frame["has_category"], "thumbnail_category_label"].astype(str)
            valid_brands = frame["brand"].dropna().astype(str)
            price_series = frame["price"] if "price" in frame.columns else pd.Series(dtype=float)
            avg_price = pd.to_numeric(price_series, errors="coerce").dropna()
            latest_crawl = frame["crawl_datetime"].max() if "crawl_datetime" in frame.columns else None
            snapshots.append(
                {
                    "snapshotId": str(snapshot_id),
                    "label": _snapshot_display_label(
                        snapshot_id,
                        latest_crawl,
                        frame["snapshot_date"].iloc[0] if "snapshot_date" in frame.columns and not frame.empty else None,
                        frame["snapshot_time"].iloc[0] if "snapshot_time" in frame.columns and not frame.empty else None,
                    ),
                    "crawlDatetime": latest_crawl.isoformat() if isinstance(latest_crawl, pd.Timestamp) and not pd.isna(latest_crawl) else None,
                    "snapshotDate": _optional_label(frame["snapshot_date"].iloc[0]) if "snapshot_date" in frame.columns and not frame.empty else None,
                    "snapshotTime": _optional_label(frame["snapshot_time"].iloc[0]) if "snapshot_time" in frame.columns and not frame.empty else None,
                    "recordCount": int(len(frame)),
                    "productCount": int(frame["product_id"].nunique()) if "product_id" in frame.columns else int(len(frame)),
                    "brandCount": int(valid_brands.nunique()),
                    "avgPrice": _safe_float(avg_price.mean()) if not avg_price.empty else None,
                    "mainImageCoveragePct": round(float(frame["has_main_image"].mean() * 100.0), 2) if len(frame) else None,
                    "detailInfoCoveragePct": round(float(frame["has_detail_info"].mean() * 100.0), 2) if len(frame) else None,
                    "categoryCoveragePct": round(float(frame["has_category"].mean() * 100.0), 2) if len(frame) else None,
                    "topBrand": valid_brands.value_counts().index[0] if not valid_brands.empty else None,
                    "topCategory": valid_categories.value_counts().index[0] if not valid_categories.empty else None,
                }
            )

        snapshots.sort(
            key=lambda row: (
                row.get("crawlDatetime") is None,
                row.get("crawlDatetime") or "",
                row.get("snapshotId") or "",
            ),
            reverse=True,
        )
        return snapshots

    def _apply_raw_filters(self, raw_df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df.copy()
        filtered = raw_df.copy()
        if "crawl_datetime" in filtered.columns:
            filtered["crawl_datetime"] = pd.to_datetime(filtered["crawl_datetime"], errors="coerce")
        if filters.source_datasets and "source_dataset" in filtered.columns:
            filtered = filtered[filtered["source_dataset"].astype(str).isin(filters.source_datasets)]
        if filters.platforms and "platform" in filtered.columns:
            filtered = filtered[filtered["platform"].astype(str).isin(filters.platforms)]
        if filters.schema_versions and "schema_version" in filtered.columns:
            filtered = filtered[filtered["schema_version"].astype(str).isin(filters.schema_versions)]
        if filters.date_from:
            if "snapshot_date" in filtered.columns:
                filtered = filtered[filtered["snapshot_date"].astype(str) >= filters.date_from]
            elif "crawl_datetime" in filtered.columns:
                filtered = filtered[filtered["crawl_datetime"] >= pd.to_datetime(filters.date_from, errors="coerce")]
        if filters.date_to:
            if "snapshot_date" in filtered.columns:
                filtered = filtered[filtered["snapshot_date"].astype(str) <= filters.date_to]
            elif "crawl_datetime" in filtered.columns:
                filtered = filtered[filtered["crawl_datetime"] <= pd.to_datetime(filters.date_to, errors="coerce")]
        return filtered.reset_index(drop=True)

    def _load_text_features(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        text_features_df = self.repository.load_table(
            filters.dataset,
            "text_features.parquet",
            "text_features",
        )
        return filter_related_table(text_features_df, filtered_fact_df)

    def _load_text_review_facts(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        review_df = self.repository.load_table(
            filters.dataset,
            "text_review_facts.parquet",
            "text_review_facts",
        )
        if review_df.empty:
            return review_df
        scoped = filter_related_table(review_df, filtered_fact_df)
        if scoped.empty:
            return scoped
        if "created_at" in scoped.columns:
            scoped["created_at"] = pd.to_datetime(scoped["created_at"], errors="coerce")
        return scoped

    def _load_text_claim_facts(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        claim_df = self.repository.load_table(
            filters.dataset,
            "text_claim_facts.parquet",
            "text_claim_facts",
        )
        if claim_df.empty:
            return claim_df
        return filter_related_table(claim_df, filtered_fact_df)

    def _load_text_gap_metrics(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        gap_df = self.repository.load_table(
            filters.dataset,
            "text_gap_metrics.parquet",
            "text_gap_metrics",
        )
        if gap_df.empty:
            return gap_df
        return filter_related_table(gap_df, filtered_fact_df)

    def _load_text_fusion_profile(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        fusion_df = self.repository.load_table(
            filters.dataset,
            "text_fusion_profile.parquet",
            "text_fusion_profile",
        )
        if fusion_df.empty:
            return fusion_df
        return filter_related_table(fusion_df, filtered_fact_df)

    def _run_dynamic_fusion(
        self,
        filters: DashboardFilters,
        filtered_fact_df: pd.DataFrame,
        text_features_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if filtered_fact_df.empty:
            return {
                "tag_performance": pd.DataFrame(),
                "brand_index": pd.DataFrame(),
                "trends": pd.DataFrame(),
                "product_profile": pd.DataFrame(),
                "category_overview": pd.DataFrame(),
                "category_relationships": pd.DataFrame(),
                "category_timeseries": pd.DataFrame(),
                "category_quality": pd.DataFrame(),
            }
        return run_analysis_fusion(
            filtered_fact_df,
            text_features_df if not text_features_df.empty else None,
        )

    def _category_level_column(self, level: str) -> str:
        return {
            "l1": "category_label_l1",
            "l2": "category_label_l2",
            "l3": "category_label_l3",
        }.get(level, "category_label_l3")

    def _build_category_frames(self, filters: DashboardFilters) -> tuple[pd.DataFrame, pd.DataFrame]:
        _, filtered_fact_df = self._load_fact(filters)
        text_features_df = self._load_text_features(filters, filtered_fact_df)
        latest = build_product_profile(filtered_fact_df, text_features_df if not text_features_df.empty else None)
        records = _prepare_category_frame(filtered_fact_df, text_features_df if not text_features_df.empty else None, latest_only=False)
        return latest, records

    def _filter_category_frame(
        self,
        frame: pd.DataFrame,
        quality_mode: str = "success_only",
        include_fallback: bool = False,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        scoped = frame.copy()
        allowed_statuses = ["success"] if quality_mode == "success_only" else ["success", "partial"]
        raw_mask = scoped["category_source"].astype(str).eq("raw_taxonomy") & scoped["category_ingest_status"].astype(str).isin(allowed_statuses)
        if include_fallback:
            fallback_mask = scoped["category_source"].astype(str).eq("fallback_name_item")
            scoped = scoped[raw_mask | fallback_mask]
        else:
            scoped = scoped[raw_mask]
        return scoped.reset_index(drop=True)

    def _category_label(self, frame: pd.DataFrame, level: str) -> pd.Series:
        column = self._category_level_column(level)
        if column not in frame.columns:
            return pd.Series(["미분류"] * len(frame), index=frame.index)
        return frame[column].fillna("미분류").astype(str)

    def _top_category_labels(self, frame: pd.DataFrame, level: str, limit: int = 12) -> list[str]:
        if frame.empty:
            return []
        labels = self._category_label(frame, level)
        return labels.value_counts(dropna=False).head(limit).index.astype(str).tolist()

    def _load_geocode_cache(self, dataset: str | None) -> dict[str, Any]:
        payload = self.repository.load_json(dataset, GEOCODE_CACHE_FILENAME)
        return payload if isinstance(payload, dict) else {}

    def _save_geocode_cache(self, dataset: str | None, payload: dict[str, Any]) -> None:
        self.repository.save_json(dataset, GEOCODE_CACHE_FILENAME, payload)

    def _geocode_location_label(self, label: str) -> dict[str, Any] | None:
        params = urlencode(
            {
                "q": f"{label}, 대한민국",
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "kr",
            }
        )
        request = Request(
            f"https://nominatim.openstreetmap.org/search?{params}",
            headers={
                "User-Agent": "SilhouetteVisualizationEngine/0.1",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, list) or not payload:
            return None
        first = payload[0]
        try:
            latitude = float(first.get("lat"))
            longitude = float(first.get("lon"))
        except Exception:
            return None
        return {
            "lat": latitude,
            "lng": longitude,
            "displayName": first.get("display_name"),
        }

    def _attach_location_points(
        self,
        filters: DashboardFilters,
        frame: pd.DataFrame,
        label_column: str = "locationLabel",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if frame.empty or label_column not in frame.columns:
            return []
        cache = self._load_geocode_cache(filters.dataset)
        updated = False
        points: list[dict[str, Any]] = []
        for index, row in enumerate(frame.head(limit).to_dict("records")):
            label = _label_or_missing(row.get(label_column), "")
            if not label:
                continue
            cached = cache.get(label)
            if cached is None:
                if index > 0:
                    time.sleep(1.05)
                cached = self._geocode_location_label(label)
                cache[label] = cached
                updated = True
            if not cached:
                continue
            points.append(
                {
                    "locationLabel": label,
                    "lat": _safe_float(cached.get("lat")),
                    "lng": _safe_float(cached.get("lng")),
                    "displayName": cached.get("displayName"),
                    "recordCount": _safe_int(row.get("record_count")),
                    "brandCount": _safe_int(row.get("brand_count")),
                    "avgRank": _safe_float(row.get("avg_rank")),
                    "avgPrice": _safe_float(row.get("avg_price")),
                    "avgDiscountPct": _safe_float(row.get("avg_discount_pct")),
                    "businessProvince": row.get("businessProvinceLabel") or row.get("business_province"),
                    "businessDistrict": row.get("businessDistrictLabel") or row.get("business_district"),
                    "businessDong": row.get("businessDongLabel") or row.get("business_dong"),
                }
            )
        if updated:
            self._save_geocode_cache(filters.dataset, cache)
        return points

    def get_datasets(self) -> dict[str, Any]:
        return {"datasets": self.repository.list_datasets()}

    def get_schema_explorer_rules(self) -> dict[str, Any]:
        return {
            "rulesFile": SCHEMA_EXPLORER_RULES_FILENAME,
            "excludeRawPrefixes": list(EXPLORATION_EXCLUDED_RAW_PREFIXES),
            "excludeNormalizedFields": sorted(EXPLORATION_EXCLUDED_NORMALIZED_FIELDS),
        }

    def _load_schema_artifact(self, dataset: str | None, scope: str) -> dict[str, Any]:
        filename = SCHEMA_RAW_FILENAME if scope == "raw" else SCHEMA_NORMALIZED_FILENAME
        return self.repository.load_schema_artifact(dataset, filename)

    def _build_scoped_raw_schema(self, filters: DashboardFilters) -> dict[str, Any]:
        raw_df = self._apply_raw_filters(self.repository.load_raw_snapshot_products(filters.dataset), filters)
        if raw_df.empty:
            return {"scope": "raw", "rowCount": 0, "fieldCount": 0, "fields": []}
        fields = {}
        for payload_column, prefix in (
            ("raw_summary_json", "summary"),
            ("raw_product_json", "product"),
            ("raw_info_map_json", "product_info"),
            ("raw_category_json", "category"),
        ):
            if payload_column not in raw_df.columns:
                continue
            for payload in raw_df[payload_column].tolist():
                decoded = _safe_json_loads(payload)
                if not isinstance(decoded, dict):
                    continue
                for key, value in decoded.items():
                    field_name = f"{prefix}.{key}"
                    if _is_deprecated_schema_field(field_name, "raw"):
                        continue
                    entry = fields.setdefault(field_name, {"field": field_name, "typeCounter": Counter(), "nonNullCount": 0})
                    entry["typeCounter"][_value_type(value)] += 1
                    if not _is_missing_value(value):
                        entry["nonNullCount"] += 1
        normalized_rows = []
        for field_name in sorted(fields.keys()):
            entry = fields[field_name]
            normalized_rows.append(
                {
                    "field": field_name,
                    "scope": "raw",
                    "nonNullCount": int(entry["nonNullCount"]),
                    "inferredType": _dominant_type(entry["typeCounter"]),
                }
            )
        return {"scope": "raw", "rowCount": int(len(raw_df)), "fieldCount": len(normalized_rows), "fields": normalized_rows}

    def _build_scoped_normalized_schema(self, filters: DashboardFilters) -> dict[str, Any]:
        full_df = self.repository.load_fact_snapshots(filters.dataset)
        filtered_df = apply_fact_filters(full_df, filters)
        if filtered_df.empty:
            return {"scope": "normalized", "rowCount": 0, "fieldCount": 0, "fields": []}
        rows = []
        for column in sorted(filtered_df.columns):
            if _is_deprecated_schema_field(column, "normalized"):
                continue
            series = filtered_df[column]
            non_null = series.dropna()
            rows.append(
                {
                    "field": column,
                    "scope": "normalized",
                    "nonNullCount": int(non_null.shape[0]),
                    "inferredType": _semantic_type_label(column, series),
                }
            )
        return {"scope": "normalized", "rowCount": int(len(filtered_df)), "fieldCount": len(rows), "fields": rows}

    def _schema_for_source(self, filters: DashboardFilters, source_dataset: str, scope: str) -> dict[str, Any]:
        scoped_filters = DashboardFilters(
            dataset=filters.dataset,
            brands=filters.brands,
            source_datasets=[source_dataset],
            platforms=filters.platforms,
            schema_versions=filters.schema_versions,
            snapshot_window=filters.snapshot_window,
            date_from=filters.date_from,
            date_to=filters.date_to,
        )
        if scope == "raw":
            return self._build_scoped_raw_schema(scoped_filters)
        return self._build_scoped_normalized_schema(scoped_filters)

    def _build_source_schema_diff(
        self,
        filters: DashboardFilters,
        compare_source_dataset: str,
        scope: str,
        current_source_dataset: str | None = None,
    ) -> dict[str, Any]:
        current_sources = filters.source_datasets or []
        chosen_current_source = str(current_source_dataset).strip() if current_source_dataset else None
        if chosen_current_source:
            current_source_dataset = chosen_current_source
            current_schema = self._schema_for_source(filters, current_source_dataset, scope)
            left_fields = {
                str(field["field"]): field
                for field in current_schema.get("fields", [])
                if not _is_deprecated_schema_field(str(field.get("field") or ""), scope)
            }
        elif not current_sources:
            current_source_dataset = "all"
            current_schema = self._load_schema_artifact(filters.dataset, scope)
            left_fields = {
                str(field["field"]): field
                for field in current_schema.get("fields", [])
                if not _is_deprecated_schema_field(str(field.get("field") or ""), scope)
            }
        else:
            current_source_dataset = current_sources[0]
            current_schema = self._schema_for_source(filters, current_source_dataset, scope)
            left_fields = {
                str(field["field"]): field
                for field in current_schema.get("fields", [])
                if not _is_deprecated_schema_field(str(field.get("field") or ""), scope)
            }
        compare_schema = self._schema_for_source(filters, compare_source_dataset, scope)
        right_fields = {
            str(field["field"]): field
            for field in compare_schema.get("fields", [])
            if not _is_deprecated_schema_field(str(field.get("field") or ""), scope)
        }
        added = [left_fields[name] for name in sorted(set(left_fields) - set(right_fields))]
        removed = [right_fields[name] for name in sorted(set(right_fields) - set(left_fields))]
        type_changed = []
        for name in sorted(set(left_fields) & set(right_fields)):
            left_type = left_fields[name].get("inferredType")
            right_type = right_fields[name].get("inferredType")
            if left_type != right_type:
                type_changed.append(
                    {
                        "field": name,
                        "currentType": left_type,
                        "compareType": right_type,
                    }
                )
        exploration_added = _filter_exploration_schema_fields(added, scope)
        exploration_removed = _filter_exploration_schema_fields(removed, scope)
        return {
            "scope": scope,
            "dataset": filters.dataset or "analytics",
            "compareSourceDataset": compare_source_dataset,
            "currentSourceDataset": current_source_dataset,
            "summary": {
                "currentFieldCount": len(left_fields),
                "compareFieldCount": len(right_fields),
                "addedFieldCount": len(added),
                "removedFieldCount": len(removed),
                "explorationAddedFieldCount": len(exploration_added),
                "explorationRemovedFieldCount": len(exploration_removed),
                "typeChangedFieldCount": len(type_changed),
            },
            "addedFields": added,
            "removedFields": removed,
            "explorationAddedFields": exploration_added,
            "explorationRemovedFields": exploration_removed,
            "typeChangedFields": type_changed,
        }

    def get_schema_inventory(self, filters: DashboardFilters) -> dict[str, Any]:
        has_source_scope = bool(filters.source_datasets or filters.platforms or filters.schema_versions or filters.date_from or filters.date_to)
        if has_source_scope:
            raw_schema = self._build_scoped_raw_schema(filters)
            normalized_schema = self._build_scoped_normalized_schema(filters)
        else:
            raw_schema = self.repository.load_schema_artifact(filters.dataset, SCHEMA_RAW_FILENAME)
            normalized_schema = self.repository.load_schema_artifact(filters.dataset, SCHEMA_NORMALIZED_FILENAME)
        schema_diff = self.repository.load_schema_artifact(filters.dataset, SCHEMA_DIFF_FILENAME)
        raw_schema["fields"] = _filter_schema_fields(raw_schema.get("fields"), "raw")
        raw_schema["fieldCount"] = len(raw_schema["fields"])
        normalized_schema["fields"] = _filter_schema_fields(normalized_schema.get("fields"), "normalized")
        normalized_schema["fieldCount"] = len(normalized_schema["fields"])
        if isinstance(schema_diff, dict):
            schema_diff["rawOnlyFields"] = _filter_schema_fields(schema_diff.get("rawOnlyFields"), "raw")
            schema_diff["explorationRawOnlyFields"] = _filter_exploration_schema_fields(schema_diff.get("rawOnlyFields"), "raw")
            schema_diff["normalizedOnlyFields"] = _filter_schema_fields(
                schema_diff.get("normalizedOnlyFields"),
                "normalized",
            )
            schema_diff["sourceMappings"] = [
                row
                for row in schema_diff.get("sourceMappings", [])
                if not _is_deprecated_schema_field(str(row.get("normalizedField") or ""), "normalized")
            ]
        kpi = self.repository.load_json(filters.dataset, "kpi_summary.json")
        full_df, filtered_df = self._load_fact(filters)
        return {
            "dataset": filters.dataset or "analytics",
            "raw": raw_schema,
            "normalized": normalized_schema,
            "diff": schema_diff,
            "availableSourceDatasets": _sorted_unique_source_datasets(full_df["source_dataset"])
            if "source_dataset" in full_df.columns
            else [],
            "availablePlatforms": sorted(full_df["platform"].dropna().astype(str).unique().tolist()) if "platform" in full_df.columns else [],
            "availableSchemaVersions": sorted(full_df["schema_version"].dropna().astype(str).unique().tolist()) if "schema_version" in full_df.columns else [],
            "filteredRecordCount": int(len(filtered_df)),
            "report": (kpi.get("extra") if isinstance(kpi, dict) else {}) or {},
        }

    def get_schema_pair_samples(
        self,
        filters: DashboardFilters,
        sample_size: int = 5,
    ) -> dict[str, Any]:
        sample_size = max(1, min(int(sample_size or 5), 20))
        explicit_pairs = [
            {"field": "price", "rawColumn": "price_raw", "normalizedColumn": "price"},
            {"field": "discount_pct", "rawColumn": "discount_raw", "normalizedColumn": "discount_pct"},
            {"field": "original_price", "rawColumn": "original_price_raw", "normalizedColumn": "original_price"},
            {"field": "discount_amount", "rawColumn": "discount_amount_raw", "normalizedColumn": "discount_amount"},
            {
                "field": "ocr_images_count_summary",
                "rawColumn": "ocr_images_count_summary_raw",
                "normalizedColumn": "ocr_images_count_summary",
            },
            {"field": "material", "rawColumn": "material_raw", "normalizedColumn": "material"},
            {"field": "color", "rawColumn": "color_raw", "normalizedColumn": "color"},
            {"field": "manufacturer", "rawColumn": "manufacturer_raw", "normalizedColumn": "manufacturer"},
            {"field": "origin_country", "rawColumn": "origin_country_raw", "normalizedColumn": "origin_country"},
            {"field": "shipping_fee", "rawColumn": "shipping_fee_raw", "normalizedColumn": "shipping_fee"},
        ]
        full_df = self.repository.load_fact_snapshots(filters.dataset)
        filtered_df = apply_fact_filters(full_df, filters)
        pairs: list[dict[str, Any]] = []
        if filtered_df.empty:
            return {"dataset": filters.dataset or "analytics", "pairs": pairs, "sampleSize": sample_size}
        for spec in explicit_pairs:
            raw_col = spec["rawColumn"]
            norm_col = spec["normalizedColumn"]
            if raw_col not in filtered_df.columns or norm_col not in filtered_df.columns:
                continue
            candidate = filtered_df[[col for col in ("snapshot_id", "product_id", raw_col, norm_col) if col in filtered_df.columns]].copy()
            mask = candidate[raw_col].notna() & candidate[norm_col].notna()
            candidate = candidate[mask]
            if candidate.empty:
                continue
            diff_mask = candidate[raw_col].astype(str) != candidate[norm_col].astype(str)
            preferred = candidate[diff_mask]
            selected = preferred if not preferred.empty else candidate
            selected = selected.head(sample_size)
            samples = []
            for _, row in selected.iterrows():
                samples.append(
                    {
                        "snapshotId": str(row.get("snapshot_id")) if "snapshot_id" in selected.columns else None,
                        "productId": str(row.get("product_id")) if "product_id" in selected.columns else None,
                        "rawValue": _normalize_value(row[raw_col]),
                        "normalizedValue": _normalize_value(row[norm_col]),
                    }
                )
            pairs.append(
                {
                    "field": spec["field"],
                    "rawField": raw_col,
                    "normalizedField": norm_col,
                    "rawInferredType": _dominant_type(
                        Counter(_value_type(val) for val in filtered_df[raw_col].dropna().head(200).tolist())
                    ),
                    "normalizedInferredType": _dominant_type(
                        Counter(_value_type(val) for val in filtered_df[norm_col].dropna().head(200).tolist())
                    ),
                    "matchCount": int(mask.sum()),
                    "diffCount": int(diff_mask.sum()),
                    "samples": samples,
                }
            )
        return {
            "dataset": filters.dataset or "analytics",
            "pairs": pairs,
            "sampleSize": sample_size,
        }

    def get_schema_diff(
        self,
        filters: DashboardFilters,
        scope: str,
        compare_source_dataset: str | None = None,
        current_source_dataset: str | None = None,
    ) -> dict[str, Any]:
        if compare_source_dataset:
            return self._build_source_schema_diff(
                filters,
                compare_source_dataset=compare_source_dataset,
                scope=scope,
                current_source_dataset=current_source_dataset,
            )
        stored_diff = self.repository.load_schema_artifact(filters.dataset, SCHEMA_DIFF_FILENAME)
        if isinstance(stored_diff, dict):
            stored_diff["rawOnlyFields"] = _filter_schema_fields(stored_diff.get("rawOnlyFields"), "raw")
            stored_diff["explorationRawOnlyFields"] = _filter_exploration_schema_fields(stored_diff.get("rawOnlyFields"), "raw")
            stored_diff["normalizedOnlyFields"] = _filter_schema_fields(
                stored_diff.get("normalizedOnlyFields"),
                "normalized",
            )
            stored_diff["sourceMappings"] = [
                row
                for row in stored_diff.get("sourceMappings", [])
                if not _is_deprecated_schema_field(str(row.get("normalizedField") or ""), "normalized")
            ]
        return {
            "scope": scope,
            "dataset": filters.dataset or "analytics",
            "compareSourceDataset": None,
            "storedDiff": stored_diff,
        }

    def _build_raw_field_frame(self, filters: DashboardFilters, field_name: str) -> pd.DataFrame:
        field_name = _resolve_schema_field_name(field_name, "raw")
        raw_df = self._apply_raw_filters(self.repository.load_raw_snapshot_products(filters.dataset), filters)
        if raw_df.empty:
            return pd.DataFrame(columns=["snapshot_id", "product_id", "field_value"])
        path_bits = field_name.split(".")
        if len(path_bits) < 2:
            return pd.DataFrame(columns=["snapshot_id", "product_id", "field_value"])
        source = path_bits[0]
        payload_column_map = {
            "summary": "raw_summary_json",
            "product": "raw_product_json",
            "product_info": "raw_info_map_json",
        }
        payload_column = payload_column_map.get(source)
        if payload_column is None or payload_column not in raw_df.columns:
            return pd.DataFrame(columns=["snapshot_id", "product_id", "field_value"])
        keep_columns = ["snapshot_id", "product_id", "source_dataset", "platform", "schema_version", payload_column]
        scoped = raw_df[[column for column in keep_columns if column in raw_df.columns]].copy()
        scoped["field_value"] = scoped[payload_column].map(
            lambda payload: _traverse_object_path(_safe_json_loads(payload), path_bits[1:])
        )
        return scoped[[column for column in ["snapshot_id", "product_id", "source_dataset", "platform", "schema_version", "field_value"] if column in scoped.columns]]

    def _build_normalized_field_frame(self, filters: DashboardFilters, field_name: str) -> pd.DataFrame:
        field_name = _resolve_schema_field_name(field_name, "normalized")
        full_df = self.repository.load_fact_snapshots(filters.dataset)
        filtered_df = apply_fact_filters(full_df, filters)
        if filtered_df.empty or field_name not in filtered_df.columns:
            return pd.DataFrame(columns=["snapshot_id", "product_id", "field_value"])
        scoped = filtered_df.copy()
        scoped["field_value"] = scoped[field_name]
        return scoped[
            [column for column in ["snapshot_id", "product_id", "source_dataset", "platform", "schema_version", "field_value"] if column in scoped.columns]
        ]

    def _field_frame(self, filters: DashboardFilters, field_name: str, scope: str) -> pd.DataFrame:
        return self._build_raw_field_frame(filters, field_name) if scope == "raw" else self._build_normalized_field_frame(filters, field_name)

    def get_field_profile(self, filters: DashboardFilters, field_name: str, scope: str) -> dict[str, Any]:
        frame = self._field_frame(filters, field_name, scope)
        if frame.empty:
            return {
                "fieldName": field_name,
                "scope": scope,
                "rowCount": 0,
                "nonNullCount": 0,
                "missingRatePct": None,
                "distinctCount": 0,
                "inferredType": "unknown",
                "sampleRows": [],
                "topValues": [],
                "histogram": [],
            }
        value_series = frame["field_value"]
        non_null = value_series.dropna()
        value_type_counter = Counter(_field_type(value) for value in non_null.head(500).tolist())
        inferred_type = _inferred_type(value_type_counter)
        sample_rows = frame[frame["field_value"].notna()].head(25).copy()
        sample_rows["field_value"] = sample_rows["field_value"].map(_normalize_value)
        distinct_count = int(non_null.map(_json_or_string).nunique()) if not non_null.empty else 0
        return {
            "fieldName": field_name,
            "scope": scope,
            "rowCount": int(len(frame)),
            "nonNullCount": int(non_null.shape[0]),
            "missingRatePct": round(float(value_series.isna().mean() * 100.0), 2) if len(frame) else None,
            "distinctCount": distinct_count,
            "inferredType": inferred_type,
            "sampleRows": [
                {
                    "snapshotId": row["snapshot_id"],
                    "productId": str(row["product_id"]),
                    "value": row["field_value"],
                }
                for row in sample_rows.to_dict("records")
            ],
            "topValues": _top_value_rows(value_series),
            "histogram": _histogram_rows(value_series) if inferred_type in {"integer", "number"} else [],
        }

    def get_candidate_insights(self, filters: DashboardFilters, field_name: str, scope: str) -> dict[str, Any]:
        field_frame = self._field_frame(filters, field_name, scope)
        full_df = self.repository.load_fact_snapshots(filters.dataset)
        fact_df = apply_fact_filters(full_df, filters)
        if field_frame.empty or fact_df.empty:
            return {"fieldName": field_name, "scope": scope, "kind": "empty", "summaryRows": [], "chartRows": []}
        joined = field_frame.merge(
            fact_df[["snapshot_id", "product_id", "rank", "price", "discount_pct", "momentum_score", "brand", "name"]].copy(),
            on=["snapshot_id", "product_id"],
            how="inner",
        )
        if joined.empty:
            return {"fieldName": field_name, "scope": scope, "kind": "empty", "summaryRows": [], "chartRows": []}
        numeric_series = pd.to_numeric(joined["field_value"], errors="coerce")
        numeric_non_null = numeric_series.dropna()
        if not numeric_non_null.empty and numeric_non_null.shape[0] >= max(5, int(len(joined) * 0.1)):
            joined = joined.assign(field_numeric=numeric_series)
            correlations = []
            for metric in ["rank", "price", "discount_pct", "momentum_score"]:
                metric_series = pd.to_numeric(joined[metric], errors="coerce")
                valid = joined["field_numeric"].notna() & metric_series.notna()
                corr = joined.loc[valid, "field_numeric"].corr(metric_series[valid]) if valid.any() else None
                correlations.append(
                    {
                        "metric": metric,
                        "correlation": _safe_float(corr),
                        "pairCount": int(valid.sum()),
                    }
                )
            scatter = joined[joined["field_numeric"].notna()].head(250)
            return {
                "fieldName": field_name,
                "scope": scope,
                "kind": "numeric",
                "summaryRows": correlations,
                "chartRows": [
                    {
                        "fieldValue": _safe_float(row.get("field_numeric")),
                        "rank": _safe_float(row.get("rank")),
                        "price": _safe_float(row.get("price")),
                        "momentumScore": _safe_float(row.get("momentum_score")),
                        "brand": row.get("brand"),
                        "name": row.get("name"),
                    }
                    for row in scatter.to_dict("records")
                ],
            }
        categorized = joined.copy()
        categorized["field_label"] = categorized["field_value"].map(lambda value: _json_or_string(value) if not _is_missing_value(value) else "미입력")
        grouped = (
            categorized.groupby("field_label", as_index=False)
            .agg(
                record_count=("product_id", "count"),
                avg_rank=("rank", "mean"),
                avg_price=("price", "mean"),
                avg_momentum_score=("momentum_score", "mean"),
            )
            .sort_values(["record_count", "avg_rank"], ascending=[False, True])
            .head(20)
        )
        return {
            "fieldName": field_name,
            "scope": scope,
            "kind": "categorical",
            "summaryRows": [
                {
                    "fieldValue": row["field_label"],
                    "recordCount": _safe_int(row["record_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                }
                for row in grouped.to_dict("records")
            ],
            "chartRows": [
                {
                    "fieldValue": row["field_label"],
                    "recordCount": _safe_int(row["record_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                }
                for row in grouped.to_dict("records")
            ],
        }

    def get_filters(self, filters: DashboardFilters) -> dict[str, Any]:
        full_df, filtered_df = self._load_fact(filters)
        brands = sorted(full_df["brand"].dropna().astype(str).unique().tolist()) if "brand" in full_df.columns else []
        source_datasets = _sorted_unique_source_datasets(full_df["source_dataset"]) if "source_dataset" in full_df.columns else []
        platforms = (
            sorted(full_df["platform"].dropna().astype(str).unique().tolist())
            if "platform" in full_df.columns
            else []
        )
        schema_versions = (
            sorted(full_df["schema_version"].dropna().astype(str).unique().tolist())
            if "schema_version" in full_df.columns
            else []
        )
        selected_snapshot_ids = (
            filtered_df["snapshot_id"].dropna().astype(str).unique().tolist()
            if "snapshot_id" in filtered_df.columns
            else []
        )
        return {
            "availableBrands": brands,
            "availableSourceDatasets": source_datasets,
            "availablePlatforms": platforms,
            "availableSchemaVersions": schema_versions,
            "selectedBrands": filters.brands,
            "selectedSourceDatasets": filters.source_datasets,
            "selectedPlatforms": filters.platforms,
            "selectedSchemaVersions": filters.schema_versions,
            "snapshotWindow": filters.snapshot_window,
            "dateRange": {
                "min": str(full_df["snapshot_date"].min()) if not full_df.empty and "snapshot_date" in full_df.columns else None,
                "max": str(full_df["snapshot_date"].max()) if not full_df.empty and "snapshot_date" in full_df.columns else None,
            },
            "selectedSnapshotIds": selected_snapshot_ids,
        }

    def get_thumbnail_snapshots(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        if filtered_fact_df.empty or "snapshot_id" not in filtered_fact_df.columns:
            return {"defaultSnapshotId": None, "snapshots": []}
        manifest_df = self.repository.load_table(filters.dataset, "image_manifest.parquet", "image_manifest")
        gallery_lookup = _thumbnail_gallery_lookup(manifest_df)
        snapshots = self._build_thumbnail_snapshot_summaries(filtered_fact_df, gallery_lookup)
        return {
            "defaultSnapshotId": snapshots[0]["snapshotId"] if snapshots else None,
            "snapshots": snapshots,
        }

    def _build_thumbnail_window_summary(
        self,
        summary_lookup: dict[str, dict[str, Any]],
        selected_snapshot_ids: list[str],
        scoped_fact: pd.DataFrame,
    ) -> dict[str, Any] | None:
        if not selected_snapshot_ids:
            return None
        selected_summaries = [summary_lookup[snapshot_id] for snapshot_id in selected_snapshot_ids if snapshot_id in summary_lookup]
        if not selected_summaries:
            return None
        if len(selected_summaries) == 1:
            return selected_summaries[0]

        selected_frame = scoped_fact[scoped_fact["snapshot_id"].astype(str).isin(selected_snapshot_ids)].copy()
        valid_brands = selected_frame["brand"].dropna().astype(str) if "brand" in selected_frame.columns else pd.Series(dtype=str)
        selected_frame["thumbnail_category_label"] = _normalize_category_series(
            selected_frame,
            ("category_label_l3", "category_l3", "category_label_l2", "category_l2", "category_label_l1", "category_l1"),
        )
        valid_categories = (
            selected_frame.loc[
                selected_frame["thumbnail_category_label"].notna() & ~_is_unclassified_series(selected_frame["thumbnail_category_label"]),
                "thumbnail_category_label",
            ].astype(str)
        )
        price_series = pd.to_numeric(selected_frame["price"], errors="coerce").dropna() if "price" in selected_frame.columns else pd.Series(dtype=float)

        first_summary = selected_summaries[0]
        last_summary = selected_summaries[-1]
        label = f'{last_summary.get("label")} ~ {first_summary.get("label")}'
        return {
            "snapshotId": f'{last_summary.get("snapshotId")}..{first_summary.get("snapshotId")}',
            "label": label,
            "snapshotCount": len(selected_snapshot_ids),
            "startSnapshotId": last_summary.get("snapshotId"),
            "endSnapshotId": first_summary.get("snapshotId"),
            "crawlDatetime": first_summary.get("crawlDatetime"),
            "snapshotDate": None,
            "snapshotTime": None,
            "recordCount": int(len(selected_frame)),
            "productCount": int(selected_frame["product_id"].nunique()) if "product_id" in selected_frame.columns else int(len(selected_frame)),
            "brandCount": int(valid_brands.nunique()),
            "avgPrice": _safe_float(price_series.mean()) if not price_series.empty else None,
            "mainImageCoveragePct": round(float(np.mean([row.get("mainImageCoveragePct") or 0 for row in selected_summaries])), 2),
            "detailInfoCoveragePct": round(float(np.mean([row.get("detailInfoCoveragePct") or 0 for row in selected_summaries])), 2),
            "categoryCoveragePct": round(float(np.mean([row.get("categoryCoveragePct") or 0 for row in selected_summaries])), 2),
            "topBrand": valid_brands.value_counts().index[0] if not valid_brands.empty else None,
            "topCategory": valid_categories.value_counts().index[0] if not valid_categories.empty else None,
        }

    def get_thumbnail_records(
        self,
        filters: DashboardFilters,
        snapshot_id: str | None,
        start_snapshot_id: str | None = None,
        end_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        empty_payload = {
            "windowMode": "point",
            "selectedSnapshotId": None,
            "selectedSnapshotIds": [],
            "snapshotSummary": None,
            "shapeLegend": _thumbnail_shape_legend(),
            "rows": [],
        }
        if filtered_fact_df.empty or "snapshot_id" not in filtered_fact_df.columns:
            return empty_payload

        scoped_fact = filtered_fact_df.copy()
        scoped_fact["snapshot_id"] = scoped_fact["snapshot_id"].astype(str)
        if "crawl_datetime" in scoped_fact.columns:
            scoped_fact["crawl_datetime"] = pd.to_datetime(scoped_fact["crawl_datetime"], errors="coerce")
        if "product_id" in scoped_fact.columns:
            scoped_fact["product_id"] = scoped_fact["product_id"].astype(str)
        if {"snapshot_id", "product_id"}.issubset(scoped_fact.columns):
            scoped_fact = scoped_fact.drop_duplicates(subset=["snapshot_id", "product_id"], keep="last").reset_index(drop=True)

        manifest_df = self.repository.load_table(filters.dataset, "image_manifest.parquet", "image_manifest")
        gallery_lookup = _thumbnail_gallery_lookup(manifest_df)
        snapshot_summaries = self._build_thumbnail_snapshot_summaries(scoped_fact, gallery_lookup)
        if not snapshot_summaries:
            return empty_payload

        summary_lookup = {str(row["snapshotId"]): row for row in snapshot_summaries}
        ordered_snapshot_ids = [str(row["snapshotId"]) for row in snapshot_summaries]

        selected_snapshot_ids: list[str]
        window_mode: str
        selected_snapshot_id = str(snapshot_id) if snapshot_id else None
        selected_start_snapshot_id = str(start_snapshot_id) if start_snapshot_id else None
        selected_end_snapshot_id = str(end_snapshot_id) if end_snapshot_id else None

        if selected_start_snapshot_id and selected_end_snapshot_id:
            if selected_start_snapshot_id not in summary_lookup:
                selected_start_snapshot_id = ordered_snapshot_ids[-1]
            if selected_end_snapshot_id not in summary_lookup:
                selected_end_snapshot_id = ordered_snapshot_ids[0]
            left_index = ordered_snapshot_ids.index(selected_start_snapshot_id)
            right_index = ordered_snapshot_ids.index(selected_end_snapshot_id)
            low = min(left_index, right_index)
            high = max(left_index, right_index)
            selected_snapshot_ids = ordered_snapshot_ids[low : high + 1]
            window_mode = "range"
        else:
            if not selected_snapshot_id or selected_snapshot_id not in summary_lookup:
                selected_snapshot_id = ordered_snapshot_ids[0]
            selected_snapshot_ids = [selected_snapshot_id]
            window_mode = "point"

        snapshot_df = scoped_fact[scoped_fact["snapshot_id"].astype(str).isin(selected_snapshot_ids)].copy()
        if snapshot_df.empty:
            return empty_payload

        raw_df = self._apply_raw_filters(self.repository.load_raw_snapshot_products(filters.dataset), filters)
        raw_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if not raw_df.empty and {"snapshot_id", "product_id"}.issubset(raw_df.columns):
            raw_df = raw_df.copy()
            raw_df["snapshot_id"] = raw_df["snapshot_id"].astype(str)
            raw_df["product_id"] = raw_df["product_id"].astype(str)
            raw_df = raw_df[raw_df["snapshot_id"].isin(selected_snapshot_ids)]
            for raw_row in raw_df.to_dict("records"):
                raw_lookup[(str(raw_row.get("snapshot_id")), str(raw_row.get("product_id")))] = raw_row

        sort_columns = [column for column in ("crawl_datetime", "snapshot_id", "rank", "product_id") if column in snapshot_df.columns]
        if sort_columns:
            ascending = [True for _ in sort_columns]
            snapshot_df = snapshot_df.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)

        rows: list[dict[str, Any]] = []
        for row in snapshot_df.to_dict("records"):
            sid = str(row.get("snapshot_id"))
            product_id = str(row.get("product_id"))
            raw_row = raw_lookup.get((sid, product_id), {})
            info_map = _safe_json_loads(raw_row.get("raw_info_map_json"))
            category_raw = _safe_json_loads(raw_row.get("raw_category_json"))
            gallery = gallery_lookup.get((sid, product_id), {})
            images = gallery.get("images", []) if isinstance(gallery, dict) else []
            category_path = _category_path_from_sources(row, category_raw)
            detail_info_rows = _mapping_rows(info_map, limit=18)
            detail_info_preview = detail_info_rows[:4]
            tags = _split_joined_tags(row.get("tags_joined"))
            image_count = int(gallery.get("imageCount") or 0) if isinstance(gallery, dict) else 0
            field_cells = [
                {
                    "key": "rank",
                    "label": "순위",
                    "shape": "number",
                    "grain": "record",
                    "value": _safe_int(row.get("rank")),
                },
                {
                    "key": "price",
                    "label": "가격",
                    "shape": "number",
                    "grain": "record",
                    "value": _safe_float(row.get("price")),
                },
                {
                    "key": "category",
                    "label": "카테고리",
                    "shape": "list",
                    "grain": "product",
                    "value": category_path,
                },
                {
                    "key": "detail_info",
                    "label": "고시정보",
                    "shape": "kv",
                    "grain": "product",
                    "value": f"{len(detail_info_rows)}개 필드" if detail_info_rows else "없음",
                },
                {
                    "key": "images",
                    "label": "이미지",
                    "shape": "image",
                    "grain": "product",
                    "value": f"{image_count}장" if image_count else "없음",
                },
            ]
            rows.append(
                {
                    "snapshotId": sid,
                    "snapshotLabel": _snapshot_display_label(
                        sid,
                        row.get("crawl_datetime"),
                        row.get("snapshot_date"),
                        row.get("snapshot_time"),
                    ),
                    "snapshotDate": _optional_label(row.get("snapshot_date")),
                    "snapshotTime": _optional_label(row.get("snapshot_time")),
                    "crawlDatetime": row.get("crawl_datetime").isoformat()
                    if isinstance(row.get("crawl_datetime"), pd.Timestamp) and not pd.isna(row.get("crawl_datetime"))
                    else None,
                    "productId": product_id,
                    "brand": _optional_label(row.get("brand")),
                    "name": _optional_label(row.get("name")) or product_id,
                    "rank": _safe_int(row.get("rank")),
                    "price": _safe_float(row.get("price")),
                    "discountPct": _safe_float(row.get("discount_pct")),
                    "categoryLabel": category_path[-1] if category_path else None,
                    "categoryPath": category_path,
                    "tags": tags,
                    "mainImagePath": _optional_fs_path(gallery.get("mainImagePath")) if isinstance(gallery, dict) else None,
                    "mainImageSource": _optional_label(gallery.get("mainImageSource")) if isinstance(gallery, dict) else None,
                    "hasMainImage": bool(gallery.get("hasMainImage")) if isinstance(gallery, dict) else False,
                    "hasExplicitMainImage": bool(gallery.get("hasExplicitMainImage")) if isinstance(gallery, dict) else False,
                    "imageCount": image_count,
                    "images": images,
                    "detailInfoCount": len(detail_info_rows),
                    "detailInfoPreview": detail_info_preview,
                    "detailInfoRows": detail_info_rows,
                    "categorySource": _optional_label(row.get("category_source")),
                    "categoryStatus": _optional_label(row.get("category_ingest_status")),
                    "productUrl": _optional_label(row.get("product_url")),
                    "sourceDataset": _optional_label(row.get("source_dataset")),
                    "platform": _optional_label(row.get("platform")),
                    "schemaVersion": _optional_label(row.get("schema_version")),
                    "fieldCells": field_cells,
                    "rawCategoryRows": _mapping_rows(category_raw, limit=12),
                }
            )

        selected_summary = self._build_thumbnail_window_summary(summary_lookup, selected_snapshot_ids, snapshot_df)
        return {
            "windowMode": window_mode,
            "selectedSnapshotId": selected_snapshot_ids[0] if len(selected_snapshot_ids) == 1 else None,
            "selectedSnapshotIds": selected_snapshot_ids,
            "snapshotSummary": selected_summary,
            "shapeLegend": _thumbnail_shape_legend(),
            "rows": _records(pd.DataFrame(rows)),
        }

    def get_overview_kpis(self, filters: DashboardFilters) -> dict[str, Any]:
        full_df, filtered_df = self._load_fact(filters)
        kpi = self.repository.load_json(filters.dataset, "kpi_summary.json")
        filtered_source_datasets: list[str] = []
        if not filtered_df.empty and "source_dataset" in filtered_df.columns:
            filtered_source_datasets = _sorted_unique_source_datasets(filtered_df["source_dataset"])
        return {
            "title": "Silhouette 분석 대시보드",
            "subtitle": "마케팅·커머스 스냅샷 데이터 기반 인사이트",
            "full": {
                "snapshotCount": int(full_df["snapshot_id"].nunique()) if not full_df.empty else 0,
                "recordCount": int(len(full_df)),
                "productCount": int(full_df["product_id"].nunique()) if not full_df.empty else 0,
                "brandCount": int(full_df["brand"].nunique()) if not full_df.empty else 0,
            },
            "filtered": {
                "snapshotCount": int(filtered_df["snapshot_id"].nunique()) if not filtered_df.empty else 0,
                "recordCount": int(len(filtered_df)),
                "productCount": int(filtered_df["product_id"].nunique()) if not filtered_df.empty else 0,
                "brandCount": int(filtered_df["brand"].nunique()) if not filtered_df.empty else 0,
            },
            "dateRange": kpi.get("date_range", {}),
            "filteredSourceDatasets": filtered_source_datasets,
        }

    def get_overview_dataset_profile(self, filters: DashboardFilters) -> dict[str, Any]:
        full_df, filtered_df = self._load_fact(filters)
        full_latest = _latest_by_product(full_df)
        filtered_latest = _latest_by_product(filtered_df)

        def safe_ratio(numerator: int, denominator: int) -> float | None:
            if denominator <= 0:
                return None
            return round((numerator / denominator) * 100.0, 2)

        full_product_count = int(full_df["product_id"].nunique()) if not full_df.empty and "product_id" in full_df.columns else 0
        filtered_product_count = (
            int(filtered_df["product_id"].nunique()) if not filtered_df.empty and "product_id" in filtered_df.columns else 0
        )
        full_repeated_count = (
            int((full_df.groupby("product_id")["snapshot_id"].nunique() > 1).sum())
            if not full_df.empty and {"product_id", "snapshot_id"}.issubset(set(full_df.columns))
            else 0
        )
        filtered_repeated_count = (
            int((filtered_df.groupby("product_id")["snapshot_id"].nunique() > 1).sum())
            if not filtered_df.empty and {"product_id", "snapshot_id"}.issubset(set(filtered_df.columns))
            else 0
        )

        profile_rows = [
            {
                "metric": "관측 행(레코드)",
                "fullValue": _safe_int(len(full_df)),
                "filteredValue": _safe_int(len(filtered_df)),
                "scope": "전체 / 현재 필터",
                "whyItMatters": "수집 시점(스냅샷)마다 상품 하나당 한 행으로 쌓인 관측 단위입니다. 표본 전체의 크기를 나타냅니다.",
            },
            {
                "metric": "고유 상품",
                "fullValue": _safe_int(full_product_count),
                "filteredValue": _safe_int(filtered_product_count),
                "scope": "전체 / 현재 필터",
                "whyItMatters": "같은 상품을 여러 시점에서 반복 관측해도, 상품 식별자(product_id)로 하나로 묶은 단위입니다.",
            },
            {
                "metric": "브랜드",
                "fullValue": _safe_int(full_df["brand"].nunique()) if "brand" in full_df.columns and not full_df.empty else 0,
                "filteredValue": _safe_int(filtered_df["brand"].nunique()) if "brand" in filtered_df.columns and not filtered_df.empty else 0,
                "scope": "전체 / 현재 필터",
                "whyItMatters": "이번 데이터에 등장한 브랜드 라인업이 얼마나 넓게 퍼져 있는지 보는 기준입니다.",
            },
            {
                "metric": "스냅샷(수집 회차)",
                "fullValue": _safe_int(full_df["snapshot_id"].nunique()) if "snapshot_id" in full_df.columns and not full_df.empty else 0,
                "filteredValue": _safe_int(filtered_df["snapshot_id"].nunique()) if "snapshot_id" in filtered_df.columns and not filtered_df.empty else 0,
                "scope": "전체 / 현재 필터",
                "whyItMatters": "수집이 실행된 시점(회차)입니다. 시간축으로 변화를 볼 때의 간격이 됩니다.",
            },
            {
                "metric": "상품당 평균 관측 행",
                "fullValue": _safe_float((len(full_df) / full_product_count) if full_product_count else None),
                "filteredValue": _safe_float((len(filtered_df) / filtered_product_count) if filtered_product_count else None),
                "scope": "전체 / 현재 필터",
                "whyItMatters": "한 상품이 스냅샷마다 평균 몇 번씩 행으로 등장했는지 나타냅니다.",
            },
            {
                "metric": "반복 관측 상품 비율",
                "fullValue": safe_ratio(full_repeated_count, full_product_count),
                "filteredValue": safe_ratio(filtered_repeated_count, filtered_product_count),
                "scope": "전체 / 현재 필터",
                "whyItMatters": "두 번 이상 관측된 상품이 전체 고유 상품 중 얼마나 되는지 보여, 시계열 비교에 쓸 수 있는 비중을 가늠합니다.",
            },
            {
                "metric": "상품 상세 스펙 반영률",
                "fullValue": safe_ratio(
                    int(full_latest["product_info_exists"].fillna(False).astype(bool).sum()) if "product_info_exists" in full_latest.columns else 0,
                    len(full_latest),
                ),
                "filteredValue": safe_ratio(
                    int(filtered_latest["product_info_exists"].fillna(False).astype(bool).sum()) if "product_info_exists" in filtered_latest.columns else 0,
                    len(filtered_latest),
                ),
                "scope": "고유 상품 기준",
                "whyItMatters": "상품 페이지의 표 형태 상세 속성이 채워져, 속성 분석 탭에서 활용하기 좋은 상품이 차지하는 비율입니다.",
            },
            {
                "metric": "텍스트 추출(OCR) 반영률",
                "fullValue": safe_ratio(
                    int(full_latest["ocr_has_data"].fillna(False).astype(bool).sum()) if "ocr_has_data" in full_latest.columns else 0,
                    len(full_latest),
                ),
                "filteredValue": safe_ratio(
                    int(filtered_latest["ocr_has_data"].fillna(False).astype(bool).sum()) if "ocr_has_data" in filtered_latest.columns else 0,
                    len(filtered_latest),
                ),
                "scope": "고유 상품 기준",
                "whyItMatters": "이미지에서 글자를 뽑아 텍스트 분석에 쓸 수 있는 상품이 차지하는 비율입니다.",
            },
            {
                "metric": "영업소재지 반영률",
                "fullValue": safe_ratio(
                    int((full_latest.get("business_address", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != "").sum()),
                    len(full_latest),
                ),
                "filteredValue": safe_ratio(
                    int((filtered_latest.get("business_address", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != "").sum()),
                    len(filtered_latest),
                ),
                "scope": "고유 상품 기준",
                "whyItMatters": "지도·공간 탭에서 주소를 좌표로 옮길 때 출발점이 되는 텍스트가 있는 상품 비율입니다.",
            },
            {
                "metric": "행정구(구·시군구) 식별률",
                "fullValue": safe_ratio(
                    int((full_latest.get("business_district", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != "").sum()),
                    len(full_latest),
                ),
                "filteredValue": safe_ratio(
                    int((filtered_latest.get("business_district", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != "").sum()),
                    len(filtered_latest),
                ),
                "scope": "고유 상품 기준",
                "whyItMatters": "구·시군구 단위로 묶어 볼 때 주소 파싱이 안정적으로 된 상품 비율입니다.",
            },
            {
                "metric": "법정동 단위 식별률",
                "fullValue": safe_ratio(
                    int((full_latest.get("business_dong", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != "").sum()),
                    len(full_latest),
                ),
                "filteredValue": safe_ratio(
                    int((filtered_latest.get("business_dong", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != "").sum()),
                    len(filtered_latest),
                ),
                "scope": "고유 상품 기준",
                "whyItMatters": "동 단위까지 세밀하게 위치를 나눠 볼 때 쓸 수 있는 상품 비율입니다.",
            },
        ]

        grain_rows = [
            {
                "concept": "관측 행(레코드)",
                "definition": "한 수집 시점(스냅샷)에서 한 상품에 대응하는 한 줄입니다.",
                "examples": "rank, price, discount_pct, crawl_datetime",
                "howToRead": "순위·가격처럼 시점마다 바뀌는 값은 이 행 단위로 읽습니다.",
            },
            {
                "concept": "고유 상품",
                "definition": "상품 식별자(product_id)로 중복을 합친 한 개의 상품 단위입니다.",
                "examples": "brand, name, material, origin_country",
                "howToRead": "속성이 얼마나 채워졌는지, 브랜드 구성은 어떤지 같은 질문은 상품 단위가 맞습니다.",
            },
            {
                "concept": "파생 지표",
                "definition": "페이지에서 바로 읽은 값이 아니라, 여러 시점을 비교해 만든 계산 결과입니다.",
                "examples": "rank_velocity, rank_acceleration, momentum_score, stability_score",
                "howToRead": "원래 찍힌 값과 같은 종류의 숫자로 보지 말고, 각 탭에서 정의를 함께 확인하는 것이 안전합니다.",
            },
            {
                "concept": "가격대 기준",
                "definition": "실판가를 나눈 구간과, 추정 정가를 나눈 구간이 함께 있습니다.",
                "examples": "price_band, estimatedOriginalPrice, priceBand",
                "howToRead": "어떤 가격 축을 보고 있는지 먼저 고르면, 밴드끼리 섞여 비교하는 실수를 줄일 수 있습니다.",
            },
        ]
        return {"profileRows": profile_rows, "grainRows": grain_rows}

    def get_overview_schema(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {"rows": []}
        rows: list[dict[str, Any]] = []
        for column in filtered_df.columns:
            series = filtered_df[column]
            non_missing = series.dropna()
            metadata = COLUMN_METADATA.get(column, {})
            rows.append(
                {
                    "column": column,
                    "dtype": _dtype_label(series),
                    "group": metadata.get("group", _infer_column_group(column)),
                    "semanticType": _semantic_type_label(column, series),
                    "grain": metadata.get("grain", _column_grain(column)),
                    "missingRatePct": round(float(series.isna().mean() * 100.0), 2) if len(series) else None,
                    "distinctCount": int(non_missing.astype(str).nunique()) if not non_missing.empty else 0,
                    "example": _example_value(series),
                    "meaning": metadata.get("meaning", "현재 데이터셋에 존재하는 컬럼입니다."),
                    "note": metadata.get("note", ""),
                }
            )
        rows.sort(key=lambda item: (str(item["group"]), str(item["column"])))
        return {"rows": rows}

    def get_overview_caveats(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        caveat_rows = [
            {
                "topic": "관측 단위",
                "rule": "기본 fact는 한 시점의 한 상품 관측 레코드입니다.",
                "impact": "관측 행과 고유 상품을 혼동하면 표본 규모를 잘못 해석할 수 있습니다.",
                "recommendation": "시계열은 행(레코드) 기준, 속성 채움률과 분포는 고유 상품 기준으로 함께 보세요.",
            },
            {
                "topic": "가격대 기준 혼재",
                "rule": "price_band는 판매가 기준이고, 가격 탭 주요 밴드는 추정 정가 기준입니다.",
                "impact": "같은 가격대라는 표현이 화면마다 다른 의미를 가질 수 있습니다.",
                "recommendation": "가격 해석 전 어떤 밴드 체계인지 제목과 설명을 먼저 확인하세요.",
            },
            {
                "topic": "product_info 범위",
                "rule": "현재는 product_info.csv 전체가 아니라 일부 승격 컬럼만 fact에 보존됩니다.",
                "impact": "속성 분석 결과를 전체 상품 정보표 전수 분석처럼 읽으면 안 됩니다.",
                "recommendation": "coverage와 컬럼 사전을 함께 보고, 없는 속성은 해석 대상에서 제외하세요.",
            },
            {
                "topic": "파생 지표",
                "rule": "momentum_score, rank_velocity, stability_score는 계산된 지표입니다.",
                "impact": "원천값처럼 해석하면 계산 규칙이 결과에 준 영향이 가려집니다.",
                "recommendation": "모멘텀 탭에서 입력값 분포와 계산 샘플을 함께 확인하세요.",
            },
            {
                "topic": "공간 데이터",
                "rule": "business_district, business_dong은 영업소재지 원문을 규칙 기반으로 파싱한 결과입니다.",
                "impact": "주소 품질이 낮으면 구/동 파싱이 비거나 부정확할 수 있습니다.",
                "recommendation": "공간 분포는 상세 주소보다 구/동 집계 projection으로 읽는 것이 안전합니다.",
            },
            {
                "topic": "지도 좌표",
                "rule": "지도 포인트는 무료 OSM 지오코딩 결과를 캐시한 근사 좌표입니다.",
                "impact": "지도는 내비게이션용 정밀 좌표가 아니라 지역 분포 해석용 표현입니다.",
                "recommendation": "브랜드·고유 상품·평균 순위 같은 집계와 함께 공간 패턴을 읽으세요.",
            },
        ]
        if "product_info_exists" in filtered_df.columns:
            missing_rate = round(float(filtered_df["product_info_exists"].fillna(False).astype(bool).mean() * 100.0), 2) if len(filtered_df) else None
            caveat_rows.append(
                {
                    "topic": "현재 필터 coverage",
                    "rule": f"현재 필터 레코드 기준 product_info 존재율은 {missing_rate if missing_rate is not None else '-'}% 입니다.",
                    "impact": "필터에 따라 속성 분석의 대표성이 달라질 수 있습니다.",
                    "recommendation": "필요하면 브랜드/기간 필터를 넓혀 coverage 변화를 함께 확인하세요.",
                }
            )
        return {"rows": caveat_rows}

    def get_price_distribution(self, filters: DashboardFilters, category: str | None = None) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "priceBandDistribution": [],
                "discountBandDistribution": [],
                "priceBandPerformance": [],
                "priceBandCategoryHeatmap": [],
                "summaryRows": [],
            }

        category_name = (category or "").strip()
        text_features_df = self._load_text_features(filters, filtered_df)
        fusion = self._run_dynamic_fusion(filters, filtered_df, text_features_df)
        product_profile_df = fusion["product_profile"].copy()
        if not product_profile_df.empty:
            if "name_item" in product_profile_df.columns:
                product_profile_df["nameItem"] = product_profile_df["name_item"].fillna("").astype(str).str.strip()
            else:
                product_profile_df["nameItem"] = "unknown"
            product_profile_df["nameItem"] = product_profile_df["nameItem"].replace(
                {"": "unknown", "nan": "unknown", "None": "unknown"}
            )

        scoped_df = filtered_df
        scoped_product_ids: set[str] | None = None
        if category_name and not product_profile_df.empty and "product_id" in product_profile_df.columns:
            matched_profile = product_profile_df[product_profile_df["nameItem"].str.lower() == category_name.lower()].copy()
            scoped_product_ids = {str(value) for value in matched_profile["product_id"].dropna().astype(str).tolist()}
            if scoped_product_ids:
                scoped_df = filtered_df[filtered_df["product_id"].astype(str).isin(scoped_product_ids)].copy()
            else:
                scoped_df = filtered_df.iloc[0:0].copy()

        if scoped_df.empty:
            return {
                "priceBandDistribution": [],
                "discountBandDistribution": [],
                "priceBandPerformance": [],
                "priceBandCategoryHeatmap": [],
                "summaryRows": [],
            }

        price_df = scoped_df.dropna(subset=["price"]).copy()
        price_df["estimatedOriginalPrice"] = price_df.apply(
            lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
            axis=1,
        )
        price_df["priceBand"] = _price_band_from_original_price(price_df["estimatedOriginalPrice"])
        price_df["priceBand"] = price_df["priceBand"].astype(str).replace("nan", "미분류")
        price_band_distribution = (
            price_df.groupby("priceBand", as_index=False)
            .agg(
                record_count=("product_id", "count"),
                avg_discount_pct=("discount_pct", "mean"),
                avg_rank=("rank", "mean"),
                avg_momentum_score=("momentum_score", "mean"),
            )
            .sort_values("record_count", ascending=False)
        )

        discount_df = scoped_df.dropna(subset=["discount_pct"]).copy()
        discount_bins = [-0.1, 0.0001, 10, 20, 30, 40, 50, 60, 70, 1000]
        discount_labels = ["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"]
        discount_df["discountBand"] = pd.cut(
            discount_df["discount_pct"],
            bins=discount_bins,
            labels=discount_labels,
            include_lowest=True,
        )
        discount_band_distribution = (
            discount_df.dropna(subset=["discountBand"])
            .groupby("discountBand", observed=False, as_index=False)
            .agg(
                record_count=("product_id", "count"),
                avg_discount_pct=("discount_pct", "mean"),
                avg_rank=("rank", "mean"),
            )
        )
        discount_band_distribution["discountBand"] = discount_band_distribution["discountBand"].astype(str)

        price_band_performance = (
            price_df.groupby("priceBand", as_index=False)
            .agg(
                avg_rank=("rank", "mean"),
                avg_rank_velocity=("rank_velocity", "mean"),
                avg_discount_pct=("discount_pct", "mean"),
                avg_momentum_score=("momentum_score", "mean"),
                record_count=("product_id", "count"),
            )
            .sort_values("avg_rank", ascending=True)
        )

        if not product_profile_df.empty:
            scoped_profile_df = product_profile_df.copy()
            if scoped_product_ids is not None and "product_id" in scoped_profile_df.columns:
                scoped_profile_df = scoped_profile_df[scoped_profile_df["product_id"].astype(str).isin(scoped_product_ids)]
            scoped_profile_df = scoped_profile_df[scoped_profile_df["nameItem"].str.lower() != "unknown"]
            scoped_profile_df["estimatedOriginalPrice"] = scoped_profile_df.apply(
                lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
                axis=1,
            )
            scoped_profile_df["priceBand"] = _price_band_from_original_price(scoped_profile_df["estimatedOriginalPrice"])
            scoped_profile_df["priceBand"] = scoped_profile_df["priceBand"].astype(str).replace("nan", "미분류")
            top_categories = (
                scoped_profile_df["nameItem"].value_counts(dropna=False).head(12).index.tolist()
            )
            category_heatmap_df = (
                scoped_profile_df[scoped_profile_df["nameItem"].isin(top_categories)]
                .groupby(["priceBand", "nameItem"], as_index=False)
                .agg(count=("product_id", "count"))
            )
        else:
            category_heatmap_df = pd.DataFrame()

        summary_rows = [
            {
                "metric": "평균 판매가",
                "value": _safe_float(price_df["price"].mean()),
                "unit": "KRW",
            },
            {
                "metric": "중앙 판매가",
                "value": _safe_float(price_df["price"].median()),
                "unit": "KRW",
            },
            {
                "metric": "추정 평균 정가",
                "value": _safe_float(price_df["estimatedOriginalPrice"].mean()),
                "unit": "KRW",
            },
            {
                "metric": "평균 할인율",
                "value": _safe_float(scoped_df["discount_pct"].mean()),
                "unit": "%",
            },
        ]

        return {
            "priceBandDistribution": [
                {
                    "priceBand": row["priceBand"],
                    "recordCount": _safe_int(row["record_count"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                }
                for row in price_band_distribution.to_dict("records")
            ],
            "discountBandDistribution": [
                {
                    "discountBand": row["discountBand"],
                    "recordCount": _safe_int(row["record_count"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                }
                for row in discount_band_distribution.to_dict("records")
            ],
            "priceBandPerformance": [
                {
                    "priceBand": row["priceBand"],
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgRankVelocity": _safe_float(row["avg_rank_velocity"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in price_band_performance.to_dict("records")
            ],
            "priceBandCategoryHeatmap": [
                {
                    "priceBand": row["priceBand"],
                    "nameItem": row["nameItem"],
                    "count": _safe_int(row["count"]),
                }
                for row in category_heatmap_df.to_dict("records")
            ],
            "summaryRows": summary_rows,
        }

    def get_overview_momentum(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {"topMomentum": [], "brandMomentum": []}
        filtered_df = _ensure_rank_energy_momentum(filtered_df)

        latest = (
            filtered_df.sort_values("crawl_datetime")
            .groupby("product_id", as_index=False)
            .tail(1)
            .sort_values(["momentum_score", "rank_velocity"], ascending=False)
        )
        top_momentum = latest[
            [
                "product_id",
                "brand",
                "name",
                "rank",
                "rank_velocity",
                "rank_acceleration",
                "momentum_score",
                "discount_pct",
                "price",
            ]
        ].head(20)
        brand_stats = (
            filtered_df.groupby("brand", as_index=False)
            .agg(
                avg_rank=("rank", "mean"),
                avg_velocity=("rank_velocity", "mean"),
                avg_momentum=("momentum_score", "mean"),
                record_count=("product_id", "count"),
            )
            .sort_values("avg_momentum", ascending=False)
            .head(20)
        )
        return {
            "topMomentum": _records(top_momentum),
            "brandMomentum": [
                {
                    "brand": row["brand"],
                    "avgMomentum": _safe_float(row["avg_momentum"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgVelocity": _safe_float(row["avg_velocity"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in brand_stats.to_dict("records")
            ],
        }

    def get_discount_reaction(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "discountChangeScatter": [],
                "discountChangeSummary": [],
                "discountLevelHeatmap": [],
                "discountLevelSummary": [],
            }

        discount_change = filtered_df.dropna(subset=["discount_velocity", "rank_velocity", "price"]).copy()
        if not discount_change.empty:
            discount_change["changeBucket"] = "유지(±5%p)"
            discount_change.loc[discount_change["discount_velocity"] > 5, "changeBucket"] = "할인 증가"
            discount_change.loc[discount_change["discount_velocity"] < -5, "changeBucket"] = "할인 감소"
            discount_change = discount_change.sample(min(len(discount_change), 2000), random_state=42)
            discount_change_summary = (
                discount_change.groupby("changeBucket", as_index=False)
                .agg(
                    avg_rank_velocity=("rank_velocity", "mean"),
                    median_rank_velocity=("rank_velocity", "median"),
                    record_count=("rank_velocity", "count"),
                )
            )
            improvement_rate = (
                discount_change.assign(rank_improved=discount_change["rank_velocity"] > 0)
                .groupby("changeBucket", as_index=False)
                .agg(improvement_rate=("rank_improved", lambda values: float(values.mean() * 100.0)))
            )
            discount_change_summary = discount_change_summary.merge(improvement_rate, on="changeBucket", how="left")
        else:
            discount_change_summary = pd.DataFrame()

        discount_level = filtered_df.dropna(subset=["discount_pct", "rank_velocity", "price"]).copy()
        if not discount_level.empty:
            discount_bins = [-0.1, 10, 20, 30, 40, 50, 60, 70, 1000]
            discount_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"]
            rank_bins = [-1000, -20, -10, -5, 0, 5, 10, 20, 1000]
            rank_labels = ["-20 이하", "-20~-10", "-10~-5", "-5~0", "0~5", "5~10", "10~20", "20 이상"]
            discount_level["discountBin"] = pd.cut(
                discount_level["discount_pct"],
                bins=discount_bins,
                labels=discount_labels,
                include_lowest=True,
            )
            discount_level["rankVelocityBand"] = pd.cut(
                discount_level["rank_velocity"],
                bins=rank_bins,
                labels=rank_labels,
                include_lowest=True,
            )
            discount_level_heatmap = (
                discount_level.dropna(subset=["discountBin", "rankVelocityBand"])
                .groupby(["discountBin", "rankVelocityBand"], observed=False, as_index=False)
                .agg(count=("product_id", "count"))
            )
            discount_level_summary = (
                discount_level.dropna(subset=["discountBin"])
                .groupby("discountBin", observed=False, as_index=False)
                .agg(
                    avg_rank_velocity=("rank_velocity", "mean"),
                    median_rank_velocity=("rank_velocity", "median"),
                    record_count=("rank_velocity", "count"),
                )
            )
            discount_level_summary["discountBin"] = discount_level_summary["discountBin"].astype(str)
            discount_level_heatmap["discountBin"] = discount_level_heatmap["discountBin"].astype(str)
            discount_level_heatmap["rankVelocityBand"] = discount_level_heatmap["rankVelocityBand"].astype(str)
        else:
            discount_level_heatmap = pd.DataFrame()
            discount_level_summary = pd.DataFrame()

        return {
            "discountChangeScatter": [
                {
                    "discountVelocity": _safe_float(row["discount_velocity"]),
                    "rankVelocity": _safe_float(row["rank_velocity"]),
                    "bucket": row["changeBucket"],
                    "brand": row["brand"],
                    "name": row["name"],
                    "rank": _safe_int(row["rank"]),
                    "discountPct": _safe_float(row["discount_pct"]),
                    "discountPrev": _safe_float(row["discount_prev"]),
                    "price": _safe_int(row["price"]),
                }
                for row in discount_change.to_dict("records")
            ],
            "discountChangeSummary": [
                {
                    "changeBucket": row["changeBucket"],
                    "avgRankVelocity": _safe_float(row["avg_rank_velocity"]),
                    "medianRankVelocity": _safe_float(row["median_rank_velocity"]),
                    "recordCount": _safe_int(row["record_count"]),
                    "improvementRate": _safe_float(row["improvement_rate"]),
                }
                for row in discount_change_summary.to_dict("records")
            ],
            "discountLevelHeatmap": [
                {
                    "discountBin": row["discountBin"],
                    "rankVelocityBand": row["rankVelocityBand"],
                    "count": _safe_int(row["count"]),
                }
                for row in discount_level_heatmap.to_dict("records")
            ],
            "discountLevelSummary": [
                {
                    "discountBin": row["discountBin"],
                    "avgRankVelocity": _safe_float(row["avg_rank_velocity"]),
                    "medianRankVelocity": _safe_float(row["median_rank_velocity"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in discount_level_summary.to_dict("records")
            ],
        }

    def get_price_reaction(self, filters: DashboardFilters) -> dict[str, Any]:
        return self.get_discount_reaction(filters)

    def get_price_timeseries(
        self,
        filters: DashboardFilters,
        product_ids: list[str],
        category: str | None = None,
    ) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "priceBandDiscountSeries": [],
                "priceBandRankSeries": [],
                "topRankedProducts": [],
            }

        selected_product_ids: set[str] | None = None
        if product_ids:
            selected_product_ids = {str(product_id) for product_id in product_ids}

        category_name = (category or "").strip()
        if category_name:
            text_features_df = self._load_text_features(filters, filtered_df)
            fusion = self._run_dynamic_fusion(filters, filtered_df, text_features_df)
            product_profile_df = fusion["product_profile"].copy()
            if not product_profile_df.empty and "name_item" in product_profile_df.columns and "product_id" in product_profile_df.columns:
                normalized_name = product_profile_df["name_item"].fillna("").astype(str).str.strip()
                category_product_ids = {
                    str(value)
                    for value in product_profile_df[normalized_name.str.lower() == category_name.lower()]["product_id"]
                    .dropna()
                    .astype(str)
                    .tolist()
                }
            else:
                category_product_ids = set()
            if selected_product_ids is None:
                selected_product_ids = category_product_ids
            else:
                selected_product_ids = selected_product_ids.intersection(category_product_ids)

        price_df = filtered_df.copy()
        if selected_product_ids is not None:
            price_df = price_df[price_df["product_id"].astype(str).isin(selected_product_ids)].copy()
        if price_df.empty:
            return {
                "priceBandDiscountSeries": [],
                "priceBandRankSeries": [],
                "topRankedProducts": [],
            }
        price_df["estimated_original_price"] = price_df.apply(
            lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
            axis=1,
        )
        price_df["original_price_band"] = _price_band_from_original_price(price_df["estimated_original_price"])
        price_df["original_price_band"] = price_df["original_price_band"].astype(str).replace("nan", "미분류")

        price_band_daily = (
            price_df.dropna(subset=["snapshot_date"])
            .groupby(["snapshot_date", "original_price_band"], as_index=False)
            .agg(
                avg_discount_pct=("discount_pct", "mean"),
                avg_rank=("rank", "mean"),
                avg_price=("price", "mean"),
                avg_original_price=("estimated_original_price", "mean"),
                record_count=("product_id", "count"),
            )
            .sort_values("snapshot_date")
        )
        product_summary = (
            price_df.groupby("product_id", as_index=False)
            .agg(
                brand=("brand", "last"),
                name=("name", "last"),
                avg_rank=("rank", "mean"),
                best_rank=("rank", "min"),
                worst_rank=("rank", "max"),
                rank_std=("rank", "std"),
                avg_discount_pct=("discount_pct", "mean"),
                min_discount_pct=("discount_pct", "min"),
                max_discount_pct=("discount_pct", "max"),
                estimated_original_price=("estimated_original_price", "median"),
                avg_price=("price", "mean"),
                observation_count=("snapshot_id", "nunique"),
            )
        )
        product_summary["discount_range"] = product_summary["max_discount_pct"] - product_summary["min_discount_pct"]
        product_summary["original_price_band"] = _price_band_from_original_price(product_summary["estimated_original_price"])
        product_summary["original_price_band"] = product_summary["original_price_band"].astype(str).replace("nan", "미분류")
        top_ranked_products = (
            product_summary[product_summary["observation_count"] >= 3]
            .sort_values(["avg_rank", "rank_std", "observation_count"], ascending=[True, True, False])
            .head(20)
        )

        return {
            "priceBandDiscountSeries": [
                {
                    "snapshotDate": str(row["snapshot_date"]),
                    "priceBand": row["original_price_band"],
                    "value": _safe_float(row["avg_discount_pct"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in price_band_daily.to_dict("records")
            ]
            ,
            "priceBandRankSeries": [
                {
                    "snapshotDate": str(row["snapshot_date"]),
                    "priceBand": row["original_price_band"],
                    "value": _safe_float(row["avg_rank"]),
                    "recordCount": _safe_int(row["record_count"]),
                    "avgOriginalPrice": _safe_float(row["avg_original_price"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                }
                for row in price_band_daily.to_dict("records")
            ],
            "topRankedProducts": [
                {
                    "productId": str(row["product_id"]),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "avgRank": _safe_float(row.get("avg_rank")),
                    "bestRank": _safe_float(row.get("best_rank")),
                    "worstRank": _safe_float(row.get("worst_rank")),
                    "rankStd": _safe_float(row.get("rank_std")),
                    "avgDiscountPct": _safe_float(row.get("avg_discount_pct")),
                    "minDiscountPct": _safe_float(row.get("min_discount_pct")),
                    "maxDiscountPct": _safe_float(row.get("max_discount_pct")),
                    "discountRange": _safe_float(row.get("discount_range")),
                    "avgPrice": _safe_float(row.get("avg_price")),
                    "estimatedOriginalPrice": _safe_float(row.get("estimated_original_price")),
                    "originalPriceBand": row.get("original_price_band"),
                    "observationCount": _safe_int(row.get("observation_count")),
                }
                for row in top_ranked_products.to_dict("records")
            ],
        }

    def get_discount_effects(
        self,
        filters: DashboardFilters,
        *,
        velocity_threshold: float = 5.0,
        control_velocity_threshold: float = 3.0,
        pre_window: int = 7,
        post_window: int = 14,
        min_obs_per_side: int = 3,
        min_control_samples: int = 2,
        merge_window_days: int = 7,
    ) -> dict[str, Any]:
        empty_payload: dict[str, Any] = {
            "summary": {
                "eventCount": 0,
                "confidentEventCount": 0,
                "improvedCount": 0,
                "neutralCount": 0,
                "worsenedCount": 0,
                "improvementRate": None,
                "medianAbnormalRankDelta": None,
                "velocityThreshold": velocity_threshold,
                "controlVelocityThreshold": control_velocity_threshold,
                "preWindowDays": pre_window,
                "postWindowDays": post_window,
                "minObsPerSide": min_obs_per_side,
                "minControlSamples": min_control_samples,
            },
            "events": [],
            "eventStudyCurves": [],
            "effectScatter": [],
        }

        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return empty_payload

        required_cols = {"product_id", "snapshot_date", "rank", "discount_pct", "price"}
        if not required_cols.issubset(filtered_df.columns):
            return empty_payload

        df = filtered_df.dropna(subset=["product_id", "snapshot_date", "rank", "discount_pct"]).copy()
        if df.empty:
            return empty_payload
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["snapshot_date"])
        if "crawl_datetime" in df.columns:
            df["crawl_datetime"] = pd.to_datetime(df["crawl_datetime"], errors="coerce")
        else:
            df["crawl_datetime"] = pd.NaT
        if "snapshot_id" not in df.columns:
            df["snapshot_id"] = ""
        # Daily close rule: for each product-day, keep the latest snapshot.
        df = (
            df.sort_values(["product_id", "snapshot_date", "crawl_datetime", "snapshot_id"])
            .groupby(["product_id", "snapshot_date"], as_index=False, group_keys=False)
            .tail(1)
            .sort_values(["product_id", "snapshot_date"])
            .reset_index(drop=True)
        )

        grouped = df.groupby("product_id", group_keys=False)
        df["discount_prev_calc"] = grouped["discount_pct"].shift(1)
        df["discount_velocity_calc"] = df["discount_pct"] - df["discount_prev_calc"]
        df["estimated_original_price"] = df.apply(
            lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
            axis=1,
        )
        df["original_price_band"] = (
            _price_band_from_original_price(df["estimated_original_price"]).astype(str).replace("nan", "미분류")
        )

        velocity_mask = df["discount_velocity_calc"].abs() >= velocity_threshold
        events_df = df.loc[velocity_mask].copy()
        if events_df.empty:
            return empty_payload
        events_df["abs_velocity"] = events_df["discount_velocity_calc"].abs()

        deduped_events: list[dict[str, Any]] = []
        for product_id, group in events_df.groupby("product_id", sort=False):
            ordered = group.sort_values("snapshot_date")
            cluster: list[dict[str, Any]] = []
            last_date: pd.Timestamp | None = None
            for _, row in ordered.iterrows():
                row_dict = row.to_dict()
                current_date = row_dict["snapshot_date"]
                if (
                    last_date is not None
                    and (current_date - last_date).days <= merge_window_days
                    and cluster
                ):
                    cluster.append(row_dict)
                else:
                    if cluster:
                        deduped_events.append(max(cluster, key=lambda r: r["abs_velocity"]))
                    cluster = [row_dict]
                last_date = current_date
            if cluster:
                deduped_events.append(max(cluster, key=lambda r: r["abs_velocity"]))

        if not deduped_events:
            return empty_payload

        events_df = pd.DataFrame(deduped_events)

        product_groups: dict[str, pd.DataFrame] = {
            str(pid): group.sort_values("snapshot_date").reset_index(drop=True)
            for pid, group in df.groupby("product_id", sort=False)
        }
        band_groups: dict[str, pd.DataFrame] = {
            str(band): frame.copy()
            for band, frame in df.groupby("original_price_band", sort=False)
        }

        effects: list[dict[str, Any]] = []
        for _, event_row in events_df.iterrows():
            product_id = str(event_row["product_id"])
            event_date = event_row["snapshot_date"]
            timeline = product_groups.get(product_id)
            if timeline is None:
                continue
            pre_start = event_date - pd.Timedelta(days=pre_window)
            post_end = event_date + pd.Timedelta(days=post_window)
            pre = timeline[(timeline["snapshot_date"] < event_date) & (timeline["snapshot_date"] >= pre_start)]
            post = timeline[(timeline["snapshot_date"] > event_date) & (timeline["snapshot_date"] <= post_end)]
            if len(pre) < min_obs_per_side or len(post) < min_obs_per_side:
                continue
            rank_pre = float(pre["rank"].mean())
            rank_post = float(post["rank"].mean())
            rank_delta = rank_pre - rank_post

            band = str(event_row.get("original_price_band") or "미분류")
            band_frame = band_groups.get(band)
            control_deltas: list[float] = []
            if band_frame is not None:
                window_frame = band_frame[
                    (band_frame["snapshot_date"] >= pre_start) & (band_frame["snapshot_date"] <= post_end)
                ]
                for ctrl_id, ctrl in window_frame.groupby("product_id", sort=False):
                    if str(ctrl_id) == product_id:
                        continue
                    ctrl_pre = ctrl[(ctrl["snapshot_date"] < event_date) & (ctrl["snapshot_date"] >= pre_start)]
                    ctrl_post = ctrl[(ctrl["snapshot_date"] > event_date) & (ctrl["snapshot_date"] <= post_end)]
                    if len(ctrl_pre) < min_obs_per_side or len(ctrl_post) < min_obs_per_side:
                        continue
                    abs_velocity = ctrl["discount_velocity_calc"].abs()
                    if abs_velocity.fillna(0).ge(control_velocity_threshold).any():
                        continue
                    control_deltas.append(float(ctrl_pre["rank"].mean() - ctrl_post["rank"].mean()))

            has_confident_control = len(control_deltas) >= min_control_samples
            control_median: float | None = (
                float(np.median(control_deltas)) if has_confident_control else None
            )
            abnormal_delta: float | None = (
                rank_delta - control_median if control_median is not None else None
            )

            best_day_offset: int | None = None
            if not post.empty:
                best_idx = post["rank"].idxmin()
                best_day_offset = int((post.loc[best_idx, "snapshot_date"] - event_date).days)

            effects.append(
                {
                    "product_id": product_id,
                    "brand": event_row.get("brand"),
                    "name": event_row.get("name"),
                    "event_date": event_date,
                    "event_type": "increase" if event_row["discount_velocity_calc"] > 0 else "decrease",
                    "discount_pct": event_row.get("discount_pct"),
                    "discount_prev": event_row.get("discount_prev_calc"),
                    "discount_delta": event_row.get("discount_velocity_calc"),
                    "rank_pre": rank_pre,
                    "rank_post": rank_post,
                    "rank_delta": rank_delta,
                    "control_rank_delta": control_median,
                    "abnormal_rank_delta": abnormal_delta,
                    "control_sample_size": len(control_deltas),
                    "low_confidence": not has_confident_control,
                    "pre_obs": int(len(pre)),
                    "post_obs": int(len(post)),
                    "best_day_offset": best_day_offset,
                    "original_price_band": band,
                }
            )

        if not effects:
            return empty_payload

        effects_df = pd.DataFrame(effects)
        confident_df = effects_df[~effects_df["low_confidence"]]
        sorted_effects = effects_df.sort_values(
            ["low_confidence", "abnormal_rank_delta"],
            ascending=[True, False],
            na_position="last",
        )

        curve_rows: list[dict[str, Any]] = []
        for event_type in ("increase", "decrease"):
            type_events = effects_df[effects_df["event_type"] == event_type]
            if type_events.empty:
                continue
            for offset in range(-pre_window, post_window + 1):
                ranks: list[float] = []
                for _, ev in type_events.iterrows():
                    timeline = product_groups.get(str(ev["product_id"]))
                    if timeline is None:
                        continue
                    target_date = ev["event_date"] + pd.Timedelta(days=offset)
                    hit = timeline[timeline["snapshot_date"] == target_date]
                    if hit.empty:
                        continue
                    ranks.append(float(hit["rank"].mean()))
                if not ranks:
                    continue
                curve_rows.append(
                    {
                        "eventType": "할인 증가" if event_type == "increase" else "할인 감소",
                        "relativeDay": offset,
                        "avgRank": float(np.mean(ranks)),
                        "sampleSize": int(len(ranks)),
                    }
                )

        improvement_threshold = 1.0
        confident_abnormal = confident_df["abnormal_rank_delta"].dropna()
        improved_mask = confident_abnormal > improvement_threshold
        worsened_mask = confident_abnormal < -improvement_threshold
        neutral_mask = confident_abnormal.abs() <= improvement_threshold

        summary = {
            "eventCount": int(len(effects_df)),
            "confidentEventCount": int(len(confident_abnormal)),
            "improvedCount": int(improved_mask.sum()),
            "neutralCount": int(neutral_mask.sum()),
            "worsenedCount": int(worsened_mask.sum()),
            "improvementRate": float(improved_mask.mean()) if len(confident_abnormal) > 0 else None,
            "medianAbnormalRankDelta": float(confident_abnormal.median())
            if len(confident_abnormal) > 0
            else None,
            "velocityThreshold": velocity_threshold,
            "controlVelocityThreshold": control_velocity_threshold,
            "preWindowDays": pre_window,
            "postWindowDays": post_window,
            "minObsPerSide": min_obs_per_side,
            "minControlSamples": min_control_samples,
        }

        def _event_type_label(value: str) -> str:
            return "할인 증가" if value == "increase" else "할인 감소"

        events_payload = [
            {
                "productId": str(row["product_id"]),
                "brand": row.get("brand"),
                "name": row.get("name"),
                "eventDate": row["event_date"].date().isoformat(),
                "eventType": _event_type_label(row["event_type"]),
                "discountPct": _safe_float(row.get("discount_pct")),
                "discountPrev": _safe_float(row.get("discount_prev")),
                "discountDelta": _safe_float(row.get("discount_delta")),
                "rankPre": _safe_float(row.get("rank_pre")),
                "rankPost": _safe_float(row.get("rank_post")),
                "rankDelta": _safe_float(row.get("rank_delta")),
                "controlRankDelta": _safe_float(row.get("control_rank_delta")),
                "abnormalRankDelta": _safe_float(row.get("abnormal_rank_delta")),
                "controlSampleSize": _safe_int(row.get("control_sample_size")),
                "lowConfidence": bool(row.get("low_confidence", False)),
                "preObs": _safe_int(row.get("pre_obs")),
                "postObs": _safe_int(row.get("post_obs")),
                "bestDayOffset": _safe_int(row.get("best_day_offset")) if row.get("best_day_offset") is not None else None,
                "originalPriceBand": row.get("original_price_band"),
            }
            for _, row in sorted_effects.iterrows()
        ]

        scatter_payload = [
            {
                "productId": str(row["product_id"]),
                "name": row.get("name"),
                "brand": row.get("brand"),
                "eventDate": row["event_date"].date().isoformat(),
                "eventType": _event_type_label(row["event_type"]),
                "discountDelta": _safe_float(row.get("discount_delta")),
                "abnormalRankDelta": _safe_float(row.get("abnormal_rank_delta")),
                "rankDelta": _safe_float(row.get("rank_delta")),
                "originalPriceBand": row.get("original_price_band"),
                "controlSampleSize": _safe_int(row.get("control_sample_size")),
                "lowConfidence": bool(row.get("low_confidence", False)),
            }
            for _, row in effects_df.iterrows()
            if not bool(row.get("low_confidence", False))
        ]

        return {
            "summary": summary,
            "events": events_payload,
            "eventStudyCurves": curve_rows,
            "effectScatter": scatter_payload,
        }

    def get_product_discount_drilldown(
        self,
        filters: DashboardFilters,
        product_id: str,
        *,
        velocity_threshold: float = 5.0,
        merge_window_days: int = 7,
    ) -> dict[str, Any]:
        empty_payload: dict[str, Any] = {
            "product": None,
            "timeline": [],
            "events": [],
            "summary": {
                "velocityThreshold": velocity_threshold,
            },
        }
        normalized_id = str(product_id).strip()
        if not normalized_id:
            return empty_payload

        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty or "product_id" not in filtered_df.columns:
            return empty_payload

        df = filtered_df[filtered_df["product_id"].astype(str) == normalized_id].copy()
        if df.empty:
            return empty_payload
        df = df.dropna(subset=["snapshot_date", "rank", "discount_pct"])
        if df.empty:
            return empty_payload

        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["snapshot_date"])
        if "crawl_datetime" in df.columns:
            df["crawl_datetime"] = pd.to_datetime(df["crawl_datetime"], errors="coerce")
        else:
            df["crawl_datetime"] = pd.NaT
        if "snapshot_id" not in df.columns:
            df["snapshot_id"] = ""
        # Daily close rule: for each product-day, keep the latest snapshot.
        df = (
            df.sort_values(["snapshot_date", "crawl_datetime", "snapshot_id"])
            .groupby(["product_id", "snapshot_date"], as_index=False, group_keys=False)
            .tail(1)
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        df["discount_prev_calc"] = df["discount_pct"].shift(1)
        df["discount_velocity_calc"] = df["discount_pct"] - df["discount_prev_calc"]
        if "price" in df.columns:
            df["estimated_original_price"] = df.apply(
                lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
                axis=1,
            )
        else:
            df["estimated_original_price"] = pd.Series(dtype=float)
        df["original_price_band"] = (
            _price_band_from_original_price(df["estimated_original_price"]).astype(str).replace("nan", "미분류")
        )

        timeline_payload = [
            {
                "snapshotDate": row["snapshot_date"].date().isoformat(),
                "rank": _safe_float(row.get("rank")),
                "discountPct": _safe_float(row.get("discount_pct")),
                "price": _safe_float(row.get("price")),
                "estimatedOriginalPrice": _safe_float(row.get("estimated_original_price")),
            }
            for _, row in df.iterrows()
        ]

        velocity_mask = df["discount_velocity_calc"].abs() >= velocity_threshold
        events_df = df.loc[velocity_mask].copy()
        events_df["abs_velocity"] = events_df["discount_velocity_calc"].abs()
        events_df = events_df.sort_values("snapshot_date").reset_index(drop=True)

        deduped_events: list[dict[str, Any]] = []
        cluster: list[dict[str, Any]] = []
        last_date: pd.Timestamp | None = None
        for _, row in events_df.iterrows():
            row_dict = row.to_dict()
            current_date = row_dict["snapshot_date"]
            if last_date is not None and (current_date - last_date).days <= merge_window_days and cluster:
                cluster.append(row_dict)
            else:
                if cluster:
                    deduped_events.append(max(cluster, key=lambda r: r["abs_velocity"]))
                cluster = [row_dict]
            last_date = current_date
        if cluster:
            deduped_events.append(max(cluster, key=lambda r: r["abs_velocity"]))

        events_payload = [
            {
                "eventDate": row["snapshot_date"].date().isoformat(),
                "eventType": "할인 증가" if row["discount_velocity_calc"] > 0 else "할인 감소",
                "discountPct": _safe_float(row.get("discount_pct")),
                "discountPrev": _safe_float(row.get("discount_prev_calc")),
                "discountDelta": _safe_float(row.get("discount_velocity_calc")),
                "rank": _safe_float(row.get("rank")),
            }
            for row in deduped_events
        ]

        latest = df.iloc[-1]
        product_payload = {
            "productId": normalized_id,
            "brand": latest.get("brand"),
            "name": latest.get("name"),
            "originalPriceBand": latest.get("original_price_band"),
            "observationCount": int(len(df)),
            "firstObservedAt": df.iloc[0]["snapshot_date"].date().isoformat(),
            "lastObservedAt": df.iloc[-1]["snapshot_date"].date().isoformat(),
            "avgDiscountPct": _safe_float(df["discount_pct"].mean()),
            "minDiscountPct": _safe_float(df["discount_pct"].min()),
            "maxDiscountPct": _safe_float(df["discount_pct"].max()),
            "avgRank": _safe_float(df["rank"].mean()),
            "bestRank": _safe_float(df["rank"].min()),
            "worstRank": _safe_float(df["rank"].max()),
        }

        summary_payload = {
            "velocityThreshold": velocity_threshold,
            "eventCount": len(events_payload),
        }

        return {
            "product": product_payload,
            "timeline": timeline_payload,
            "events": events_payload,
            "summary": summary_payload,
        }

    def get_rank_trends(self, filters: DashboardFilters, product_ids: list[str]) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "availableProducts": [],
                "defaultProductIds": [],
                "selectedProductIds": product_ids,
                "series": [],
            }

        latest_products = (
            filtered_df.sort_values("crawl_datetime")
            .groupby("product_id", as_index=False)
            .tail(1)[["product_id", "rank", "crawl_datetime", "name", "brand"]]
            .rename(columns={"rank": "latest_rank"})
        )
        observation_counts = (
            filtered_df.groupby("product_id", as_index=False)
            .agg(observation_count=("snapshot_id", "nunique"))
        )
        candidate_products = latest_products.merge(observation_counts, on="product_id", how="left")
        candidate_products = candidate_products.sort_values(
            ["observation_count", "latest_rank", "crawl_datetime"],
            ascending=[False, True, False],
        )
        default_product_ids = candidate_products.head(3)["product_id"].astype(str).tolist()
        requested_product_ids = [str(product_id) for product_id in product_ids if str(product_id).strip()]
        selected = list(dict.fromkeys(default_product_ids + requested_product_ids))
        return {
            "availableProducts": [
                {
                    "productId": str(row["product_id"]),
                    "name": row.get("name"),
                    "brand": row.get("brand"),
                    "latestRank": _safe_float(row.get("latest_rank")),
                    "observationCount": _safe_int(row.get("observation_count")),
                }
                for row in candidate_products.to_dict("records")
            ],
            "defaultProductIds": default_product_ids,
            "selectedProductIds": selected,
            "series": [
                {
                    "crawlDatetime": row["crawl_datetime"].isoformat() if isinstance(row["crawl_datetime"], pd.Timestamp) else None,
                    "productId": str(row["product_id"]),
                    "brand": row["brand"],
                    "name": row["name"],
                    "rank": _safe_float(row["rank"]),
                    "price": _safe_int(row["price"]),
                    "discountPct": _safe_float(row["discount_pct"]),
                }
                for row in filtered_df.sort_values(["crawl_datetime", "product_id"]).to_dict("records")
            ],
        }

    def get_momentum_inputs(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "rankVelocityDistribution": [],
                "rankAccelerationDistribution": [],
                "energyVelocityDistribution": [],
                "energyAccelerationDistribution": [],
                "discountVelocityDistribution": [],
                "stabilityDistribution": [],
            }

        momentum_df = _ensure_rank_energy_momentum(filtered_df)
        rank_velocity_bins = [-1000, -20, -10, -5, -1, 1, 5, 10, 20, 1000]
        rank_velocity_labels = ["-20 이하", "-20~-10", "-10~-5", "-5~-1", "-1~1", "1~5", "5~10", "10~20", "20 이상"]
        energy_bins = [-1000, -60, -30, -10, -1, 1, 10, 30, 60, 1000]
        energy_labels = ["-60 이하", "-60~-30", "-30~-10", "-10~-1", "-1~1", "1~10", "10~30", "30~60", "60 이상"]
        stability_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        stability_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

        def band_count(frame: pd.DataFrame, source: str, target: str, bins: list[float], labels: list[str]) -> pd.DataFrame:
            scoped = frame.dropna(subset=[source]).copy()
            scoped[target] = pd.cut(scoped[source], bins=bins, labels=labels, include_lowest=True)
            grouped = (
                scoped.dropna(subset=[target])
                .groupby(target, observed=False, as_index=False)
                .agg(count=("product_id", "count"))
            )
            grouped[target] = grouped[target].astype(str)
            return grouped

        rank_velocity_distribution = band_count(momentum_df, "rank_velocity", "band", rank_velocity_bins, rank_velocity_labels)
        rank_acceleration_distribution = band_count(momentum_df, "rank_acceleration", "band", rank_velocity_bins, rank_velocity_labels)
        energy_velocity_distribution = band_count(momentum_df, "energy_velocity", "band", energy_bins, energy_labels)
        energy_acceleration_distribution = band_count(momentum_df, "energy_acceleration", "band", energy_bins, energy_labels)
        discount_velocity_distribution = band_count(momentum_df, "discount_velocity", "band", rank_velocity_bins, rank_velocity_labels)
        stability_distribution = band_count(momentum_df, "consistency_score", "band", stability_bins, stability_labels)

        return {
            "rankVelocityDistribution": [
                {"band": row["band"], "count": _safe_int(row["count"])}
                for row in rank_velocity_distribution.to_dict("records")
            ],
            "rankAccelerationDistribution": [
                {"band": row["band"], "count": _safe_int(row["count"])}
                for row in rank_acceleration_distribution.to_dict("records")
            ],
            "energyVelocityDistribution": [
                {"band": row["band"], "count": _safe_int(row["count"])}
                for row in energy_velocity_distribution.to_dict("records")
            ],
            "energyAccelerationDistribution": [
                {"band": row["band"], "count": _safe_int(row["count"])}
                for row in energy_acceleration_distribution.to_dict("records")
            ],
            "discountVelocityDistribution": [
                {"band": row["band"], "count": _safe_int(row["count"])}
                for row in discount_velocity_distribution.to_dict("records")
            ],
            "stabilityDistribution": [
                {"band": row["band"], "count": _safe_int(row["count"])}
                for row in stability_distribution.to_dict("records")
            ],
        }

    def get_momentum_formula_samples(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "formula": "momentum_score=rank_energy(t)-rank_energy(t-1) only for consecutive observations; entry_score=rank_energy(t); exit_score=rank_energy(t-1)",
                "rows": [],
            }
        filtered_df = _ensure_rank_energy_momentum(filtered_df)

        latest = (
            filtered_df.sort_values("crawl_datetime")
            .groupby("product_id", as_index=False)
            .tail(1)
            .copy()
        )
        latest["trend_contribution"] = latest["energy_velocity"].fillna(0)
        latest["acceleration_contribution"] = latest["energy_acceleration"].fillna(0)
        latest["consistency_contribution"] = latest["consistency_score"].fillna(0.5)
        latest["abs_momentum"] = latest["momentum_score"].fillna(0).abs()
        samples = latest.sort_values(["abs_momentum", "energy_acceleration"], ascending=False).head(20)
        return {
            "formula": "momentum_score=rank_energy(t)-rank_energy(t-1) only for consecutive observations; entry_score=rank_energy(t); exit_score=rank_energy(t-1)",
            "rows": [
                {
                    "productId": str(row["product_id"]),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "rank": _safe_float(row.get("rank")),
                    "standardScore": _safe_float(row.get("score")),
                    "rankEnergy": _safe_float(row.get("rank_energy")),
                    "rankVelocity": _safe_float(row.get("rank_velocity")),
                    "rankAcceleration": _safe_float(row.get("rank_acceleration")),
                    "energyVelocity": _safe_float(row.get("energy_velocity")),
                    "energyAcceleration": _safe_float(row.get("energy_acceleration")),
                    "entryScore": _safe_float(row.get("entry_score")),
                    "exitScore": _safe_float(row.get("exit_score")),
                    "trendScore": _safe_float(row.get("trend_score")),
                    "breakoutScore": _safe_float(row.get("breakout_score")),
                    "isReentry": bool(row.get("is_reentry")) if row.get("is_reentry") is not None else None,
                    "presenceRatio5": _safe_float(row.get("presence_ratio_5")),
                    "rankTierWeight": _safe_float(row.get("rank_tier_weight")),
                    "scaleK": _safe_float(row.get("momentum_scale_k")),
                    "eventBonus": _safe_float(row.get("event_bonus")),
                    "trendContribution": _safe_float(row.get("trend_contribution")),
                    "accelerationContribution": _safe_float(row.get("acceleration_contribution")),
                    "consistencyContribution": _safe_float(row.get("consistency_contribution")),
                    "momentumScore": _safe_float(row.get("momentum_score")),
                    "stabilityScore": _safe_float(row.get("stability_score")),
                    "persistence": _safe_float(row.get("consistency_score")),
                    "eventState": row.get("momentum_event_state"),
                    "eventLabel": row.get("momentum_event_label"),
                    "discountVelocity": _safe_float(row.get("discount_velocity")),
                }
                for row in samples.to_dict("records")
            ],
        }

    def get_momentum_distribution(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_df = self._load_fact(filters)
        if filtered_df.empty:
            return {
                "momentumBandDistribution": [],
                "eventStateDistribution": [],
                "brandMomentum": [],
                "priceBandMomentum": [],
                "topMomentum": [],
            }
        filtered_df = _ensure_rank_energy_momentum(filtered_df)

        latest = (
            filtered_df.sort_values("crawl_datetime")
            .groupby("product_id", as_index=False)
            .tail(1)
            .copy()
        )
        latest["estimated_original_price"] = latest.apply(
            lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
            axis=1,
        )
        latest["estimated_original_price_band"] = _price_band_from_original_price(latest["estimated_original_price"])
        latest["estimated_original_price_band"] = (
            latest["estimated_original_price_band"].astype(str).replace("nan", "미분류")
        )
        momentum_bins = [-1000, -60, -30, -10, -1, 1, 10, 30, 60, 1000]
        momentum_labels = ["-60 이하", "-60~-30", "-30~-10", "-10~-1", "-1~1", "1~10", "10~30", "30~60", "60 이상"]
        latest["momentumBand"] = pd.cut(
            latest["momentum_score"],
            bins=momentum_bins,
            labels=momentum_labels,
            include_lowest=True,
        )
        band_distribution = (
            latest.dropna(subset=["momentumBand"])
            .groupby("momentumBand", observed=False, as_index=False)
            .agg(count=("product_id", "count"), avg_rank=("rank", "mean"))
        )
        band_distribution["momentumBand"] = band_distribution["momentumBand"].astype(str)
        event_state_distribution = (
            latest.assign(eventLabel=latest["momentum_event_label"].fillna("정체"))
            .groupby("eventLabel", as_index=False)
            .agg(count=("product_id", "count"), avg_momentum=("momentum_score", "mean"), avg_acceleration=("energy_acceleration", "mean"))
            .sort_values("count", ascending=False)
        )
        lifecycle_event_distribution = _momentum_lifecycle_event_rows(filtered_df)
        if not lifecycle_event_distribution.empty:
            event_state_distribution = (
                pd.concat([event_state_distribution, lifecycle_event_distribution], ignore_index=True)
                .groupby("eventLabel", as_index=False)
                .agg(count=("count", "sum"), avg_momentum=("avg_momentum", "mean"), avg_acceleration=("avg_acceleration", "mean"))
                .sort_values("count", ascending=False)
            )

        snapshot_count = max(int(filtered_df["snapshot_id"].nunique()) if "snapshot_id" in filtered_df.columns else 1, 1)
        product_energy = (
            filtered_df.groupby("product_id", as_index=False)
            .agg(
                observation_count=("snapshot_id", "nunique"),
                cumulative_rank_energy=("rank_energy", "sum"),
                avg_rank_energy=("rank_energy", "mean"),
                best_rank=("rank", "min"),
                best_rank_energy=("rank_energy", "max"),
                avg_momentum=("momentum_score", "mean"),
                avg_acceleration=("energy_acceleration", "mean"),
            )
        )
        product_energy["presence_ratio"] = product_energy["observation_count"] / snapshot_count
        product_energy["sustained_rank_energy"] = product_energy["cumulative_rank_energy"] * (0.5 + product_energy["presence_ratio"])
        latest = latest.merge(product_energy, on="product_id", how="left")

        brand_stats = (
            latest.groupby("brand", as_index=False)
            .agg(
                avg_rank=("rank", "mean"),
                avg_velocity=("rank_velocity", "mean"),
                avg_energy_velocity=("energy_velocity", "mean"),
                avg_acceleration=("energy_acceleration", "mean"),
                avg_momentum=("sustained_rank_energy", "mean"),
                total_sustained_energy=("sustained_rank_energy", "sum"),
                record_count=("product_id", "count"),
            )
            .sort_values("total_sustained_energy", ascending=False)
            .head(20)
        )

        price_band_source = (
            latest["price_band"]
            if "price_band" in latest.columns
            else pd.Series(["미분류"] * len(latest), index=latest.index)
        )
        price_band_stats = (
            latest.assign(priceBand=price_band_source.astype(str).replace("nan", "미분류"))
            .groupby("priceBand", as_index=False)
            .agg(
                avg_rank=("rank", "mean"),
                avg_momentum=("sustained_rank_energy", "mean"),
                avg_velocity=("rank_velocity", "mean"),
                avg_energy_velocity=("energy_velocity", "mean"),
                avg_acceleration=("energy_acceleration", "mean"),
                total_sustained_energy=("sustained_rank_energy", "sum"),
                record_count=("product_id", "count"),
            )
            .sort_values("total_sustained_energy", ascending=False)
        )

        top_momentum = latest[
            [
                "product_id",
                "brand",
                "name",
                "rank",
                "rank_velocity",
                "rank_acceleration",
                "score",
                "rank_energy",
                "energy_velocity",
                "energy_acceleration",
                "entry_score",
                "momentum_score",
                "observation_count",
                "presence_ratio",
                "cumulative_rank_energy",
                "avg_rank_energy",
                "best_rank",
                "best_rank_energy",
                "sustained_rank_energy",
                "consistency_score",
                "momentum_event_state",
                "momentum_event_label",
                "stability_score",
                "discount_pct",
                "price",
                "price_band",
                "estimated_original_price_band",
            ]
        ].copy()
        top_momentum["action_score"] = top_momentum["sustained_rank_energy"]
        top_momentum = top_momentum.sort_values(
            ["sustained_rank_energy", "observation_count", "best_rank_energy"],
            ascending=False,
        )

        return {
            "momentumBandDistribution": [
                {
                    "momentumBand": row["momentumBand"],
                    "count": _safe_int(row["count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                }
                for row in band_distribution.to_dict("records")
            ],
            "eventStateDistribution": [
                {
                    "eventLabel": row["eventLabel"],
                    "count": _safe_int(row["count"]),
                    "avgMomentum": _safe_float(row["avg_momentum"]),
                    "avgAcceleration": _safe_float(row["avg_acceleration"]),
                }
                for row in event_state_distribution.to_dict("records")
            ],
            "brandMomentum": [
                {
                    "brand": row["brand"],
                    "avgMomentum": _safe_float(row["avg_momentum"]),
                    "totalSustainedEnergy": _safe_float(row["total_sustained_energy"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgVelocity": _safe_float(row["avg_velocity"]),
                    "avgEnergyVelocity": _safe_float(row["avg_energy_velocity"]),
                    "avgAcceleration": _safe_float(row["avg_acceleration"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in brand_stats.to_dict("records")
            ],
            "priceBandMomentum": [
                {
                    "priceBand": row["priceBand"],
                    "avgMomentum": _safe_float(row["avg_momentum"]),
                    "totalSustainedEnergy": _safe_float(row["total_sustained_energy"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgVelocity": _safe_float(row["avg_velocity"]),
                    "avgEnergyVelocity": _safe_float(row["avg_energy_velocity"]),
                    "avgAcceleration": _safe_float(row["avg_acceleration"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in price_band_stats.to_dict("records")
            ],
            "topMomentum": [
                {
                    "productId": str(row["product_id"]),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "rank": _safe_float(row.get("rank")),
                    "standardScore": _safe_float(row.get("score")),
                    "rankEnergy": _safe_float(row.get("rank_energy")),
                    "rankVelocity": _safe_float(row.get("rank_velocity")),
                    "rankAcceleration": _safe_float(row.get("rank_acceleration")),
                    "energyVelocity": _safe_float(row.get("energy_velocity")),
                    "energyAcceleration": _safe_float(row.get("energy_acceleration")),
                    "entryScore": _safe_float(row.get("entry_score")),
                    "actionScore": _safe_float(row.get("action_score")),
                    "observationCount": _safe_int(row.get("observation_count")),
                    "presenceRatio": _safe_float(row.get("presence_ratio")),
                    "cumulativeRankEnergy": _safe_float(row.get("cumulative_rank_energy")),
                    "avgRankEnergy": _safe_float(row.get("avg_rank_energy")),
                    "bestRank": _safe_float(row.get("best_rank")),
                    "bestRankEnergy": _safe_float(row.get("best_rank_energy")),
                    "sustainedRankEnergy": _safe_float(row.get("sustained_rank_energy")),
                    "momentumScore": _safe_float(row.get("momentum_score")),
                    "persistence": _safe_float(row.get("consistency_score")),
                    "eventState": row.get("momentum_event_state"),
                    "eventLabel": row.get("momentum_event_label"),
                    "stabilityScore": _safe_float(row.get("stability_score")),
                    "discountPct": _safe_float(row.get("discount_pct")),
                    "price": _safe_int(row.get("price")),
                    "priceBand": _label_or_missing(row.get("price_band"), "미분류"),
                    "estimatedOriginalPriceBand": _label_or_missing(
                        row.get("estimated_original_price_band"),
                        "미분류",
                    ),
                }
                for row in top_momentum.head(60).to_dict("records")
            ],
        }

    def get_semantic_text_features(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        text_features_df = self._load_text_features(filters, filtered_fact_df)
        empty_response = {
            "itemDistribution": [],
            "coverageRows": [],
            "colorDistribution": [],
            "materialDistribution": [],
            "originCountryDistribution": [],
            "districtDistribution": [],
            "dongDistribution": [],
            "locationMapPoints": [],
            "materialPriceBandHeatmap": [],
            "rows": [],
        }
        if filtered_fact_df.empty:
            return empty_response

        latest = _latest_by_product(filtered_fact_df)
        for column in [
            "business_address",
            "business_province",
            "business_district",
            "business_dong",
            "shipping_fee",
            "origin_country",
            "material",
            "color",
        ]:
            if column not in latest.columns:
                latest[column] = None
        latest["estimatedOriginalPrice"] = latest.apply(
            lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
            axis=1,
        )
        latest["priceBand"] = _price_band_from_original_price(latest["estimatedOriginalPrice"])
        latest["priceBand"] = latest["priceBand"].astype(str).replace("nan", "미분류")

        merge_columns = ["snapshot_id", "product_id", "name_item", "color_normalized", "material_normalized", "text_richness_score"]
        available_merge_columns = [column for column in merge_columns if column in text_features_df.columns]
        if {"snapshot_id", "product_id"}.issubset(set(available_merge_columns)):
            latest = latest.merge(
                text_features_df[available_merge_columns].drop_duplicates(subset=["snapshot_id", "product_id"], keep="last"),
                how="left",
                on=["snapshot_id", "product_id"],
            )

        latest["nameItem"] = (
            latest["name_item"].map(lambda value: _label_or_missing(value, "미분류 상품유형"))
            if "name_item" in latest.columns
            else "미분류 상품유형"
        )
        latest["colorNormalized"] = (
            latest["color_normalized"].map(lambda value: _label_or_missing(value, "미정규화"))
            if "color_normalized" in latest.columns
            else latest["color"].map(lambda value: _label_or_missing(value, "미입력"))
        )
        latest["materialNormalized"] = (
            latest["material_normalized"].map(lambda value: _label_or_missing(value, "미정규화"))
            if "material_normalized" in latest.columns
            else latest["material"].map(lambda value: _label_or_missing(value, "미입력"))
        )
        latest["originCountryLabel"] = latest["origin_country"].map(lambda value: _label_or_missing(value, "미입력"))
        latest["shippingFeeStatus"] = latest["shipping_fee"].map(
            lambda value: "입력됨" if _label_or_missing(value, "") else "미입력"
        )
        latest["businessProvinceLabel"] = latest["business_province"].map(lambda value: _label_or_missing(value, "미입력"))
        latest["businessDistrictLabel"] = latest["business_district"].map(lambda value: _label_or_missing(value, "미입력"))
        latest["businessDongLabel"] = latest["business_dong"].map(lambda value: _label_or_missing(value, "미입력"))
        latest["locationLabel"] = latest.apply(
            lambda row: _compose_area_label(
                row.get("business_province"),
                row.get("business_district"),
                row.get("business_dong"),
            )
            or _compose_area_label(row.get("business_province"), row.get("business_district"), None)
            or _label_or_missing(row.get("business_address"), "미입력"),
            axis=1,
        )

        def attribute_distribution(source: str, label: str, limit: int = 12) -> pd.DataFrame:
            grouped = (
                latest.groupby(source, as_index=False)
                .agg(
                    record_count=("product_id", "count"),
                    brand_count=("brand", "nunique"),
                    avg_rank=("rank", "mean"),
                    avg_price=("price", "mean"),
                    avg_discount_pct=("discount_pct", "mean"),
                )
                .sort_values(["record_count", "avg_rank"], ascending=[False, True])
                .head(limit)
                .copy()
            )
            grouped.columns = [label if column == source else column for column in grouped.columns]
            return grouped

        item_dist = attribute_distribution("nameItem", "nameItem", limit=15)[["nameItem", "record_count"]]
        item_dist.columns = ["nameItem", "count"]
        color_dist = attribute_distribution("colorNormalized", "colorValue", limit=15)
        material_dist = attribute_distribution("materialNormalized", "materialValue", limit=12)
        origin_dist = attribute_distribution("originCountryLabel", "originCountry", limit=12)
        district_dist = attribute_distribution("businessDistrictLabel", "businessDistrict", limit=15)
        dong_dist = attribute_distribution("businessDongLabel", "businessDong", limit=15)

        top_materials = material_dist["materialValue"].tolist()
        material_heatmap = (
            latest[latest["materialNormalized"].isin(top_materials)]
            .groupby(["materialNormalized", "priceBand"], as_index=False)
            .agg(count=("product_id", "count"))
            .rename(columns={"materialNormalized": "materialValue"})
        )
        location_agg = (
            latest[latest["locationLabel"].map(lambda value: _label_or_missing(value, "") != "")]
            .groupby(["locationLabel", "businessProvinceLabel", "businessDistrictLabel", "businessDongLabel"], as_index=False)
            .agg(
                record_count=("product_id", "count"),
                brand_count=("brand", "nunique"),
                avg_rank=("rank", "mean"),
                avg_price=("price", "mean"),
                avg_discount_pct=("discount_pct", "mean"),
            )
            .sort_values(["record_count", "avg_rank"], ascending=[False, True])
        )
        location_points = self._attach_location_points(filters, location_agg, label_column="locationLabel", limit=20)

        product_count = int(len(latest))
        coverage_rows = [
            {
                "metric": "상품 정보 보유 상품 수",
                "filledCount": _safe_int(latest["product_info_exists"].fillna(False).astype(bool).sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    (latest["product_info_exists"].fillna(False).astype(bool).mean() * 100.0) if product_count else None
                ),
            },
            {
                "metric": "색상 입력 상품 수",
                "filledCount": _safe_int((latest["color"].map(lambda value: _label_or_missing(value, "")) != "").sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    ((latest["color"].map(lambda value: _label_or_missing(value, "")) != "").mean() * 100.0) if product_count else None
                ),
            },
            {
                "metric": "소재 입력 상품 수",
                "filledCount": _safe_int((latest["material"].map(lambda value: _label_or_missing(value, "")) != "").sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    ((latest["material"].map(lambda value: _label_or_missing(value, "")) != "").mean() * 100.0) if product_count else None
                ),
            },
            {
                "metric": "제조국 입력 상품 수",
                "filledCount": _safe_int((latest["origin_country"].map(lambda value: _label_or_missing(value, "")) != "").sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    ((latest["origin_country"].map(lambda value: _label_or_missing(value, "")) != "").mean() * 100.0) if product_count else None
                ),
            },
            {
                "metric": "영업소재지 입력 상품 수",
                "filledCount": _safe_int((latest["business_address"].map(lambda value: _label_or_missing(value, "")) != "").sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    ((latest["business_address"].map(lambda value: _label_or_missing(value, "")) != "").mean() * 100.0) if product_count else None
                ),
            },
            {
                "metric": "구 단위 파싱 가능 상품 수",
                "filledCount": _safe_int((latest["business_district"].map(lambda value: _label_or_missing(value, "")) != "").sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    ((latest["business_district"].map(lambda value: _label_or_missing(value, "")) != "").mean() * 100.0) if product_count else None
                ),
            },
            {
                "metric": "동 단위 파싱 가능 상품 수",
                "filledCount": _safe_int((latest["business_dong"].map(lambda value: _label_or_missing(value, "")) != "").sum()),
                "productCount": product_count,
                "filledRatePct": _safe_float(
                    ((latest["business_dong"].map(lambda value: _label_or_missing(value, "")) != "").mean() * 100.0) if product_count else None
                ),
            },
        ]

        sample = latest[
            [
                "product_id",
                "brand",
                "name",
                "nameItem",
                "colorNormalized",
                "materialNormalized",
                "originCountryLabel",
                "businessProvinceLabel",
                "businessDistrictLabel",
                "businessDongLabel",
                "business_address",
                "shippingFeeStatus",
                "priceBand",
                "rank",
                "price",
                "discount_pct",
            ]
        ].sort_values(["rank", "price"], ascending=[True, True]).head(50)

        return {
            "itemDistribution": _records(item_dist),
            "coverageRows": coverage_rows,
            "colorDistribution": [
                {
                    "colorValue": row["colorValue"],
                    "recordCount": _safe_int(row["record_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                }
                for row in color_dist.to_dict("records")
            ],
            "materialDistribution": [
                {
                    "materialValue": row["materialValue"],
                    "recordCount": _safe_int(row["record_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                }
                for row in material_dist.to_dict("records")
            ],
            "originCountryDistribution": [
                {
                    "originCountry": row["originCountry"],
                    "recordCount": _safe_int(row["record_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                }
                for row in origin_dist.to_dict("records")
            ],
            "districtDistribution": [
                {
                    "businessDistrict": row["businessDistrict"],
                    "recordCount": _safe_int(row["record_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                }
                for row in district_dist.to_dict("records")
            ],
            "dongDistribution": [
                {
                    "businessDong": row["businessDong"],
                    "recordCount": _safe_int(row["record_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                }
                for row in dong_dist.to_dict("records")
            ],
            "locationMapPoints": location_points,
            "materialPriceBandHeatmap": [
                {
                    "materialValue": row["materialValue"],
                    "priceBand": row["priceBand"],
                    "count": _safe_int(row["count"]),
                }
                for row in material_heatmap.to_dict("records")
            ],
            "rows": [
                {
                    "productId": str(row["product_id"]),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "nameItem": row.get("nameItem"),
                    "colorNormalized": row.get("colorNormalized"),
                    "materialNormalized": row.get("materialNormalized"),
                    "originCountry": row.get("originCountryLabel"),
                    "businessProvince": row.get("businessProvinceLabel"),
                    "businessDistrict": row.get("businessDistrictLabel"),
                    "businessDong": row.get("businessDongLabel"),
                    "businessAddress": row.get("business_address"),
                    "shippingFeeStatus": row.get("shippingFeeStatus"),
                    "priceBand": row.get("priceBand"),
                    "rank": _safe_float(row.get("rank")),
                    "price": _safe_int(row.get("price")),
                    "discountPct": _safe_float(row.get("discount_pct")),
                }
                for row in sample.to_dict("records")
            ],
        }

    def get_text_overview(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        empty_response = {
            "kpiRows": [],
            "sentimentDistribution": [],
            "reviewTypeDistribution": [],
            "aspectVolume": [],
            "qualityRows": [],
            "tpoDistribution": [],
            "reviewsPerProductDistribution": [],
            "reviewsPerProductBins": [],
            "reviewsPerProductStats": [],
        }
        if filtered_fact_df.empty:
            return empty_response

        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        claim_df = self._load_text_claim_facts(filters, filtered_fact_df)
        fusion_df = self._load_text_fusion_profile(filters, filtered_fact_df)
        if review_df.empty and claim_df.empty and fusion_df.empty:
            return empty_response

        product_count = int(filtered_fact_df["product_id"].astype(str).nunique()) if "product_id" in filtered_fact_df.columns else 0
        review_count = int(review_df["review_id"].nunique()) if "review_id" in review_df.columns and not review_df.empty else 0
        sentence_count = int(len(review_df))
        claim_count = int(len(claim_df))
        avg_rating = _safe_float(review_df["rating"].mean()) if "rating" in review_df.columns and not review_df.empty else None
        fusion_profile_count = int(len(fusion_df))
        avg_fusion_score = _safe_float(fusion_df["fusion_score"].mean()) if "fusion_score" in fusion_df.columns and not fusion_df.empty else None
        avg_confidence_score = _safe_float(fusion_df["confidence_score"].mean()) if "confidence_score" in fusion_df.columns and not fusion_df.empty else None
        avg_agreement_rate = _safe_float(fusion_df["agreement_rate"].dropna().mean()) if "agreement_rate" in fusion_df.columns and not fusion_df.empty else None

        kpi_rows = [
            {"metric": "분석 대상 상품 수", "value": product_count, "unit": "개"},
            {"metric": "리뷰 수", "value": review_count, "unit": "건"},
            {"metric": "리뷰 문장 수", "value": sentence_count, "unit": "문장"},
            {"metric": "클레임 문장 수", "value": claim_count, "unit": "문장"},
            {"metric": "평균 평점", "value": _safe_float(avg_rating), "unit": "점"},
            {"metric": "융합 프로파일 수", "value": fusion_profile_count, "unit": "행"},
            {"metric": "평균 융합 점수", "value": avg_fusion_score, "unit": "점"},
            {"metric": "평균 신뢰도", "value": avg_confidence_score, "unit": "점"},
            {"metric": "평균 합의도", "value": avg_agreement_rate, "unit": "비율"},
        ]

        sentiment_dist: list[dict[str, Any]] = []
        if not review_df.empty and {"sentiment_label", "rating", "sentiment_score"}.issubset(review_df.columns):
            grouped = (
                review_df.groupby("sentiment_label", as_index=False)
                .agg(
                    count=("sentence_text", "count"),
                    avg_rating=("rating", "mean"),
                    avg_sentiment_score=("sentiment_score", "mean"),
                )
                .sort_values("count", ascending=False)
            )
            sentiment_dist = [
                {
                    "sentiment": row["sentiment_label"],
                    "count": _safe_int(row["count"]),
                    "avgRating": _safe_float(row["avg_rating"]),
                    "avgSentimentScore": _safe_float(row["avg_sentiment_score"]),
                }
                for row in grouped.to_dict("records")
            ]

        review_type_dist: list[dict[str, Any]] = []
        if not review_df.empty and {"review_type", "photo_review", "rating"}.issubset(review_df.columns):
            grouped = (
                review_df.groupby("review_type", as_index=False)
                .agg(
                    count=("review_id", "nunique"),
                    sentence_count=("sentence_text", "count"),
                    photo_ratio=("photo_review", "mean"),
                    avg_rating=("rating", "mean"),
                )
                .sort_values("count", ascending=False)
            )
            review_type_dist = [
                {
                    "reviewType": _label_or_missing(row["review_type"], "unknown"),
                    "count": _safe_int(row["count"]),
                    "sentenceCount": _safe_int(row["sentence_count"]),
                    "photoRatioPct": _safe_float((row["photo_ratio"] or 0) * 100.0),
                    "avgRating": _safe_float(row["avg_rating"]),
                }
                for row in grouped.to_dict("records")
            ]

        aspect_volume: list[dict[str, Any]] = []
        if not fusion_df.empty and {"aspect", "fusion_score", "confidence_score"}.issubset(fusion_df.columns):
            grouped = (
                fusion_df.groupby("aspect", as_index=False)
                .agg(
                    fusion_profile_count=("product_id", "count"),
                    avg_fusion_score=("fusion_score", "mean"),
                    avg_confidence_score=("confidence_score", "mean"),
                    avg_agreement_rate=("agreement_rate", "mean"),
                )
                .sort_values("avg_confidence_score", ascending=False)
            )
            aspect_volume = [
                {
                    "aspect": _label_or_missing(row["aspect"], "general"),
                    "fusionProfileCount": _safe_int(row["fusion_profile_count"]),
                    "avgFusionScore": _safe_float(row["avg_fusion_score"]),
                    "avgConfidenceScore": _safe_float(row["avg_confidence_score"]),
                    "avgAgreementRate": _safe_float(row["avg_agreement_rate"]),
                }
                for row in grouped.to_dict("records")
            ]

        quality_rows = []
        if not fusion_df.empty:
            review_covered = int((fusion_df["review_sentence_count"].fillna(0) > 0).sum()) if "review_sentence_count" in fusion_df.columns else 0
            claim_covered = int((fusion_df["claim_count"].fillna(0) > 0).sum()) if "claim_count" in fusion_df.columns else 0
            profile_count = max(int(len(fusion_df)), 1)
            avg_source_coverage = _safe_float(fusion_df["claim_source_coverage_pct"].mean()) if "claim_source_coverage_pct" in fusion_df.columns else None
            avg_evidence_density = _safe_float(fusion_df["evidence_density"].mean()) if "evidence_density" in fusion_df.columns else None
            quality_rows = [
                {"metric": "리뷰 근거 커버리지", "value": _safe_float((review_covered / profile_count) * 100.0), "unit": "%"},
                {"metric": "클레임 근거 커버리지", "value": _safe_float((claim_covered / profile_count) * 100.0), "unit": "%"},
                {"metric": "클레임 소스 커버리지", "value": avg_source_coverage, "unit": "%"},
                {"metric": "평균 근거 밀도", "value": avg_evidence_density, "unit": "개"},
            ]

        tpo_dist: list[dict[str, Any]] = []
        if not review_df.empty and "tpo" in review_df.columns:
            tpo_grouped = (
                review_df.groupby("tpo", as_index=False)
                .agg(
                    count=("sentence_text", "count"),
                    avg_sentiment_score=("sentiment_score", "mean"),
                )
                .sort_values("count", ascending=False)
            )
            tpo_dist = [
                {
                    "tpo": _label_or_missing(row["tpo"], "unspecified"),
                    "count": _safe_int(row["count"]),
                    "avgSentimentScore": _safe_float(row["avg_sentiment_score"]),
                }
                for row in tpo_grouped.to_dict("records")
            ]

        reviews_per_product_dist: list[dict[str, Any]] = []
        reviews_per_product_bins: list[dict[str, Any]] = []
        reviews_per_product_stats: list[dict[str, Any]] = []
        if product_count > 0 and "product_id" in filtered_fact_df.columns:
            latest_fact = _latest_by_product(filtered_fact_df)
            product_ids = latest_fact["product_id"].astype(str).drop_duplicates()
            meta_totals = self._load_total_reviews_from_meta(product_ids)
            total_review_col = "meta.total_reviews_reported" if not meta_totals.empty else None
            per_product: pd.Series
            if total_review_col is not None:
                per_product = meta_totals.reindex(product_ids, fill_value=0).astype("int64")
            else:
                per_product = pd.Series(dtype="int64")
                raw_products_df = self.repository.load_raw_snapshot_products(filters.dataset)
                if not raw_products_df.empty and {"snapshot_id", "product_id"}.issubset(raw_products_df.columns):
                    key_df = latest_fact[["snapshot_id", "product_id"]].copy()
                    key_df["snapshot_id"] = key_df["snapshot_id"].astype(str)
                    key_df["product_id"] = key_df["product_id"].astype(str)
                    raw_scoped = raw_products_df.copy()
                    raw_scoped["snapshot_id"] = raw_scoped["snapshot_id"].astype(str)
                    raw_scoped["product_id"] = raw_scoped["product_id"].astype(str)
                    raw_scoped = raw_scoped.merge(key_df, on=["snapshot_id", "product_id"], how="inner")
                    total_review_col = next(
                        (
                            col
                            for col in ("product.reviews_count", "reviews_count", "product.review_count", "review_count")
                            if col in raw_scoped.columns
                        ),
                        None,
                    )
                    if total_review_col is not None:
                        raw = raw_scoped[total_review_col].map(
                            lambda v: str(v).replace(",", "").strip() if not _is_missing_value(v) else None
                        )
                        parsed = pd.to_numeric(raw, errors="coerce").fillna(0).clip(lower=0).astype("int64")
                        per_product = pd.Series(parsed.values, index=raw_scoped["product_id"].astype(str)).groupby(level=0).max()
                        per_product = per_product.reindex(product_ids, fill_value=0).astype("int64")

            if per_product.empty:
                product_ids = latest_fact["product_id"].astype(str).drop_duplicates()
                per_product = pd.Series(0, index=product_ids, dtype="int64")
                if not review_df.empty and {"product_id", "review_id"}.issubset(review_df.columns):
                    rc = (
                        review_df.assign(_pid=review_df["product_id"].astype(str))
                        .groupby("_pid", as_index=True)["review_id"]
                        .nunique()
                    )
                    per_product = per_product.add(rc, fill_value=0).astype("int64")

            total_products = int(len(per_product))
            # ---- 로그 구간 히스토그램 (정적 bin) ----
            # 각 구간은 [low, high) (high=None이면 [low, ∞)).
            bin_defs: list[tuple[int, int | None, str]] = [
                (0, 1, "0"),
                (1, 5, "1–4"),
                (5, 10, "5–9"),
                (10, 30, "10–29"),
                (30, 100, "30–99"),
                (100, 300, "100–299"),
                (300, 500, "300–499"),
                (500, 1000, "500–999"),
                (1000, 3000, "1k–2.9k"),
                (3000, 10000, "3k–9.9k"),
                (10000, 30000, "10k–29.9k"),
                (30000, 100000, "30k–99.9k"),
                (100000, None, "100k+"),
            ]
            cumulative_count = 0
            for low, high, label in bin_defs:
                if high is None:
                    mask = per_product >= low
                else:
                    mask = (per_product >= low) & (per_product < high)
                count = int(mask.sum())
                cumulative_count += count
                cumulative_percent = (cumulative_count / total_products * 100.0) if total_products else 0.0
                reviews_per_product_bins.append(
                    {
                        "label": label,
                        "low": _safe_int(low),
                        "high": None if high is None else _safe_int(high),
                        "productCount": _safe_int(count),
                        "cumulativeCount": _safe_int(cumulative_count),
                        "cumulativePercent": _safe_float(round(cumulative_percent, 2)),
                    }
                )

            # ---- 백분위/요약 통계 ----
            def _q(series: pd.Series, q: float) -> float:
                if series.empty:
                    return 0.0
                try:
                    return float(series.quantile(q))
                except Exception:
                    return 0.0

            over_cap = int((per_product > 300).sum())
            zero_ct = int((per_product == 0).sum())
            source_label = "전체 리뷰 수" if total_review_col is not None else "수집 리뷰 수"
            reviews_per_product_stats = [
                {"metric": f"{source_label} 0건 상품", "value": zero_ct, "unit": "개"},
                {"metric": f"{source_label} 300건 초과 상품", "value": over_cap, "unit": "개"},
                {"metric": "상품당 리뷰 P25", "value": _safe_float(_q(per_product, 0.25)), "unit": "건"},
                {"metric": "상품당 리뷰 중앙값", "value": _safe_float(_q(per_product, 0.50)), "unit": "건"},
                {"metric": "상품당 리뷰 P75", "value": _safe_float(_q(per_product, 0.75)), "unit": "건"},
                {"metric": "상품당 리뷰 P90", "value": _safe_float(_q(per_product, 0.90)), "unit": "건"},
                {"metric": "상품당 리뷰 P95", "value": _safe_float(_q(per_product, 0.95)), "unit": "건"},
                {"metric": "상품당 리뷰 P99", "value": _safe_float(_q(per_product, 0.99)), "unit": "건"},
                {"metric": "상품당 리뷰 평균", "value": _safe_float(float(per_product.mean()) if len(per_product) else 0.0), "unit": "건"},
                {"metric": "상품당 리뷰 최대", "value": _safe_int(int(per_product.max()) if len(per_product) else 0), "unit": "건"},
            ]

        return {
            "kpiRows": kpi_rows,
            "sentimentDistribution": sentiment_dist,
            "reviewTypeDistribution": review_type_dist,
            "aspectVolume": aspect_volume,
            "qualityRows": quality_rows,
            "tpoDistribution": tpo_dist,
            "reviewsPerProductDistribution": reviews_per_product_dist,
            "reviewsPerProductBins": reviews_per_product_bins,
            "reviewsPerProductStats": reviews_per_product_stats,
        }

    def get_text_aspects(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        if filtered_fact_df.empty:
            return {"aspectSentiment": [], "aspectByCategory": [], "aspectByBrand": [], "rows": []}

        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        fusion_df = self._load_text_fusion_profile(filters, filtered_fact_df)
        if review_df.empty and fusion_df.empty:
            return {"aspectSentiment": [], "aspectByCategory": [], "aspectByBrand": [], "rows": []}

        latest = _latest_by_product(filtered_fact_df)
        for column in ["category_l1", "brand", "name"]:
            if column not in latest.columns:
                latest[column] = None
        dim_cols = ["product_id", "brand", "name", "category_l1"]

        if not review_df.empty:
            review_df = review_df.merge(latest[dim_cols].drop_duplicates(subset=["product_id"]), on="product_id", how="left", suffixes=("", "_fact"))
            review_df["brand"] = review_df["brand"].fillna(review_df.get("brand_fact"))
            review_df["name"] = review_df["name"].fillna(review_df.get("name_fact"))
            review_df["category_l1"] = review_df["category_l1"].map(lambda value: _label_or_missing(value, "미분류"))
            aspect_sentiment = (
                review_df.groupby(["aspect", "sentiment_label"], as_index=False)
                .agg(
                    sentence_count=("sentence_text", "count"),
                    review_count=("review_id", "nunique"),
                    avg_rating=("rating", "mean"),
                    avg_sentiment_score=("sentiment_score", "mean"),
                )
                .sort_values(["sentence_count", "review_count"], ascending=[False, False])
            )
        else:
            aspect_sentiment = pd.DataFrame(columns=["aspect", "sentiment_label", "sentence_count", "review_count", "avg_rating", "avg_sentiment_score"])

        if not fusion_df.empty:
            fusion = fusion_df.copy()
            fusion["category_l1"] = fusion.get("category_l1", pd.Series(index=fusion.index, dtype="object"))
            fusion["category_l1"] = fusion["category_l1"].map(lambda value: _label_or_missing(value, "미분류"))
            fusion["brand"] = fusion["brand"].map(lambda value: _label_or_missing(value, "미확인"))
            aspect_by_category = (
                fusion.groupby(["aspect", "category_l1"], as_index=False)
                .agg(
                    profile_count=("product_id", "count"),
                    avg_fusion_score=("fusion_score", "mean"),
                    avg_confidence_score=("confidence_score", "mean"),
                )
                .sort_values("profile_count", ascending=False)
            )
            aspect_by_brand = (
                fusion.groupby(["aspect", "brand"], as_index=False)
                .agg(
                    profile_count=("product_id", "count"),
                    avg_fusion_score=("fusion_score", "mean"),
                    avg_confidence_score=("confidence_score", "mean"),
                )
                .sort_values("profile_count", ascending=False)
            )
            sample_rows = fusion.sort_values(["confidence_score", "evidence_count"], ascending=[False, False], na_position="last").head(120)
        else:
            aspect_by_category = pd.DataFrame(columns=["aspect", "category_l1", "profile_count", "avg_fusion_score", "avg_confidence_score"])
            aspect_by_brand = pd.DataFrame(columns=["aspect", "brand", "profile_count", "avg_fusion_score", "avg_confidence_score"])
            sample_rows = pd.DataFrame()

        return {
            "aspectSentiment": [
                {
                    "aspect": row["aspect"],
                    "sentiment": row["sentiment_label"],
                    "sentenceCount": _safe_int(row["sentence_count"]),
                    "reviewCount": _safe_int(row["review_count"]),
                    "avgRating": _safe_float(row["avg_rating"]),
                    "avgSentimentScore": _safe_float(row["avg_sentiment_score"]),
                }
                for row in aspect_sentiment.to_dict("records")
            ],
            "aspectByCategory": [
                {
                    "aspect": row["aspect"],
                    "category": row["category_l1"],
                    "profileCount": _safe_int(row["profile_count"]),
                    "avgFusionScore": _safe_float(row["avg_fusion_score"]),
                    "avgConfidenceScore": _safe_float(row["avg_confidence_score"]),
                }
                for row in aspect_by_category.head(120).to_dict("records")
            ],
            "aspectByBrand": [
                {
                    "aspect": row["aspect"],
                    "brand": _label_or_missing(row["brand"], "미확인"),
                    "profileCount": _safe_int(row["profile_count"]),
                    "avgFusionScore": _safe_float(row["avg_fusion_score"]),
                    "avgConfidenceScore": _safe_float(row["avg_confidence_score"]),
                }
                for row in aspect_by_brand.head(120).to_dict("records")
            ],
            "rows": [
                {
                    "productId": str(row.get("product_id")),
                    "brand": _label_or_missing(row.get("brand"), "미확인"),
                    "name": row.get("name"),
                    "category": _label_or_missing(row.get("category_l1"), "미분류"),
                    "aspect": row.get("aspect"),
                    "fusionScore": _safe_float(row.get("fusion_score")),
                    "confidenceScore": _safe_float(row.get("confidence_score")),
                    "agreementRate": _safe_float(row.get("agreement_rate")),
                    "evidenceCount": _safe_int(row.get("evidence_count")),
                }
                for row in sample_rows.to_dict("records")
            ],
        }

    def get_text_fusion(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        if filtered_fact_df.empty:
            return {
                "fusionByAspect": [],
                "fusionByCategory": [],
                "fusionByBrand": [],
                "fusionTopProducts": [],
            }

        fusion_df = self._load_text_fusion_profile(filters, filtered_fact_df)
        if fusion_df.empty:
            return {
                "fusionByAspect": [],
                "fusionByCategory": [],
                "fusionByBrand": [],
                "fusionTopProducts": [],
            }

        fusion_by_aspect = (
            fusion_df.groupby("aspect", as_index=False)
            .agg(
                product_count=("product_id", "nunique"),
                review_sentence_count=("review_sentence_count", "sum"),
                claim_count=("claim_count", "sum"),
                avg_fusion_score=("fusion_score", "mean"),
                avg_confidence_score=("confidence_score", "mean"),
                avg_agreement_rate=("agreement_rate", "mean"),
            )
            .sort_values("avg_confidence_score", ascending=False)
        )
        fusion_by_category = (
            fusion_df.groupby(["category_l1", "aspect"], as_index=False)
            .agg(
                product_count=("product_id", "nunique"),
                avg_fusion_score=("fusion_score", "mean"),
                avg_confidence_score=("confidence_score", "mean"),
            )
            .sort_values("product_count", ascending=False)
        )
        fusion_by_brand = (
            fusion_df.groupby(["brand", "aspect"], as_index=False)
            .agg(
                product_count=("product_id", "nunique"),
                avg_fusion_score=("fusion_score", "mean"),
                avg_confidence_score=("confidence_score", "mean"),
            )
            .sort_values("product_count", ascending=False)
        )
        top_products = fusion_df.sort_values(
            ["confidence_score", "evidence_count", "fusion_score"],
            ascending=[False, False, False],
            na_position="last",
        ).head(120)

        return {
            "fusionByAspect": [
                {
                    "aspect": row["aspect"],
                    "productCount": _safe_int(row["product_count"]),
                    "reviewSentenceCount": _safe_int(row["review_sentence_count"]),
                    "claimCount": _safe_int(row["claim_count"]),
                    "avgFusionScore": _safe_float(row["avg_fusion_score"]),
                    "avgConfidenceScore": _safe_float(row["avg_confidence_score"]),
                    "avgAgreementRate": _safe_float(row["avg_agreement_rate"]),
                }
                for row in fusion_by_aspect.to_dict("records")
            ],
            "fusionByCategory": [
                {
                    "category": _label_or_missing(row["category_l1"], "미분류"),
                    "aspect": row["aspect"],
                    "productCount": _safe_int(row["product_count"]),
                    "avgFusionScore": _safe_float(row["avg_fusion_score"]),
                    "avgConfidenceScore": _safe_float(row["avg_confidence_score"]),
                }
                for row in fusion_by_category.head(160).to_dict("records")
            ],
            "fusionByBrand": [
                {
                    "brand": _label_or_missing(row["brand"], "미확인"),
                    "aspect": row["aspect"],
                    "productCount": _safe_int(row["product_count"]),
                    "avgFusionScore": _safe_float(row["avg_fusion_score"]),
                    "avgConfidenceScore": _safe_float(row["avg_confidence_score"]),
                }
                for row in fusion_by_brand.head(160).to_dict("records")
            ],
            "fusionTopProducts": [
                {
                    "productId": str(row.get("product_id")),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "category": row.get("category_l1"),
                    "aspect": row.get("aspect"),
                    "reviewSentenceCount": _safe_int(row.get("review_sentence_count")),
                    "reviewCount": _safe_int(row.get("review_count")),
                    "claimCount": _safe_int(row.get("claim_count")),
                    "reviewSignal": _safe_float(row.get("review_signal")),
                    "claimSignal": _safe_float(row.get("claim_signal")),
                    "fusionScore": _safe_float(row.get("fusion_score")),
                    "confidenceScore": _safe_float(row.get("confidence_score")),
                    "agreementRate": _safe_float(row.get("agreement_rate")),
                    "evidenceCount": _safe_int(row.get("evidence_count")),
                }
                for row in top_products.to_dict("records")
            ],
        }

    def get_text_evidence(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        if filtered_fact_df.empty:
            return {"rows": []}

        fusion_df = self._load_text_fusion_profile(filters, filtered_fact_df)
        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        claim_df = self._load_text_claim_facts(filters, filtered_fact_df)
        if fusion_df.empty:
            return {"rows": []}

        review_examples = pd.DataFrame()
        if not review_df.empty and {"snapshot_id", "product_id", "aspect", "sentence_text"}.issubset(review_df.columns):
            review_examples = (
                review_df.sort_values(["helpful_count", "sentence_text"], ascending=[False, True], na_position="last")
                .groupby(["snapshot_id", "product_id", "aspect"], as_index=False)
                .head(1)[["snapshot_id", "product_id", "aspect", "sentence_text", "sentiment_score", "review_type"]]
                .rename(
                    columns={
                        "sentence_text": "representative_review_sentence",
                        "sentiment_score": "representative_review_signal",
                        "review_type": "representative_review_type",
                    }
                )
            )

        claim_examples = pd.DataFrame()
        if not claim_df.empty and {"snapshot_id", "product_id", "aspect", "claim_text"}.issubset(claim_df.columns):
            claim_examples = (
                claim_df.sort_values(["confidence", "claim_text"], ascending=[False, True], na_position="last")
                .groupby(["snapshot_id", "product_id", "aspect"], as_index=False)
                .head(1)[["snapshot_id", "product_id", "aspect", "claim_text", "source_field", "claim_type"]]
                .rename(
                    columns={
                        "claim_text": "representative_claim",
                        "source_field": "representative_claim_source",
                        "claim_type": "representative_claim_type",
                    }
                )
            )

        evidence_df = fusion_df.copy()
        if not review_examples.empty:
            evidence_df = evidence_df.merge(review_examples, on=["snapshot_id", "product_id", "aspect"], how="left")
        if not claim_examples.empty:
            evidence_df = evidence_df.merge(claim_examples, on=["snapshot_id", "product_id", "aspect"], how="left")

        evidence_df = evidence_df.sort_values(
            ["confidence_score", "evidence_count", "fusion_score"],
            ascending=[False, False, False],
            na_position="last",
        )
        rows = [
            {
                "productId": str(row.get("product_id")),
                "brand": row.get("brand"),
                "name": row.get("name"),
                "category": row.get("category_l1"),
                "aspect": row.get("aspect"),
                "fusionScore": _safe_float(row.get("fusion_score")),
                "confidenceScore": _safe_float(row.get("confidence_score")),
                "agreementRate": _safe_float(row.get("agreement_rate")),
                "evidenceCount": _safe_int(row.get("evidence_count")),
                "representativeReviewSentence": row.get("representative_review_sentence"),
                "representativeReviewSignal": _safe_float(row.get("representative_review_signal")),
                "representativeReviewType": row.get("representative_review_type"),
                "representativeClaim": row.get("representative_claim"),
                "representativeClaimType": row.get("representative_claim_type"),
                "representativeClaimSource": row.get("representative_claim_source"),
            }
            for row in evidence_df.head(160).to_dict("records")
        ]
        return {"rows": rows}

    def get_text_gaps(self, filters: DashboardFilters) -> dict[str, Any]:
        # 이전 API 호환을 위해 유지 (융합 응답 반환)
        return self.get_text_fusion(filters)

    def _load_text_trend_keywords(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        trend_df = self.repository.load_table(
            filters.dataset,
            "text_trend_keywords.parquet",
            "text_trend_keywords",
        )
        if trend_df.empty:
            return trend_df
        if "snapshot_id" in trend_df.columns and "snapshot_id" in filtered_fact_df.columns:
            valid_snapshots = set(filtered_fact_df["snapshot_id"].dropna().unique())
            trend_df = trend_df[trend_df["snapshot_id"].isin(valid_snapshots)]
        return trend_df

    # ------------------------------------------------------------------
    # 태스크 3: 미충족 니즈
    # ------------------------------------------------------------------

    def get_text_unmet_needs(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        empty = {"summary": [], "byAspect": [], "topSentences": []}
        if filtered_fact_df.empty:
            return empty

        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        if review_df.empty or "is_unmet_need" not in review_df.columns:
            return empty

        unmet = review_df[review_df["is_unmet_need"].fillna(False).astype(bool)]
        total_count = int(len(review_df))
        unmet_count = int(len(unmet))

        summary = [
            {"metric": "전체 리뷰 문장 수", "value": total_count, "unit": "문장"},
            {"metric": "미충족 니즈 문장 수", "value": unmet_count, "unit": "문장"},
            {"metric": "미충족 비율", "value": _safe_float(unmet_count / max(total_count, 1) * 100), "unit": "%"},
        ]

        by_aspect: list[dict[str, Any]] = []
        if not unmet.empty and "aspect" in unmet.columns:
            aspect_total = review_df.groupby("aspect", as_index=False).agg(totalCount=("sentence_text", "count"))
            aspect_unmet = unmet.groupby("aspect", as_index=False).agg(unmetCount=("sentence_text", "count"))
            merged = aspect_unmet.merge(aspect_total, on="aspect", how="left")
            merged["unmetRatio"] = (merged["unmetCount"] / merged["totalCount"].clip(lower=1)) * 100.0
            merged = merged.sort_values("unmetCount", ascending=False)
            by_aspect = [
                {
                    "aspect": row["aspect"],
                    "unmetCount": _safe_int(row["unmetCount"]),
                    "totalCount": _safe_int(row["totalCount"]),
                    "unmetRatio": _safe_float(row["unmetRatio"]),
                }
                for row in merged.to_dict("records")
            ]

        top_sentences: list[dict[str, Any]] = []
        if not unmet.empty:
            sample = unmet.sort_values("helpful_count", ascending=False, na_position="last").head(50)
            top_sentences = [
                {
                    "productId": str(row.get("product_id")),
                    "brand": _label_or_missing(row.get("brand"), "미확인"),
                    "name": row.get("name"),
                    "aspect": row.get("aspect"),
                    "sentence": row.get("sentence_text"),
                    "sentiment": _safe_float(row.get("sentiment_score")),
                }
                for row in sample.to_dict("records")
            ]

        return {"summary": summary, "byAspect": by_aspect, "topSentences": top_sentences}

    # ------------------------------------------------------------------
    # 태스크 4: 사이즈 경향 가이드
    # ------------------------------------------------------------------

    def get_text_size_guide(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        empty: dict[str, Any] = {"productSizeTendency": []}
        if filtered_fact_df.empty:
            return empty

        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        if review_df.empty or "size_tendency" not in review_df.columns:
            return empty

        size_df = review_df[
            (review_df["aspect"] == "size_fit") & review_df["size_tendency"].notna()
        ].copy()
        if size_df.empty:
            return empty

        def _dominant(group: pd.DataFrame) -> pd.Series:
            counts = group["size_tendency"].value_counts()
            dominant = counts.index[0] if not counts.empty else "mixed"
            total = int(counts.sum())
            confidence = float(counts.iloc[0] / total) if total > 0 else 0.0
            small_c = int(counts.get("small", 0))
            true_c = int(counts.get("true_to_size", 0))
            large_c = int(counts.get("large", 0))
            return pd.Series({
                "smallCount": small_c,
                "trueCount": true_c,
                "largeCount": large_c,
                "totalCount": total,
                "dominantTendency": dominant,
                "confidenceLevel": round(confidence, 2),
            })

        grouped = size_df.groupby("product_id", as_index=False).apply(_dominant, include_groups=False).reset_index()
        if "product_id" not in grouped.columns and "level_0" in grouped.columns:
            grouped = grouped.rename(columns={"level_0": "product_id"})

        latest = _latest_by_product(filtered_fact_df)
        dim_cols = [c for c in ["product_id", "brand", "name", "category_l1"] if c in latest.columns]
        if dim_cols:
            grouped = grouped.merge(
                latest[dim_cols].drop_duplicates(subset=["product_id"]),
                on="product_id",
                how="left",
            )
        grouped = grouped.sort_values("totalCount", ascending=False)

        return {
            "productSizeTendency": [
                {
                    "productId": str(row.get("product_id")),
                    "brand": _label_or_missing(row.get("brand"), "미확인"),
                    "name": row.get("name"),
                    "category": _label_or_missing(row.get("category_l1"), "미분류"),
                    "smallCount": _safe_int(row.get("smallCount")),
                    "trueCount": _safe_int(row.get("trueCount")),
                    "largeCount": _safe_int(row.get("largeCount")),
                    "totalCount": _safe_int(row.get("totalCount")),
                    "dominantTendency": row.get("dominantTendency"),
                    "confidenceLevel": _safe_float(row.get("confidenceLevel")),
                }
                for row in grouped.head(120).to_dict("records")
            ]
        }

    # ------------------------------------------------------------------
    # 태스크 5: 트렌드 키워드
    # ------------------------------------------------------------------

    def get_text_trends(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        empty: dict[str, Any] = {"keywordTimeseries": [], "risingKeywords": []}
        if filtered_fact_df.empty:
            return empty

        trend_df = self._load_text_trend_keywords(filters, filtered_fact_df)
        if trend_df.empty:
            return empty

        timeseries_df = _aggregate_keyword_trend_timeseries_by_day(trend_df)
        timeseries = [
            {
                "keyword": row.get("keyword"),
                "keywordType": row.get("keyword_type"),
                "snapshotDate": str(row.get("snapshot_date") or ""),
                "mentionCount": _safe_int(row.get("mention_count")),
                "productCount": _safe_int(row.get("product_count")),
            }
            for row in timeseries_df.to_dict("records")
        ]

        snapshot_dates = sorted(trend_df["snapshot_date"].dropna().unique())
        rising: list[dict[str, Any]] = []
        if len(snapshot_dates) >= 2:
            recent = str(snapshot_dates[-1])
            prior = str(snapshot_dates[-2])
            recent_df = trend_df[trend_df["snapshot_date"].astype(str) == recent]
            prior_df = trend_df[trend_df["snapshot_date"].astype(str) == prior]
            recent_agg = recent_df.groupby(["keyword", "keyword_type"], as_index=False).agg(
                recentCount=("mention_count", "sum"), avgSentiment=("avg_sentiment", "mean")
            )
            prior_agg = prior_df.groupby(["keyword", "keyword_type"], as_index=False).agg(
                priorCount=("mention_count", "sum")
            )
            merged = recent_agg.merge(prior_agg, on=["keyword", "keyword_type"], how="left")
            merged["priorCount"] = merged["priorCount"].fillna(0).astype(int)
            merged["growthRate"] = merged.apply(
                lambda r: ((r["recentCount"] - r["priorCount"]) / max(r["priorCount"], 1)) * 100.0, axis=1
            )
            merged = merged.sort_values("growthRate", ascending=False)
            rising = [
                {
                    "keyword": row["keyword"],
                    "keywordType": row["keyword_type"],
                    "recentCount": _safe_int(row["recentCount"]),
                    "priorCount": _safe_int(row["priorCount"]),
                    "growthRate": _safe_float(row["growthRate"]),
                    "avgSentiment": _safe_float(row["avgSentiment"]),
                }
                for row in merged.head(30).to_dict("records")
            ]
        else:
            total_agg = trend_df.groupby(["keyword", "keyword_type"], as_index=False).agg(
                recentCount=("mention_count", "sum"), avgSentiment=("avg_sentiment", "mean")
            ).sort_values("recentCount", ascending=False)
            rising = [
                {
                    "keyword": row["keyword"],
                    "keywordType": row["keyword_type"],
                    "recentCount": _safe_int(row["recentCount"]),
                    "priorCount": 0,
                    "growthRate": None,
                    "avgSentiment": _safe_float(row["avgSentiment"]),
                }
                for row in total_agg.head(30).to_dict("records")
            ]

        return {"keywordTimeseries": timeseries, "risingKeywords": rising}

    # ------------------------------------------------------------------
    # 태스크 6: 브랜드 이미지 맵핑
    # ------------------------------------------------------------------

    def _load_brand_style_embedding_agg(self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame) -> pd.DataFrame:
        df = self.repository.load_table(filters.dataset, "brand_style_embedding_agg.parquet", "brand_style_embedding_agg")
        if df.empty or filtered_fact_df.empty:
            return pd.DataFrame()
        if not {"snapshot_id", "brand"}.issubset(df.columns):
            return pd.DataFrame()
        keys = filtered_fact_df[["snapshot_id", "brand"]].drop_duplicates()
        return df.merge(keys, on=["snapshot_id", "brand"], how="inner")

    def _load_brand_style_embedding_evidence(
        self, filters: DashboardFilters, filtered_fact_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = self.repository.load_table(
            filters.dataset, "brand_style_embedding_evidence.parquet", "brand_style_embedding_evidence"
        )
        if df.empty or filtered_fact_df.empty:
            return pd.DataFrame()
        if not {"snapshot_id", "brand"}.issubset(df.columns):
            return pd.DataFrame()
        keys = filtered_fact_df[["snapshot_id", "brand"]].drop_duplicates()
        return df.merge(keys, on=["snapshot_id", "brand"], how="inner")

    def _brand_image_from_embedding_agg(
        self,
        agg_df: pd.DataFrame,
        evidence_df: pd.DataFrame,
        embedding_meta: dict[str, Any],
        filtered_fact_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """임베딩 집계 parquet → brandProfile / brandStyleMatrix / evidence."""
        empty: dict[str, Any] = {
            "brandProfile": [],
            "brandStyleMatrix": [],
            "scoringMethod": "embedding",
            "embeddingMeta": embedding_meta,
            "brandImageEvidence": [],
        }
        if agg_df.empty:
            return empty

        try:
            axes_cfg, _ = load_style_axes(AXES_PATH)
        except Exception:
            axes_cfg = []
        label_map = {a["axis_id"]: str(a.get("label_ko", a["axis_id"])) for a in axes_cfg}
        axis_order = [a["axis_id"] for a in axes_cfg] if axes_cfg else list(BRAND_IMAGE_STYLE_AXIS_ORDER)

        brand_products: dict[str, int] = {}
        if "brand" in filtered_fact_df.columns and "product_id" in filtered_fact_df.columns:
            brand_products = (
                filtered_fact_df.dropna(subset=["brand"])
                .groupby("brand")["product_id"]
                .nunique()
                .to_dict()
            )

        summed = (
            agg_df.groupby(["brand", "axis_id"], as_index=False)
            .agg(intent_raw=("intent_raw", "sum"), perceived_raw=("perceived_raw", "sum"))
        )

        def _dict_to_unit_vec(counts: dict[str, float], keys: list[str]) -> list[float]:
            total = sum(counts.get(a, 0.0) for a in keys)
            if total <= 0:
                return [0.0] * len(keys)
            return [counts.get(a, 0.0) / total for a in keys]

        def _cosine_vec(a: list[float], b: list[float]) -> float | None:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na <= 1e-15 or nb <= 1e-15:
                return None
            return dot / (na * nb)

        active_brands = sorted(
            set(summed["brand"].astype(str).unique()),
            key=lambda br: brand_products.get(br, 0),
            reverse=True,
        )[:30]

        brand_profiles: list[dict[str, Any]] = []
        brand_style_matrix: list[dict[str, Any]] = []

        for brand in active_brands:
            sub = summed[summed["brand"].astype(str) == brand]
            i_counts = {str(r["axis_id"]): float(r["intent_raw"]) for _, r in sub.iterrows()}
            p_counts = {str(r["axis_id"]): float(r["perceived_raw"]) for _, r in sub.iterrows()}
            iv = _dict_to_unit_vec(i_counts, axis_order)
            pv = _dict_to_unit_vec(p_counts, axis_order)
            align = _cosine_vec(iv, pv)
            gaps = [pv[j] - iv[j] for j in range(len(axis_order))]

            def _top_styles(vec: list[float], k: int = 3) -> list[dict[str, Any]]:
                ranked = sorted(
                    ((axis_order[j], vec[j]) for j in range(len(vec))),
                    key=lambda t: t[1],
                    reverse=True,
                )
                out: list[dict[str, Any]] = []
                for aid, share in ranked[:k]:
                    if share <= 1e-9:
                        continue
                    lbl = label_map.get(aid, brand_image_style_label_ko(aid))
                    out.append({"style": aid, "styleLabel": lbl, "sharePct": round(share * 100.0, 1)})
                return out

            max_gap = max(gaps) if gaps else 0.0
            min_gap = min(gaps) if gaps else 0.0
            max_j = int(gaps.index(max_gap)) if gaps else 0
            min_j = int(gaps.index(min_gap)) if gaps else 0
            lab_max = label_map.get(axis_order[max_j], brand_image_style_label_ko(axis_order[max_j]))
            lab_min = label_map.get(axis_order[min_j], brand_image_style_label_ko(axis_order[min_j]))

            brand_profiles.append({
                "brand": brand,
                "productCount": int(brand_products.get(brand, 0)),
                "intentStyleTop": _top_styles(iv),
                "perceivedStyleTop": _top_styles(pv),
                "imageAlignment": round(align, 4) if align is not None else None,
                "claimStyleMass": round(sum(i_counts.values()), 4),
                "reviewStyleMass": round(sum(p_counts.values()), 4),
                "customerLedImageNote": f"{lab_max} (+{max_gap:.2f})" if max_gap > 0.02 else None,
                "brandLedImageNote": f"{lab_min} ({min_gap:.2f})" if min_gap < -0.02 else None,
            })

            for j, aid in enumerate(axis_order):
                lbl = label_map.get(aid, brand_image_style_label_ko(aid))
                brand_style_matrix.append({
                    "brand": brand,
                    "style": aid,
                    "styleLabel": lbl,
                    "intentShare": round(iv[j], 4),
                    "perceivedShare": round(pv[j], 4),
                    "styleGap": round(gaps[j], 4),
                })

        evidence_rows: list[dict[str, Any]] = []
        if not evidence_df.empty:
            ev = evidence_df.copy()
            for _, row in ev.head(400).iterrows():
                aid = str(row.get("axis_id", ""))
                evidence_rows.append({
                    "brand": str(row.get("brand", "")),
                    "style": aid,
                    "styleLabel": label_map.get(aid, brand_image_style_label_ko(aid)),
                    "source": str(row.get("source", "")),
                    "rank": int(row.get("rank", 0) or 0),
                    "snippet": str(row.get("snippet", ""))[:400],
                    "contribScore": _safe_float(row.get("contrib_score")),
                    "productId": row.get("product_id"),
                })

        return {
            "brandProfile": brand_profiles,
            "brandStyleMatrix": brand_style_matrix,
            "scoringMethod": "embedding",
            "embeddingMeta": embedding_meta,
            "brandImageEvidence": evidence_rows,
        }

    def get_text_brand_image(self, filters: DashboardFilters) -> dict[str, Any]:
        """브랜드 이미지: 상품 카피(클레임) 기반 의도 vs 리뷰 기반 지각, 스타일 축 정렬."""
        _, filtered_fact_df = self._load_fact(filters)
        empty: dict[str, Any] = {
            "brandProfile": [],
            "brandStyleMatrix": [],
            "scoringMethod": "keyword",
            "embeddingMeta": None,
            "brandImageEvidence": [],
        }
        if filtered_fact_df.empty:
            return empty

        claim_df = self._load_text_claim_facts(filters, filtered_fact_df)
        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        if claim_df.empty and review_df.empty:
            return empty

        agg_emb = self._load_brand_style_embedding_agg(filters, filtered_fact_df)
        if not agg_emb.empty:
            ev_emb = self._load_brand_style_embedding_evidence(filters, filtered_fact_df)
            meta = self.repository.load_json(filters.dataset, "brand_style_embedding_meta.json")
            out = self._brand_image_from_embedding_agg(agg_emb, ev_emb, meta if meta else {}, filtered_fact_df)
            return out

        axis_order = list(BRAND_IMAGE_STYLE_AXIS_ORDER)
        brand_products: dict[str, int] = {}
        if "brand" in filtered_fact_df.columns and "product_id" in filtered_fact_df.columns:
            brand_products = (
                filtered_fact_df.dropna(subset=["brand"])
                .groupby("brand")["product_id"]
                .nunique()
                .to_dict()
            )

        intent_by_brand: dict[str, dict[str, float]] = {}
        if not claim_df.empty and {"brand", "claim_text"}.issubset(claim_df.columns):
            for row in claim_df.itertuples(index=False):
                brand = getattr(row, "brand", None)
                if _is_missing_value(brand):
                    continue
                text = str(getattr(row, "claim_text", "") or "").replace("\u200b", " ").strip()
                if not text:
                    continue
                styles = detect_brand_image_styles(text)
                if not styles:
                    continue
                conf = _safe_float(getattr(row, "confidence", 1.0)) or 1.0
                inc = conf / float(len(styles))
                b = str(brand)
                bucket = intent_by_brand.setdefault(b, {a: 0.0 for a in axis_order})
                for s in styles:
                    bucket[s] = bucket.get(s, 0.0) + inc

        perceived_by_brand: dict[str, dict[str, float]] = {}
        if not review_df.empty and {"brand", "sentence_text"}.issubset(review_df.columns):
            for row in review_df.itertuples(index=False):
                brand = getattr(row, "brand", None)
                if _is_missing_value(brand):
                    continue
                text = str(getattr(row, "sentence_text", "") or "").replace("\u200b", " ").strip()
                if not text:
                    continue
                styles = detect_brand_image_styles(text)
                if not styles:
                    continue
                inc = 1.0 / float(len(styles))
                b = str(brand)
                bucket = perceived_by_brand.setdefault(b, {a: 0.0 for a in axis_order})
                for s in styles:
                    bucket[s] = bucket.get(s, 0.0) + inc

        active_brands = set(intent_by_brand.keys()) | set(perceived_by_brand.keys())
        if not active_brands:
            return empty

        def _to_unit_vec(counts: dict[str, float]) -> list[float]:
            total = sum(counts.get(a, 0.0) for a in axis_order)
            if total <= 0:
                return [0.0] * len(axis_order)
            return [counts.get(a, 0.0) / total for a in axis_order]

        def _cosine(a: list[float], b: list[float]) -> float | None:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na <= 1e-15 or nb <= 1e-15:
                return None
            return dot / (na * nb)

        brand_profiles: list[dict[str, Any]] = []
        brand_style_matrix: list[dict[str, Any]] = []

        sorted_brands = sorted(active_brands, key=lambda br: brand_products.get(br, 0), reverse=True)[:30]
        for brand in sorted_brands:
            i_counts = intent_by_brand.get(brand, {a: 0.0 for a in axis_order})
            p_counts = perceived_by_brand.get(brand, {a: 0.0 for a in axis_order})
            iv = _to_unit_vec(i_counts)
            pv = _to_unit_vec(p_counts)
            align = _cosine(iv, pv)
            gaps = [pv[j] - iv[j] for j in range(len(axis_order))]

            def _top_styles(vec: list[float], k: int = 3) -> list[dict[str, Any]]:
                ranked = sorted(
                    ((axis_order[j], vec[j]) for j in range(len(vec))),
                    key=lambda t: t[1],
                    reverse=True,
                )
                out: list[dict[str, Any]] = []
                for aid, share in ranked[:k]:
                    if share <= 1e-9:
                        continue
                    out.append({
                        "style": aid,
                        "styleLabel": brand_image_style_label_ko(aid),
                        "sharePct": round(share * 100.0, 1),
                    })
                return out

            max_gap = max(gaps) if gaps else 0.0
            min_gap = min(gaps) if gaps else 0.0
            max_j = int(gaps.index(max_gap)) if gaps else 0
            min_j = int(gaps.index(min_gap)) if gaps else 0
            lab_max = brand_image_style_label_ko(axis_order[max_j])
            lab_min = brand_image_style_label_ko(axis_order[min_j])

            brand_profiles.append({
                "brand": brand,
                "productCount": int(brand_products.get(brand, 0)),
                "intentStyleTop": _top_styles(iv),
                "perceivedStyleTop": _top_styles(pv),
                "imageAlignment": round(align, 4) if align is not None else None,
                "claimStyleMass": round(sum(i_counts.values()), 2),
                "reviewStyleMass": round(sum(p_counts.values()), 2),
                "customerLedImageNote": f"{lab_max} (+{max_gap:.2f})" if max_gap > 0.02 else None,
                "brandLedImageNote": f"{lab_min} ({min_gap:.2f})" if min_gap < -0.02 else None,
            })

            for j, aid in enumerate(axis_order):
                brand_style_matrix.append({
                    "brand": brand,
                    "style": aid,
                    "styleLabel": brand_image_style_label_ko(aid),
                    "intentShare": round(iv[j], 4),
                    "perceivedShare": round(pv[j], 4),
                    "styleGap": round(gaps[j], 4),
                })

        return {
            "brandProfile": brand_profiles,
            "brandStyleMatrix": brand_style_matrix,
            "scoringMethod": "keyword",
            "embeddingMeta": None,
            "brandImageEvidence": [],
        }

    # ------------------------------------------------------------------
    # 워드클라우드용 단어 빈도
    # ------------------------------------------------------------------

    _KOREAN_STOPWORDS: set[str] = {
        "이", "그", "저", "것", "수", "등", "들", "및", "좀", "잘",
        "더", "를", "을", "에", "의", "가", "은", "는", "로", "으로",
        "와", "과", "도", "에서", "까지", "부터", "만", "이나", "나",
        "하고", "이고", "인데", "인", "한", "할", "합니다", "하는",
        "있는", "없는", "했", "된", "됩니다", "않", "않은", "같은",
        "위해", "대한", "통해", "하면", "해서", "하게", "되는",
        "너무", "정말", "진짜", "매우", "아주", "조금", "약간",
        "같아요", "있어요", "없어요", "했어요", "됩니다", "합니다",
        "입니다", "에요", "이에요", "거든요", "네요", "요",
        "그냥", "다른", "많이", "하나", "때", "안", "못", "또",
        "다시", "어떤", "그래서", "근데", "그리고", "그런데",
        "되게", "좋아요", "같아", "있어", "없어",
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "with", "by", "it", "this", "that",
    }

    @staticmethod
    def _tokenize_korean(text: str) -> list[str]:
        """공백 기반 토큰화 + 한글/영문 외 문자 제거."""
        import re
        cleaned = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9]", " ", text)
        tokens = cleaned.lower().split()
        return [t for t in tokens if len(t) >= 2]

    def get_text_word_frequency(self, filters: DashboardFilters, max_words: int = 80) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        if filtered_fact_df.empty:
            return {"words": []}

        review_df = self._load_text_review_facts(filters, filtered_fact_df)
        if review_df.empty or "sentence_text" not in review_df.columns:
            return {"words": []}

        counter: dict[str, int] = {}
        for text in review_df["sentence_text"].dropna():
            for token in self._tokenize_korean(str(text)):
                if token not in self._KOREAN_STOPWORDS:
                    counter[token] = counter.get(token, 0) + 1

        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:max_words]
        return {
            "words": [
                {"word": word, "frequency": count}
                for word, count in sorted_words
            ],
        }

    def get_category_overview(
        self,
        filters: DashboardFilters,
        category_level: str = "l3",
        quality_mode: str = "success_only",
        include_fallback: bool = False,
    ) -> dict[str, Any]:
        latest, _ = self._build_category_frames(filters)
        scoped = self._filter_category_frame(latest, quality_mode=quality_mode, include_fallback=include_fallback)
        if scoped.empty:
            return {"summaryRows": [], "marketMap": [], "leaderRows": []}
        scoped["categoryLabel"] = self._category_label(scoped, category_level)
        grouped = (
            scoped.groupby("categoryLabel", as_index=False)
            .agg(
                record_count=("product_id", "count"),
                product_count=("product_id", "nunique"),
                brand_count=("brand", "nunique"),
                avg_rank=("rank", "mean"),
                avg_momentum_score=("momentum_score", "mean"),
                avg_price=("price", "mean"),
                avg_discount_pct=("discount_pct", "mean"),
                fallback_count=("category_is_fallback", "sum"),
            )
            .sort_values(["product_count", "avg_rank"], ascending=[False, True])
        )
        total_products = max(int(grouped["product_count"].sum()), 1)
        grouped["share_of_catalog"] = (grouped["product_count"] / total_products) * 100.0
        summary_rows = [
            {"metric": "표시 카테고리 수", "value": _safe_int(grouped["categoryLabel"].nunique()), "unit": "개"},
            {"metric": "분석 포함 상품 수", "value": _safe_int(scoped["product_id"].nunique()), "unit": "개"},
            {"metric": "fallback 상품 수", "value": _safe_int(scoped["category_is_fallback"].sum()), "unit": "개"},
            {"metric": "품질 모드", "value": "success only" if quality_mode == "success_only" else "success + partial", "unit": "-"},
        ]
        return {
            "summaryRows": summary_rows,
            "marketMap": [
                {
                    "categoryLabel": row["categoryLabel"],
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                    "recordCount": _safe_int(row["record_count"]),
                    "productCount": _safe_int(row["product_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "shareOfCatalog": _safe_float(row["share_of_catalog"]),
                }
                for row in grouped.head(20).to_dict("records")
            ],
            "leaderRows": [
                {
                    "categoryLabel": row["categoryLabel"],
                    "productCount": _safe_int(row["product_count"]),
                    "brandCount": _safe_int(row["brand_count"]),
                    "shareOfCatalog": _safe_float(row["share_of_catalog"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                    "fallbackCount": _safe_int(row["fallback_count"]),
                }
                for row in grouped.head(30).to_dict("records")
            ],
        }

    def get_category_relationships(
        self,
        filters: DashboardFilters,
        category_level: str = "l3",
        quality_mode: str = "success_only",
        include_fallback: bool = False,
    ) -> dict[str, Any]:
        latest, _ = self._build_category_frames(filters)
        scoped = self._filter_category_frame(latest, quality_mode=quality_mode, include_fallback=include_fallback)
        if scoped.empty:
            return {"priceHeatmap": [], "materialHeatmap": [], "colorHeatmap": []}
        scoped["categoryLabel"] = self._category_label(scoped, category_level)
        top_categories = self._top_category_labels(scoped, category_level, limit=12)

        def build_heatmap_rows(source_column: str, target_key: str, limit: int = 12) -> list[dict[str, Any]]:
            if source_column not in scoped.columns:
                return []
            relation = scoped[scoped["categoryLabel"].isin(top_categories)].copy()
            relation[target_key] = (
                relation[source_column].map(lambda value: _label_or_missing(value, "미분류")).astype(str)
            )
            top_values = relation[target_key].value_counts(dropna=False).head(limit).index.astype(str).tolist()
            grouped = (
                relation[relation[target_key].isin(top_values)]
                .groupby(["categoryLabel", target_key], as_index=False, observed=True)
                .agg(
                    count=("product_id", "count"),
                    avg_rank=("rank", "mean"),
                    avg_price=("price", "mean"),
                    avg_momentum_score=("momentum_score", "mean"),
                )
            )
            return [
                {
                    "categoryLabel": row["categoryLabel"],
                    target_key: row[target_key],
                    "count": _safe_int(row["count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgPrice": _safe_float(row["avg_price"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                }
                for row in grouped.to_dict("records")
            ]

        return {
            "priceHeatmap": build_heatmap_rows("price_band", "priceBand"),
            "materialHeatmap": build_heatmap_rows("material_normalized" if "material_normalized" in scoped.columns else "material", "materialValue"),
            "colorHeatmap": build_heatmap_rows("color_normalized" if "color_normalized" in scoped.columns else "color", "colorValue"),
        }

    def get_category_timeseries(
        self,
        filters: DashboardFilters,
        category_level: str = "l3",
        quality_mode: str = "success_only",
        include_fallback: bool = False,
    ) -> dict[str, Any]:
        _, records = self._build_category_frames(filters)
        scoped = self._filter_category_frame(records, quality_mode=quality_mode, include_fallback=include_fallback)
        if scoped.empty or "snapshot_date" not in scoped.columns:
            return {"shareSeries": [], "rankSeries": [], "momentumSeries": []}
        scoped["categoryLabel"] = self._category_label(scoped, category_level)
        top_categories = self._top_category_labels(scoped, category_level, limit=8)
        grouped_all = (
            scoped.groupby(["snapshot_date", "categoryLabel"], as_index=False)
            .agg(
                product_count=("product_id", "nunique"),
                avg_rank=("rank", "mean"),
                avg_momentum_score=("momentum_score", "mean"),
            )
            .sort_values(["snapshot_date", "categoryLabel"])
        )
        daily_totals = grouped_all.groupby("snapshot_date", as_index=False).agg(total_product_count=("product_count", "sum"))

        # 점유율은 "상위 + 기타"로 구성해 시점별 합계가 항상 100%가 되도록 맞춘다.
        grouped_share = grouped_all.copy()
        grouped_share["categoryLabel"] = grouped_share["categoryLabel"].where(
            grouped_share["categoryLabel"].isin(top_categories),
            "기타",
        )
        grouped_share = (
            grouped_share.groupby(["snapshot_date", "categoryLabel"], as_index=False)
            .agg(product_count=("product_count", "sum"))
            .sort_values(["snapshot_date", "categoryLabel"])
        )
        grouped_share = grouped_share.merge(daily_totals, on="snapshot_date", how="left")
        grouped_share["share_of_catalog"] = (
            grouped_share["product_count"] / grouped_share["total_product_count"].replace({0: pd.NA})
        ) * 100.0

        # 순위/모멘텀은 해석 일관성을 위해 상위 카테고리만 유지한다.
        grouped_top = (
            grouped_all[grouped_all["categoryLabel"].isin(top_categories)]
            .copy()
            .sort_values(["snapshot_date", "categoryLabel"])
        )
        grouped_top = grouped_top.merge(daily_totals, on="snapshot_date", how="left")
        grouped_top["share_of_catalog"] = (
            grouped_top["product_count"] / grouped_top["total_product_count"].replace({0: pd.NA})
        ) * 100.0

        return {
            "shareSeries": [
                {
                    "snapshotDate": str(row["snapshot_date"]),
                    "categoryLabel": row["categoryLabel"],
                    "value": _safe_float(row["share_of_catalog"]),
                }
                for row in grouped_share.to_dict("records")
            ],
            "rankSeries": [
                {
                    "snapshotDate": str(row["snapshot_date"]),
                    "categoryLabel": row["categoryLabel"],
                    "value": _safe_float(row["avg_rank"]),
                }
                for row in grouped_top.to_dict("records")
            ],
            "momentumSeries": [
                {
                    "snapshotDate": str(row["snapshot_date"]),
                    "categoryLabel": row["categoryLabel"],
                    "value": _safe_float(row["avg_momentum_score"]),
                }
                for row in grouped_top.to_dict("records")
            ],
        }

    def get_category_quality(
        self,
        filters: DashboardFilters,
        category_level: str = "l3",
    ) -> dict[str, Any]:
        latest, _ = self._build_category_frames(filters)
        if latest.empty:
            return {"statusRows": [], "sourceRows": [], "issueRows": []}
        latest["categoryLabel"] = self._category_label(latest, category_level)
        status_rows = (
            latest.groupby("category_ingest_status", as_index=False)
            .agg(product_count=("product_id", "nunique"))
            .sort_values("product_count", ascending=False)
        )
        source_rows = (
            latest.groupby("category_source", as_index=False)
            .agg(product_count=("product_id", "nunique"))
            .sort_values("product_count", ascending=False)
        )
        issue_mask = (
            latest["category_ingest_status"].astype(str).isin(["failure", "skipped", "partial"])
            | latest["category_quality_tier"].astype(str).eq("low")
        )
        issue_rows = latest[issue_mask].copy()
        issue_rows = issue_rows.sort_values(["category_ingest_status", "rank"], ascending=[True, True])
        return {
            "statusRows": [
                {
                    "status": row["category_ingest_status"],
                    "productCount": _safe_int(row["product_count"]),
                }
                for row in status_rows.to_dict("records")
            ],
            "sourceRows": [
                {
                    "source": row["category_source"],
                    "productCount": _safe_int(row["product_count"]),
                }
                for row in source_rows.to_dict("records")
            ],
            "issueRows": [
                {
                    "productId": str(row.get("product_id")),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "categoryLabel": row.get("categoryLabel"),
                    "categoryIngestStatus": row.get("category_ingest_status"),
                    "categoryQualityTier": row.get("category_quality_tier"),
                    "categorySource": row.get("category_source"),
                    "categoryRawStatus": row.get("category_raw_status"),
                    "categorySkipReason": row.get("category_skip_reason"),
                    "categoryDecisionSource": row.get("category_decision_source"),
                    "categoryEvidenceReason": row.get("category_evidence_reason"),
                    "categoryReviewReasons": row.get("category_review_reasons_json"),
                    "rank": _safe_float(row.get("rank")),
                    "price": _safe_int(row.get("price")),
                }
                for row in issue_rows.head(40).to_dict("records")
            ],
        }

    def get_category_examples(
        self,
        filters: DashboardFilters,
        category_level: str = "l3",
        quality_mode: str = "success_only",
        include_fallback: bool = False,
    ) -> dict[str, Any]:
        latest, _ = self._build_category_frames(filters)
        scoped = self._filter_category_frame(latest, quality_mode=quality_mode, include_fallback=include_fallback)
        if scoped.empty:
            return {"rows": []}
        scoped["categoryLabel"] = self._category_label(scoped, category_level)
        scoped = scoped.sort_values(["rank", "price"], ascending=[True, True])
        return {
            "rows": [
                {
                    "productId": str(row.get("product_id")),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "categoryLabel": row.get("categoryLabel"),
                    "categorySource": row.get("category_source"),
                    "categoryIsFallback": bool(row.get("category_is_fallback")),
                    "categoryIngestStatus": row.get("category_ingest_status"),
                    "categoryQualityTier": row.get("category_quality_tier"),
                    "rank": _safe_float(row.get("rank")),
                    "price": _safe_int(row.get("price")),
                    "discountPct": _safe_float(row.get("discount_pct")),
                }
                for row in scoped.head(50).to_dict("records")
            ]
        }

    def get_tag_correlation(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        text_features_df = self._load_text_features(filters, filtered_fact_df)
        fusion = self._run_dynamic_fusion(filters, filtered_fact_df, text_features_df)
        tag_perf_df = fusion["tag_performance"].head(50)
        return {
            "chart": [
                {
                    "tag": row["tag"],
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                    "avgStabilityScore": _safe_float(row["avg_stability_score"]),
                    "recordCount": _safe_int(row["record_count"]),
                }
                for row in tag_perf_df.head(25).to_dict("records")
            ],
            "rows": _records(tag_perf_df),
        }

    def get_timeseries(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        text_features_df = self._load_text_features(filters, filtered_fact_df)
        fusion = self._run_dynamic_fusion(filters, filtered_fact_df, text_features_df)
        trends_df = fusion["trends"].copy()
        if trends_df.empty:
            return {"series": [], "rows": []}
        return {
            "series": [
                {
                    "snapshotDate": str(row["snapshot_date"]),
                    "recordCount": _safe_int(row["record_count"]),
                    "productCount": _safe_int(row["product_count"]),
                    "avgRank": _safe_float(row["avg_rank"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                }
                for row in trends_df.sort_values("snapshot_date").to_dict("records")
            ],
            "rows": _records(trends_df),
        }

    def get_brand_index(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        text_features_df = self._load_text_features(filters, filtered_fact_df)
        fusion = self._run_dynamic_fusion(filters, filtered_fact_df, text_features_df)
        brand_index_df = fusion["brand_index"].head(50)
        return {
            "chart": [
                {
                    "brand": row["brand"],
                    "avgRank": _safe_float(row["avg_rank"]),
                    "productCount": _safe_int(row["product_count"]),
                    "avgStabilityScore": _safe_float(row["avg_stability_score"]),
                    "avgDiscountPct": _safe_float(row["avg_discount_pct"]),
                    "avgMomentumScore": _safe_float(row["avg_momentum_score"]),
                }
                for row in brand_index_df.head(25).to_dict("records")
            ],
            "rows": _records(brand_index_df),
        }

    def get_embedding_projection(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        projection_df = self.repository.load_table(
            filters.dataset,
            "analysis_embedding_projection.parquet",
            "analysis_embedding_projection",
        )
        projection_df = filter_related_table(projection_df, filtered_fact_df)
        projection_df = _merge_embedding_image_paths(projection_df, self.repository, filters.dataset)
        if projection_df.empty:
            return {"frames": []}

        manifest_df = self.repository.load_table(filters.dataset, "image_manifest.parquet", "image_manifest")
        thumb_paths = _thumbnail_path_by_snapshot_product(manifest_df)
        scoped = projection_df.copy()
        scoped["x"] = pd.to_numeric(scoped.get("x"), errors="coerce")
        scoped["y"] = pd.to_numeric(scoped.get("y"), errors="coerce")
        scoped = scoped.dropna(subset=["x", "y", "product_id"]).copy()
        if scoped.empty:
            return {"frames": []}
        scoped["product_id"] = scoped["product_id"].astype(str)
        scoped["crawl_datetime"] = pd.to_datetime(scoped.get("crawl_datetime"), errors="coerce")

        merge_cols = [
            col
            for col in ("snapshot_id", "product_id", "category_label_l1", "category_l1", "category_label_l2", "category_l2")
            if col in filtered_fact_df.columns
        ]
        if {"snapshot_id", "product_id"}.issubset(set(merge_cols)):
            category_frame = filtered_fact_df[merge_cols].copy()
            category_frame["product_id"] = category_frame["product_id"].astype(str)
            category_frame = category_frame.drop_duplicates(["snapshot_id", "product_id"])
            scoped = scoped.merge(category_frame, on=["snapshot_id", "product_id"], how="left")
        scoped["category_label"], _ = _build_adaptive_embedding_labels(scoped)

        frame_specs = []
        for snapshot_id, frame_df in scoped.groupby("snapshot_id"):
            sort_time = frame_df["crawl_datetime"].min() if "crawl_datetime" in frame_df.columns else pd.NaT
            frame_specs.append((snapshot_id, frame_df.copy(), sort_time))
        frame_specs.sort(key=lambda item: (pd.isna(item[2]), item[2], str(item[0])))

        frames = []
        prev_centers: dict[str, np.ndarray] = {}
        next_cluster_index = 1
        prior_sets: list[set[str]] = []
        last_seen_points: dict[str, dict[str, Any]] = {}
        for idx, (snapshot_id, frame_df, _) in enumerate(frame_specs):
            sorted_frame = frame_df.sort_values(["rank", "product_id"], na_position="last").copy()
            sorted_frame, cluster_rows, prev_centers, next_cluster_index = _frame_cluster_payload(
                sorted_frame,
                prev_centers,
                next_cluster_index,
            )
            current_ids = set(sorted_frame["product_id"].astype(str).tolist())
            prev_1 = prior_sets[-1] if len(prior_sets) >= 1 else set()
            prev_2 = prior_sets[-2] if len(prior_sets) >= 2 else set()

            if idx == 0:
                sorted_frame["lifecycle_state"] = "new"
            elif idx == 1:
                sorted_frame["lifecycle_state"] = sorted_frame["product_id"].map(
                    lambda pid: "retained" if str(pid) in prev_1 else "new"
                )
            else:
                sorted_frame["lifecycle_state"] = sorted_frame["product_id"].map(
                    lambda pid: "retained" if (str(pid) in prev_1 or str(pid) in prev_2) else "new"
                )
            sorted_frame["is_ghost"] = False

            exited_ids = (prev_1 | prev_2) - current_ids if idx > 0 else set()
            ghost_rows: list[dict[str, Any]] = []
            for product_id in sorted(exited_ids):
                last_row = last_seen_points.get(product_id)
                if not last_row:
                    continue
                ghost_rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "frame_label": sorted_frame["frame_label"].iloc[0] if "frame_label" in sorted_frame.columns and not sorted_frame.empty else str(snapshot_id),
                        "product_id": product_id,
                        "brand": last_row.get("brand"),
                        "name": last_row.get("name"),
                        "rank": last_row.get("rank"),
                        "rank_velocity": last_row.get("rank_velocity"),
                        "movement_group": last_row.get("movement_group"),
                        "cluster_id": last_row.get("cluster_id"),
                        "dominant_category": last_row.get("dominant_category"),
                        "dominant_share_pct": last_row.get("dominant_share_pct"),
                        "x": last_row.get("x"),
                        "y": last_row.get("y"),
                        "lifecycle_state": "exited",
                        "is_ghost": True,
                    }
                )
            frame_with_ghosts = pd.concat([sorted_frame, pd.DataFrame(ghost_rows)], ignore_index=True, sort=False)

            if not sorted_frame.empty:
                for row in sorted_frame.to_dict("records"):
                    last_seen_points[str(row.get("product_id"))] = row

            sid_key = str(snapshot_id)
            frame_label = frame_with_ghosts["frame_label"].iloc[0] if "frame_label" in frame_with_ghosts.columns and not frame_with_ghosts.empty else snapshot_id
            points = [
                {
                    "productId": str(row["product_id"]),
                    "brand": row.get("brand"),
                    "name": row.get("name"),
                    "rank": _safe_float(row.get("rank")),
                    "rankVelocity": _safe_float(row.get("rank_velocity")),
                    "movementGroup": row.get("movement_group"),
                    "clusterId": row.get("cluster_id"),
                    "dominantCategory": row.get("dominant_category"),
                    "x": _safe_float(row.get("x")),
                    "y": _safe_float(row.get("y")),
                    "mainImage": _optional_fs_path(thumb_paths.get((sid_key, str(row["product_id"])))),
                    "lifecycleState": row.get("lifecycle_state"),
                    "isGhost": bool(row.get("is_ghost")),
                }
                for row in frame_with_ghosts.to_dict("records")
            ]

            new_count = int((sorted_frame["lifecycle_state"] == "new").sum()) if not sorted_frame.empty else 0
            retained_count = int((sorted_frame["lifecycle_state"] == "retained").sum()) if not sorted_frame.empty else 0
            exited_count = len(ghost_rows)
            frames.append(
                {
                    "snapshotId": snapshot_id,
                    "label": frame_label,
                    "newCount": new_count,
                    "retainedCount": retained_count,
                    "exitedCount": exited_count,
                    "clusterShareTopN": cluster_rows[:5],
                    "points": points,
                }
            )
            prior_sets.append(current_ids)
        return {"frames": frames}

    def get_embedding_overview(self, filters: DashboardFilters) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        projection_df = self.repository.load_table(
            filters.dataset,
            "analysis_embedding_projection.parquet",
            "analysis_embedding_projection",
        )
        projection_df = filter_related_table(projection_df, filtered_fact_df)
        projection_df = _merge_embedding_image_paths(projection_df, self.repository, filters.dataset)
        if projection_df.empty:
            return {"points": [], "clusters": [], "summary": []}

        manifest_df = self.repository.load_table(filters.dataset, "image_manifest.parquet", "image_manifest")
        thumb_paths = _thumbnail_path_by_snapshot_product(manifest_df)

        scoped = projection_df.copy()
        scoped["x"] = pd.to_numeric(scoped.get("x"), errors="coerce")
        scoped["y"] = pd.to_numeric(scoped.get("y"), errors="coerce")
        scoped = scoped.dropna(subset=["x", "y", "product_id"])
        if scoped.empty:
            return {"points": [], "clusters": [], "summary": []}

        merge_cols = [
            col
            for col in (
                "snapshot_id",
                "product_id",
                "category_label_l1",
                "category_l1",
                "category_label_l2",
                "category_l2",
                "category_label_l3",
                "category_l3",
                "name_item",
                "color_normalized",
                "color",
                "material_normalized",
                "material",
            )
            if col in filtered_fact_df.columns
        ]
        if {"snapshot_id", "product_id"}.issubset(set(merge_cols)):
            category_frame = filtered_fact_df[merge_cols].drop_duplicates(["snapshot_id", "product_id"])
            scoped = scoped.merge(category_frame, on=["snapshot_id", "product_id"], how="left")
        scoped["category_label"], category_strategy = _build_adaptive_embedding_labels(scoped)
        scoped["is_unclassified"] = _is_unclassified_series(scoped["category_label"])

        if "crawl_datetime" in scoped.columns:
            scoped["crawl_datetime"] = pd.to_datetime(scoped["crawl_datetime"], errors="coerce")
            scoped = scoped.sort_values(["crawl_datetime", "snapshot_id", "rank"], na_position="last")
        else:
            scoped = scoped.sort_values(["snapshot_id", "rank"], na_position="last")
        latest = scoped.drop_duplicates(subset=["product_id"], keep="last").reset_index(drop=True)
        excluded_unclassified_count = int(latest["is_unclassified"].sum()) if "is_unclassified" in latest.columns else 0
        latest = latest[~latest["is_unclassified"]].copy().reset_index(drop=True)
        if latest.empty:
            return {
                "points": [],
                "clusters": [],
                "summary": [
                    {"metric": "고유 상품 수(분포 반영)", "value": 0, "unit": "개"},
                    {"metric": "미분류 제외 건수", "value": excluded_unclassified_count, "unit": "개"},
                    {"metric": "카테고리 표시 전략", "value": category_strategy, "unit": "-"},
                ],
                "strategy": category_strategy,
            }

        points_array = latest[["x", "y"]].to_numpy(dtype=np.float64)
        n_points = len(latest)
        if n_points <= 1:
            labels = np.zeros((n_points,), dtype=np.int32)
        else:
            cluster_count = max(4, min(14, int(np.sqrt(max(n_points, 4) / 2))))
            labels = _kmeans_assignments(points_array, cluster_count)
        latest["cluster_id"] = [f"cluster_{int(label) + 1}" for label in labels]

        cluster_rows = []
        total_weighted_purity = 0.0

        def summarize_top_labels(
            series: pd.Series,
            *,
            limit: int = 2,
            missing: set[str] | None = None,
            drop_junk: bool = False,
        ) -> str:
            cleaned = series.dropna().map(lambda value: str(value).strip())
            if missing:
                cleaned = cleaned[~cleaned.isin(missing)]
            cleaned = cleaned[cleaned != ""]
            if drop_junk:
                cleaned = cleaned[~cleaned.map(_is_junk_attribute_label)]
            if cleaned.empty:
                return "-"
            counts = cleaned.value_counts().head(limit)
            total = float(counts.sum()) if counts.sum() else 1.0
            parts = [f"{label} {count / total * 100.0:.0f}%" for label, count in counts.items()]
            return " / ".join(parts)

        for cluster_id, frame in latest.groupby("cluster_id"):
            category_counts = frame["category_label"].value_counts()
            point_count = int(len(frame))
            dominant_category = str(category_counts.index[0]) if not category_counts.empty else "미분류"
            dominant_ratio = float(category_counts.iloc[0] / point_count) if point_count > 0 and not category_counts.empty else 0.0
            total_weighted_purity += dominant_ratio * point_count
            l3_series = (
                frame["category_label_l3"]
                if "category_label_l3" in frame.columns
                else frame["category_l3"]
                if "category_l3" in frame.columns
                else pd.Series([], dtype="object")
            )
            item_series = frame["name_item"] if "name_item" in frame.columns else pd.Series([], dtype="object")
            color_series = (
                frame["color_normalized"]
                if "color_normalized" in frame.columns
                else frame["color"]
                if "color" in frame.columns
                else pd.Series([], dtype="object")
            )
            material_series = (
                frame["material_normalized"]
                if "material_normalized" in frame.columns
                else frame["material"]
                if "material" in frame.columns
                else pd.Series([], dtype="object")
            )
            l3_preview = summarize_top_labels(l3_series, missing={"미분류", "미입력", "미정규화"})
            item_preview = summarize_top_labels(item_series, missing={"미입력", "미정규화"})
            color_preview = summarize_top_labels(color_series, missing={"미입력", "미정규화", "기타"}, drop_junk=True)
            material_preview = summarize_top_labels(material_series, missing={"미입력", "미정규화", "기타"}, drop_junk=True)
            avg_rank = float(frame["rank"].dropna().mean()) if "rank" in frame.columns and frame["rank"].dropna().size else None
            cluster_rows.append(
                {
                    "clusterId": cluster_id,
                    "pointCount": point_count,
                    "dominantCategory": dominant_category,
                    "dominantSharePct": dominant_ratio * 100.0,
                    "categoryCount": int(category_counts.size),
                    "l3Preview": l3_preview,
                    "itemPreview": item_preview,
                    "colorPreview": color_preview,
                    "materialPreview": material_preview,
                    "avgRank": avg_rank,
                }
            )
        cluster_rows.sort(key=lambda row: row["pointCount"], reverse=True)
        weighted_purity = (total_weighted_purity / n_points) if n_points else 0.0

        summary_rows = [
            {"metric": "고유 상품 수(분포 반영)", "value": n_points, "unit": "개"},
            {"metric": "미분류 제외 건수", "value": excluded_unclassified_count, "unit": "개"},
            {"metric": "카테고리 표시 전략", "value": category_strategy, "unit": "-"},
            {"metric": "클러스터 수", "value": len(cluster_rows), "unit": "개"},
            {"metric": "클러스터 순도(가중)", "value": weighted_purity * 100.0, "unit": "%"},
        ]

        points = [
            {
                "productId": str(row.get("product_id")),
                "snapshotId": row.get("snapshot_id"),
                "name": row.get("name"),
                "brand": row.get("brand"),
                "rank": _safe_float(row.get("rank")),
                "categoryLabel": row.get("category_label"),
                "clusterId": row.get("cluster_id"),
                "x": _safe_float(row.get("x")),
                "y": _safe_float(row.get("y")),
                "mainImage": _optional_fs_path(
                    thumb_paths.get((str(row.get("snapshot_id")), str(row.get("product_id")))),
                ),
            }
            for row in latest.to_dict("records")
        ]
        return {"points": points, "clusters": cluster_rows, "summary": summary_rows, "strategy": category_strategy}

    def get_rank_race(self, filters: DashboardFilters, entity_type: str) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        race_df = self.repository.load_table(
            filters.dataset,
            "analysis_rank_race.parquet",
            "analysis_rank_race",
        )
        race_df = filter_related_table(race_df, filtered_fact_df)
        if race_df.empty:
            return {"entityType": entity_type, "frames": []}

        race_df = race_df[race_df["entity_type"] == entity_type].copy()
        frames = []
        for snapshot_id, frame_df in race_df.groupby("snapshot_id"):
            sorted_frame = frame_df.sort_values(["rank", "entity_id"])
            frames.append(
                {
                    "snapshotId": snapshot_id,
                    "label": sorted_frame["frame_label"].iloc[0] if "frame_label" in sorted_frame.columns else snapshot_id,
                    "bars": [
                        {
                            "entityId": str(row["entity_id"]),
                            "entityLabel": row["entity_label"],
                            "rank": _safe_float(row["rank"]),
                            "rankDelta": _safe_float(row["rank_delta"]),
                            "momentumScore": _safe_float(row["momentum_score"]),
                        }
                        for row in sorted_frame.to_dict("records")
                    ],
                }
            )
        return {"entityType": entity_type, "frames": frames}

    def get_rank_trajectories(self, filters: DashboardFilters, entity_type: str) -> dict[str, Any]:
        _, filtered_fact_df = self._load_fact(filters)
        filtered_fact_df = _ensure_rank_energy_momentum(filtered_fact_df)
        # 1) Pre-built table이 있으면 사용
        trajectories_df = self.repository.load_table(
            filters.dataset,
            "analysis_rank_trajectories.parquet",
            "analysis_rank_trajectories",
        )
        trajectories_df = filter_related_table(trajectories_df, filtered_fact_df)
        if trajectories_df.empty and not filtered_fact_df.empty:
            # 2) fact에 날짜·순위가 있으면 그대로 시계열로 사용 (수집된 전체 시퀀스 = 순위 변동)
            need_cols = ["snapshot_id", "crawl_datetime", "product_id", "rank"]
            if entity_type == "product":
                available = [c for c in need_cols if c in filtered_fact_df.columns]
                if len(available) == len(need_cols):
                    filtered_fact_df["crawl_datetime"] = pd.to_datetime(
                        filtered_fact_df["crawl_datetime"], errors="coerce"
                    )
                    trajectories_df = filtered_fact_df.copy()
                    trajectories_df["entity_type"] = "product"
                    trajectories_df["entity_id"] = trajectories_df["product_id"].astype(str)
                    trajectories_df["entity_label"] = (
                        trajectories_df["name"].fillna(trajectories_df["product_id"].astype(str))
                        if "name" in trajectories_df.columns
                        else trajectories_df["product_id"].astype(str)
                    )
                    trajectories_df["rank_delta"] = (
                        trajectories_df["rank_velocity"]
                        if "rank_velocity" in trajectories_df.columns
                        else pd.NA
                    )
                    trajectories_df["momentum_score"] = (
                        trajectories_df["momentum_score"]
                        if "momentum_score" in trajectories_df.columns
                        else pd.NA
                    )
                    for column in ["rank_energy", "energy_velocity", "energy_acceleration", "consistency_score", "momentum_event_state", "momentum_event_label"]:
                        trajectories_df[column] = trajectories_df[column] if column in trajectories_df.columns else pd.NA
                    trajectories_df["record_count"] = 1
            elif entity_type == "brand" and "brand" in filtered_fact_df.columns:
                agg_dict = {
                    "rank": ("rank", "mean"),
                    "record_count": ("product_id", "count"),
                }
                if "rank_velocity" in filtered_fact_df.columns:
                    agg_dict["rank_delta"] = ("rank_velocity", "mean")
                if "momentum_score" in filtered_fact_df.columns:
                    agg_dict["momentum_score"] = ("momentum_score", "mean")
                if "rank_energy" in filtered_fact_df.columns:
                    agg_dict["rank_energy"] = ("rank_energy", "mean")
                if "energy_velocity" in filtered_fact_df.columns:
                    agg_dict["energy_velocity"] = ("energy_velocity", "mean")
                if "energy_acceleration" in filtered_fact_df.columns:
                    agg_dict["energy_acceleration"] = ("energy_acceleration", "mean")
                if "consistency_score" in filtered_fact_df.columns:
                    agg_dict["consistency_score"] = ("consistency_score", "mean")
                agg = filtered_fact_df.groupby(
                    ["snapshot_id", "crawl_datetime", "brand"], as_index=False
                ).agg(**agg_dict)
                agg["crawl_datetime"] = pd.to_datetime(agg["crawl_datetime"], errors="coerce")
                agg["entity_type"] = "brand"
                agg["entity_id"] = agg["brand"].astype(str)
                agg["entity_label"] = agg["brand"].astype(str)
                if "rank_delta" not in agg.columns:
                    agg["rank_delta"] = None
                if "momentum_score" not in agg.columns:
                    agg["momentum_score"] = None
                for column in ["rank_energy", "energy_velocity", "energy_acceleration", "consistency_score"]:
                    if column not in agg.columns:
                        agg[column] = None
                agg["momentum_event_state"] = None
                agg["momentum_event_label"] = None
                trajectories_df = agg

        if trajectories_df.empty:
            return {"entityType": entity_type, "series": []}

        trajectories_df = trajectories_df[trajectories_df["entity_type"] == entity_type].copy()
        if entity_type == "product" and not filtered_fact_df.empty and {"snapshot_id", "product_id"}.issubset(set(filtered_fact_df.columns)):
            fact_lookup = filtered_fact_df.copy()
            fact_lookup["entity_id"] = fact_lookup["product_id"].astype(str)
            fact_lookup["estimated_original_price"] = fact_lookup.apply(
                lambda row: _estimated_original_price(row.get("price"), row.get("discount_pct")),
                axis=1,
            )
            fact_lookup["estimated_original_price_band"] = _price_band_from_original_price(fact_lookup["estimated_original_price"])
            fact_lookup["estimated_original_price_band"] = (
                fact_lookup["estimated_original_price_band"].astype(str).replace("nan", "미분류")
            )
            if "price_band" in fact_lookup.columns:
                fact_lookup["price_band"] = fact_lookup["price_band"].astype(str).replace("nan", "미분류")
            else:
                fact_lookup["price_band"] = "미분류"
            trajectories_df["entity_id"] = trajectories_df["entity_id"].astype(str)
            override_columns = [
                "momentum_score",
                "rank_energy",
                "energy_velocity",
                "energy_acceleration",
                "consistency_score",
                "momentum_event_state",
                "momentum_event_label",
            ]
            trajectories_df = trajectories_df.drop(columns=[column for column in override_columns if column in trajectories_df.columns])
            fact_columns = ["snapshot_id", "entity_id"] + [
                column
                for column in [
                    "brand",
                    "price_band",
                    "estimated_original_price_band",
                    "momentum_score",
                    "rank_energy",
                    "energy_velocity",
                    "energy_acceleration",
                    "consistency_score",
                    "momentum_event_state",
                    "momentum_event_label",
                ]
                if column not in trajectories_df.columns
            ]
            fact_columns = [column for column in fact_columns if column in fact_lookup.columns]
            trajectories_df = trajectories_df.merge(
                fact_lookup[fact_columns].drop_duplicates(
                    subset=["snapshot_id", "entity_id"],
                    keep="last",
                ),
                how="left",
                on=["snapshot_id", "entity_id"],
            )
        return {
            "entityType": entity_type,
            "series": [
                {
                    "snapshotId": row["snapshot_id"],
                    "crawlDatetime": row["crawl_datetime"].isoformat()
                    if isinstance(row["crawl_datetime"], pd.Timestamp)
                    else None,
                    "entityId": str(row["entity_id"]),
                    "entityLabel": row["entity_label"],
                    "brand": row.get("brand"),
                    "rank": _safe_float(row["rank"]),
                    "rankDelta": _safe_float(row["rank_delta"]) if "rank_delta" in row and row.get("rank_delta") is not None else None,
                    "momentumScore": _safe_float(row["momentum_score"]) if "momentum_score" in row and row.get("momentum_score") is not None else None,
                    "rankEnergy": _safe_float(row.get("rank_energy")),
                    "energyVelocity": _safe_float(row.get("energy_velocity")),
                    "energyAcceleration": _safe_float(row.get("energy_acceleration")),
                    "persistence": _safe_float(row.get("consistency_score")),
                    "eventState": row.get("momentum_event_state"),
                    "eventLabel": row.get("momentum_event_label"),
                    "recordCount": _safe_int(row["record_count"]) if "record_count" in row else 1,
                    "priceBand": row.get("price_band"),
                    "estimatedOriginalPriceBand": row.get("estimated_original_price_band"),
                }
                for row in trajectories_df.sort_values(["crawl_datetime", "rank", "entity_id"]).to_dict("records")
            ],
        }
