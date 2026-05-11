#!/usr/bin/env python3
"""
구조화 텍스트 기반 분류 및 정규화.
제품명 파싱, 태그 분류, 색상/소재 정규화.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- 제품명 파싱 패턴 ----
NAME_FIT_RE = re.compile(
    r"(세미\s*와이드|와이드|오버|슬림|레귤러|루즈|배기|스트레이트|테이퍼드|벌룬|오버핏|슬림핏|레귤러핏)"
)
NAME_ITEM_RE = re.compile(
    r"(슬랙스|팬츠|바지|자켓|코트|셔츠|티셔츠|맨투맨|후드|스웨트|패딩|니트|원피스|스커트|"
    r"스니커즈|운동화|신발|부츠|로퍼|백팩|가방|크로스백|지갑|벨트|모자|캡)"
)
NAME_COLOR_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
NAME_FEATURE_RE = re.compile(
    r"(밴딩|기모|스트레치|방수|경량|오가닉|히든|다기능|폴리|코튼|데님|레더|합성)"
)

# ---- 태그 상위 카테고리 키워드 (소문자 매칭용) ----
TAG_TAXONOMY: Dict[str, List[str]] = {
    "fit_style": [
        "와이드", "오버", "슬림", "레귤러", "루즈", "배기", "벌룬", "핏", "테이퍼", "스트레이트",
    ],
    "item_type": [
        "팬츠", "바지", "자켓", "코트", "셔츠", "티", "맨투맨", "후드", "스웨트", "패딩", "니트",
        "스니커즈", "운동화", "신발", "부츠", "로퍼", "백팩", "가방", "크로스백", "모자", "캡", "벨트",
    ],
    "occasion": [
        "데일리", "캐주얼", "아우터", "간절기", "등산", "트레이닝", "출퇴근", "룩북",
    ],
    "aesthetics": [
        "미니멀", "클래식", "빈티지", "스트릿", "젠더리스", "유니섹스", "무지",
    ],
    "brand_collection": [
        "specialdrop", "콜라보", "한정", "lab", "capsule",
    ],
}

# ---- 색상 정규화 (표준 -> 동의어 목록) ----
COLOR_NORMALIZE: Dict[str, List[str]] = {
    "블랙": ["블랙", "black", "검정", "검정색", "bk"],
    "화이트": ["화이트", "white", "흰색", "wh"],
    "그레이": ["그레이", "grey", "gray", "회색", "멜란지"],
    "네이비": ["네이비", "navy"],
    "베이지": ["베이지", "beige"],
    "브라운": ["브라운", "brown"],
    "카키": ["카키", "khaki"],
    "올리브": ["올리브", "olive"],
    "레드": ["레드", "red", "와인"],
    "블루": ["블루", "blue"],
    "핑크": ["핑크", "pink"],
    "그린": ["그린", "green"],
    "옐로우": ["옐로우", "yellow"],
    "오렌지": ["오렌지", "orange"],
    "퍼플": ["퍼플", "purple"],
    "아이보리": ["아이보리", "ivory"],
    "크림": ["크림", "cream"],
    "차콜": ["차콜", "charcoal"],
    "실버": ["실버", "silver"],
    "골드": ["골드", "gold"],
}

# ---- 소재 정규화 ----
MATERIAL_NORMALIZE: Dict[str, List[str]] = {
    "면": ["면", "코튼", "cotton", "오가닉"],
    "폴리에스터": ["폴리에스터", "폴리", "polyester", "poly"],
    "나일론": ["나일론", "nylon"],
    "울": ["울", "wool"],
    "레더": ["레더", "가죽", "leather", "인조가죽"],
    "데님": ["데님", "denim"],
    "기타": ["혼용", "기타", "other"],
}

def _first_match(text: Optional[str], pattern: re.Pattern) -> Optional[str]:
    if not text:
        return None
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _all_matches(text: Optional[str], pattern: re.Pattern) -> List[str]:
    if not text:
        return []
    return [g.strip() for g in pattern.findall(text) if g and g.strip()]


def parse_product_name(name: Optional[str]) -> Dict[str, Any]:
    """제품명에서 핏/아이템/색상/기능 키워드 추출."""
    if not name or not str(name).strip():
        return {"name_fit": None, "name_item": None, "name_color": None, "name_features": []}
    s = str(name).strip()
    bracket = _first_match(s, NAME_COLOR_BRACKET_RE)
    return {
        "name_fit": _first_match(s, NAME_FIT_RE),
        "name_item": _first_match(s, NAME_ITEM_RE),
        "name_color": bracket if bracket and len(bracket) < 30 else None,
        "name_features": _all_matches(s, NAME_FEATURE_RE),
    }


def classify_tag(tag: str) -> Optional[str]:
    """단일 태그를 상위 카테고리로 분류."""
    if not tag:
        return None
    t = tag.replace("#", "").strip().lower()
    for category, keywords in TAG_TAXONOMY.items():
        for kw in keywords:
            if kw in t:
                return category
    return "other"


def normalize_color(raw: Any) -> Optional[str]:
    """색상 문자열을 표준 값으로 정규화."""
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    raw_str = str(raw).strip()
    if not raw_str or raw_str.lower() == "nan":
        return None
    s = raw_str.lower()
    for canonical, synonyms in COLOR_NORMALIZE.items():
        for syn in synonyms:
            if syn.lower() in s or s in syn.lower():
                return canonical
    return raw_str if len(raw_str) < 50 else None


def normalize_material(raw: Any) -> Optional[str]:
    """소재 문자열을 표준 값으로 정규화."""
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    raw_str = str(raw).strip()
    if not raw_str or raw_str.lower() == "nan":
        return None
    s = raw_str.lower()
    for canonical, synonyms in MATERIAL_NORMALIZE.items():
        for syn in synonyms:
            if syn.lower() in s:
                return canonical
    return raw_str if len(raw_str) < 100 else None


def build_text_features(featured_df: pd.DataFrame) -> pd.DataFrame:
    """
    fact_snapshots를 기준으로 제품별 텍스트 피처 테이블 생성.
    featured_df: fact_snapshots 스타일 (snapshot_id, product_id, name, tags_joined, color, material, ...)
    """
    if featured_df.empty:
        return pd.DataFrame()

    keys = featured_df[["snapshot_id", "product_id"]].drop_duplicates()
    rows: List[Dict[str, Any]] = []

    for _, row in keys.iterrows():
        sid = row["snapshot_id"]
        pid = row["product_id"]
        feat = featured_df[(featured_df["snapshot_id"] == sid) & (featured_df["product_id"] == pid)].iloc[0]

        name = feat.get("name")
        parsed = parse_product_name(name)
        tags_joined = feat.get("tags_joined")
        if tags_joined is None:
            tags_joined = feat.get("tags")
        if isinstance(tags_joined, (list, tuple, set)):
            tags_list = [str(t).strip() for t in tags_joined if str(t).strip()]
        elif isinstance(tags_joined, str):
            tags_list = [t.strip() for t in tags_joined.split(",") if t.strip()]
        elif hasattr(tags_joined, "tolist"):
            converted = tags_joined.tolist()
            if isinstance(converted, list):
                tags_list = [str(t).strip() for t in converted if str(t).strip()]
            elif converted is None:
                tags_list = []
            else:
                tags_list = [t.strip() for t in str(converted).split(",") if t.strip()]
        elif tags_joined is None:
            tags_list = []
        else:
            tags_list = [t.strip() for t in str(tags_joined).split(",") if t.strip()]
        tag_categories: Dict[str, List[str]] = {}
        for t in tags_list:
            cat = classify_tag(t)
            if cat:
                tag_categories.setdefault(cat, []).append(t)
        color_raw = feat.get("color")
        material_raw = feat.get("material")
        color_normalized = normalize_color(color_raw)
        material_normalized = normalize_material(material_raw)
        name_feature_count = len(parsed["name_features"])
        text_richness = (
            (len(tags_list) * 2)
            + name_feature_count
            + (1 if parsed["name_item"] else 0)
            + (1 if parsed["name_fit"] else 0)
            + (1 if color_normalized else 0)
            + (1 if material_normalized else 0)
        )

        rows.append({
            "snapshot_id": sid,
            "product_id": pid,
            "name_fit": parsed["name_fit"],
            "name_item": parsed["name_item"],
            "name_color": parsed["name_color"],
            "name_features": json.dumps(parsed["name_features"], ensure_ascii=False) if parsed["name_features"] else None,
            "tag_categories": json.dumps(tag_categories, ensure_ascii=False) if tag_categories else None,
            "color_normalized": color_normalized,
            "material_normalized": material_normalized,
            "text_richness_score": round(text_richness, 2),
        })

    return pd.DataFrame(rows)
