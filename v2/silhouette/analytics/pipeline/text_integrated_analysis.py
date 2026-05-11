#!/usr/bin/env python3
"""
리뷰 원문 + OCR/메타 텍스트를 통합해 텍스트 분석 산출물을 생성한다.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REVIEW_COLUMNS = [
    "snapshot_id",
    "snapshot_date",
    "snapshot_time",
    "crawl_datetime",
    "product_id",
    "brand",
    "name",
    "review_id",
    "created_at",
    "rating",
    "review_type",
    "sort_source",
    "photo_review",
    "helpful_count",
    "option_text",
    "sentence_id",
    "sentence_text",
    "sentence_length",
    "sentiment_label",
    "sentiment_score",
    "aspect",
    "tpo",
    "is_unmet_need",
    "size_tendency",
]

CLAIM_COLUMNS = [
    "snapshot_id",
    "snapshot_date",
    "snapshot_time",
    "crawl_datetime",
    "product_id",
    "brand",
    "name",
    "aspect",
    "claim_text",
    "claim_type",
    "claim_polarity",
    "claim_score",
    "source_field",
    "confidence",
]

GAP_COLUMNS = [
    "snapshot_id",
    "snapshot_date",
    "snapshot_time",
    "crawl_datetime",
    "product_id",
    "brand",
    "name",
    "category_code",
    "category_l1",
    "category_l2",
    "category_l3",
    "price_band",
    "rank",
    "aspect",
    "review_sentence_count",
    "review_count",
    "avg_rating",
    "review_sentiment_score",
    "claim_count",
    "claim_score",
    "gap_score",
    "gap_abs",
    "gap_direction",
]

FUSION_COLUMNS = [
    "snapshot_id",
    "snapshot_date",
    "snapshot_time",
    "crawl_datetime",
    "product_id",
    "brand",
    "name",
    "category_code",
    "category_l1",
    "category_l2",
    "category_l3",
    "price_band",
    "rank",
    "aspect",
    "review_sentence_count",
    "review_count",
    "review_signal",
    "review_signal_std",
    "claim_count",
    "claim_signal",
    "claim_signal_std",
    "claim_confidence",
    "claim_source_coverage_pct",
    "source_field_count",
    "evidence_count",
    "evidence_density",
    "agreement_rate",
    "diversity_penalty",
    "fusion_score",
    "confidence_score",
]

TREND_COLUMNS = [
    "snapshot_id",
    "snapshot_date",
    "keyword",
    "keyword_type",
    "mention_count",
    "product_count",
    "avg_sentiment",
]

# ---------------------------------------------------------------------------
# 태스크 1: ABSA 확장 키워드 사전
# ---------------------------------------------------------------------------

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "size_fit": ["사이즈", "정사이즈", "작게", "크게", "핏", "fit", "기장", "허리", "어깨", "통"],
    "comfort": ["편함", "편해", "편안", "착용감", "쿠션", "무게", "가볍", "부담없", "무겁"],
    "design": ["디자인", "코디", "예쁘", "이쁘", "스타일", "컬러", "색감", "실루엣", "라인"],
    "quality": ["마감", "퀄리티", "내구", "재질", "소재", "봉제", "보풀", "올풀림"],
    "price_value": ["가격", "가성비", "할인", "비싸", "저렴", "합리적"],
    "delivery_service": ["배송", "포장", "교환", "반품", "cs", "고객센터"],
    "scent_beauty": ["향", "지속력", "발림", "흡수", "끈적", "트러블"],
    "color_accuracy": ["실물", "화면", "모니터", "색상차", "어둡", "밝"],
    "thickness": ["두께", "두꺼", "얇", "비침", "비치", "시스루", "안비침"],
    "durability": ["세탁", "변형", "줄어", "늘어", "색빠짐", "물빠짐", "구김"],
    "stretch": ["신축", "스판", "스트레치", "늘어나", "탄성"],
}

# ---------------------------------------------------------------------------
# 태스크 2: TPO / 라이프스타일 키워드
# ---------------------------------------------------------------------------

TPO_KEYWORDS: dict[str, list[str]] = {
    "commute": ["출근", "오피스", "사무실", "회사"],
    "wedding_guest": ["하객", "결혼식", "피로연"],
    "casual": ["데일리", "일상", "편하게", "캐주얼"],
    "date": ["데이트", "소개팅", "미팅"],
    "travel": ["여행", "휴가", "나들이"],
    "exercise": ["운동", "헬스", "요가", "필라테스", "러닝", "등산"],
    "school": ["학교", "대학", "등교", "강의"],
    "season_summer": ["여름", "시원", "반팔", "린넨"],
    "season_winter": ["겨울", "보온", "따뜻", "기모", "패딩"],
}

# ---------------------------------------------------------------------------
# 태스크 3: 미충족 니즈 패턴
# ---------------------------------------------------------------------------

UNMET_NEED_PATTERNS: list[str] = [
    "였으면", "있었으면", "으면 좋겠", "아쉽", "아쉬운",
    "생각보다", "예상보다", "기대보다",
    "없어서", "없는게", "없는 게",
    "추가되", "추가했으면", "개선",
    "불편", "불만", "애매",
]

# ---------------------------------------------------------------------------
# 태스크 4: 사이즈 경향 키워드
# ---------------------------------------------------------------------------

SIZE_SMALL_KEYWORDS = ["작", "타이트", "짧", "좁", "빡빡", "끼"]
SIZE_LARGE_KEYWORDS = ["크", "넉넉", "넓", "길", "오버", "루즈", "헐렁"]
SIZE_TRUE_KEYWORDS = ["정사이즈", "딱 맞", "사이즈 맞", "정확"]

# ---------------------------------------------------------------------------
# 태스크 5: 트렌드 키워드 사전
# ---------------------------------------------------------------------------

TREND_COLOR_KEYWORDS: list[str] = [
    "버터", "옐로", "라벤더", "민트", "카키", "베이지",
    "아이보리", "그레이", "블랙", "화이트", "네이비",
    "올리브", "모카", "브라운", "핑크", "레드", "블루",
    "크림", "차콜", "카멜", "코발트", "머스타드",
]
TREND_STYLE_KEYWORDS: list[str] = [
    "오버핏", "슬림", "와이드", "크롭", "롱", "미니",
    "발레코어", "올드머니", "고프코어", "미니멀", "시크",
    "레트로", "스트릿", "페미닌", "캐주얼", "포멀",
]

# ---------------------------------------------------------------------------
# 브랜드 이미지(스타일) 축 — 상품 카피(클레임) vs 리뷰 정렬
# ---------------------------------------------------------------------------

BRAND_IMAGE_STYLE_AXES: dict[str, list[str]] = {
    "minimal": [
        "미니멀", "미니멀리즘", "심플", "클린", "베이직", "단정",
        "minimal", "simple", "clean",
    ],
    "chic": [
        "시크", "세련", "도회", "urban", "글램",
    ],
    "street": [
        "스트릿", "힙합", "힙", "스케이트", "그래픽",
    ],
    "casual": [
        "캐주얼", "데일리", "편안", "일상", "루즈", "이지", "relaxed",
    ],
    "sporty": [
        "스포츠", "스포티", "애슬레저", "운동", "러닝", "트레이닝", "짐웨어", "gym",
    ],
    "americana": [
        "아메카지", "워크", "밀리터리", "카모",
    ],
    "formal": [
        "포멀", "오피스", "정장", "드레시", "비즈니스", "셔츠룩",
    ],
    "feminine": [
        "페미닌", "로맨틱", "러블리", "여성스", "프릴", "레이스",
    ],
    "luxury": [
        "럭셔리", "하이엔드", "프리미엄", "고급", "명품",
    ],
    "retro": [
        "레트로", "빈티지", "y2k", "올드스쿨",
    ],
    "outdoor": [
        "아웃도어", "고프코어", "캠핑", "등산", "방풍",
    ],
}

BRAND_IMAGE_STYLE_AXIS_ORDER: tuple[str, ...] = tuple(BRAND_IMAGE_STYLE_AXES.keys())

_BRAND_IMAGE_STYLE_LABELS_KO: dict[str, str] = {
    "minimal": "미니멀/심플",
    "chic": "시크/세련",
    "street": "스트릿",
    "casual": "캐주얼/데일리",
    "sporty": "스포츠/애슬레저",
    "americana": "아메카지/워크",
    "formal": "포멀/오피스",
    "feminine": "페미닌",
    "luxury": "럭셔리/프리미엄",
    "retro": "레트로/빈티지",
    "outdoor": "아웃도어/고프코어",
}


def brand_image_style_label_ko(axis_id: str) -> str:
    return _BRAND_IMAGE_STYLE_LABELS_KO.get(axis_id, axis_id)


def detect_brand_image_styles(text: str) -> list[str]:
    """문장·스니펫에서 스타일 축 키워드 매칭(축당 최대 1회)."""
    if not text or not str(text).strip():
        return []
    lowered = str(text).lower()
    hits: list[str] = []
    for axis_id in BRAND_IMAGE_STYLE_AXIS_ORDER:
        kws = BRAND_IMAGE_STYLE_AXES.get(axis_id, ())
        if any(kw.lower() in lowered for kw in kws):
            hits.append(axis_id)
    return hits


# ---------------------------------------------------------------------------
# 기존 감성 키워드
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS = [
    "좋",
    "만족",
    "편",
    "예쁘",
    "이쁘",
    "추천",
    "괜찮",
    "빠르",
    "탄탄",
    "가성비",
    "고급",
    "부드럽",
    "시원",
]
NEGATIVE_KEYWORDS = [
    "아쉽",
    "불편",
    "별로",
    "작",
    "크",
    "비싸",
    "늦",
    "느리",
    "문제",
    "하자",
    "구김",
    "까짐",
    "좁",
    "딱딱",
]
NEGATIVE_CLAIM_KEYWORDS = ["주의", "불가", "제한", "오차", "불편", "금지"]
CLAIM_SOURCE_WEIGHTS = {
    "name": 0.9,
    "material": 0.85,
    "color": 0.85,
    "tags": 0.7,
    "ocr_text_joined": 0.6,
}
CLAIM_SOURCE_FIELD_COUNT = 5

SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")
WHITESPACE_RE = re.compile(r"\s+")


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = WHITESPACE_RE.sub(" ", str(value).replace("\u200b", " ")).strip()
    return text


def _split_sentences(text: str, max_sentences: int = 6) -> list[str]:
    if not text:
        return []
    raw = [WHITESPACE_RE.sub(" ", chunk).strip() for chunk in SENTENCE_SPLIT_RE.split(text)]
    deduped: list[str] = []
    seen = set()
    for sentence in raw:
        if len(sentence) < 2 or sentence in seen:
            continue
        seen.add(sentence)
        deduped.append(sentence[:280])
        if len(deduped) >= max_sentences:
            break
    return deduped


def _detect_aspects(text: str) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            hits.append(aspect)
    return hits or ["general"]


def _detect_tpo(text: str) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for tpo, keywords in TPO_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            hits.append(tpo)
    return hits or ["unspecified"]


def _detect_unmet_need(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in UNMET_NEED_PATTERNS)


def _classify_size_tendency(text: str) -> str | None:
    lowered = text.lower()
    small = any(kw in lowered for kw in SIZE_SMALL_KEYWORDS)
    large = any(kw in lowered for kw in SIZE_LARGE_KEYWORDS)
    true_fit = any(kw in lowered for kw in SIZE_TRUE_KEYWORDS)
    if true_fit and not small and not large:
        return "true_to_size"
    if small and not large:
        return "small"
    if large and not small:
        return "large"
    if small and large:
        return "mixed"
    return None


def _extract_trend_keywords(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    colors = [kw for kw in TREND_COLOR_KEYWORDS if kw in lowered]
    styles = [kw for kw in TREND_STYLE_KEYWORDS if kw in lowered]
    return {"color": colors, "style": styles}


def _sentiment_score(text: str) -> float:
    lowered = text.lower()
    pos = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in lowered)
    neg = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in lowered)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def _sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def _claim_polarity(text: str) -> tuple[str, float]:
    lowered = text.lower()
    if any(keyword in lowered for keyword in NEGATIVE_CLAIM_KEYWORDS):
        return "caution", -1.0
    return "positive", 1.0


def _iter_reviews(master_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with master_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


def _build_review_facts(featured_df: pd.DataFrame, reviews_root: Path) -> pd.DataFrame:
    if featured_df.empty:
        return _empty_df(REVIEW_COLUMNS)
    latest = (
        featured_df.sort_values(["crawl_datetime", "snapshot_id"], ascending=[True, True], na_position="last")
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for row in latest.to_dict("records"):
        product_id = str(row.get("product_id") or "").strip()
        if not product_id:
            continue
        master_path = reviews_root / "products" / product_id / "master.jsonl"
        if not master_path.exists():
            continue
        for review in _iter_reviews(master_path):
            review_id = review.get("review_id")
            rating = review.get("rating")
            review_type = _normalize_text(review.get("review_type")) or "unknown"
            sort_source = _normalize_text(review.get("sort_source")) or "unknown"
            photo_review = bool(review.get("photo_review"))
            helpful_count = int(review.get("helpful_count") or 0)
            option_text = _normalize_text(review.get("option_text"))
            created_at = _normalize_text(review.get("created_at")) or None
            text = _normalize_text(review.get("review_text"))
            if not text:
                continue
            sentences = _split_sentences(text, max_sentences=6)
            if not sentences:
                continue
            for sentence_idx, sentence in enumerate(sentences, start=1):
                score = _sentiment_score(sentence)
                label = _sentiment_label(score)
                aspects = _detect_aspects(sentence)
                tpo_list = _detect_tpo(sentence)
                is_unmet = _detect_unmet_need(sentence)

                for aspect in aspects:
                    size_tend = _classify_size_tendency(sentence) if aspect == "size_fit" else None
                    for tpo in tpo_list:
                        rows.append(
                            {
                                "snapshot_id": row.get("snapshot_id"),
                                "snapshot_date": row.get("snapshot_date"),
                                "snapshot_time": row.get("snapshot_time"),
                                "crawl_datetime": row.get("crawl_datetime"),
                                "product_id": product_id,
                                "brand": row.get("brand"),
                                "name": row.get("name"),
                                "review_id": review_id,
                                "created_at": created_at,
                                "rating": rating,
                                "review_type": review_type,
                                "sort_source": sort_source,
                                "photo_review": photo_review,
                                "helpful_count": helpful_count,
                                "option_text": option_text or None,
                                "sentence_id": sentence_idx,
                                "sentence_text": sentence,
                                "sentence_length": len(sentence),
                                "sentiment_label": label,
                                "sentiment_score": score,
                                "aspect": aspect,
                                "tpo": tpo,
                                "is_unmet_need": is_unmet,
                                "size_tendency": size_tend,
                            }
                        )
    if not rows:
        return _empty_df(REVIEW_COLUMNS)
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def _build_claim_facts(featured_df: pd.DataFrame) -> pd.DataFrame:
    if featured_df.empty:
        return _empty_df(CLAIM_COLUMNS)
    latest = (
        featured_df.sort_values(["crawl_datetime", "snapshot_id"], ascending=[True, True], na_position="last")
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    rows: list[dict[str, Any]] = []
    source_fields = ["name", "material", "color", "tags_joined", "ocr_text_joined"]
    for row in latest.to_dict("records"):
        product_id = str(row.get("product_id") or "").strip()
        if not product_id:
            continue
        for source_field in source_fields:
            raw_text = _normalize_text(row.get(source_field))
            if not raw_text:
                continue
            snippets = _split_sentences(raw_text, max_sentences=8 if source_field == "ocr_text_joined" else 3)
            for snippet in snippets:
                if len(snippet) < 4:
                    continue
                polarity, claim_score = _claim_polarity(snippet)
                claim_type = "caution_claim" if polarity == "caution" else "feature_claim"
                aspects = _detect_aspects(snippet)
                confidence = CLAIM_SOURCE_WEIGHTS.get(source_field, 0.5)
                for aspect in aspects:
                    rows.append(
                        {
                            "snapshot_id": row.get("snapshot_id"),
                            "snapshot_date": row.get("snapshot_date"),
                            "snapshot_time": row.get("snapshot_time"),
                            "crawl_datetime": row.get("crawl_datetime"),
                            "product_id": product_id,
                            "brand": row.get("brand"),
                            "name": row.get("name"),
                            "aspect": aspect,
                            "claim_text": snippet[:280],
                            "claim_type": claim_type,
                            "claim_polarity": polarity,
                            "claim_score": claim_score,
                            "source_field": source_field,
                            "confidence": confidence,
                        }
                    )
    if not rows:
        return _empty_df(CLAIM_COLUMNS)
    return pd.DataFrame(rows, columns=CLAIM_COLUMNS).drop_duplicates(
        subset=["snapshot_id", "product_id", "aspect", "claim_text", "source_field"], keep="first"
    )


def _build_gap_metrics(
    featured_df: pd.DataFrame,
    review_df: pd.DataFrame,
    claim_df: pd.DataFrame,
) -> pd.DataFrame:
    if featured_df.empty:
        return _empty_df(GAP_COLUMNS)
    latest = (
        featured_df.sort_values(["crawl_datetime", "snapshot_id"], ascending=[True, True], na_position="last")
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    for column in ["category_code", "category_l1", "category_l2", "category_l3", "price_band", "rank"]:
        if column not in latest.columns:
            latest[column] = None
    base_cols = [
        "snapshot_id",
        "snapshot_date",
        "snapshot_time",
        "crawl_datetime",
        "product_id",
        "brand",
        "name",
        "category_code",
        "category_l1",
        "category_l2",
        "category_l3",
        "price_band",
        "rank",
    ]
    base = latest[base_cols].drop_duplicates(subset=["snapshot_id", "product_id"], keep="last")

    if review_df.empty:
        review_agg = _empty_df(
            ["snapshot_id", "product_id", "aspect", "review_sentence_count", "review_count", "avg_rating", "review_sentiment_score"]
        )
    else:
        review_agg = (
            review_df.groupby(["snapshot_id", "product_id", "aspect"], as_index=False)
            .agg(
                review_sentence_count=("sentence_text", "count"),
                review_count=("review_id", "nunique"),
                avg_rating=("rating", "mean"),
                review_sentiment_score=("sentiment_score", "mean"),
            )
            .copy()
        )

    if claim_df.empty:
        claim_agg = _empty_df(["snapshot_id", "product_id", "aspect", "claim_count", "claim_score"])
    else:
        claim_agg = (
            claim_df.groupby(["snapshot_id", "product_id", "aspect"], as_index=False)
            .agg(
                claim_count=("claim_text", "count"),
                claim_score=("claim_score", "mean"),
            )
            .copy()
        )

    merged = review_agg.merge(
        claim_agg,
        on=["snapshot_id", "product_id", "aspect"],
        how="outer",
    )
    if merged.empty:
        return _empty_df(GAP_COLUMNS)
    merged = merged.merge(base, on=["snapshot_id", "product_id"], how="left")
    merged["review_sentence_count"] = pd.to_numeric(merged["review_sentence_count"], errors="coerce").fillna(0).astype(int)
    merged["review_count"] = pd.to_numeric(merged["review_count"], errors="coerce").fillna(0).astype(int)
    merged["claim_count"] = pd.to_numeric(merged["claim_count"], errors="coerce").fillna(0).astype(int)
    merged["review_sentiment_score"] = pd.to_numeric(merged["review_sentiment_score"], errors="coerce").fillna(0.0)
    merged["claim_score"] = pd.to_numeric(merged["claim_score"], errors="coerce").fillna(0.0)
    merged["gap_score"] = merged["claim_score"] - merged["review_sentiment_score"]
    merged["gap_abs"] = merged["gap_score"].abs()
    merged["gap_direction"] = merged["gap_score"].apply(
        lambda score: "aligned" if abs(score) < 0.3 else ("over_claimed" if score > 0 else "under_claimed")
    )
    return merged[GAP_COLUMNS].sort_values(["gap_abs", "review_sentence_count"], ascending=[False, False]).reset_index(drop=True)


def _build_fusion_profile(
    featured_df: pd.DataFrame,
    review_df: pd.DataFrame,
    claim_df: pd.DataFrame,
) -> pd.DataFrame:
    if featured_df.empty:
        return _empty_df(FUSION_COLUMNS)
    latest = (
        featured_df.sort_values(["crawl_datetime", "snapshot_id"], ascending=[True, True], na_position="last")
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    for column in ["category_code", "category_l1", "category_l2", "category_l3", "price_band", "rank"]:
        if column not in latest.columns:
            latest[column] = None
    base_cols = [
        "snapshot_id",
        "snapshot_date",
        "snapshot_time",
        "crawl_datetime",
        "product_id",
        "brand",
        "name",
        "category_code",
        "category_l1",
        "category_l2",
        "category_l3",
        "price_band",
        "rank",
    ]
    base = latest[base_cols].drop_duplicates(subset=["snapshot_id", "product_id"], keep="last")

    if review_df.empty:
        review_agg = _empty_df(
            [
                "snapshot_id",
                "product_id",
                "aspect",
                "review_sentence_count",
                "review_count",
                "review_signal",
                "review_signal_std",
            ]
        )
    else:
        review_agg = (
            review_df.groupby(["snapshot_id", "product_id", "aspect"], as_index=False)
            .agg(
                review_sentence_count=("sentence_text", "count"),
                review_count=("review_id", "nunique"),
                review_signal=("sentiment_score", "mean"),
                review_signal_std=("sentiment_score", "std"),
            )
            .copy()
        )

    if claim_df.empty:
        claim_agg = _empty_df(
            [
                "snapshot_id",
                "product_id",
                "aspect",
                "claim_count",
                "claim_signal",
                "claim_signal_std",
                "claim_confidence",
                "source_field_count",
            ]
        )
    else:
        claims = claim_df.copy()
        claims["claim_score"] = pd.to_numeric(claims["claim_score"], errors="coerce").fillna(0.0)
        claims["confidence"] = pd.to_numeric(claims["confidence"], errors="coerce").fillna(0.5)
        claims["weighted_claim_score"] = claims["claim_score"] * claims["confidence"]
        claim_agg = (
            claims.groupby(["snapshot_id", "product_id", "aspect"], as_index=False)
            .agg(
                claim_count=("claim_text", "count"),
                weighted_claim_sum=("weighted_claim_score", "sum"),
                confidence_sum=("confidence", "sum"),
                claim_signal_std=("claim_score", "std"),
                claim_confidence=("confidence", "mean"),
                source_field_count=("source_field", "nunique"),
            )
            .copy()
        )
        claim_agg["claim_signal"] = claim_agg.apply(
            lambda row: row["weighted_claim_sum"] / row["confidence_sum"] if row["confidence_sum"] else 0.0,
            axis=1,
        )
        claim_agg = claim_agg.drop(columns=["weighted_claim_sum", "confidence_sum"])

    merged = review_agg.merge(
        claim_agg,
        on=["snapshot_id", "product_id", "aspect"],
        how="outer",
    )
    if merged.empty:
        return _empty_df(FUSION_COLUMNS)

    merged = merged.merge(base, on=["snapshot_id", "product_id"], how="left")
    merged["review_sentence_count"] = pd.to_numeric(merged["review_sentence_count"], errors="coerce").fillna(0).astype(int)
    merged["review_count"] = pd.to_numeric(merged["review_count"], errors="coerce").fillna(0).astype(int)
    merged["claim_count"] = pd.to_numeric(merged["claim_count"], errors="coerce").fillna(0).astype(int)
    merged["review_signal"] = pd.to_numeric(merged["review_signal"], errors="coerce").fillna(0.0)
    merged["review_signal_std"] = pd.to_numeric(merged["review_signal_std"], errors="coerce").fillna(0.0)
    merged["claim_signal"] = pd.to_numeric(merged["claim_signal"], errors="coerce").fillna(0.0)
    merged["claim_signal_std"] = pd.to_numeric(merged["claim_signal_std"], errors="coerce").fillna(0.0)
    merged["claim_confidence"] = pd.to_numeric(merged["claim_confidence"], errors="coerce").fillna(0.0)
    merged["source_field_count"] = pd.to_numeric(merged["source_field_count"], errors="coerce").fillna(0).astype(int)
    merged["claim_source_coverage_pct"] = (merged["source_field_count"] / CLAIM_SOURCE_FIELD_COUNT) * 100.0

    merged["evidence_count"] = merged["review_sentence_count"] + merged["claim_count"]
    merged["evidence_density"] = merged["evidence_count"] / (merged["review_count"].clip(lower=1))

    def _signal_direction(score: float) -> int:
        if score >= 0.2:
            return 1
        if score <= -0.2:
            return -1
        return 0

    def _agreement_rate(row: pd.Series) -> float | None:
        if row["review_sentence_count"] <= 0 or row["claim_count"] <= 0:
            return None
        review_dir = _signal_direction(float(row["review_signal"]))
        claim_dir = _signal_direction(float(row["claim_signal"]))
        if review_dir == 0 or claim_dir == 0:
            return 0.5
        return 1.0 if review_dir == claim_dir else 0.0

    def _confidence_score(row: pd.Series) -> float:
        evidence = float(row["evidence_count"])
        if evidence <= 0:
            return 0.0
        evidence_weight = min(1.0, math.log1p(evidence) / math.log(12.0))
        review_presence = min(1.0, float(row["review_sentence_count"]) / 8.0)
        coverage_ratio = min(1.0, float(row["claim_source_coverage_pct"]) / 100.0)
        source_quality = min(1.0, 0.7 * float(row["claim_confidence"]) + 0.3 * coverage_ratio)
        return evidence_weight * (0.55 * review_presence + 0.45 * source_quality)

    merged["agreement_rate"] = merged.apply(_agreement_rate, axis=1)
    merged["diversity_penalty"] = (merged["review_signal_std"] + merged["claim_signal_std"]) / 2.0
    merged["diversity_penalty"] = merged["diversity_penalty"].clip(lower=0, upper=1)
    merged["fusion_score"] = 0.7 * merged["review_signal"] + 0.3 * merged["claim_signal"]
    merged["confidence_score"] = merged.apply(_confidence_score, axis=1)

    return merged[FUSION_COLUMNS].sort_values(
        ["confidence_score", "evidence_count", "fusion_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_trend_facts(
    review_df: pd.DataFrame,
    claim_df: pd.DataFrame,
) -> pd.DataFrame:
    """리뷰+클레임 텍스트에서 색상/스타일 키워드 빈도를 스냅샷별로 집계한다."""
    keyword_rows: list[dict[str, Any]] = []

    def _scan_text(snapshot_id: Any, snapshot_date: Any, product_id: Any, text: str, sentiment: float) -> None:
        hits = _extract_trend_keywords(text)
        for kw_type, kw_list in hits.items():
            for kw in kw_list:
                keyword_rows.append({
                    "snapshot_id": snapshot_id,
                    "snapshot_date": snapshot_date,
                    "product_id": product_id,
                    "keyword": kw,
                    "keyword_type": kw_type,
                    "sentiment": sentiment,
                })

    if not review_df.empty and "sentence_text" in review_df.columns:
        for row in review_df.to_dict("records"):
            _scan_text(
                row.get("snapshot_id"),
                row.get("snapshot_date"),
                row.get("product_id"),
                str(row.get("sentence_text") or ""),
                float(row.get("sentiment_score") or 0),
            )

    if not claim_df.empty and "claim_text" in claim_df.columns:
        for row in claim_df.to_dict("records"):
            _scan_text(
                row.get("snapshot_id"),
                row.get("snapshot_date"),
                row.get("product_id"),
                str(row.get("claim_text") or ""),
                float(row.get("claim_score") or 0),
            )

    if not keyword_rows:
        return _empty_df(TREND_COLUMNS)

    raw = pd.DataFrame(keyword_rows)
    grouped = (
        raw.groupby(["snapshot_id", "snapshot_date", "keyword", "keyword_type"], as_index=False)
        .agg(
            mention_count=("product_id", "count"),
            product_count=("product_id", "nunique"),
            avg_sentiment=("sentiment", "mean"),
        )
        .sort_values(["snapshot_date", "mention_count"], ascending=[True, False])
    )
    return grouped[TREND_COLUMNS].reset_index(drop=True)


def build_text_integrated_artifacts(featured_df: pd.DataFrame, reviews_root: Path) -> Dict[str, pd.DataFrame]:
    review_df = _build_review_facts(featured_df, reviews_root)
    claim_df = _build_claim_facts(featured_df)
    gap_df = _build_gap_metrics(featured_df, review_df, claim_df)
    fusion_df = _build_fusion_profile(featured_df, review_df, claim_df)
    trend_df = _build_trend_facts(review_df, claim_df)
    return {
        "text_review_facts": review_df,
        "text_claim_facts": claim_df,
        "text_gap_metrics": gap_df,
        "text_fusion_profile": fusion_df,
        "text_trend_keywords": trend_df,
    }
