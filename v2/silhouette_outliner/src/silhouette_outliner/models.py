from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .endpoints import DEFAULT_STORE_CODE


_WINDOW_ID_TO_DAYS = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "rt": 1,
    "realtime": 1,
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "day": 1,
    "week": 7,
    "month": 30,
}


def infer_window_days(window_id: str, label: str | None = None) -> int | None:
    """Best-effort guess of how many days a ranking window covers.

    Used to weight 'sustained energy' and to space the rank-line chart x-axis.
    Returning None means the analyzer falls back to a neutral weighting (1.0).
    """

    if window_id:
        key = window_id.lower()
        if key in _WINDOW_ID_TO_DAYS:
            return _WINDOW_ID_TO_DAYS[key]
    if label:
        for token in ("일간", "1일", "데일리", "오늘"):
            if token in label:
                return 1
        for token in ("주간", "1주", "위클리", "이번주"):
            if token in label:
                return 7
        for token in ("월간", "1개월", "1달", "먼슬리", "이번달"):
            if token in label:
                return 30
        if "실시간" in label:
            return 1
    return None


@dataclass(frozen=True)
class RankingWindowSpec:
    """One ranking period slice (e.g. daily / weekly / monthly) merged into API URLs."""

    id: str
    label: str
    query_params: dict[str, str] = field(default_factory=dict)
    days: int | None = None

    @property
    def days_effective(self) -> int | None:
        """Days the window covers; falls back to a heuristic on id/label."""
        return self.days if self.days is not None else infer_window_days(self.id, self.label)


DEFAULT_RANKING_WINDOW = RankingWindowSpec(id="default", label="기본", query_params={}, days=None)


@dataclass(frozen=True)
class SectionTarget:
    label: str
    section_id: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CategoryTarget:
    label: str
    code: str
    parent_label: str | None = None

    @property
    def major_code(self) -> str:
        normalized = normalize_category_code(self.code)
        return normalized[:3]

    @property
    def minor_code(self) -> str:
        normalized = normalize_category_code(self.code)
        return normalized[3:]

    @property
    def is_parent_total(self) -> bool:
        return self.minor_code == "000"


@dataclass(frozen=True)
class CollectionTarget:
    section: SectionTarget
    category: CategoryTarget
    store_code: str = DEFAULT_STORE_CODE
    gender_filter: str = "A"
    gender_label: str = "전체"
    age_band: str = "AGE_BAND_ALL"
    age_label: str = "전체 연령"
    sub_pan: str = "product"
    limit: int = 100
    ranking_window: RankingWindowSpec = DEFAULT_RANKING_WINDOW

    @property
    def key(self) -> str:
        base = f"{self.section.section_id}_{self.category.code}"
        if self.sub_pan != "product":
            base = f"{base}_{self.sub_pan}"
        if self.gender_filter != "A":
            base = f"{base}_gf{self.gender_filter}"
        if self.age_band != "AGE_BAND_ALL":
            base = f"{base}_{self.age_band}"
        if self.ranking_window.id != "default":
            return f"{base}_{self.ranking_window.id}"
        return base


@dataclass
class RawCollection:
    target: CollectionTarget
    url: str
    collected_at: str
    source: str
    ok: bool
    payload: Any = None
    error: str | None = None
    response_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = {
            "section": asdict(self.target.section),
            "category": asdict(self.target.category),
            "store_code": self.target.store_code,
            "gender_filter": self.target.gender_filter,
            "gender_label": self.target.gender_label,
            "age_band": self.target.age_band,
            "age_label": self.target.age_label,
            "sub_pan": self.target.sub_pan,
            "limit": self.target.limit,
            "key": self.target.key,
            "ranking_window": {
                "id": self.target.ranking_window.id,
                "label": self.target.ranking_window.label,
                "query_params": dict(self.target.ranking_window.query_params),
                "days": self.target.ranking_window.days,
            },
        }
        return data


@dataclass
class RankingItem:
    rank: int | None
    brand: str
    product: str
    product_clean: str
    price: int | None
    original_price: int | None
    discount_rate: float | None
    discount_amount: int | None
    product_id: str | None
    product_url: str | None
    image_url: str | None
    brand_id: str | None
    review_count: int | None
    review_score: int | None
    viewers_now: int | None
    buyers_now: int | None
    is_sold_out: bool
    labels: list[str]
    section_label: str
    section_id: str
    category_label: str
    category_code: str
    category_major_code: str
    category_minor_code: str
    category_parent_label: str | None
    source: str
    collected_at: str
    ranking_window_id: str = "default"
    ranking_window_label: str = "기본"
    ranking_window_days: int | None = None
    sub_pan: str = "product"
    gender_filter: str = "A"
    gender_label: str = "전체"
    age_band: str = "AGE_BAND_ALL"
    age_label: str = "전체 연령"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContentSignalProduct:
    product_id: str
    product_order: int
    brand: str | None = None
    product: str | None = None
    price: int | None = None
    product_url: str | None = None
    image_url: str | None = None
    period_rank: int | None = None
    realtime_rank: int | None = None
    buyers_now: int | None = None
    review_count: int | None = None
    validation: float | None = None
    matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContentSignal:
    rank: int
    content_id: str
    title: str
    brand: str
    content_type: str
    view_count_text: str | None
    comment_count_text: str | None
    popularity_score_text: str | None
    url: str
    product_ids: list[str]
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KeywordSignal:
    rank: int
    keyword: str
    fluctuation_type: str
    fluctuation_amount: int | None
    fluctuation_label: str
    search_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KeywordGenderBoard:
    gender_code: str
    gender_label: str
    rows: list[KeywordSignal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gender_code": self.gender_code,
            "gender_label": self.gender_label,
            "rows": [row.to_dict() for row in self.rows],
            "has_data": bool(self.rows),
        }


@dataclass
class CustomerSignalDataset:
    generated_at: str
    contents: list[ContentSignal]
    keywords_by_gender: list[KeywordGenderBoard]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "contents": [item.to_dict() for item in self.contents],
            "keywords_by_gender": [board.to_dict() for board in self.keywords_by_gender],
            "errors": list(self.errors),
        }


@dataclass
class NormalizedDataset:
    generated_at: str
    items: list[RankingItem]
    collections: list[RawCollection]
    customer_signals: CustomerSignalDataset | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at,
            "items": [item.to_dict() for item in self.items],
            "collections": [collection.to_dict() for collection in self.collections],
        }
        if self.customer_signals is not None:
            payload["customer_signals"] = self.customer_signals.to_dict()
        return payload


def normalize_category_code(code: str) -> str:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) <= 3:
        return digits.zfill(3) + "000"
    return digits.zfill(6)


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
