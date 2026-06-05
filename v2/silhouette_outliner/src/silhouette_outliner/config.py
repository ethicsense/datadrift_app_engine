from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .endpoints import (
    DEFAULT_STORE_CODE,
    RANKING_CLIENT_API,
    RANKING_PAGE_PATH,
    RANKING_WEB_ORIGIN,
)
from .bcave_portfolio import BCAVE_BRAND_PANES, SUB_PAN_BRAND
from .demographics import (
    DEFAULT_AGE_BANDS,
    DEFAULT_GENDER_FILTERS,
    AgeBandSpec,
    GenderSpec,
)
from .models import (
    CategoryTarget,
    CollectionTarget,
    DEFAULT_RANKING_WINDOW,
    RankingWindowSpec,
    SectionTarget,
)
from .runtime_paths import bundled_config_path

BASE_RANKING_URL = f"{RANKING_WEB_ORIGIN}{RANKING_PAGE_PATH}"
BASE_RANKING_API_URL = RANKING_CLIENT_API


def default_collect_config_path() -> Path:
    """Full Musinsa preset: daily/weekly/monthly windows + gender/age demographics."""
    return bundled_config_path("periodic-multag.json")


DEFAULT_SECTIONS = [
    SectionTarget(label="전체", section_id="199"),
]

DEFAULT_CATEGORIES = [
    CategoryTarget(label="상의", code="001000", parent_label="의류"),
    CategoryTarget(label="아우터", code="002000", parent_label="의류"),
    CategoryTarget(label="바지", code="003000", parent_label="의류"),
    CategoryTarget(label="가방", code="004000", parent_label="잡화"),
]

# Period track uses these baselines; demographics track uses `demographics_window`.
PERIOD_GENDER_CODE = "A"
PERIOD_AGE_BAND_CODE = "AGE_BAND_ALL"

DEFAULT_DEMOGRAPHICS_WINDOW = RankingWindowSpec(
    id="1w",
    label="주간",
    query_params={"period": "WEEKLY"},
    days=7,
)

# Age-ranking table uses this window; heatmap/demographics keep `demographics_window`.
DEFAULT_AGE_RANKINGS_WINDOW = RankingWindowSpec(
    id="rt",
    label="실시간",
    query_params={"period": "REALTIME"},
    days=1,
)


def _parse_ranking_window_entry(entry: Any) -> RankingWindowSpec | None:
    if not (isinstance(entry, dict) and entry.get("id")):
        return None
    days_value = entry.get("days")
    try:
        days_typed = int(days_value) if days_value is not None else None
    except (TypeError, ValueError):
        days_typed = None
    return RankingWindowSpec(
        id=str(entry["id"]),
        label=str(entry.get("label", entry["id"])),
        query_params={
            str(k): str(v) for k, v in entry.get("query_params", {}).items() if k is not None
        },
        days=days_typed,
    )


def _parse_gender_specs(raw: Any) -> tuple[GenderSpec, ...]:
    if not isinstance(raw, list) or not raw:
        return DEFAULT_GENDER_FILTERS
    parsed: list[GenderSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code", "")).strip()
        if not code:
            continue
        parsed.append(
            GenderSpec(
                id=str(entry.get("id", code)),
                code=code,
                label=str(entry.get("label", code)),
            )
        )
    return tuple(parsed) if parsed else DEFAULT_GENDER_FILTERS


def _parse_age_band_specs(raw: Any) -> tuple[AgeBandSpec, ...]:
    if not isinstance(raw, list) or not raw:
        return DEFAULT_AGE_BANDS
    parsed: list[AgeBandSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code", "")).strip()
        if not code:
            continue
        parsed.append(
            AgeBandSpec(
                id=str(entry.get("id", code)),
                code=code,
                label=str(entry.get("label", code)),
            )
        )
    return tuple(parsed) if parsed else DEFAULT_AGE_BANDS


@dataclass(frozen=True)
class AppConfig:
    sections: list[SectionTarget]
    categories: list[CategoryTarget]
    ranking_windows: tuple[RankingWindowSpec, ...] = (DEFAULT_RANKING_WINDOW,)
    store_code: str = DEFAULT_STORE_CODE
    gender_filter: str = "A"
    age_band: str = "AGE_BAND_ALL"
    gender_filters: tuple[GenderSpec, ...] = DEFAULT_GENDER_FILTERS
    age_bands: tuple[AgeBandSpec, ...] = DEFAULT_AGE_BANDS
    demographics_window: RankingWindowSpec = DEFAULT_DEMOGRAPHICS_WINDOW
    age_rankings_window: RankingWindowSpec | None = None
    sub_pan: str = "product"
    limit: int = 100
    request_delay_seconds: float = 0.4
    track_bcave_portfolio: bool = True
    collect_customer_signals: bool = True
    content_top_n: int = 10
    keyword_top_n: int = 100
    keyword_gender_filters: tuple[str, ...] = ("A", "M", "F")

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls(
            sections=DEFAULT_SECTIONS,
            categories=DEFAULT_CATEGORIES,
            ranking_windows=(DEFAULT_RANKING_WINDOW,),
            gender_filters=(DEFAULT_GENDER_FILTERS[0],),
            age_bands=(DEFAULT_AGE_BANDS[0],),
            demographics_window=DEFAULT_DEMOGRAPHICS_WINDOW,
            collect_customer_signals=True,
            content_top_n=10,
            keyword_top_n=100,
            keyword_gender_filters=("A", "M", "F"),
        )

    def expands_demographics(self) -> bool:
        return len(self.gender_filters) > 1 or len(self.age_bands) > 1

    def resolved_genders(self) -> tuple[GenderSpec, ...]:
        if len(self.gender_filters) > 1:
            return self.gender_filters
        for spec in DEFAULT_GENDER_FILTERS:
            if spec.code == self.gender_filter:
                return (spec,)
        return (
            GenderSpec(id=self.gender_filter, code=self.gender_filter, label=self.gender_filter),
        )

    def resolved_age_bands(self) -> tuple[AgeBandSpec, ...]:
        if len(self.age_bands) > 1:
            return self.age_bands
        for spec in DEFAULT_AGE_BANDS:
            if spec.code == self.age_band:
                return (spec,)
        return (AgeBandSpec(id=self.age_band, code=self.age_band, label=self.age_band),)

    def targets(self) -> list[CollectionTarget]:
        period_gender = next(
            (g for g in DEFAULT_GENDER_FILTERS if g.code == PERIOD_GENDER_CODE),
            DEFAULT_GENDER_FILTERS[0],
        )
        period_age = next(
            (a for a in DEFAULT_AGE_BANDS if a.code == PERIOD_AGE_BAND_CODE),
            DEFAULT_AGE_BANDS[0],
        )

        # Period track: category × ranking_windows, 성별/연령 전체 고정.
        product_targets = [
            CollectionTarget(
                section=section,
                category=category,
                store_code=self.store_code,
                gender_filter=period_gender.code,
                gender_label=period_gender.label,
                age_band=period_age.code,
                age_label=period_age.label,
                sub_pan=self.sub_pan,
                limit=self.limit,
                ranking_window=window,
            )
            for section in self.sections
            for category in self.categories
            for window in self.ranking_windows
        ]

        # Demographics track: category × demographics_window × 성별 × 연령 (기간과 분리).
        if self.expands_demographics():
            demo_window = self.demographics_window
            period_window_ids = {window.id for window in self.ranking_windows}
            genders = self.resolved_genders()
            ages = self.resolved_age_bands()
            product_targets.extend(
                CollectionTarget(
                    section=section,
                    category=category,
                    store_code=self.store_code,
                    gender_filter=gender.code,
                    gender_label=gender.label,
                    age_band=age.code,
                    age_label=age.label,
                    sub_pan=self.sub_pan,
                    limit=self.limit,
                    ranking_window=demo_window,
                )
                for section in self.sections
                for category in self.categories
                for gender in genders
                for age in ages
                if not (
                    gender.code == period_gender.code
                    and age.code == period_age.code
                    and demo_window.id in period_window_ids
                )
            )

        # Age-ranking table track: 성별=전체 고정 × 연령별 × age_rankings_window (기본 실시간).
        ar_window = self.age_rankings_window
        if self.expands_demographics() and ar_window is not None:
            period_window_ids = {window.id for window in self.ranking_windows}
            ages = self.resolved_age_bands()
            demo_window = self.demographics_window
            product_targets.extend(
                CollectionTarget(
                    section=section,
                    category=category,
                    store_code=self.store_code,
                    gender_filter=period_gender.code,
                    gender_label=period_gender.label,
                    age_band=age.code,
                    age_label=age.label,
                    sub_pan=self.sub_pan,
                    limit=self.limit,
                    ranking_window=ar_window,
                )
                for section in self.sections
                for category in self.categories
                for age in ages
                if not (
                    age.code == period_age.code
                    and ar_window.id in period_window_ids
                )
                and not (
                    ar_window.id == demo_window.id
                    and period_gender.code == PERIOD_GENDER_CODE
                )
            )

        if not self.track_bcave_portfolio:
            return product_targets

        brand_targets = [
            CollectionTarget(
                section=pane.section,
                category=pane.category,
                store_code=self.store_code,
                gender_filter=period_gender.code,
                gender_label=period_gender.label,
                age_band=period_age.code,
                age_label=period_age.label,
                sub_pan=SUB_PAN_BRAND,
                limit=self.limit,
                ranking_window=window,
            )
            for pane in BCAVE_BRAND_PANES
            for window in self.ranking_windows
        ]
        if self.expands_demographics():
            demo_window = self.demographics_window
            period_window_ids = {window.id for window in self.ranking_windows}
            genders = self.resolved_genders()
            ages = self.resolved_age_bands()
            brand_targets.extend(
                CollectionTarget(
                    section=pane.section,
                    category=pane.category,
                    store_code=self.store_code,
                    gender_filter=gender.code,
                    gender_label=gender.label,
                    age_band=age.code,
                    age_label=age.label,
                    sub_pan=SUB_PAN_BRAND,
                    limit=self.limit,
                    ranking_window=demo_window,
                )
                for pane in BCAVE_BRAND_PANES
                for gender in genders
                for age in ages
                if not (
                    gender.code == period_gender.code
                    and age.code == period_age.code
                    and demo_window.id in period_window_ids
                )
            )
        return product_targets + brand_targets

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [asdict(section) for section in self.sections],
            "categories": [asdict(category) for category in self.categories],
            "ranking_windows": [
                {
                    "id": w.id,
                    "label": w.label,
                    "query_params": dict(w.query_params),
                    "days": w.days,
                }
                for w in self.ranking_windows
            ],
            "store_code": self.store_code,
            "gender_filter": self.gender_filter,
            "age_band": self.age_band,
            "gender_filters": [asdict(g) for g in self.gender_filters],
            "age_bands": [asdict(a) for a in self.age_bands],
            "demographics_window": {
                "id": self.demographics_window.id,
                "label": self.demographics_window.label,
                "query_params": dict(self.demographics_window.query_params),
                "days": self.demographics_window.days,
            },
            "age_rankings_window": (
                {
                    "id": self.age_rankings_window.id,
                    "label": self.age_rankings_window.label,
                    "query_params": dict(self.age_rankings_window.query_params),
                    "days": self.age_rankings_window.days,
                }
                if self.age_rankings_window is not None
                else None
            ),
            "sub_pan": self.sub_pan,
            "limit": self.limit,
            "request_delay_seconds": self.request_delay_seconds,
            "track_bcave_portfolio": self.track_bcave_portfolio,
            "collect_customer_signals": self.collect_customer_signals,
            "content_top_n": self.content_top_n,
            "keyword_top_n": self.keyword_top_n,
            "keyword_gender_filters": list(self.keyword_gender_filters),
        }


def load_config(path: Path | None = None) -> AppConfig:
    if path is None or not path.exists():
        return AppConfig.defaults()

    data = json.loads(path.read_text(encoding="utf-8"))
    sections = [
        SectionTarget(
            label=str(item["label"]),
            section_id=str(item["section_id"]),
            params={str(k): str(v) for k, v in item.get("params", {}).items()},
        )
        for item in data.get("sections", [])
    ] or DEFAULT_SECTIONS
    categories = [
        CategoryTarget(
            label=str(item["label"]),
            code=str(item["code"]),
            parent_label=item.get("parent_label"),
        )
        for item in data.get("categories", [])
    ] or DEFAULT_CATEGORIES

    raw_windows = data.get("ranking_windows")
    if isinstance(raw_windows, list) and raw_windows:
        parsed_windows = [w for w in (_parse_ranking_window_entry(e) for e in raw_windows) if w]
        ranking_windows = tuple(parsed_windows) if parsed_windows else (DEFAULT_RANKING_WINDOW,)
    else:
        ranking_windows = (DEFAULT_RANKING_WINDOW,)

    demographics_window = _parse_ranking_window_entry(data.get("demographics_window"))
    if demographics_window is None:
        demo_id = str(data.get("demographics_window_id", DEFAULT_DEMOGRAPHICS_WINDOW.id))
        demographics_window = next(
            (w for w in ranking_windows if w.id == demo_id),
            DEFAULT_DEMOGRAPHICS_WINDOW,
        )

    age_rankings_window: RankingWindowSpec | None = None
    if "age_rankings_window" in data:
        age_rankings_window = _parse_ranking_window_entry(data.get("age_rankings_window"))
    elif isinstance(data.get("age_bands"), list) and len(data["age_bands"]) > 1:
        # periodic-multag 등 멀티 연령: 히트맵은 demographics_window, 연령 표는 실시간.
        age_rankings_window = DEFAULT_AGE_RANKINGS_WINDOW

    gender_filter = str(data.get("gender_filter", "A"))
    age_band = str(data.get("age_band", "AGE_BAND_ALL"))
    if "gender_filters" in data:
        gender_filters = _parse_gender_specs(data.get("gender_filters"))
    else:
        gender_filters = tuple(spec for spec in DEFAULT_GENDER_FILTERS if spec.code == gender_filter)
        if not gender_filters:
            gender_filters = (
                GenderSpec(id="all", code=gender_filter, label=gender_label_for_legacy(gender_filter)),
            )
    if "age_bands" in data:
        age_bands = _parse_age_band_specs(data.get("age_bands"))
    else:
        age_bands = tuple(spec for spec in DEFAULT_AGE_BANDS if spec.code == age_band)
        if not age_bands:
            age_bands = (AgeBandSpec(id="all", code=age_band, label=age_band),)

    return AppConfig(
        sections=sections,
        categories=categories,
        ranking_windows=ranking_windows,
        store_code=str(data.get("store_code", DEFAULT_STORE_CODE)),
        gender_filter=gender_filter,
        age_band=age_band,
        gender_filters=gender_filters,
        age_bands=age_bands,
        demographics_window=demographics_window,
        age_rankings_window=age_rankings_window,
        sub_pan=str(data.get("sub_pan", "product")),
        limit=int(data.get("limit", 100)),
        request_delay_seconds=float(data.get("request_delay_seconds", 0.4)),
        track_bcave_portfolio=bool(data.get("track_bcave_portfolio", True)),
        collect_customer_signals=bool(data.get("collect_customer_signals", True)),
        content_top_n=int(data.get("content_top_n", 10)),
        keyword_top_n=int(data.get("keyword_top_n", 100)),
        keyword_gender_filters=_parse_keyword_gender_filters(data.get("keyword_gender_filters")),
    )


def _parse_keyword_gender_filters(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list) and raw:
        parsed = tuple(str(code).strip().upper() for code in raw if str(code).strip())
        return parsed if parsed else ("A", "M", "F")
    return ("A", "M", "F")


def gender_label_for_legacy(code: str) -> str:
    for spec in DEFAULT_GENDER_FILTERS:
        if spec.code == code:
            return spec.label
    return code


def build_ranking_url(target: CollectionTarget) -> str:
    params = {
        "gf": target.gender_filter,
        "storeCode": target.store_code,
        "sectionId": target.section.section_id,
        "contentsId": "",
        "categoryCode": target.category.code,
        "ageBand": target.age_band,
        "subPan": target.sub_pan,
    }
    params.update(target.section.params)
    params.update(target.ranking_window.query_params)
    return f"{BASE_RANKING_URL}?{urlencode(params)}"


def build_ranking_api_url(target: CollectionTarget) -> str:
    params = {
        "storeCode": target.store_code,
        "sectionId": target.section.section_id,
        "gf": target.gender_filter,
        "contentsId": "",
        "categoryCode": target.category.code,
        "ageBand": target.age_band,
        "subPan": target.sub_pan,
    }
    params.update(target.section.params)
    params.update(target.ranking_window.query_params)
    return f"{BASE_RANKING_API_URL}?{urlencode(params)}"
