"""Gender and age-band axes for ranking collection and heatmap analysis.

API codes follow Musinsa ranking query params (`gf`, `ageBand`). Values were
aligned to the public filter bar labels (2026-05); re-verify via Network if the
upstream UI changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenderSpec:
    id: str
    code: str
    label: str


@dataclass(frozen=True)
class AgeBandSpec:
    id: str
    code: str
    label: str


# Default ranking period for gender/age (demographics) analysis. Override via config
# `demographics_window` (e.g. change id to `1m` for monthly).
DEFAULT_DEMOGRAPHICS_WINDOW_ID = "1w"

DEFAULT_GENDER_FILTERS: tuple[GenderSpec, ...] = (
    GenderSpec(id="all", code="A", label="전체"),
    GenderSpec(id="male", code="M", label="남성"),
    GenderSpec(id="female", code="F", label="여성"),
)

# Order matches the ranking UI age chips (left → right).
DEFAULT_AGE_BANDS: tuple[AgeBandSpec, ...] = (
    AgeBandSpec(id="all", code="AGE_BAND_ALL", label="전체 연령"),
    AgeBandSpec(id="u19", code="AGE_BAND_00", label="19세 이하"),
    AgeBandSpec(id="20_24", code="AGE_BAND_20", label="20-24"),
    AgeBandSpec(id="25_29", code="AGE_BAND_25", label="25-29"),
    AgeBandSpec(id="30_34", code="AGE_BAND_30", label="30-34"),
    AgeBandSpec(id="35_39", code="AGE_BAND_35", label="35-39"),
    AgeBandSpec(id="40p", code="AGE_BAND_40", label="40세 이상"),
)

_AGE_CODE_TO_INDEX = {spec.code: idx for idx, spec in enumerate(DEFAULT_AGE_BANDS)}
_GENDER_CODE_TO_KEY = {spec.code: spec.id for spec in DEFAULT_GENDER_FILTERS}


def age_band_index(code: str) -> int | None:
    return _AGE_CODE_TO_INDEX.get(code)


def gender_key(code: str) -> str:
    return _GENDER_CODE_TO_KEY.get(code, code)


def gender_label_for_code(code: str) -> str:
    for spec in DEFAULT_GENDER_FILTERS:
        if spec.code == code:
            return spec.label
    return code


def age_label_for_code(code: str) -> str:
    for spec in DEFAULT_AGE_BANDS:
        if spec.code == code:
            return spec.label
    return code
