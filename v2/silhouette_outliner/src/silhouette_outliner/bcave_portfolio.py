"""BCave portfolio definitions for the Musinsa **브랜드 탭** ranking tracker.

The brand tab is a separate pan from the product tab. It uses
``subPan=brand`` (singular) and dedicated section IDs that differ from the
product-side style sections (199 / 202 ...).

Section IDs were discovered via Playwright by clicking each style chip
under the brand tab. They may shift on a UI redeploy; refresh by re-running
the discovery script when something looks off.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CategoryTarget, SectionTarget

# brand-tab sub-pan identifier as used by the upstream client API
SUB_PAN_BRAND = "brand"

# brand-tab style section IDs (discovered 2026-05-15)
STYLE_SECTION_OVERALL = "1054"  # 브랜드 탭 · 전체/전체
STYLE_SECTION_YOUNG_CASUAL = "1056"  # 영캐주얼
STYLE_SECTION_WOMENS_CASUAL = "1063"  # 여성캐주얼
STYLE_SECTION_STREET_CASUAL = "1066"  # 스트릿캐주얼


@dataclass(frozen=True)
class BcaveBrandPane:
    """One brand-tab pane (전체/전체 / 스타일별)."""

    label: str
    section: SectionTarget
    category: CategoryTarget
    is_overall: bool = False


_ALL_CATEGORY = CategoryTarget(label="전체", code="000000")


BCAVE_BRAND_PANES: tuple[BcaveBrandPane, ...] = (
    BcaveBrandPane(
        label="전체/전체",
        section=SectionTarget(
            label="브랜드 · 전체/전체",
            section_id=STYLE_SECTION_OVERALL,
            params={"categoryCode": ""},
        ),
        category=_ALL_CATEGORY,
        is_overall=True,
    ),
    BcaveBrandPane(
        label="영캐주얼",
        section=SectionTarget(
            label="브랜드 · 영캐주얼",
            section_id=STYLE_SECTION_YOUNG_CASUAL,
            params={"categoryCode": ""},
        ),
        category=_ALL_CATEGORY,
    ),
    BcaveBrandPane(
        label="여성캐주얼",
        section=SectionTarget(
            label="브랜드 · 여성캐주얼",
            section_id=STYLE_SECTION_WOMENS_CASUAL,
            params={"categoryCode": ""},
        ),
        category=_ALL_CATEGORY,
    ),
    BcaveBrandPane(
        label="스트릿캐주얼",
        section=SectionTarget(
            label="브랜드 · 스트릿캐주얼",
            section_id=STYLE_SECTION_STREET_CASUAL,
            params={"categoryCode": ""},
        ),
        category=_ALL_CATEGORY,
    ),
)


@dataclass(frozen=True)
class BcaveBrandSpec:
    id: str
    label_ko: str
    musinsa_names: tuple[str, ...]
    style_lane_label: str
    style_section_id: str


BCAVE_PORTFOLIO: tuple[BcaveBrandSpec, ...] = (
    BcaveBrandSpec(
        id="covernat",
        label_ko="커버낫",
        musinsa_names=("커버낫",),
        style_lane_label="영캐주얼",
        style_section_id=STYLE_SECTION_YOUNG_CASUAL,
    ),
    BcaveBrandSpec(
        id="wackywilly",
        label_ko="와키윌리",
        musinsa_names=("와키윌리",),
        style_lane_label="스트릿캐주얼",
        style_section_id=STYLE_SECTION_STREET_CASUAL,
    ),
    BcaveBrandSpec(
        id="fallett",
        label_ko="팔렛",
        musinsa_names=("팔렛",),
        style_lane_label="여성캐주얼",
        style_section_id=STYLE_SECTION_WOMENS_CASUAL,
    ),
    BcaveBrandSpec(
        id="lee",
        label_ko="리",
        musinsa_names=("리",),
        style_lane_label="영캐주얼",
        style_section_id=STYLE_SECTION_YOUNG_CASUAL,
    ),
)


def match_bcave_brand(display_name: str) -> BcaveBrandSpec | None:
    name = (display_name or "").strip()
    if not name:
        return None
    for spec in BCAVE_PORTFOLIO:
        if name in spec.musinsa_names:
            return spec
    return None
