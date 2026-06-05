from silhouette_outliner.analyze import _bcave_brand_tracker
from silhouette_outliner.bcave_portfolio import (
    BCAVE_BRAND_PANES,
    STYLE_SECTION_OVERALL,
    STYLE_SECTION_STREET_CASUAL,
    STYLE_SECTION_WOMENS_CASUAL,
    STYLE_SECTION_YOUNG_CASUAL,
)
from silhouette_outliner.models import (
    CategoryTarget,
    CollectionTarget,
    NormalizedDataset,
    RankingItem,
    RankingWindowSpec,
    RawCollection,
    SectionTarget,
)
from silhouette_outliner.normalize import normalize_collections


def _brand_item(**overrides):
    base = dict(
        rank=1,
        brand="커버낫",
        product="",
        product_clean="",
        price=None,
        original_price=None,
        discount_rate=None,
        discount_amount=None,
        product_id=None,
        product_url=None,
        image_url=None,
        brand_id="covernat",
        review_count=None,
        review_score=None,
        viewers_now=None,
        buyers_now=None,
        is_sold_out=False,
        labels=[],
        section_label="전체/전체",
        section_id=STYLE_SECTION_OVERALL,
        category_label="전체",
        category_code="000000",
        category_major_code="001",
        category_minor_code="000",
        category_parent_label="의류",
        source="api",
        collected_at="2026-05-15T00:00:00Z",
        ranking_window_id="1w",
        ranking_window_label="주간",
        ranking_window_days=7,
        sub_pan="brand",
    )
    base.update(overrides)
    return RankingItem(**base)


def test_normalize_parses_ranking_brand_cards():
    section = SectionTarget(
        label="브랜드 · 전체/전체",
        section_id=STYLE_SECTION_OVERALL,
        params={"categoryCode": ""},
    )
    category = CategoryTarget(label="전체", code="000000")
    window = RankingWindowSpec("1w", "주간", {"period": "WEEKLY"}, days=7)
    target = CollectionTarget(
        section=section,
        category=category,
        sub_pan="brand",
        ranking_window=window,
    )
    payload = {
        "data": {
            "modules": [
                {
                    "type": "RANKING_BRAND",
                    "title": {
                        "rank": "1",
                        "title": {"text": "무신사 스탠다드"},
                        "imageUrl": "https://img/1.png",
                        "onClick": {"url": "https://www.musinsa.com/brand/musinsastandard"},
                        "fluctuation": {"type": "NONE"},
                    },
                },
                {
                    "type": "RANKING_BRAND",
                    "title": {
                        "rank": "20",
                        "title": {"text": "커버낫"},
                        "imageUrl": "https://img/2.png",
                        "onClick": {"url": "https://www.musinsa.com/brand/covernat"},
                    },
                },
            ]
        }
    }
    collection = RawCollection(
        target=target,
        url="https://example",
        collected_at="2026-05-15T00:00:00Z",
        source="api",
        ok=True,
        payload=payload,
    )
    dataset = normalize_collections([collection])
    brands = sorted((item.rank, item.brand) for item in dataset.items)
    assert brands == [(1, "무신사 스탠다드"), (20, "커버낫")]
    assert all(item.sub_pan == "brand" for item in dataset.items)


def test_bcave_brand_panes_overall_uses_empty_category_code():
    overall = BCAVE_BRAND_PANES[0]
    assert overall.label == "전체/전체"
    assert overall.is_overall is True
    assert overall.category.code == "000000"
    assert overall.section.params.get("categoryCode") == ""


def test_bcave_tracker_reports_overall_and_lane_ranks():
    items = [
        _brand_item(rank=20, brand="커버낫", section_id=STYLE_SECTION_OVERALL, section_label="전체/전체"),
        _brand_item(rank=23, brand="리", section_id=STYLE_SECTION_OVERALL, section_label="전체/전체"),
        _brand_item(rank=98, brand="와키윌리", section_id=STYLE_SECTION_OVERALL, section_label="전체/전체"),
        _brand_item(rank=4, brand="리", section_id=STYLE_SECTION_YOUNG_CASUAL, section_label="영캐주얼"),
        _brand_item(rank=8, brand="커버낫", section_id=STYLE_SECTION_YOUNG_CASUAL, section_label="영캐주얼"),
        _brand_item(rank=133, brand="팔렛", section_id=STYLE_SECTION_WOMENS_CASUAL, section_label="여성캐주얼"),
        _brand_item(rank=11, brand="와키윌리", section_id=STYLE_SECTION_STREET_CASUAL, section_label="스트릿캐주얼"),
    ]
    dataset = NormalizedDataset(generated_at="2026-05-15T00:00:00Z", items=items, collections=[])

    result = _bcave_brand_tracker(dataset)
    assert result["has_data"] is True
    by_id = {row["id"]: row for row in result["brands"]}
    assert by_id["covernat"]["overall"]["rank"] == 20
    assert by_id["covernat"]["style_lane"]["rank"] == 8
    assert by_id["covernat"]["style_lane_label"] == "영캐주얼"
    assert by_id["lee"]["style_lane"]["rank"] == 4
    assert by_id["wackywilly"]["overall"]["rank"] == 98
    assert by_id["wackywilly"]["style_lane"]["rank"] == 11
    assert by_id["wackywilly"]["style_lane_label"] == "스트릿캐주얼"
    assert by_id["fallett"]["overall"]["in_list"] is False
    assert by_id["fallett"]["style_lane"]["rank"] == 133

    assert result["top_k"] == 10

    sections = {sec["label"]: sec for sec in result["sections"]}
    assert sections["전체/전체"]["is_overall"] is True
    assert sections["전체/전체"]["uses_compressed_view"] is True
    overall_targets = {
        row["brand"]
        for row in sections["전체/전체"]["table_rows"]
        if row.get("kind") == "brand" and row.get("highlight")
    }
    assert overall_targets == {"커버낫", "리", "와키윌리", "팔렛"}
    fallett_overall = next(
        row for row in sections["전체/전체"]["table_rows"]
        if row.get("brand") == "팔렛"
    )
    assert fallett_overall["missing"] is True
    assert any(row.get("kind") == "ellipsis" for row in sections["전체/전체"]["table_rows"])

    assert sections["영캐주얼"]["uses_compressed_view"] is False
    young_targets = [
        row["brand"]
        for row in sections["영캐주얼"]["table_rows"]
        if row.get("kind") == "brand" and row.get("highlight")
    ]
    assert young_targets == ["리", "커버낫"]

    fallett_rows = [
        row
        for row in sections["여성캐주얼"]["table_rows"]
        if row.get("kind") == "brand" and row.get("brand") == "팔렛"
    ]
    assert len(fallett_rows) == 1 and fallett_rows[0]["highlight"] is True

    overall_rows = [
        row
        for row in sections["전체/전체"]["table_rows"]
        if row.get("kind") == "brand"
    ]
    head_rows = [row for row in overall_rows if not row["missing"]]
    assert head_rows == overall_rows[: len(head_rows)]
    assert head_rows[0]["rank"] <= head_rows[-1]["rank"]

    street_targets = [
        row["brand"]
        for row in sections["스트릿캐주얼"]["table_rows"]
        if row.get("kind") == "brand" and row.get("highlight")
    ]
    assert street_targets == ["와키윌리"]
