from __future__ import annotations

from silhouette_outliner.analyze import (
    _analyze_slice,
    _build_customer_signals_payload,
    analyze_dataset,
    customer_signals_report_filename,
)
from silhouette_outliner.config import AppConfig, load_config
from silhouette_outliner.customer_signals import (
    _fetch_keyword_rankings,
    _fetch_top_viewed_contents,
    _fluctuation_label,
    _format_count_text,
    _product_ids_from_initial,
    _product_ids_from_list_item,
    join_content_products,
)
from silhouette_outliner.models import (
    ContentSignal,
    CustomerSignalDataset,
    KeywordGenderBoard,
    KeywordSignal,
    NormalizedDataset,
    RankingItem,
    RawCollection,
    utc_timestamp,
)
from silhouette_outliner.report import render_report
from tests.test_normalize_analyze import _make_item


def test_fluctuation_label_maps_types():
    assert _fluctuation_label("UP", 2) == "상승 +2"
    assert _fluctuation_label("DOWN", 1) == "하락 -1"
    assert _fluctuation_label("NONE", None) == "유지"
    assert _fluctuation_label("NEW", None) == "신규"


def test_product_ids_from_initial_merges_meta_and_markers():
    initial = {
        "meta": {"goodsList": [101, 202, 101]},
        "modules": [
            {
                "contents": {
                    "resources": [
                        {
                            "markers": [
                                {"goodsId": "303"},
                                {"goodsId": 202},
                            ]
                        }
                    ]
                }
            }
        ],
    }
    assert _product_ids_from_initial(initial) == ["101", "202", "303"]


def test_format_count_text_uses_korean_units():
    assert _format_count_text(999) == "999"
    assert _format_count_text(12_340) == "1.2만"
    assert _format_count_text(2_470_000) == "247만"


def test_product_ids_from_list_item_dedupes():
    item = {"relatedGoodsList": ["101", 202, "101", "303"]}
    assert _product_ids_from_list_item(item) == ["101", "202", "303"]


def test_fetch_top_viewed_contents_sorts_by_view_count(monkeypatch):
    payload = {
        "data": {
            "list": [
                {
                    "id": "1",
                    "cmsIndex": "100",
                    "landingUrl": "https://www.musinsa.com/content/100",
                    "title": "낮은 조회",
                    "viewCount": 50,
                    "commentCount": 1,
                    "brandNameList": ["A"],
                    "attributeDictionaryName": "스페셜",
                    "relatedGoodsList": ["9001"],
                },
                {
                    "id": "2",
                    "cmsIndex": "200",
                    "landingUrl": "https://www.musinsa.com/content/200",
                    "title": "높은 조회",
                    "viewCount": 5000,
                    "commentCount": 3,
                    "brandNameList": ["B"],
                    "attributeDictionaryName": "에디토리얼",
                    "relatedGoodsList": [],
                },
            ]
        }
    }

    def fake_http_json(url: str, referer: str):
        assert "contentCategoryCode=001" in url
        assert "CONTENT_POPULARITY_ONE_WEEK_SCORE" in url
        return payload

    monkeypatch.setattr(
        "silhouette_outliner.customer_signals._http_json",
        fake_http_json,
    )
    rows = _fetch_top_viewed_contents(1)
    assert len(rows) == 1
    assert rows[0]["content_id"] == "200"
    assert rows[0]["rank"] == 1
    assert rows[0]["view_count_text"] == "5,000"
    assert rows[0]["comment_count_text"] == "3"


def test_fetch_keyword_rankings_parses_modules(monkeypatch):
    payload = {
        "data": {
            "modules": [
                {
                    "type": "RANKING_SEARCH",
                    "rank": "1",
                    "fluctuation": {"type": "NONE"},
                    "title": {"text": "반팔"},
                    "onClick": {"url": "https://www.musinsa.com/search/goods?keyword=%EB%B0%98%ED%8C%94"},
                },
                {
                    "type": "RANKING_SEARCH",
                    "rank": "2",
                    "fluctuation": {"type": "UP", "amount": "3"},
                    "title": {"text": "아디다스"},
                    "onClick": {"url": "https://example.com/search"},
                },
            ]
        }
    }

    def fake_http_json(url: str, referer: str):
        return payload

    monkeypatch.setattr(
        "silhouette_outliner.customer_signals._http_json",
        fake_http_json,
    )
    rows = _fetch_keyword_rankings("A", 10)
    assert len(rows) == 2
    assert rows[0].keyword == "반팔"
    assert rows[1].fluctuation_label == "상승 +3"


def test_config_targets_unchanged_by_customer_signal_settings():
    from dataclasses import replace
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "periodic-multag.json")
    cfg_off = replace(cfg, collect_customer_signals=False)
    assert len(cfg.targets()) == len(cfg_off.targets())


def test_analyze_slice_omits_price_age_heatmap():
    from silhouette_outliner.config import DEFAULT_CATEGORIES, DEFAULT_SECTIONS
    from silhouette_outliner.models import CollectionTarget

    item = _make_item(rank=1, product_id="1")
    target = CollectionTarget(section=DEFAULT_SECTIONS[0], category=DEFAULT_CATEGORIES[0])
    collection = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="test",
        ok=True,
    )
    result = _analyze_slice([item], [collection], [])
    assert "price_age_heatmap" not in result


def test_join_content_products_uses_goods_lookup_for_unmatched():
    content = ContentSignal(
        rank=1,
        content_id="111",
        title="테스트",
        brand="브랜드",
        content_type="스페셜",
        view_count_text="10",
        comment_count_text="1",
        popularity_score_text=None,
        url="https://www.musinsa.com/content/111",
        product_ids=["missing"],
    )
    products = join_content_products(
        content,
        product_lookup={},
        goods_lookup={
            "missing": {
                "product": "콘텐츠 전용 티셔츠",
                "brand": "브랜드B",
                "product_url": "https://www.musinsa.com/products/1",
                "image_url": "https://example.com/thumb.jpg",
                "price": 39000,
                "review_count": 5,
            }
        },
        period_rank_by_id={},
        realtime_rank_by_id={},
        validation_by_item_id={},
    )
    assert len(products) == 1
    assert products[0].matched is False
    assert products[0].product == "콘텐츠 전용 티셔츠"
    assert products[0].brand == "브랜드B"
    assert products[0].product_url is not None


def test_build_customer_signals_joins_ranking_items(monkeypatch):
    def fake_goods_catalog(product_ids: list[str]):
        return {
            "missing": {
                "product": "미매칭 상품",
                "brand": "브랜드B",
                "product_url": "https://www.musinsa.com/products/missing",
            }
        }

    monkeypatch.setattr(
        "silhouette_outliner.analyze.fetch_goods_catalog",
        fake_goods_catalog,
    )
    items = [
        _make_item(
            rank=12,
            product_id="9001",
            brand="브랜드A",
            product="상품A",
            product_clean="상품A",
            price=39000,
            buyers_now=10,
            review_count=100,
            ranking_window_id="1w",
            gender_filter="A",
            age_band="AGE_BAND_ALL",
        ),
        _make_item(
            rank=5,
            product_id="9001",
            brand="브랜드A",
            product="상품A",
            product_clean="상품A",
            price=39000,
            ranking_window_id="rt",
            gender_filter="A",
            age_band="AGE_BAND_20",
        ),
    ]
    signals = CustomerSignalDataset(
        generated_at=utc_timestamp(),
        contents=[
            ContentSignal(
                rank=1,
                content_id="111",
                title="테스트 콘텐츠",
                brand="브랜드A",
                content_type="스페셜",
                view_count_text="100",
                comment_count_text="2",
                popularity_score_text="10",
                url="https://www.musinsa.com/content/111",
                product_ids=["9001", "missing"],
            )
        ],
        keywords_by_gender=[
            KeywordGenderBoard(
                gender_code="A",
                gender_label="전체",
                rows=[
                    KeywordSignal(
                        rank=1,
                        keyword="반팔",
                        fluctuation_type="UP",
                        fluctuation_amount=1,
                        fluctuation_label="상승 +1",
                        search_url="https://example.com",
                    )
                ],
            )
        ],
    )
    payload = _build_customer_signals_payload(
        items,
        signals,
        demographics_window_label="주간",
        realtime_window_label="실시간",
        primary_window_id="1w",
        realtime_window_id="rt",
    )
    assert payload["has_data"] is True
    content = payload["contents"][0]
    assert content["matched_product_count"] == 1
    matched = content["products"][0]
    assert matched["period_rank"] == 12
    assert matched["realtime_rank"] == 5
    assert matched["buyers_now"] == 10
    unmatched = content["products"][1]
    assert unmatched["matched"] is False
    assert unmatched["product"] == "미매칭 상품"
    assert payload["keywords"]["genders"][0]["rows"][0]["keyword"] == "반팔"


def test_analyze_dataset_includes_customer_signals():
    item = _make_item(
        rank=1,
        product_id="1",
        category_code="001000",
        category_label="상의",
        ranking_window_id="default",
        gender_filter="A",
        age_band="AGE_BAND_ALL",
    )
    dataset = NormalizedDataset(
        generated_at=utc_timestamp(),
        items=[item],
        collections=[],
        customer_signals=CustomerSignalDataset(
            generated_at=utc_timestamp(),
            contents=[],
            keywords_by_gender=[],
        ),
    )
    analysis = analyze_dataset(dataset)
    assert "customer_signals" in analysis
    assert analysis["customer_signals"]["has_data"] is False
    # CS dataset is empty (no contents, no keyword rows) so the single CS
    # report is not built and the field stays None.
    assert analysis["customer_signals_report"] is None


def test_analyze_dataset_builds_single_customer_signals_report():
    items = [
        _make_item(
            rank=1,
            product_id="1",
            category_code="001000",
            category_label="상의",
            ranking_window_id="default",
            gender_filter="A",
            age_band="AGE_BAND_ALL",
        ),
        _make_item(
            rank=1,
            product_id="2",
            category_code="002000",
            category_label="아우터",
            ranking_window_id="default",
            gender_filter="A",
            age_band="AGE_BAND_ALL",
        ),
    ]
    dataset = NormalizedDataset(
        generated_at=utc_timestamp(),
        items=items,
        collections=[],
        customer_signals=CustomerSignalDataset(
            generated_at=utc_timestamp(),
            contents=[],
            keywords_by_gender=[
                KeywordGenderBoard(
                    gender_code="A",
                    gender_label="전체",
                    rows=[
                        KeywordSignal(
                            rank=1,
                            keyword="반팔",
                            fluctuation_type="NONE",
                            fluctuation_amount=None,
                            fluctuation_label="유지",
                            search_url="https://example.com",
                        )
                    ],
                )
            ],
        ),
    )
    analysis = analyze_dataset(dataset)
    cs_report = analysis["customer_signals_report"]
    assert cs_report is not None
    assert cs_report["meta"]["report_kind"] == "customer_signals"
    assert "price_age_heatmap_panels" in cs_report
    # Two categories → two heatmap panels in the single CS report.
    assert len(cs_report["price_age_heatmap_panels"]["categories"]) == 2
    # Nav includes the customer signals tab and the brand tab alongside both
    # categories.
    nav_kinds = [entry["kind"] for entry in cs_report["category_nav"]]
    assert nav_kinds.count("customer_signals") == 1
    assert nav_kinds.count("brand_portfolio") == 1
    assert nav_kinds.count("category") == 2


def test_brand_report_keeps_customer_signals_nav_entry():
    item = _make_item(
        rank=1,
        product_id="1",
        category_code="001000",
        category_label="상의",
        ranking_window_id="default",
        gender_filter="A",
        age_band="AGE_BAND_ALL",
    )
    dataset = NormalizedDataset(
        generated_at=utc_timestamp(),
        items=[item],
        collections=[],
        customer_signals=CustomerSignalDataset(
            generated_at=utc_timestamp(),
            contents=[],
            keywords_by_gender=[
                KeywordGenderBoard(
                    gender_code="A",
                    gender_label="전체",
                    rows=[
                        KeywordSignal(
                            rank=1,
                            keyword="반팔",
                            fluctuation_type="NONE",
                            fluctuation_amount=None,
                            fluctuation_label="유지",
                            search_url="https://example.com",
                        )
                    ],
                )
            ],
        ),
    )
    analysis = analyze_dataset(dataset)
    brand_report = analysis["brand_portfolio_report"]
    nav_kinds = [entry["kind"] for entry in brand_report["category_nav"]]
    assert "customer_signals" in nav_kinds


def test_render_report_includes_customer_signals_section(tmp_path):
    item = _make_item(
        rank=1,
        product_id="1",
        category_code="001000",
        category_label="상의",
        gender_filter="A",
        age_band="AGE_BAND_ALL",
    )
    dataset = NormalizedDataset(
        generated_at=utc_timestamp(),
        items=[item],
        collections=[],
        customer_signals=CustomerSignalDataset(
            generated_at=utc_timestamp(),
            contents=[
                ContentSignal(
                    rank=1,
                    content_id="c1",
                    title="인기 콘텐츠",
                    brand="브랜드",
                    content_type="스페셜",
                    view_count_text="10",
                    comment_count_text="1",
                    popularity_score_text=None,
                    url="https://www.musinsa.com/content/c1",
                    product_ids=[],
                )
            ],
            keywords_by_gender=[
                KeywordGenderBoard(
                    gender_code="A",
                    gender_label="전체",
                    rows=[
                        KeywordSignal(
                            rank=1,
                            keyword="반팔",
                            fluctuation_type="NONE",
                            fluctuation_amount=None,
                            fluctuation_label="유지",
                            search_url="https://example.com",
                        )
                    ],
                )
            ],
        ),
    )
    analysis = analyze_dataset(dataset)
    category_html = render_report(dataset, analysis, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "sec-customer-signals" not in category_html
    # The category report's top nav must always include the 고객 신호 entry.
    assert customer_signals_report_filename() in category_html

    cs_analysis = analysis["customer_signals_report"]
    assert cs_analysis is not None
    cs_html = render_report(
        dataset,
        cs_analysis,
        tmp_path / customer_signals_report_filename(),
    ).read_text(encoding="utf-8")
    assert "sec-customer-signals" in cs_html
    assert "고객 신호" in cs_html
    assert "검색어 랭킹" in cs_html
    assert "콘텐츠판 인기 콘텐츠" in cs_html
    assert "cs-product-chips" in cs_html
    assert "cs-cat-switch" not in cs_html
    # No per-category customer-signals files anymore.
    assert "report_customer_signals_001000.html" not in cs_html
