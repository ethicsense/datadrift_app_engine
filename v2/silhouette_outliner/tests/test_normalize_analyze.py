from silhouette_outliner.analyze import (
    BRAND_PORTFOLIO_NAV_KEY,
    HEAT_WEIGHT_BUYERS,
    HHI_CONCENTRATED_MIN,
    HHI_DIFFUSE_MAX,
    VALIDATION_WEIGHT_BUYERS,
    VALIDATION_WEIGHT_REVIEW,
    VALIDATION_WEIGHT_SALES_LABEL,
    _age_ranking_tables,
    _brand_concentration,
    _heat,
    _linear_regression,
    _heatmap_cell_alpha,
    _price_age_heatmap,
    _price_dot_strip,
    _resolve_realtime_window_id,
    _ranking_validation_scatter,
    _sales_label_score,
    _validation_score,
    analyze_dataset,
    brand_portfolio_report_filename,
)
from silhouette_outliner.demographics import DEFAULT_AGE_BANDS, DEFAULT_GENDER_FILTERS
from pathlib import Path

from silhouette_outliner.config import DEFAULT_CATEGORIES, DEFAULT_SECTIONS, load_config
from silhouette_outliner.models import (
    CategoryTarget,
    CollectionTarget,
    RankingItem,
    RankingWindowSpec,
    RawCollection,
    normalize_category_code,
)
from silhouette_outliner.normalize import _clean_product_name, normalize_collections


def _make_item(**overrides):
    base = dict(
        rank=1,
        brand="B",
        product="P",
        product_clean="P",
        price=10000,
        original_price=None,
        discount_rate=None,
        discount_amount=None,
        product_id="1",
        product_url="https://x/1",
        image_url=None,
        brand_id=None,
        review_count=None,
        review_score=None,
        viewers_now=None,
        buyers_now=None,
        is_sold_out=False,
        labels=[],
        section_label="전체",
        section_id="199",
        category_label="상의",
        category_code="001000",
        category_major_code="001",
        category_minor_code="000",
        category_parent_label="의류",
        source="test",
        collected_at="2026-05-14T02:13:44Z",
    )
    base.update(overrides)
    return RankingItem(**base)


def test_normalize_category_code_supports_three_and_six_digits():
    assert normalize_category_code("000") == "000000"
    assert normalize_category_code("001000") == "001000"
    assert normalize_category_code("1001") == "001001"


def test_product_clean_strips_trailing_sku():
    assert _clean_product_name("뉴 타비 발레리나 플랫 슈즈 - 블랙 / S58WZ0127P3753T8013") == "뉴 타비 발레리나 플랫 슈즈 - 블랙"
    assert _clean_product_name("스탠 스미스 LO 발레 JQ6939") == "스탠 스미스 LO 발레 JQ6939"
    assert _clean_product_name("삼바 OG - 화이트:블랙 / B75806") == "삼바 OG - 화이트:블랙"
    assert _clean_product_name("폴로 치노 베이스볼 캡-누벅") == "폴로 치노 베이스볼 캡-누벅"
    assert _clean_product_name("색상 / RED") == "색상 / RED"  # too short to be SKU


def test_normalize_extracts_product_column_fields():
    target = CollectionTarget(
        section=DEFAULT_SECTIONS[0],
        category=CategoryTarget(label="상의", code="001000", parent_label="의류"),
    )
    collection = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="network-json",
        ok=True,
        payload={
            "data": {
                "modules": [
                    {"type": "TAB_OUTLINED"},
                    {
                        "type": "MULTICOLUMN",
                        "items": [
                            {
                                "type": "PRODUCT_COLUMN",
                                "id": "1387964",
                                "image": {
                                    "rank": 1,
                                    "url": "https://image.example/x.jpg",
                                    "labels": [{"text": "판매 1.1만개"}],
                                },
                                "info": {
                                    "brandName": "무신사 스탠다드",
                                    "productName": "베이식 피케 폴로 셔츠 / TEST123ABC",
                                    "finalPrice": 23890,
                                    "discountRatio": 20,
                                    "isSoldOut": False,
                                    "additionalInformation": [
                                        {"text": "13명이 보는 중"},
                                        {"text": "102명이 구매 중"},
                                    ],
                                },
                                "onClick": {
                                    "url": "https://www.musinsa.com/products/1387964",
                                    "eventLog": {
                                        "amplitude": {
                                            "payload": {
                                                "reviewCount": "1934",
                                                "reviewScore": "96",
                                                "original_price": "29900",
                                                "brand_id": "musinsastandard",
                                            }
                                        },
                                        "ga4": {"payload": {"discount": 6010}},
                                    },
                                },
                            }
                        ],
                    },
                ]
            }
        },
    )

    dataset = normalize_collections([collection])

    assert len(dataset.items) == 1
    item = dataset.items[0]
    assert item.rank == 1
    assert item.brand == "무신사 스탠다드"
    assert item.brand_id == "musinsastandard"
    assert item.product == "베이식 피케 폴로 셔츠 / TEST123ABC"
    assert item.product_clean == "베이식 피케 폴로 셔츠"
    assert item.price == 23890
    assert item.original_price == 29900
    assert item.discount_rate == 20
    assert item.discount_amount == 6010
    assert item.review_count == 1934
    assert item.review_score == 96
    assert item.viewers_now == 13
    assert item.buyers_now == 102
    assert item.is_sold_out is False
    assert item.labels == ["판매 1.1만개"]


def test_normalize_falls_back_to_flat_items_payload():
    target = CollectionTarget(section=DEFAULT_SECTIONS[0], category=DEFAULT_CATEGORIES[0])
    collection = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {
                    "rank": 1,
                    "brandName": "브랜드A",
                    "productName": "상품A",
                    "price": 100000,
                    "discountRate": 20,
                    "productUrl": "/products/1",
                }
            ]
        },
    )
    dataset = normalize_collections([collection])
    assert len(dataset.items) == 1
    assert dataset.items[0].brand == "브랜드A"
    assert dataset.items[0].product_clean == "상품A"


def test_normalize_respects_collection_limit():
    target = CollectionTarget(section=DEFAULT_SECTIONS[0], category=DEFAULT_CATEGORIES[0], limit=100)
    collection = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="client-api",
        ok=True,
        payload={
            "items": [
                {
                    "rank": rank,
                    "brandName": f"브랜드{rank}",
                    "productName": f"상품{rank}",
                    "price": 10000 + rank,
                    "productUrl": f"/products/{rank}",
                }
                for rank in range(1, 102)
            ]
        },
    )

    dataset = normalize_collections([collection])

    assert len(dataset.items) == 100
    assert max(item.rank for item in dataset.items if item.rank is not None) == 100


def test_heat_score_uses_weighted_buyers():
    item = _make_item(viewers_now=10, buyers_now=5)
    assert _heat(item) == 10 + 5 * HEAT_WEIGHT_BUYERS


def test_validation_score_weights_review_buyers_and_sales_label():
    item = _make_item(review_count=9, buyers_now=99, labels=["판매 10만개"])
    # 0.5*log10(10)=0.5  +  0.3*log10(100)=0.6  +  0.2*log10(10)=0.2
    assert abs(_validation_score(item) - (0.5 + 0.6 + 0.2)) < 1e-6


def test_validation_score_handles_missing_signals():
    item = _make_item(review_count=None, buyers_now=None, labels=[])
    assert _validation_score(item) == 0.0


def test_validation_weights_sum_to_one():
    # We don't strictly *need* this, but it's a sanity check on intent.
    assert abs(VALIDATION_WEIGHT_REVIEW + VALIDATION_WEIGHT_BUYERS + VALIDATION_WEIGHT_SALES_LABEL - 1.0) < 1e-9


def test_sales_label_score_extracts_volume_tiers():
    assert _sales_label_score(["판매 1만개"]) == 0.0  # log10(1) = 0
    assert _sales_label_score(["판매 10만개"]) == 1.0
    assert _sales_label_score(["판매 1.1만개"]) > 0.0
    assert _sales_label_score(["누적 판매 50만 돌파"]) > 1.6  # log10(50) ~ 1.69
    # BEST mention should give a small but non-zero bonus.
    assert _sales_label_score(["BEST"]) == 0.5
    assert _sales_label_score(["베스트 ITEM"]) == 0.5
    # Strongest label wins when multiple are present.
    assert _sales_label_score(["BEST", "판매 100만개"]) == 2.0
    assert _sales_label_score([]) == 0.0
    assert _sales_label_score(None) == 0.0


def test_linear_regression_recovers_known_line():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 2x + 1
    slope, intercept = _linear_regression(xs, ys)
    assert abs(slope - 2.0) < 1e-9
    assert abs(intercept - 1.0) < 1e-9


def test_scatter_residuals_split_into_both_sides():
    """With a clear positive correlation between rank-helpfulness and validation,
    we should see both above-line and below-line points populated."""
    items = []
    # Group A: top ranks (1-15) with low activity → below trend (push suspect)
    for rank in range(1, 16):
        items.append(_make_item(rank=rank, review_count=5, buyers_now=0, product_id=f"a{rank}"))
    # Group B: middle ranks (40-60) with very high activity → above trend
    for rank in range(40, 60):
        items.append(_make_item(rank=rank, review_count=20000, buyers_now=80, product_id=f"b{rank}"))
    # Group C: tail ranks (85-100) with mixed activity
    for rank in range(85, 101):
        items.append(_make_item(rank=rank, review_count=100, buyers_now=5, product_id=f"c{rank}"))

    scores = {id(item): _validation_score(item) for item in items}
    result = _ranking_validation_scatter(items, scores)
    assert result["has_data"] is True
    assert result["summary"]["above"] > 0
    assert result["summary"]["below"] > 0
    # Each labeled outlier should reference a real point.
    assert len(result["labels"]) >= 1
    for lab in result["labels"]:
        assert lab["kind"] in {"above", "below"}
        assert lab.get("key")
    assert result["default_preset"] == "top20"
    assert "outliers" in result["presets"]
    assert all(point.get("key") for point in result["points"])
    assert all(point.get("rank_band") for point in result["points"])


def test_brand_concentration_classifies_into_three_levels():
    # Diffuse: many brands evenly distributed
    diffuse = [_make_item(brand=f"B{i}") for i in range(20)]
    bc = _brand_concentration(diffuse)
    assert bc["level"] == "diffuse"
    assert bc["hhi"] < HHI_DIFFUSE_MAX

    # Concentrated: one brand dominates
    concentrated = [_make_item(brand="Top") for _ in range(7)] + [_make_item(brand=f"B{i}") for i in range(3)]
    bc = _brand_concentration(concentrated)
    assert bc["level"] == "concentrated"
    assert bc["hhi"] >= HHI_CONCENTRATED_MIN


def test_normalize_keeps_same_product_id_across_ranking_windows():
    section = DEFAULT_SECTIONS[0]
    category = CategoryTarget(label="상의", code="001000", parent_label="의류")
    w1 = RankingWindowSpec("1d", "일간", {})
    w2 = RankingWindowSpec("1w", "주간", {})
    payload = {
        "items": [
            {
                "rank": 3,
                "brandName": "브랜드A",
                "productName": "상품A",
                "price": 100000,
                "discountRate": 20,
                "productUrl": "/products/1",
            }
        ]
    }
    c1 = RawCollection(
        target=CollectionTarget(section=section, category=category, ranking_window=w1),
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload=payload,
    )
    c2 = RawCollection(
        target=CollectionTarget(section=section, category=category, ranking_window=w2),
        url="https://example.com",
        collected_at="2026-05-14T02:13:45Z",
        source="dom",
        ok=True,
        payload=payload,
    )
    dataset = normalize_collections([c1, c2])
    assert len(dataset.items) == 2
    ids = {item.ranking_window_id for item in dataset.items}
    assert ids == {"1d", "1w"}
    assert {item.rank for item in dataset.items} == {3}


def test_analyze_cross_window_two_periods():
    section = DEFAULT_SECTIONS[0]
    category = CategoryTarget(label="상의", code="001000", parent_label="의류")
    w1 = RankingWindowSpec("1d", "일간", {})
    w2 = RankingWindowSpec("1w", "주간", {})
    c1 = RawCollection(
        target=CollectionTarget(section=section, category=category, ranking_window=w1),
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {
                    "rank": 5,
                    "brandName": "브랜드A",
                    "productName": "상품A",
                    "price": 100000,
                    "discountRate": 20,
                    "productUrl": "/products/1",
                },
                {
                    "rank": 10,
                    "brandName": "브랜드B",
                    "productName": "상품B",
                    "price": 50000,
                    "productUrl": "/products/2",
                },
            ]
        },
    )
    c2 = RawCollection(
        target=CollectionTarget(section=section, category=category, ranking_window=w2),
        url="https://example.com",
        collected_at="2026-05-14T02:13:45Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {
                    "rank": 12,
                    "brandName": "브랜드A",
                    "productName": "상품A",
                    "price": 100000,
                    "discountRate": 20,
                    "productUrl": "/products/1",
                },
            ]
        },
    )
    dataset = normalize_collections([c1, c2])
    analysis = analyze_dataset(dataset)

    cw = analysis["cross_window"]
    assert cw["has_data"] is True
    # Chart order is oldest → newest (largest days first); '1w' has days=7, '1d' has days=1
    assert cw["chart_window_order"] == ["1w", "1d"]
    # Legacy pair summary remains for adjacent windows.
    assert len(cw["legacy"]["pair_summaries"]) == 1
    pair = cw["legacy"]["pair_summaries"][0]
    assert pair["from_window"] == "1w"
    assert pair["to_window"] == "1d"
    assert pair["both_count"] == 1
    assert pair["only_from_count"] == 0
    assert pair["only_to_count"] == 1
    assert analysis["meta"]["multiple_windows"] is True
    assert analysis["headline"]["item_count"] == 2
    assert analysis["headline"]["total_item_count"] == 3
    assert analysis["headline"]["limit_target"] == 100
    assert analysis["kpis"]["discounted_count"] == 1
    assert analysis["kpis"]["discount_application_pct"] == 50.0
    assert "windows" in analysis and set(analysis["windows"].keys()) == {"1d", "1w"}
    # New per-window TOP10 is populated even when fewer than 10 items exist.
    assert {row["rank"] for row in analysis["windows"]["1d"]["top10"]} == {5, 10}
    assert analysis["windows"]["1w"]["top10"][0]["product"] == "상품A"


def _make_collection(section, category, window, items):
    return RawCollection(
        target=CollectionTarget(section=section, category=category, ranking_window=window),
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={"items": items},
    )


def _three_window_dataset():
    """Build a synthetic dataset that exercises every momentum pattern."""

    section = DEFAULT_SECTIONS[0]
    category = CategoryTarget(label="상의", code="001000", parent_label="의류")
    w_d = RankingWindowSpec("1d", "일간", {"period": "DAILY"}, days=1)
    w_w = RankingWindowSpec("1w", "주간", {"period": "WEEKLY"}, days=7)
    w_m = RankingWindowSpec("1m", "월간", {"period": "MONTHLY"}, days=30)

    items_d = [
        {"rank": 1, "brandName": "A", "productName": "Breakout", "price": 50000, "productUrl": "/p/A"},
        {"rank": 3, "brandName": "B", "productName": "Climber", "price": 30000, "productUrl": "/p/B"},
        {"rank": 80, "brandName": "C", "productName": "Fading", "price": 25000, "productUrl": "/p/C"},
        {"rank": 15, "brandName": "E", "productName": "DayWeek", "price": 40000, "productUrl": "/p/E"},
    ]
    items_w = [
        {"rank": 30, "brandName": "B", "productName": "Climber", "price": 30000, "productUrl": "/p/B"},
        {"rank": 50, "brandName": "C", "productName": "Fading", "price": 25000, "productUrl": "/p/C"},
        {"rank": 25, "brandName": "E", "productName": "DayWeek", "price": 40000, "productUrl": "/p/E"},
        {"rank": 18, "brandName": "F", "productName": "MonthWeek", "price": 60000, "productUrl": "/p/F"},
    ]
    items_m = [
        {"rank": 70, "brandName": "B", "productName": "Climber", "price": 30000, "productUrl": "/p/B"},
        {"rank": 5, "brandName": "C", "productName": "Fading", "price": 25000, "productUrl": "/p/C"},
        {"rank": 12, "brandName": "D", "productName": "Classic", "price": 120000, "productUrl": "/p/D"},
        {"rank": 22, "brandName": "F", "productName": "MonthWeek", "price": 60000, "productUrl": "/p/F"},
    ]
    collections = [
        _make_collection(section, category, w_d, items_d),
        _make_collection(section, category, w_w, items_w),
        _make_collection(section, category, w_m, items_m),
    ]
    return normalize_collections(collections)


def _find_product(products, url_suffix):
    return next(p for p in products if (p["url"] or "").endswith(url_suffix))


def test_rank_energy_and_momentum_span_values():
    dataset = _three_window_dataset()
    analysis = analyze_dataset(dataset)
    cw = analysis["cross_window"]
    assert cw["has_data"] is True
    # Oldest → newest order on the chart.
    assert cw["chart_window_order"] == ["1m", "1w", "1d"]

    climber = _find_product(cw["products"], "/p/B")
    # rank_energy(rank, limit=100) = (101 - rank) / 100
    assert climber["rank_energy"]["1m"] == round((101 - 70) / 100, 6)
    assert climber["rank_energy"]["1w"] == round((101 - 30) / 100, 6)
    assert climber["rank_energy"]["1d"] == round((101 - 3) / 100, 6)
    # momentum_span = rank_energy(newest) - rank_energy(oldest)
    expected_span = round(climber["rank_energy"]["1d"] - climber["rank_energy"]["1m"], 4)
    assert climber["momentum_span"] == expected_span
    assert climber["pattern"] == "steady_climb"


def test_sustained_rank_energy_uses_sqrt_days_weights():
    import math

    dataset = _three_window_dataset()
    analysis = analyze_dataset(dataset)
    climber = _find_product(analysis["cross_window"]["products"], "/p/B")
    expected = (
        math.sqrt(30) * climber["rank_energy"]["1m"]
        + math.sqrt(7) * climber["rank_energy"]["1w"]
        + math.sqrt(1) * climber["rank_energy"]["1d"]
    )
    assert abs(climber["sustained_rank_energy"] - round(expected, 4)) < 1e-3


def test_pattern_classification_covers_main_buckets():
    dataset = _three_window_dataset()
    cw = analyze_dataset(dataset)["cross_window"]
    by_suffix = {p["url"].rsplit("/", 1)[-1]: p for p in cw["products"]}
    assert by_suffix["A"]["pattern"] == "entry_breakout"
    assert by_suffix["B"]["pattern"] == "steady_climb"
    assert by_suffix["C"]["pattern"] == "fading"
    assert by_suffix["D"]["pattern"] == "classic_drop"
    assert by_suffix["E"]["pattern"] == "transient_dw"
    assert by_suffix["F"]["pattern"] == "transient_wm"


def test_momentum_chart_lines_include_all_windows_with_observation_flag():
    dataset = _three_window_dataset()
    cw = analyze_dataset(dataset)["cross_window"]
    chart = cw["momentum_chart"]
    by_suffix = {p["url"].rsplit("/", 1)[-1]: p for p in cw["products"]}
    assert chart["has_data"] is True
    assert chart["default_preset"] == "top12"
    assert "top12" in chart["presets"]
    assert len(chart["presets"]["top12"]["keys"]) <= 12
    assert chart["canvas"]["h"] == 480
    assert {ax["id"] for ax in chart["axis_windows"]} == {"1m", "1w", "1d"}
    b_line = next(line for line in chart["lines"] if "Climber" in line["product"])
    assert len(b_line["points"]) == 3
    assert len(b_line["observed_coords"]) == 3
    assert b_line["momentum_span"] is not None
    entry = by_suffix["A"]
    assert entry["momentum_span"] is None
    assert entry["event_cluster"] == "entry_latest_only"
    a_line = next(line for line in chart["lines"] if "Breakout" in line["product"])
    assert a_line["event_cluster"] == "entry_latest_only"
    span_keys = {p["key"] for p in cw["products"] if p.get("momentum_span") is not None}
    assert set(chart["presets"]["top12"]["keys"]).issubset(span_keys)


def test_momentum_distribution_bin_keys_match_spans():
    from silhouette_outliner.analyze import _momentum_bin_index

    cw = analyze_dataset(_three_window_dataset())["cross_window"]
    dist = cw["momentum_distribution"]
    assert dist["has_data"] is True
    products_with_span = [p for p in cw["products"] if p.get("momentum_span") is not None]
    assert dist["total"] == len(products_with_span)
    all_keys: list[str] = []
    by_key = {p["key"]: p for p in cw["products"]}
    for cell in dist["cells"]:
        assert cell["count"] == len(cell["product_keys"])
        all_keys.extend(cell["product_keys"])
        for key in cell["product_keys"]:
            assert _momentum_bin_index(by_key[key]["momentum_span"]) == cell["index"]
    assert len(all_keys) == dist["total"]


def test_event_distribution_includes_product_rows():
    cw = analyze_dataset(_three_window_dataset())["cross_window"]
    evdist = cw["event_distribution"]
    entry = next(c for c in evdist["cells"] if c["id"] == "entry_latest_only")
    assert entry["count"] >= 1
    assert len(entry["rows"]) == entry["count"]
    assert entry["picks"] == entry["rows"][:3]
    assert entry["is_primary"] is True
    assert "일간" in entry["subtitle"]
    assert entry["rows"][0]["event_strength"] is not None
    assert entry["rows"][0]["brand"]
    exit_cell = next(c for c in evdist["cells"] if c["id"] == "exit_oldest_only")
    assert exit_cell["count"] >= 1
    assert exit_cell["rows"][0]["event_strength"] == exit_cell["rows"][0].get("event_strength")


def test_momentum_distribution_includes_product_rows():
    cw = analyze_dataset(_three_window_dataset())["cross_window"]
    dist = cw["momentum_distribution"]
    assert dist["has_data"] is True
    populated = [c for c in dist["cells"] if c["count"]]
    assert populated
    cell = populated[0]
    assert len(cell["rows"]) == cell["count"]
    assert cell["picks"] == cell["rows"][:3]
    assert cell["label"].startswith("[")
    assert "momentum_span" in cell["subtitle"]
    assert cell["rows"][0]["momentum_span"] is not None
    assert cell["rows"][0]["brand"]


def test_entry_exit_use_event_cluster_not_momentum_span():
    cw = analyze_dataset(_three_window_dataset())["cross_window"]
    by_suffix = {p["url"].rsplit("/", 1)[-1]: p for p in cw["products"]}
    entry = by_suffix["A"]
    assert entry["momentum_span"] is None
    assert entry["event_cluster"] == "entry_latest_only"
    assert entry["event_strength"] == entry["rank_energy"]["1d"]
    drop = by_suffix["D"]
    assert drop["momentum_span"] is None
    assert drop["event_cluster"] == "exit_oldest_only"
    assert drop["event_strength"] == drop["rank_energy"]["1m"]


def test_momentum_distribution_and_events_partition_join():
    cw = analyze_dataset(_three_window_dataset())["cross_window"]
    chart = cw["momentum_chart"]
    dist = cw["momentum_distribution"]
    evdist = cw["event_distribution"]
    join_n = len(cw["products"])
    span_n = sum(1 for p in cw["products"] if p.get("momentum_span") is not None)
    event_n = sum(1 for p in cw["products"] if p.get("event_cluster") not in (None, "none"))
    assert dist["total"] == span_n
    assert evdist["total"] == event_n
    assert span_n + event_n == join_n
    assert chart["plotted_count"] == join_n
    assert chart["momentum_cohort_size"] == span_n
    assert chart["event_cohort_size"] == event_n


def test_weekly_only_product_is_middle_event_cluster():
    dataset = _three_window_dataset()
    section = DEFAULT_SECTIONS[0]
    category = CategoryTarget(label="상의", code="001000", parent_label="의류")
    w_w = RankingWindowSpec("1w", "주간", {"period": "WEEKLY"}, days=7)
    extra_w = _make_collection(
        section,
        category,
        w_w,
        [{"rank": 20, "brandName": "W", "productName": "WeeklyOnly", "price": 10000, "productUrl": "/p/W"}],
    )
    # Rebuild collections: keep 1d/1m from dataset, replace 1w with base + weekly-only row.
    base_cols = dataset.collections
    merged_w_items = list(base_cols[1].payload["items"]) + list(extra_w.payload["items"])
    collections = [
        base_cols[0],
        _make_collection(section, category, w_w, merged_w_items),
        base_cols[2],
    ]
    cw = analyze_dataset(normalize_collections(collections))["cross_window"]
    product = next(p for p in cw["products"] if (p["url"] or "").endswith("/p/W"))
    assert product["momentum_span"] is None
    assert product["event_cluster"] == "middle_only"
    assert product["event_strength"] == product["rank_energy"]["1w"]


def test_transient_pairs_are_event_clusters_not_momentum_span():
    cw = analyze_dataset(_three_window_dataset())["cross_window"]
    by_suffix = {p["url"].rsplit("/", 1)[-1]: p for p in cw["products"]}

    day_week = by_suffix["E"]
    assert day_week["momentum_span"] is None
    assert day_week["event_cluster"] == "transient_new"
    assert day_week["event_strength"] == max(day_week["rank_energy"].values())

    month_week = by_suffix["F"]
    assert month_week["momentum_span"] is None
    assert month_week["event_cluster"] == "transient_old"
    assert month_week["event_strength"] == max(month_week["rank_energy"].values())


def test_insight_rows_and_momentum_extrema():
    dataset = _three_window_dataset()
    cw = analyze_dataset(dataset)["cross_window"]
    for card in cw["insights"]:
        assert len(card["rows"]) == card["count"]
        assert card["picks"] == card["rows"][:3]
        for row in card["rows"]:
            assert row.get("key")
    me = cw["momentum_extrema"]
    assert me["has_data"] is True
    assert me["n"] == 5
    assert len(me["top"]) <= 5
    assert len(me["bottom"]) <= 5
    if len(me["top"]) >= 2:
        assert me["top"][0]["momentum_span"] >= me["top"][1]["momentum_span"]
    if len(me["bottom"]) >= 2:
        assert me["bottom"][0]["momentum_span"] <= me["bottom"][1]["momentum_span"]


def test_single_window_skips_momentum_chart():
    target = CollectionTarget(section=DEFAULT_SECTIONS[0], category=DEFAULT_CATEGORIES[0])
    raw = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {"rank": 1, "brandName": "A", "productName": "X", "price": 30000, "productUrl": "/p/1"},
            ]
        },
    )
    analysis = analyze_dataset(normalize_collections([raw]))
    assert analysis["meta"]["multiple_windows"] is False
    assert analysis["cross_window"]["has_data"] is False
    assert analysis["cross_window"]["momentum_chart"]["has_data"] is False
    assert analysis["cross_window"]["momentum_extrema"]["has_data"] is False


def test_window_spec_days_inferred_from_label_when_missing():
    spec = RankingWindowSpec("custom_a", "일간 베스트")
    assert spec.days_effective == 1
    spec_w = RankingWindowSpec("custom_b", "주간 베스트")
    assert spec_w.days_effective == 7
    spec_m = RankingWindowSpec("custom_c", "월간 베스트")
    assert spec_m.days_effective == 30
    spec_d_id = RankingWindowSpec("1d", "Anything")
    assert spec_d_id.days_effective == 1


def test_analyze_dataset_full_shape():
    target = CollectionTarget(section=DEFAULT_SECTIONS[0], category=DEFAULT_CATEGORIES[0])
    collection = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {"rank": 1, "brandName": "브랜드A", "productName": "상품A", "price": 100000, "discountRate": 20, "productUrl": "/products/1"},
                {"rank": 2, "brandName": "브랜드A", "productName": "상품B", "price": 120000, "discountRate": 10, "productUrl": "/products/2"},
            ]
        },
    )
    dataset = normalize_collections([collection])
    analysis = analyze_dataset(dataset)

    # KPI key no longer collides with dict.items
    assert analysis["kpis"]["item_count"] == 2
    assert analysis["kpis"]["brand_count"] == 1
    assert analysis["kpis"]["validated_count"] == 0  # no sales labels in fixture
    assert analysis["kpis"]["discounted_count"] == 2
    assert analysis["kpis"]["discount_application_pct"] == 100.0
    assert analysis["kpis"]["avg_discount_rate"] == 15.0
    assert analysis["headline"]["item_count"] == 2
    assert analysis["headline"]["limit_target"] == 100
    assert analysis["headline"]["category"] == "상의"
    assert "KST" in analysis["headline"]["collected_at_kst_pretty"]
    assert analysis["brand_concentration"]["has_data"] is True
    assert analysis["price_dot_strip"]["has_data"] is True
    assert "axis" in analysis["price_dot_strip"]
    assert "presets" in analysis["price_dot_strip"]
    assert len(analysis["price_dot_strip"]["presets"]) == 6
    assert analysis["price_dot_strip"]["dots"][0]["key"]
    assert analysis["age_rankings"]["has_data"] is False
    assert analysis["quality"]["sentence"].startswith("수집 2건")
    # Scatter exists even without reviews — every item has a rank, validation
    # just stays at 0.0 for items with no activity signal.
    assert analysis["scatter"]["has_data"] is True
    assert len(analysis["scatter"]["points"]) == 2


def test_analyze_dataset_groups_category_reports():
    tops = CategoryTarget(label="상의", code="001000", parent_label="의류")
    outer = CategoryTarget(label="아우터", code="002000", parent_label="의류")
    section = DEFAULT_SECTIONS[0]
    collections = [
        RawCollection(
            target=CollectionTarget(section=section, category=tops),
            url="https://example.com/tops",
            collected_at="2026-05-14T02:13:44Z",
            source="dom",
            ok=True,
            payload={
                "items": [
                    {"rank": 1, "brandName": "A", "productName": "상의상품", "price": 10000, "productUrl": "/products/1"},
                ]
            },
        ),
        RawCollection(
            target=CollectionTarget(section=section, category=outer),
            url="https://example.com/outer",
            collected_at="2026-05-14T02:13:44Z",
            source="dom",
            ok=True,
            payload={
                "items": [
                    {"rank": 1, "brandName": "B", "productName": "아우터상품", "price": 20000, "productUrl": "/products/2"},
                ]
            },
        ),
    ]

    analysis = analyze_dataset(normalize_collections(collections))

    assert set(analysis["category_reports"]) == {"001000", "002000"}
    assert analysis["headline"]["category"] == "상의"
    assert analysis["category_reports"]["002000"]["headline"]["category"] == "아우터"
    assert analysis["category_reports"]["002000"]["bcave_tracker"]["has_data"] is False
    assert len(analysis["category_nav"]) == 3
    assert analysis["category_nav"][-1]["label"] == "브랜드"
    assert analysis["category_nav"][-1]["kind"] == "brand_portfolio"
    assert analysis["category_nav"][-1]["href"] == brand_portfolio_report_filename()
    assert analysis["meta"]["multiple_categories"] is True

    portfolio = analysis["brand_portfolio"]
    assert portfolio["has_data"] is True
    assert len(portfolio["category_share_rows"]) == 2
    assert portfolio["category_count"] == 2
    assert portfolio["target_count"] == 4

    brand_report = analysis["brand_portfolio_report"]
    assert brand_report["meta"]["report_kind"] == "brand_portfolio"
    assert brand_report["category_nav"][-1]["active"] is True
    assert brand_report["category_nav"][-1]["code"] == BRAND_PORTFOLIO_NAV_KEY


def test_brand_portfolio_marks_target_brand_in_category_share():
    tops = CategoryTarget(label="상의", code="001000", parent_label="의류")
    section = DEFAULT_SECTIONS[0]
    collection = RawCollection(
        target=CollectionTarget(section=section, category=tops),
        url="https://example.com/tops",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {"rank": 1, "brandName": "커버낫", "productName": "후드", "price": 10000, "productUrl": "/products/1"},
                {"rank": 2, "brandName": "커버낫", "productName": "티셔츠", "price": 12000, "productUrl": "/products/2"},
                {"rank": 3, "brandName": "기타브랜드", "productName": "셔츠", "price": 9000, "productUrl": "/products/3"},
            ]
        },
    )
    analysis = analyze_dataset(normalize_collections([collection]))
    row = analysis["brand_portfolio"]["category_share_rows"][0]
    covernat = next(bar for bar in row["top_brands"] if bar["brand"] == "커버낫")
    assert covernat["is_target"] is True
    assert covernat["target_label"] == "커버낫"
    assert covernat["bar_scale_pct"] == 100.0
    assert row["has_targets"] is True

    matrix_row = next(
        row for row in analysis["brand_portfolio"]["target_brand_matrix"] if row["id"] == "covernat"
    )
    cell = matrix_row["categories"][0]
    assert cell["in_list"] is True
    assert cell["share_pct"] == 66.7
    assert len(cell["products"]) == 2
    assert cell["products"][0]["rank"] == 1
    assert cell["products"][0]["product"] == "후드"
    assert cell["products"][1]["rank"] == 2


def test_category_brand_distribution_tables_sorted_by_count_and_best_rank():
    tops = CategoryTarget(label="상의", code="001000", parent_label="의류")
    section = DEFAULT_SECTIONS[0]
    collection = RawCollection(
        target=CollectionTarget(section=section, category=tops),
        url="https://example.com/tops",
        collected_at="2026-05-14T02:13:44Z",
        source="dom",
        ok=True,
        payload={
            "items": [
                {"rank": 1, "brandName": "커버낫", "productName": "후드", "price": 10000, "productUrl": "/products/1"},
                {"rank": 5, "brandName": "커버낫", "productName": "티셔츠", "price": 12000, "productUrl": "/products/2"},
                {"rank": 2, "brandName": "기타브랜드", "productName": "셔츠", "price": 9000, "productUrl": "/products/3"},
                {"rank": 8, "brandName": "단일브랜드", "productName": "팬츠", "price": 8000, "productUrl": "/products/4"},
            ]
        },
    )
    analysis = analyze_dataset(normalize_collections([collection]))
    table = analysis["brand_portfolio"]["category_brand_tables"][0]
    assert table["category_label"] == "상의"
    assert table["item_count"] == 4
    assert table["brand_count"] == 3
    assert table["rows"][0]["brand"] == "커버낫"
    assert table["rows"][0]["product_count"] == 2
    assert table["rows"][0]["best_rank"] == 1
    assert table["rows"][0]["best_rank_label"] == "#1"
    assert table["rows"][0]["is_target"] is True
    assert table["rows"][1]["brand"] == "기타브랜드"
    assert table["rows"][1]["product_count"] == 1
    assert table["rows"][1]["best_rank"] == 2
    assert len(table["rows"][0]["top_products"]) == 2
    assert table["rows"][0]["top_products"][0]["rank"] == 1
    assert table["rows"][0]["top_products"][0]["product"] == "후드"


def test_overall_brand_distribution_table_top_ten_and_products():
    tops = CategoryTarget(label="상의", code="001000", parent_label="의류")
    outer = CategoryTarget(label="아우터", code="002000", parent_label="의류")
    section = DEFAULT_SECTIONS[0]
    items = []
    for rank, brand in enumerate(
        [
            "B1",
            "B1",
            "B2",
            "B2",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
        ],
        start=1,
    ):
        items.append(
            {
                "rank": rank,
                "brandName": brand,
                "productName": f"P{rank}",
                "price": 10000,
                "productUrl": f"/products/{rank}",
            }
        )
    collections = [
        RawCollection(
            target=CollectionTarget(section=section, category=tops),
            url="https://example.com/tops",
            collected_at="2026-05-14T02:13:44Z",
            source="dom",
            ok=True,
            payload={"items": items},
        ),
        RawCollection(
            target=CollectionTarget(section=section, category=outer),
            url="https://example.com/outer",
            collected_at="2026-05-14T02:13:44Z",
            source="dom",
            ok=True,
            payload={
                "items": [
                    {
                        "rank": 1,
                        "brandName": "B1",
                        "productName": "아우터1",
                        "price": 20000,
                        "productUrl": "/products/o1",
                    }
                ]
            },
        ),
    ]
    analysis = analyze_dataset(normalize_collections(collections))
    overall = analysis["brand_portfolio"]["overall_brand_table"]
    assert overall is not None
    assert overall["category_label"] == "전체 상품"
    assert overall["is_overall"] is True
    assert len(overall["rows"]) == 10
    assert overall["rows"][0]["brand"] == "B1"
    assert overall["rows"][0]["product_count"] == 3
    assert len(overall["rows"][0]["top_products"]) == 3
    assert overall["rows"][0]["top_products"][0]["category_label"] == "상의"
    assert overall["rows"][0]["top_products"][0]["validation"] >= 0
    assert "image_url" in overall["rows"][0]["top_products"][0]
    assert overall["rows"][0]["lead_product"]["rank"] == 1


def test_normalize_preserves_gender_and_age_metadata():
    gender = DEFAULT_GENDER_FILTERS[1]
    age = DEFAULT_AGE_BANDS[2]
    target = CollectionTarget(
        section=DEFAULT_SECTIONS[0],
        category=DEFAULT_CATEGORIES[0],
        gender_filter=gender.code,
        gender_label=gender.label,
        age_band=age.code,
        age_label=age.label,
    )
    collection = RawCollection(
        target=target,
        url="https://example.com",
        collected_at="2026-05-20T00:00:00Z",
        source="client-api",
        ok=True,
        payload={
            "items": [
                {
                    "rank": 1,
                    "brandName": "브랜드A",
                    "productName": "상품A",
                    "price": 45000,
                    "productUrl": "/products/1",
                }
            ]
        },
    )
    item = normalize_collections([collection]).items[0]
    assert item.gender_filter == "M"
    assert item.gender_label == "남성"
    assert item.age_band == "AGE_BAND_20"
    assert item.age_label == "20-24"


def test_normalize_keeps_same_product_across_age_segments():
    base_payload = {
        "items": [
            {
                "rank": 1,
                "brandName": "브랜드A",
                "productName": "상품A",
                "price": 45000,
                "productUrl": "/products/shared",
            }
        ]
    }
    ages = (DEFAULT_AGE_BANDS[1], DEFAULT_AGE_BANDS[2])
    collections = [
        RawCollection(
            target=CollectionTarget(
                section=DEFAULT_SECTIONS[0],
                category=DEFAULT_CATEGORIES[0],
                age_band=age.code,
                age_label=age.label,
            ),
            url="https://example.com",
            collected_at="2026-05-20T00:00:00Z",
            source="client-api",
            ok=True,
            payload=base_payload,
        )
        for age in ages
    ]
    dataset = normalize_collections(collections)
    assert len(dataset.items) == 2
    assert {item.age_band for item in dataset.items} == {ages[0].code, ages[1].code}


def test_heatmap_cell_alpha_spreads_clustered_counts():
    assert _heatmap_cell_alpha(0, 5, 20) == 0.0
    low = _heatmap_cell_alpha(5, 5, 20)
    mid = _heatmap_cell_alpha(12, 5, 20)
    high = _heatmap_cell_alpha(20, 5, 20)
    assert low < mid < high
    assert high == 1.0
    assert mid < 12 / 20


def test_price_age_heatmap_builds_gender_grids_and_totals():
    items = [
        _make_item(
            rank=1,
            price=25_000,
            gender_filter="A",
            gender_label="전체",
            age_band="AGE_BAND_20",
            age_label="20-24",
            product_id="p1",
        ),
        _make_item(
            rank=2,
            price=28_000,
            gender_filter="A",
            gender_label="전체",
            age_band="AGE_BAND_20",
            age_label="20-24",
            product_id="p2",
        ),
        _make_item(
            rank=3,
            price=120_000,
            gender_filter="M",
            gender_label="남성",
            age_band="AGE_BAND_30",
            age_label="30-34",
            product_id="p3",
        ),
    ]
    result = _price_age_heatmap(items)
    assert result["has_data"] is True
    all_grid = next(g for g in result["genders"] if g["key"] == "all")
    assert all_grid["has_data"] is True
    age_20_col = next(i for i, a in enumerate(DEFAULT_AGE_BANDS) if a.id == "20_24")
    age_30_col = next(i for i, a in enumerate(DEFAULT_AGE_BANDS) if a.id == "30_34")
    low_price_row = next(r for r in all_grid["rows"] if r["price_label"] == "~3만")
    assert low_price_row["cells"][age_20_col]["count"] == 2
    male_grid = next(g for g in result["genders"] if g["key"] == "male")
    mid_price_row = next(r for r in male_grid["rows"] if r["price_label"] == "10~20만")
    assert mid_price_row["cells"][age_30_col]["count"] == 1


def test_cross_window_excludes_realtime_age_rankings_window():
    """실시간(rt) and 일간(1d) both have days=1; mixing them stacks two Y at one X."""
    section = DEFAULT_SECTIONS[0]
    category = CategoryTarget(label="상의", code="001000", parent_label="의류")
    w_d = RankingWindowSpec("1d", "일간", {"period": "DAILY"}, days=1)
    w_w = RankingWindowSpec("1w", "주간", {"period": "WEEKLY"}, days=7)
    w_m = RankingWindowSpec("1m", "월간", {"period": "MONTHLY"}, days=30)
    w_rt = RankingWindowSpec("rt", "실시간", {"period": "REALTIME"}, days=1)
    product = {
        "rank": 1,
        "brandName": "브랜드A",
        "productName": "상품A",
        "price": 100000,
        "productUrl": "/products/1",
    }
    collections = [
        _make_collection(section, category, w_m, [{**product, "rank": 50}]),
        _make_collection(section, category, w_w, [{**product, "rank": 20}]),
        _make_collection(section, category, w_d, [{**product, "rank": 5}]),
        _make_collection(section, category, w_rt, [{**product, "rank": 48}]),
    ]
    dataset = normalize_collections(collections)
    cw = analyze_dataset(dataset)["cross_window"]
    assert cw["has_data"] is True
    assert "rt" not in cw["chart_window_order"]
    assert cw["chart_window_order"] == ["1m", "1w", "1d"]
    chart = cw["momentum_chart"]
    axis_ids = [ax["id"] for ax in chart["axis_windows"]]
    assert axis_ids == ["1m", "1w", "1d"]
    line = next(ln for ln in chart["lines"] if ln["url"].endswith("/products/1"))
    xs = {pt["x"] for pt in line["observed_coords"]}
    assert len(xs) == len(line["observed_coords"])


def test_periodic_multag_config_splits_period_and_demographics_tracks():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "periodic-multag.json"
    config = load_config(config_path)
    assert config.demographics_window.id == "1w"
    assert config.demographics_window.label == "주간"
    assert config.age_rankings_window is not None
    assert config.age_rankings_window.id == "rt"
    assert config.age_rankings_window.query_params["period"] == "REALTIME"
    targets = config.targets()
    product_targets = [t for t in targets if t.sub_pan == "product"]
    brand_targets = [t for t in targets if t.sub_pan == "brand"]
    assert config.expands_demographics() is True

    # Period 12 + demographics 80 + age-rankings 28 (성별 전체 × 연령 7 × 실시간)
    assert len(product_targets) == 12 + 4 * (3 * 7 - 1) + 4 * 7
    assert len(brand_targets) == 12 + 4 * (3 * 7 - 1)
    rt_product = [
        t for t in product_targets if t.ranking_window.id == "rt" and t.gender_filter == "A"
    ]
    assert len(rt_product) == 28


def test_category_reports_get_distinct_price_age_heatmaps():
    tops = CategoryTarget(label="상의", code="001000", parent_label="의류")
    outer = CategoryTarget(label="아우터", code="002000", parent_label="의류")
    section = DEFAULT_SECTIONS[0]
    window = RankingWindowSpec("1w", "주간", {"period": "WEEKLY"}, days=7)
    age = DEFAULT_AGE_BANDS[2]

    def collection(category, price, product_suffix):
        return RawCollection(
            target=CollectionTarget(
                section=section,
                category=category,
                ranking_window=window,
                age_band=age.code,
                age_label=age.label,
            ),
            url=f"https://example.com/{product_suffix}",
            collected_at="2026-05-20T00:00:00Z",
            source="client-api",
            ok=True,
            payload={
                "items": [
                    {
                        "rank": 1,
                        "brandName": "B",
                        "productName": f"P-{product_suffix}",
                        "price": price,
                        "productUrl": f"/products/{product_suffix}",
                    }
                ]
            },
        )

    dataset = normalize_collections([
        collection(tops, 25_000, "tops"),
        collection(outer, 150_000, "outer"),
    ])
    analysis = analyze_dataset(dataset)
    tops_hm = analysis["category_reports"]["001000"]["price_age_heatmap"]
    outer_hm = analysis["category_reports"]["002000"]["price_age_heatmap"]
    assert tops_hm["has_data"] is True
    assert outer_hm["has_data"] is True
    assert tops_hm["ranking_window_label"] == "주간"
    assert tops_hm["ranking_window_id"] == "1w"
    assert analysis["category_reports"]["001000"]["meta"]["demographics_window_id"] == "1w"
    tops_low = next(
        r for r in tops_hm["active"]["rows"] if r["price_label"] == "~3만"
    )
    outer_high = next(
        r for r in outer_hm["active"]["rows"] if r["price_label"] == "10~20만"
    )
    age_col = next(i for i, a in enumerate(DEFAULT_AGE_BANDS) if a.id == "20_24")
    assert tops_low["cells"][age_col]["count"] == 1
    assert outer_high["cells"][age_col]["count"] == 1


def test_price_dot_strip_payload_fields():
    items = [
        _make_item(rank=1, price=25000, product_id="a"),
        _make_item(rank=2, price=80000, product_id="b"),
    ]
    result = _price_dot_strip(items)
    assert result["has_data"] is True
    assert result["axis"]["lo_log"] < result["axis"]["hi_log"]
    assert result["presets"][0]["label"] == "~3만"
    assert result["dots"][0]["key"] == "id:a"
    assert result["dots"][0]["url"] == "https://x/1"


def test_realtime_multag_config_targets():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "realtime-multag.json"
    config = load_config(config_path)
    assert config.ranking_windows[0].query_params["period"] == "REALTIME"
    assert config.demographics_window.id == "rt"
    targets = [t for t in config.targets() if t.sub_pan == "product"]
    # Period 4 categories + demographics 4 × (3×7 - 1 duplicate all×all)
    assert len(targets) == 4 + 4 * (3 * 7 - 1)


def test_realtime_age_rankings_by_age_band():
    rt = RankingWindowSpec("rt", "실시간", {"period": "REALTIME"}, days=1)
    section = DEFAULT_SECTIONS[0]
    category = DEFAULT_CATEGORIES[0]
    collections = []
    for age in (DEFAULT_AGE_BANDS[0], DEFAULT_AGE_BANDS[2]):
        collections.append(
            RawCollection(
                target=CollectionTarget(
                    section=section,
                    category=category,
                    gender_filter="A",
                    gender_label="전체",
                    age_band=age.code,
                    age_label=age.label,
                    ranking_window=rt,
                ),
                url=f"https://example.com/{age.id}",
                collected_at="2026-05-20T00:00:00Z",
                source="client-api",
                ok=True,
                payload={
                    "items": [
                        {
                            "rank": 1,
                            "brandName": "브랜드",
                            "productName": f"상품-{age.id}",
                            "price": 39000,
                            "productUrl": f"/products/{age.id}",
                        }
                    ]
                },
            )
        )
    analysis = analyze_dataset(normalize_collections(collections))
    ar = analysis["age_rankings"]
    assert ar["has_data"] is True
    assert ar["window_id"] == "rt"
    assert ar["window_label"] == "실시간"
    assert ar["top_n"] == 30
    all_age = next(a for a in ar["ages"] if a["id"] == "all")
    band_20 = next(a for a in ar["ages"] if a["id"] == "20_24")
    assert all_age["has_data"] is True
    assert band_20["has_data"] is True
    assert all_age["rows"][0]["product"] == "상품-all"
    assert band_20["rows"][0]["product"] == "상품-20_24"


def test_age_rankings_empty_without_realtime_window():
    weekly = RankingWindowSpec("1w", "주간", {"period": "WEEKLY"}, days=7)
    section = DEFAULT_SECTIONS[0]
    category = DEFAULT_CATEGORIES[0]
    collection = RawCollection(
        target=CollectionTarget(
            section=section,
            category=category,
            gender_filter="A",
            gender_label="전체",
            age_band=DEFAULT_AGE_BANDS[1].code,
            age_label=DEFAULT_AGE_BANDS[1].label,
            ranking_window=weekly,
        ),
        url="https://example.com",
        collected_at="2026-05-20T00:00:00Z",
        source="client-api",
        ok=True,
        payload={
            "items": [
                {
                    "rank": 1,
                    "brandName": "B",
                    "productName": "P",
                    "price": 10000,
                    "productUrl": "/products/1",
                }
            ]
        },
    )
    dataset = normalize_collections([collection])
    wid, label = _resolve_realtime_window_id(dataset.items, dataset.collections)
    assert wid is None
    tables = _age_ranking_tables(dataset.items, window_id=wid, window_label=label)
    assert tables["has_data"] is False
    analysis = analyze_dataset(dataset)
    assert analysis["age_rankings"]["has_data"] is False
