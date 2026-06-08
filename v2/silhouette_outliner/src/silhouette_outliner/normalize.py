from __future__ import annotations

import re
from typing import Any

from .discover import find_product_like_records
from .endpoints import RANKING_WEB_ORIGIN
from .models import NormalizedDataset, RankingItem, RawCollection, normalize_category_code, utc_timestamp

# Tabs that return brand-only ranking cards (RANKING_BRAND) instead of product cards.
# Both singular ("brand") and plural ("brands") have been observed on the upstream API.
_SUB_PAN_BRANDS = frozenset({"brand", "brands"})

NAME_KEYS = ("productName", "goodsName", "goodsNm", "productNm", "name", "title", "goods_name")
BRAND_KEYS = ("brandName", "brandNm", "brand", "brandNameKor", "brand_name", "brandLabel")
RANK_KEYS = ("rank", "ranking", "rankNo", "displayRank", "sortNo")
PRICE_KEYS = ("price", "salePrice", "finalPrice", "goodsPrice", "lowestPrice")
ORIGINAL_PRICE_KEYS = ("originalPrice", "normalPrice", "consumerPrice", "listPrice")
DISCOUNT_KEYS = ("discountRate", "discountRatio", "discount", "saleRate", "discountRateText")
PRODUCT_ID_KEYS = ("productId", "goodsNo", "goodsId", "id")
URL_KEYS = ("productUrl", "goodsUrl", "url", "linkUrl")
IMAGE_KEYS = ("imageUrl", "image", "thumbnail", "thumbnailUrl", "imgUrl")
REVIEW_COUNT_KEYS = ("reviewCount", "review_count")
REVIEW_SCORE_KEYS = ("reviewScore", "review_score")
BRAND_ID_KEYS = ("brand_id", "brandId")

# Strip trailing SKU like " / S58WZ0127P3753T8013" or " / JQ6939" from product names.
# Conservative: only trims tokens that are at least 6 chars and consist of uppercase ASCII,
# digits, colon, or hyphen. Color names like "BLACK" are kept since they don't reach 6 chars
# typical SKU patterns do (e.g. "B75806", "212683-90H").
_SKU_TAIL = re.compile(r"\s*/\s*[A-Z0-9][A-Z0-9:\-]{5,}\s*$")
_VIEWERS_RE = re.compile(r"(\d[\d,]*)\s*명이\s*보는\s*중")
_BUYERS_RE = re.compile(r"(\d[\d,]*)\s*명이\s*(구매|판매)\s*중")


def normalize_collections(collections: list[RawCollection]) -> NormalizedDataset:
    items: list[RankingItem] = []
    seen: set[tuple[str, ...]] = set()

    for collection in collections:
        records = _records_from_collection(collection)
        for index, record in enumerate(records, start=1):
            item = _record_to_item(collection, record, fallback_rank=index)
            if not _within_target_limit(item, collection.target.limit):
                continue
            segment = (item.gender_filter, item.age_band)
            if collection.target.sub_pan in _SUB_PAN_BRANDS:
                dedupe_key = (
                    item.section_id,
                    item.category_code,
                    item.ranking_window_id,
                    item.sub_pan,
                    *segment,
                    item.brand or "",
                )
            else:
                dedupe_key = (
                    item.section_id,
                    item.category_code,
                    item.ranking_window_id,
                    item.sub_pan,
                    *segment,
                    item.product_id or "",
                    item.product_url or item.product,
                )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            items.append(item)

    items = _collapse_brand_sub_pan_rows(items)
    return NormalizedDataset(generated_at=utc_timestamp(), items=items, collections=collections)


def _within_target_limit(item: RankingItem, limit: int) -> bool:
    if limit <= 0:
        return True
    if item.rank is None:
        return False
    return item.rank <= limit


def _collapse_brand_sub_pan_rows(items: list[RankingItem]) -> list[RankingItem]:
    """Brand ranking pan lists multiple SKUs per brand; keep the best (lowest) rank only."""

    brand_rows: dict[tuple[str, ...], RankingItem] = {}
    passthrough: list[RankingItem] = []

    for item in items:
        if item.sub_pan not in _SUB_PAN_BRANDS or not item.brand:
            passthrough.append(item)
            continue
        key = (
            item.section_id,
            item.category_code,
            item.ranking_window_id,
            item.sub_pan,
            item.brand,
        )
        existing = brand_rows.get(key)
        if existing is None:
            brand_rows[key] = item
            continue
        if item.rank is not None and (existing.rank is None or item.rank < existing.rank):
            brand_rows[key] = item

    merged = passthrough + list(brand_rows.values())
    merged.sort(
        key=lambda row: (
            row.ranking_window_id,
            row.section_id,
            row.rank is None,
            row.rank or 9999,
        )
    )
    return merged


def _records_from_collection(collection: RawCollection) -> list[dict[str, Any]]:
    payload = collection.payload

    if collection.target.sub_pan in _SUB_PAN_BRANDS:
        brand_cards = _extract_ranking_brand_cards(payload)
        if brand_cards:
            return [_flatten_ranking_brand(card) for card in brand_cards]
        # Fall through; some legacy responses may still ship product cards.

    # Preferred: walk modules[*].items[*] looking for PRODUCT_COLUMN-like cards.
    product_columns = _extract_product_columns(payload)
    if product_columns:
        return [_flatten_product_column(card) for card in product_columns]

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    records = find_product_like_records(payload)
    if records:
        return records
    return []


def _extract_ranking_brand_cards(payload: Any) -> list[dict[str, Any]]:
    """Collect every dict whose type == 'RANKING_BRAND'."""
    found: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "RANKING_BRAND":
                found.append(value)
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found


def _flatten_ranking_brand(card: dict[str, Any]) -> dict[str, Any]:
    """Convert a RANKING_BRAND card into the flat record shape the picker expects."""
    flat: dict[str, Any] = {}
    title = card.get("title") if isinstance(card.get("title"), dict) else {}

    rank_value = title.get("rank")
    rank_int = _to_int(rank_value)
    if rank_int is not None:
        flat["rank"] = rank_int

    name_block = title.get("title") if isinstance(title.get("title"), dict) else {}
    brand_name = name_block.get("text") if isinstance(name_block, dict) else None
    if brand_name:
        flat["brandName"] = str(brand_name).strip()

    image_url = title.get("imageUrl")
    if image_url:
        flat["imageUrl"] = image_url

    on_click = title.get("onClick") if isinstance(title.get("onClick"), dict) else {}
    if on_click.get("url"):
        flat["url"] = on_click.get("url")

    amp_payload = _safe_get(on_click, "eventLog", "amplitude", "payload")
    if isinstance(amp_payload, dict):
        brand_id = amp_payload.get("brand_id")
        if brand_id:
            flat["brand_id"] = brand_id

    fluctuation = title.get("fluctuation") if isinstance(title.get("fluctuation"), dict) else {}
    fl_type = fluctuation.get("type")
    if fl_type:
        flat["__labels"] = [str(fl_type)]

    return flat


def _extract_product_columns(payload: Any) -> list[dict[str, Any]]:
    """Walk the payload and collect every dict whose type starts with PRODUCT_ and has an `info` block."""
    found: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            type_value = value.get("type")
            info = value.get("info")
            if (
                isinstance(type_value, str)
                and type_value.startswith("PRODUCT")
                and isinstance(info, dict)
            ):
                found.append(value)
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found


def _flatten_product_column(card: dict[str, Any]) -> dict[str, Any]:
    """Convert a PRODUCT_COLUMN card into a flat dict that the existing pickers understand."""
    flat: dict[str, Any] = {}
    info = card.get("info") if isinstance(card.get("info"), dict) else {}
    image = card.get("image") if isinstance(card.get("image"), dict) else {}
    on_click = card.get("onClick") if isinstance(card.get("onClick"), dict) else {}

    if card.get("id") is not None:
        flat["id"] = card.get("id")

    if image:
        if image.get("rank") is not None:
            flat["rank"] = image.get("rank")
        if image.get("url"):
            flat["imageUrl"] = image.get("url")
        labels = image.get("labels")
        if isinstance(labels, list):
            flat["__labels"] = [
                str(label.get("text")).strip()
                for label in labels
                if isinstance(label, dict) and label.get("text")
            ]
        like = image.get("onClickLike")
        if isinstance(like, dict) and like.get("productId") is not None:
            flat.setdefault("productId", like.get("productId"))

    for key in ("brandName", "productName", "finalPrice", "discountRatio", "isSoldOut"):
        if info.get(key) is not None:
            flat[key] = info.get(key)

    additional = info.get("additionalInformation")
    if isinstance(additional, list):
        flat["__additional"] = [
            str(entry.get("text"))
            for entry in additional
            if isinstance(entry, dict) and entry.get("text")
        ]

    if on_click.get("url"):
        flat["url"] = on_click.get("url")

    amp_payload = _safe_get(on_click, "eventLog", "amplitude", "payload")
    if isinstance(amp_payload, dict):
        for amp_key, dest_key in (
            ("reviewCount", "reviewCount"),
            ("reviewScore", "reviewScore"),
            ("original_price", "originalPrice"),
            ("brand_id", "brand_id"),
            ("product_id", "productId"),
            ("product_name", "productName"),
            ("brand_name", "brandName"),
        ):
            if amp_payload.get(amp_key) not in (None, ""):
                flat.setdefault(dest_key, amp_payload.get(amp_key))

    ga4_payload = _safe_get(on_click, "eventLog", "ga4", "payload")
    if isinstance(ga4_payload, dict) and ga4_payload.get("discount") is not None:
        flat["__discount_amount"] = ga4_payload.get("discount")

    return flat


def _safe_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _record_to_item(collection: RawCollection, record: dict[str, Any], fallback_rank: int) -> RankingItem:
    target = collection.target
    category_code = normalize_category_code(target.category.code)
    rank = _to_int(_pick(record, RANK_KEYS)) or fallback_rank
    price = _to_int(_pick(record, PRICE_KEYS))
    original_price = _to_int(_pick(record, ORIGINAL_PRICE_KEYS))
    discount_rate = _to_float_percent(_pick(record, DISCOUNT_KEYS))
    if discount_rate is None and price and original_price and original_price > price:
        discount_rate = round((original_price - price) / original_price * 100, 1)

    discount_amount = _to_int(record.get("__discount_amount"))
    if discount_amount is None and price is not None and original_price is not None and original_price > price:
        discount_amount = original_price - price

    product_url = _to_str(_pick(record, URL_KEYS))
    product_id = _to_str(_pick(record, PRODUCT_ID_KEYS)) or _extract_product_id(product_url)
    if not product_url and product_id:
        product_url = f"/products/{product_id}"

    product_name = _to_str(_pick(record, NAME_KEYS))
    product_clean = _clean_product_name(product_name)

    labels_raw = record.get("__labels")
    labels = [str(l) for l in labels_raw] if isinstance(labels_raw, list) else []

    additional = record.get("__additional")
    viewers_now = None
    buyers_now = None
    if isinstance(additional, list):
        viewers_now, buyers_now = _parse_additional_info(additional)

    is_sold_out_raw = record.get("isSoldOut")
    is_sold_out = bool(is_sold_out_raw) if is_sold_out_raw is not None else False

    return RankingItem(
        rank=rank,
        brand=_to_str(_pick(record, BRAND_KEYS)),
        product=product_name,
        product_clean=product_clean,
        price=price,
        original_price=original_price,
        discount_rate=discount_rate,
        discount_amount=discount_amount,
        product_id=product_id,
        product_url=_absolute_url(product_url),
        image_url=_to_str(_pick(record, IMAGE_KEYS)),
        brand_id=_to_str(_pick(record, BRAND_ID_KEYS)) or None,
        review_count=_to_int(_pick(record, REVIEW_COUNT_KEYS)),
        review_score=_to_int(_pick(record, REVIEW_SCORE_KEYS)),
        viewers_now=viewers_now,
        buyers_now=buyers_now,
        is_sold_out=is_sold_out,
        labels=labels,
        section_label=target.section.label,
        section_id=target.section.section_id,
        category_label=target.category.label,
        category_code=target.category.code,
        category_major_code=category_code[:3],
        category_minor_code=category_code[3:],
        category_parent_label=target.category.parent_label,
        source=collection.source,
        collected_at=collection.collected_at,
        ranking_window_id=target.ranking_window.id,
        ranking_window_label=target.ranking_window.label,
        ranking_window_days=target.ranking_window.days_effective,
        sub_pan=target.sub_pan,
        gender_filter=target.gender_filter,
        gender_label=target.gender_label,
        age_band=target.age_band,
        age_label=target.age_label,
    )


def _clean_product_name(name: str) -> str:
    if not name:
        return ""
    return _SKU_TAIL.sub("", name).strip()


def _parse_additional_info(entries: list[str]) -> tuple[int | None, int | None]:
    viewers: int | None = None
    buyers: int | None = None
    for entry in entries:
        text = str(entry)
        if viewers is None:
            match = _VIEWERS_RE.search(text)
            if match:
                viewers = _to_int(match.group(1))
                continue
        if buyers is None:
            match = _BUYERS_RE.search(text)
            if match:
                buyers = _to_int(match.group(1))
    return viewers, buyers


def _pick(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {key.lower(): key for key in record.keys()}
    for key in keys:
        original_key = lowered.get(key.lower())
        if original_key is not None:
            value = record.get(original_key)
            if value not in (None, ""):
                return value
    return None


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    return " ".join(str(value).split())


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"-?\d[\d,]*", str(value))
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _to_float_percent(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_product_id(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/products/(\d+)", url)
    return match.group(1) if match else None


def _absolute_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return RANKING_WEB_ORIGIN + url
    return url
