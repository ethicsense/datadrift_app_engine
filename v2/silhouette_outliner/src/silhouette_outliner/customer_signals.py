from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AppConfig
from .demographics import gender_label_for_code
from .endpoints import (
    CONTENT_DETAIL_PATH,
    CONTENT_GOODS_API,
    CONTENT_LIST_API,
    CONTENT_LIST_PATH,
    RANKING_PAN_API,
    RANKING_WEB_ORIGIN,
)
from .models import (
    ContentSignal,
    ContentSignalProduct,
    CustomerSignalDataset,
    KeywordGenderBoard,
    KeywordSignal,
    utc_timestamp,
)

_LOG = logging.getLogger("silhouette_outliner")

_CONTENT_CATEGORY_ALL = "001"
_CONTENT_SORT_POPULAR_WEEK = "CONTENT_POPULARITY_ONE_WEEK_SCORE"
_CONTENT_FETCH_POOL_MIN = 50
_GOODS_LOOKUP_CHUNK = 50
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">([\s\S]*?)</script>'
)
_VIEW_COUNT_RE = re.compile(r"조회\s*(?:<!--\s*-->)?\s*([0-9.,만천억]+)")
_COMMENT_COUNT_RE = re.compile(r"댓글\s*(?:<!--\s*-->)?\s*([0-9.,만천억]+)")


def collect_customer_signals(config: AppConfig) -> CustomerSignalDataset:
    if not config.collect_customer_signals:
        return CustomerSignalDataset(
            generated_at=utc_timestamp(),
            contents=[],
            keywords_by_gender=[],
        )

    errors: list[str] = []
    contents: list[ContentSignal] = []
    keywords_by_gender: list[KeywordGenderBoard] = []

    try:
        popular = _fetch_top_viewed_contents(config.content_top_n)
        for entry in popular:
            product_ids = list(entry.get("product_ids", []))
            title = str(entry.get("title") or "")
            brand = str(entry.get("brand") or "")
            content_type = str(entry.get("content_type") or "")
            view_count_text = entry.get("view_count_text")
            comment_count_text = entry.get("comment_count_text")
            published_at = entry.get("published_at")

            if not product_ids:
                try:
                    detail = _fetch_content_detail(entry["content_id"])
                    product_ids = _merge_product_ids(product_ids, detail.get("product_ids", []))
                    title = str(detail.get("title") or title)
                    brand = str(detail.get("brand") or brand)
                    content_type = str(detail.get("content_type") or content_type)
                    view_count_text = detail.get("view_count_text") or view_count_text
                    comment_count_text = detail.get("comment_count_text") or comment_count_text
                    published_at = detail.get("published_at") or published_at
                except Exception as exc:
                    errors.append(f"content detail {entry.get('content_id')}: {exc}")
                if config.request_delay_seconds > 0:
                    time.sleep(config.request_delay_seconds)

            contents.append(
                ContentSignal(
                    rank=int(entry["rank"]),
                    content_id=str(entry["content_id"]),
                    title=title,
                    brand=brand,
                    content_type=content_type,
                    view_count_text=view_count_text,
                    comment_count_text=comment_count_text,
                    popularity_score_text=None,
                    url=str(
                        entry.get("url")
                        or f"{RANKING_WEB_ORIGIN}{CONTENT_DETAIL_PATH}/{entry['content_id']}"
                    ),
                    product_ids=product_ids,
                    published_at=published_at,
                )
            )
    except Exception as exc:
        errors.append(f"content list: {exc}")

    for gender_code in config.keyword_gender_filters:
        try:
            rows = _fetch_keyword_rankings(gender_code, config.keyword_top_n)
            keywords_by_gender.append(
                KeywordGenderBoard(
                    gender_code=gender_code,
                    gender_label=gender_label_for_code(gender_code),
                    rows=rows,
                )
            )
        except Exception as exc:
            errors.append(f"keyword ranking gf={gender_code}: {exc}")
            keywords_by_gender.append(
                KeywordGenderBoard(
                    gender_code=gender_code,
                    gender_label=gender_label_for_code(gender_code),
                    rows=[],
                )
            )
        if config.request_delay_seconds > 0:
            time.sleep(config.request_delay_seconds)

    _LOG.info(
        "고객 신호 수집 완료: 콘텐츠 %d건, 검색어 성별 %d종, 오류 %d건",
        len(contents),
        len(keywords_by_gender),
        len(errors),
    )
    return CustomerSignalDataset(
        generated_at=utc_timestamp(),
        contents=contents,
        keywords_by_gender=keywords_by_gender,
        errors=errors,
    )


def _http_json(url: str, referer: str) -> Any:
    request = Request(
        url,
        headers={
            "accept": "application/json",
            "user-agent": _USER_AGENT,
            "referer": referer,
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_text(url: str, referer: str) -> str:
    request = Request(
        url,
        headers={
            "user-agent": _USER_AGENT,
            "referer": referer,
        },
    )
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_top_viewed_contents(limit: int) -> list[dict[str, Any]]:
    """콘텐츠 목록(전체 · 인기순 1주일) 후보를 가져와 조회수 기준 상위 N건을 반환."""
    pool_size = max(_CONTENT_FETCH_POOL_MIN, limit * 10)
    params = urlencode(
        {
            "contentCategoryCode": _CONTENT_CATEGORY_ALL,
            "sort": _CONTENT_SORT_POPULAR_WEEK,
            "page": 1,
            "size": pool_size,
        }
    )
    payload = _http_json(
        f"{CONTENT_LIST_API}?{params}",
        referer=f"{RANKING_WEB_ORIGIN}{CONTENT_LIST_PATH}",
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("content list API returned no data object")
    raw_list = data.get("list")
    if not isinstance(raw_list, list) or not raw_list:
        raise RuntimeError("content list API returned an empty list")

    ranked: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        view_count = _to_int(item.get("viewCount"))
        if view_count is None:
            continue
        content_id = _content_id_from_item(item)
        if not content_id:
            continue
        brands = item.get("brandNameList")
        brand = ""
        if isinstance(brands, list) and brands:
            brand = str(brands[0]).strip()
        content_type = str(item.get("attributeDictionaryName") or "").strip()
        if not content_type:
            content_type = str(item.get("contentsType2DepthLabel") or item.get("contentsType1DepthLabel") or "").strip()
        ranked.append(
            {
                "content_id": content_id,
                "title": str(item.get("title", "")).strip(),
                "brand": brand,
                "content_type": content_type,
                "view_count": view_count,
                "view_count_text": _format_count_text(view_count),
                "comment_count_text": _format_count_text(_to_int(item.get("commentCount"))),
                "url": str(item.get("landingUrl") or "").strip()
                or f"{RANKING_WEB_ORIGIN}{CONTENT_DETAIL_PATH}/{content_id}",
                "product_ids": _product_ids_from_list_item(item),
                "published_at": item.get("displayStartDate"),
            }
        )

    ranked.sort(key=lambda row: row["view_count"], reverse=True)
    entries: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:limit], start=1):
        entry = dict(row)
        entry["rank"] = rank
        entry.pop("view_count", None)
        entries.append(entry)
    return entries


def _content_id_from_item(item: dict[str, Any]) -> str:
    landing = str(item.get("landingUrl") or "").strip()
    content_id = _content_id_from_url(landing)
    if content_id:
        return content_id
    cms_index = item.get("cmsIndex")
    if cms_index is not None and str(cms_index).strip():
        return str(cms_index).strip()
    raw_id = item.get("id")
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id).strip()
    return ""


def _product_ids_from_list_item(item: dict[str, Any]) -> list[str]:
    goods = item.get("relatedGoodsList")
    if not isinstance(goods, list):
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_id in goods:
        product_id = str(raw_id).strip()
        if product_id and product_id not in seen:
            seen.add(product_id)
            ordered.append(product_id)
    return ordered


def _format_count_text(value: int | None) -> str | None:
    if value is None:
        return None
    if value >= 100_000_000:
        scaled = value / 100_000_000
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}억"
    if value >= 10_000:
        scaled = value / 10_000
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}만"
    return f"{value:,}"


def _fetch_content_detail(content_id: str) -> dict[str, Any]:
    url = f"{RANKING_WEB_ORIGIN}{CONTENT_DETAIL_PATH}/{content_id}"
    html = _http_text(url, referer=f"{RANKING_WEB_ORIGIN}{CONTENT_LIST_PATH}")
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise RuntimeError(f"__NEXT_DATA__ not found for content {content_id}")

    page_data = json.loads(match.group(1))
    initial = page_data.get("props", {}).get("pageProps", {}).get("initialData", {})
    meta = initial.get("meta") if isinstance(initial.get("meta"), dict) else {}
    content_types = initial.get("contentTypes") if isinstance(initial.get("contentTypes"), list) else []

    brands = meta.get("brands") if isinstance(meta.get("brands"), list) else []
    brand = ""
    if brands:
        brand = str(brands[0])
    modules = initial.get("modules") if isinstance(initial.get("modules"), list) else []

    content_type = ""
    for entry in content_types:
        if isinstance(entry, dict) and entry.get("isRepresentative"):
            content_type = str(entry.get("name", "")).strip()
            break
    if not content_type and content_types and isinstance(content_types[0], dict):
        content_type = str(content_types[0].get("name", "")).strip()

    product_ids = _product_ids_from_initial(initial)
    view_match = _VIEW_COUNT_RE.search(html)
    comment_match = _COMMENT_COUNT_RE.search(html)

    return {
        "title": str(meta.get("title", "")).strip(),
        "brand": brand,
        "content_type": content_type,
        "view_count_text": view_match.group(1).strip() if view_match else None,
        "comment_count_text": comment_match.group(1).strip() if comment_match else None,
        "published_at": meta.get("displayedFrom"),
        "product_ids": product_ids,
    }


def _product_ids_from_initial(initial: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    meta = initial.get("meta") if isinstance(initial.get("meta"), dict) else {}
    goods_list = meta.get("goodsList") if isinstance(meta.get("goodsList"), list) else []
    for raw_id in goods_list:
        product_id = str(raw_id).strip()
        if product_id and product_id not in seen:
            seen.add(product_id)
            ordered.append(product_id)

    modules = initial.get("modules") if isinstance(initial.get("modules"), list) else []
    for module in modules:
        contents = module.get("contents")
        if not isinstance(contents, dict):
            continue
        resources = contents.get("resources")
        if not isinstance(resources, list):
            continue
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            markers = resource.get("markers")
            if not isinstance(markers, list):
                continue
            for marker in markers:
                if not isinstance(marker, dict):
                    continue
                goods_id = marker.get("goodsId")
                if goods_id is None:
                    continue
                product_id = str(goods_id).strip()
                if product_id and product_id not in seen:
                    seen.add(product_id)
                    ordered.append(product_id)
    return ordered


def _fetch_keyword_rankings(gender_code: str, limit: int) -> list[KeywordSignal]:
    params = urlencode({"subPan": "keyword", "gf": gender_code})
    payload = _http_json(
        f"{RANKING_PAN_API}&{params}",
        referer=f"{RANKING_WEB_ORIGIN}/main/musinsa/ranking?subPan=keyword",
    )
    modules = payload.get("data", {}).get("modules", [])
    rows: list[KeywordSignal] = []
    for module in modules:
        if module.get("type") != "RANKING_SEARCH":
            continue
        rank = _to_int(module.get("rank"))
        if rank is None:
            continue
        title = module.get("title") if isinstance(module.get("title"), dict) else {}
        keyword = str(title.get("text", "")).strip()
        if not keyword:
            continue
        fluctuation = module.get("fluctuation") if isinstance(module.get("fluctuation"), dict) else {}
        fl_type = str(fluctuation.get("type", "NONE")).upper()
        fl_amount = _to_int(fluctuation.get("amount"))
        on_click = module.get("onClick") if isinstance(module.get("onClick"), dict) else {}
        rows.append(
            KeywordSignal(
                rank=rank,
                keyword=keyword,
                fluctuation_type=fl_type,
                fluctuation_amount=fl_amount,
                fluctuation_label=_fluctuation_label(fl_type, fl_amount),
                search_url=str(on_click.get("url", "")).strip(),
            )
        )
        if len(rows) >= limit:
            break
    rows.sort(key=lambda row: row.rank)
    return rows


def _fluctuation_label(fl_type: str, amount: int | None) -> str:
    if fl_type == "UP" and amount is not None:
        return f"상승 +{amount}"
    if fl_type == "DOWN" and amount is not None:
        return f"하락 -{amount}"
    if fl_type == "NEW":
        return "신규"
    return "유지"


def _content_id_from_url(url: Any) -> str | None:
    if not url:
        return None
    text = str(url).strip()
    if "/content/" not in text:
        return None
    tail = text.split("/content/", 1)[1]
    return tail.split("?", 1)[0].split("/", 1)[0] or None


def _merge_product_ids(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for raw_id in group:
            product_id = str(raw_id).strip()
            if not product_id or product_id in seen:
                continue
            seen.add(product_id)
            merged.append(product_id)
    return merged


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def fetch_goods_catalog(product_ids: list[str]) -> dict[str, dict[str, Any]]:
    """랭킹에 없는 소개 상품의 이름·브랜드·URL을 콘텐츠 상품 API로 조회."""
    unique = [product_id for product_id in dict.fromkeys(str(pid).strip() for pid in product_ids) if product_id]
    if not unique:
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    referer = f"{RANKING_WEB_ORIGIN}{CONTENT_LIST_PATH}"
    for offset in range(0, len(unique), _GOODS_LOOKUP_CHUNK):
        chunk = unique[offset : offset + _GOODS_LOOKUP_CHUNK]
        params = urlencode(
            {
                "goodsIds": ",".join(chunk),
                "saleStates": "SALE,SOLD_OUT",
            }
        )
        payload = _http_json(f"{CONTENT_GOODS_API}?{params}", referer=referer)
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        goods_list = data.get("list")
        if not isinstance(goods_list, list):
            continue
        for goods in goods_list:
            if not isinstance(goods, dict):
                continue
            goods_no = goods.get("goodsNo")
            if goods_no is None:
                continue
            product_id = str(goods_no).strip()
            if not product_id:
                continue
            catalog[product_id] = {
                "product": str(goods.get("goodsName") or "").strip() or None,
                "brand": str(goods.get("brandName") or goods.get("brand") or "").strip() or None,
                "product_url": str(goods.get("goodsLinkUrl") or "").strip() or None,
                "image_url": str(goods.get("thumbnail") or "").strip() or None,
                "price": _to_int(goods.get("price") or goods.get("finalPrice")),
                "review_count": _to_int(goods.get("reviewCount")),
            }
    return catalog


def join_content_products(
    content: ContentSignal,
    *,
    product_lookup: dict[str, Any],
    goods_lookup: dict[str, dict[str, Any]] | None = None,
    period_rank_by_id: dict[str, int],
    realtime_rank_by_id: dict[str, int],
    validation_by_item_id: dict[int, float],
    max_products: int = 8,
) -> list[ContentSignalProduct]:
    rows: list[ContentSignalProduct] = []
    goods_lookup = goods_lookup or {}
    for order, product_id in enumerate(content.product_ids[:max_products], start=1):
        item = product_lookup.get(product_id)
        if item is None:
            goods = goods_lookup.get(product_id)
            if goods:
                rows.append(
                    ContentSignalProduct(
                        product_id=product_id,
                        product_order=order,
                        brand=goods.get("brand"),
                        product=goods.get("product"),
                        price=goods.get("price"),
                        product_url=goods.get("product_url"),
                        image_url=goods.get("image_url"),
                        review_count=goods.get("review_count"),
                        matched=False,
                    )
                )
            else:
                rows.append(
                    ContentSignalProduct(
                        product_id=product_id,
                        product_order=order,
                        matched=False,
                    )
                )
            continue
        rows.append(
            ContentSignalProduct(
                product_id=product_id,
                product_order=order,
                brand=item.brand,
                product=item.product_clean or item.product,
                price=item.price,
                product_url=item.product_url,
                image_url=item.image_url,
                period_rank=period_rank_by_id.get(product_id),
                realtime_rank=realtime_rank_by_id.get(product_id),
                buyers_now=item.buyers_now,
                review_count=item.review_count,
                validation=round(validation_by_item_id.get(id(item), 0.0), 2),
                matched=True,
            )
        )
    return rows
