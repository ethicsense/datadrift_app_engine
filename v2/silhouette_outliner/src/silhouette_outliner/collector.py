from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .config import AppConfig, build_ranking_api_url, build_ranking_url
from .exceptions import CollectCancelled
from .discover import find_product_like_records
from .models import CollectionTarget, RawCollection, utc_timestamp

_LOG = logging.getLogger("silhouette_outliner")


def collect_all(
    config: AppConfig,
    run_dir: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> list[RawCollection]:
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    collections: list[RawCollection] = []
    targets = list(config.targets())
    total = len(targets)
    for index, target in enumerate(targets, start=1):
        if should_cancel is not None and should_cancel():
            raise CollectCancelled("사용자가 분석을 중단했습니다.")
        _LOG.info(
            "수집 중 (%d/%d) key=%s …",
            index,
            total,
            target.key,
        )
        collection = collect_target(target, raw_dir)
        status = "성공" if collection.ok else "실패"
        err = f", 오류={collection.error}" if collection.error else ""
        _LOG.info(
            "수집 완료 (%d/%d) key=%s → %s, 소스=%s%s",
            index,
            total,
            target.key,
            status,
            collection.source,
            err,
        )
        collections.append(collection)
        path = raw_dir / f"{index:03d}_{target.key}.json"
        path.write_text(json.dumps(collection.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if config.request_delay_seconds > 0:
            time.sleep(config.request_delay_seconds)
    return collections


def collect_target(target: CollectionTarget, raw_dir: Path | None = None, timeout_ms: int = 25000) -> RawCollection:
    url = build_ranking_url(target)
    collected_at = utc_timestamp()
    api_collection = _collect_target_api(target, collected_at)
    if api_collection.ok:
        return api_collection

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on optional runtime setup
        return RawCollection(
            target=target,
            url=url,
            collected_at=collected_at,
            source="playwright",
            ok=False,
            error=f"api failed: {api_collection.error}; playwright import failed: {exc}",
        )

    captured: list[tuple[str, Any]] = []
    dom_payload: list[dict[str, Any]] = []
    error: str | None = None

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 390, "height": 1400},
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            ),
        )

        def handle_response(response: Any) -> None:
            if not _looks_collectable_response(response.url, response.headers.get("content-type")):
                return
            try:
                payload = response.json()
            except Exception:
                return
            if find_product_like_records(payload):
                captured.append((response.url, payload))

        page.on("response", handle_response)
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            if raw_dir is not None:
                html_path = raw_dir / f"{target.key}.html"
                html_path.write_text(page.content(), encoding="utf-8")
            dom_payload = collect_dom_products(page, limit=target.limit)
        except Exception as exc:
            error = str(exc)
        finally:
            browser.close()

    if captured:
        response_url, payload = max(captured, key=lambda item: len(find_product_like_records(item[1])))
        return RawCollection(
            target=target,
            url=url,
            collected_at=collected_at,
            source="network-json",
            ok=True,
            payload=payload,
            response_url=response_url,
        )

    if dom_payload:
        return RawCollection(
            target=target,
            url=url,
            collected_at=collected_at,
            source="dom",
            ok=True,
            payload={"items": dom_payload},
            error=error,
        )

    return RawCollection(
        target=target,
        url=url,
        collected_at=collected_at,
        source="none",
        ok=False,
        payload={"items": []},
        error=error or "no product-like records found",
    )


def _collect_target_api(target: CollectionTarget, collected_at: str) -> RawCollection:
    api_url = build_ranking_api_url(target)
    request = Request(
        api_url,
        headers={
            "accept": "application/json",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "referer": build_ranking_url(target),
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return RawCollection(
            target=target,
            url=build_ranking_url(target),
            collected_at=collected_at,
            source="client-api",
            ok=False,
            payload={"items": []},
            error=str(exc),
            response_url=api_url,
        )

    records = find_product_like_records(payload)
    return RawCollection(
        target=target,
        url=build_ranking_url(target),
        collected_at=collected_at,
        source="client-api",
        ok=bool(records),
        payload=payload,
        error=None if records else "client api returned no product-like records",
        response_url=api_url,
    )


def collect_dom_products(page: Any, limit: int = 100) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    links = page.locator('a[href*="/products/"]')
    count = min(links.count(), limit * 3)
    for index in range(count):
        link = links.nth(index)
        try:
            href = link.get_attribute("href")
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)
            text = _clean_text(link.inner_text(timeout=1000))
            container_text = _clean_text(
                link.evaluate(
                    """node => {
                      const card = node.closest('li, article, div');
                      return card ? card.innerText : node.innerText;
                    }"""
                )
            )
            image_url = link.evaluate(
                """node => {
                  const card = node.closest('li, article, div') || node;
                  const img = card.querySelector('img');
                  return img ? (img.currentSrc || img.src || img.getAttribute('data-src')) : null;
                }"""
            )
            products.append(
                {
                    "rank": len(products) + 1,
                    "productName": text or _first_line(container_text),
                    "brandName": _guess_brand(container_text, text),
                    "price": _guess_price(container_text),
                    "discountRate": _guess_discount(container_text),
                    "productUrl": href,
                    "imageUrl": image_url,
                    "rawText": container_text,
                }
            )
            if len(products) >= limit:
                break
        except Exception:
            continue
    return products


def _looks_collectable_response(url: str, content_type: str | None) -> bool:
    lowered = url.lower()
    if content_type and "json" not in content_type.lower():
        return False
    return any(token in lowered for token in ("ranking", "goods", "product", "display", "api"))


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _first_line(value: str) -> str:
    return value.splitlines()[0].strip() if value else ""


def _guess_brand(container_text: str, product_text: str) -> str:
    parts = [part.strip() for part in container_text.splitlines() if part.strip()]
    if len(parts) > 1 and product_text:
        for part in parts:
            if part != product_text and not any(ch.isdigit() for ch in part):
                return part[:80]
    return ""


def _guess_price(text: str) -> int | None:
    import re

    matches = re.findall(r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})\s*원?", text)
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _guess_discount(text: str) -> float | None:
    import re

    matches = re.findall(r"(\d{1,2})\s*%", text)
    if not matches:
        return None
    return float(matches[0])
