from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import build_ranking_url
from .models import CollectionTarget, utc_timestamp


@dataclass
class DiscoveredResponse:
    url: str
    status: int | None
    content_type: str | None
    item_count: int
    keys: list[str]
    sample_path: str | None = None


@dataclass
class DiscoveryResult:
    target_url: str
    discovered_at: str
    responses: list[DiscoveredResponse]
    dom_product_count: int
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_url": self.target_url,
            "discovered_at": self.discovered_at,
            "responses": [asdict(response) for response in self.responses],
            "dom_product_count": self.dom_product_count,
            "errors": self.errors,
        }


def discover_target(target: CollectionTarget, output_dir: Path, timeout_ms: int = 20000) -> DiscoveryResult:
    target_url = build_ranking_url(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses: list[DiscoveredResponse] = []
    errors: list[str] = []
    dom_product_count = 0

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on optional runtime setup
        return DiscoveryResult(
            target_url=target_url,
            discovered_at=utc_timestamp(),
            responses=[],
            dom_product_count=0,
            errors=[f"playwright import failed: {exc}"],
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 1200})

        def handle_response(response: Any) -> None:
            url = response.url
            if not _looks_relevant_url(url):
                return
            content_type = response.headers.get("content-type")
            if content_type and "json" not in content_type.lower():
                return
            try:
                payload = response.json()
            except Exception:
                return
            items = find_product_like_records(payload)
            if not items:
                return
            sample_path = output_dir / f"discovered_{len(responses) + 1}.json"
            sample_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            responses.append(
                DiscoveredResponse(
                    url=url,
                    status=response.status,
                    content_type=content_type,
                    item_count=len(items),
                    keys=sorted({key for item in items[:10] for key in item.keys()}),
                    sample_path=str(sample_path),
                )
            )

        page.on("response", handle_response)
        try:
            page.goto(target_url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(1200)
            dom_product_count = page.locator('a[href*="/products/"]').count()
        except Exception as exc:
            errors.append(str(exc))
        finally:
            browser.close()

    return DiscoveryResult(
        target_url=target_url,
        discovered_at=utc_timestamp(),
        responses=responses,
        dom_product_count=dom_product_count,
        errors=errors,
    )


def find_product_like_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            merged = _merged_product_card(value)
            if merged is not None:
                records.append(merged)
                return
            if _looks_like_product(value):
                records.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return records


def _looks_relevant_url(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in ("ranking", "goods", "product", "display", "api"))


def _looks_like_product(value: dict[str, Any]) -> bool:
    keys = {key.lower() for key in value.keys()}
    has_name = bool(keys & {"goodsnm", "goodsname", "productname", "productnm", "name", "title"})
    has_brand = bool(keys & {"brand", "brandnm", "brandname", "brandnamekor", "brandnameeng"})
    has_price = bool(keys & {"price", "saleprice", "normalprice", "goodsprice", "finalprice"})
    has_rank = bool(keys & {"rank", "ranking", "rankno", "displayrank", "sortno"})
    return has_name and (has_brand or has_price or has_rank)


def _merged_product_card(value: dict[str, Any]) -> dict[str, Any] | None:
    info = value.get("info")
    if not isinstance(info, dict) or not _looks_like_product(info):
        return None
    merged = dict(info)
    if value.get("id") is not None:
        merged.setdefault("id", value.get("id"))
    image = value.get("image")
    if isinstance(image, dict):
        merged.setdefault("rank", image.get("rank"))
        merged.setdefault("imageUrl", image.get("url"))
        like = image.get("onClickLike")
        if isinstance(like, dict):
            merged.setdefault("productId", like.get("productId"))
    return merged
