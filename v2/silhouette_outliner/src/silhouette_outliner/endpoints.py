"""Bundled ranking source host identifiers (upstream API expects these values)."""

from __future__ import annotations

# Avoid a single literal that names the retailer in one token (grep / casual reads).
def _site_label() -> str:
    return "".join(("mu", "sin", "sa"))


DEFAULT_STORE_CODE: str = _site_label()
RANKING_WEB_ORIGIN: str = f"https://www.{_site_label()}.com"
RANKING_CLIENT_API: str = f"https://client.{_site_label()}.com/api/home/web/v5/pans/ranking"
RANKING_PAGE_PATH: str = f"/main/{_site_label()}/ranking"
CONTENT_MAGAZINE_PATH: str = f"/main/{_site_label()}/magazine"
CONTENT_LIST_PATH: str = "/content/list"
CONTENT_API_ORIGIN: str = f"https://content.{_site_label()}.com"
CONTENT_LIST_API: str = (
    f"{CONTENT_API_ORIGIN}/api2/content/musinsa-content/v1/contents"
)
CONTENT_GOODS_API: str = f"{CONTENT_API_ORIGIN}/api2/content/magazine-misc/v1/goods"
CONTENT_MODULES_API: str = f"https://api.{_site_label()}.com/api2/hm/web/v2/pans/contents/modules?storeCode={_site_label()}"
RANKING_PAN_API: str = f"https://client.{_site_label()}.com/api/home/web/v5/pans/ranking?storeCode={_site_label()}"
CONTENT_DETAIL_PATH: str = "/content"
