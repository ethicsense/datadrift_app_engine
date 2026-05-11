from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService, apply_core_aliases

router = APIRouter(prefix="/api/price", tags=["price"])


@router.get("/distribution")
def price_distribution(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    category: str | None = Query(default=None),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_price_distribution(filters, category=category))


@router.get("/timeseries")
def price_timeseries(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    product_ids: list[str] = Query(default_factory=list, alias="product_ids"),
    category: str | None = Query(default=None),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_price_timeseries(filters, product_ids, category=category))


@router.get("/reaction")
def price_reaction(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_price_reaction(filters)


@router.get("/discount-effects")
def price_discount_effects(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_discount_effects(filters))


@router.get("/discount-drilldown")
def price_discount_drilldown(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    product_id: str = Query(..., min_length=1),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_product_discount_drilldown(filters, product_id))
