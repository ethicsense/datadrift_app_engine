from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService

router = APIRouter(prefix="/api/overview", tags=["overview"])


@router.get("/kpis")
def overview_kpis(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_overview_kpis(filters)


@router.get("/momentum")
def overview_momentum(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_overview_momentum(filters)


@router.get("/dataset-profile")
def overview_dataset_profile(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_overview_dataset_profile(filters)


@router.get("/schema")
def overview_schema(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_overview_schema(filters)


@router.get("/caveats")
def overview_caveats(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_overview_caveats(filters)


@router.get("/discount-reaction")
def overview_discount_reaction(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_discount_reaction(filters)


@router.get("/rank-trends")
def overview_rank_trends(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    product_ids: list[str] = Query(default_factory=list, alias="product_ids"),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_rank_trends(filters, product_ids)
