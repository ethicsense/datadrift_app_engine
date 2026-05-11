from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService

router = APIRouter(prefix="/api/category", tags=["category"])


@router.get("/overview")
def category_overview(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    level: str = Query(default="l3", pattern="^(l1|l2|l3)$"),
    quality_mode: str = Query(default="success_only", pattern="^(success_only|success_partial)$"),
    include_fallback: bool = Query(default=False),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_category_overview(filters, category_level=level, quality_mode=quality_mode, include_fallback=include_fallback)


@router.get("/relationships")
def category_relationships(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    level: str = Query(default="l3", pattern="^(l1|l2|l3)$"),
    quality_mode: str = Query(default="success_only", pattern="^(success_only|success_partial)$"),
    include_fallback: bool = Query(default=False),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_category_relationships(filters, category_level=level, quality_mode=quality_mode, include_fallback=include_fallback)


@router.get("/timeseries")
def category_timeseries(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    level: str = Query(default="l3", pattern="^(l1|l2|l3)$"),
    quality_mode: str = Query(default="success_only", pattern="^(success_only|success_partial)$"),
    include_fallback: bool = Query(default=False),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_category_timeseries(filters, category_level=level, quality_mode=quality_mode, include_fallback=include_fallback)


@router.get("/quality")
def category_quality(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    level: str = Query(default="l3", pattern="^(l1|l2|l3)$"),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_category_quality(filters, category_level=level)


@router.get("/examples")
def category_examples(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    level: str = Query(default="l3", pattern="^(l1|l2|l3)$"),
    quality_mode: str = Query(default="success_only", pattern="^(success_only|success_partial)$"),
    include_fallback: bool = Query(default=False),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_category_examples(filters, category_level=level, quality_mode=quality_mode, include_fallback=include_fallback)
