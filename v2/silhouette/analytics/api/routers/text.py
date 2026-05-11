from __future__ import annotations

from fastapi import APIRouter, Depends

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService

router = APIRouter(prefix="/api/text", tags=["text"])


@router.get("/overview")
def text_overview(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_overview(filters)


@router.get("/aspects")
def text_aspects(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_aspects(filters)


@router.get("/fusion")
def text_fusion(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_fusion(filters)


@router.get("/evidence")
def text_evidence(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_evidence(filters)


@router.get("/gaps")
def text_gaps(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    # 이전 경로 호환용
    return service.get_text_gaps(filters)


@router.get("/unmet-needs")
def text_unmet_needs(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_unmet_needs(filters)


@router.get("/size-guide")
def text_size_guide(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_size_guide(filters)


@router.get("/trends")
def text_trends(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_trends(filters)


@router.get("/brand-image")
def text_brand_image(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_brand_image(filters)


@router.get("/word-frequency")
def text_word_frequency(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_text_word_frequency(filters)
