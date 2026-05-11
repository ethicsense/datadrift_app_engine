from __future__ import annotations

from fastapi import APIRouter, Depends

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/tag-correlation")
def performance_tag_correlation(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_tag_correlation(filters)
