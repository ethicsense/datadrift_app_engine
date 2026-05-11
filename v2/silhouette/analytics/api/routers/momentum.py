from __future__ import annotations

from fastapi import APIRouter, Depends

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService, apply_core_aliases

router = APIRouter(prefix="/api/momentum", tags=["momentum"])


@router.get("/inputs")
def momentum_inputs(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_momentum_inputs(filters))


@router.get("/formula-samples")
def momentum_formula_samples(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_momentum_formula_samples(filters))


@router.get("/distribution")
def momentum_distribution(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_momentum_distribution(filters))
