from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService, apply_core_aliases

router = APIRouter(prefix="/api/animation", tags=["animation"])


@router.get("/embedding-projection")
def embedding_projection(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_embedding_projection(filters))


@router.get("/embedding-overview")
def embedding_overview(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_embedding_overview(filters))


@router.get("/rank-race")
def rank_race(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    entity_type: str = Query(default="brand", pattern="^(brand|product)$"),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_rank_race(filters, entity_type)


@router.get("/rank-trajectories")
def rank_trajectories(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    entity_type: str = Query(default="product", pattern="^(brand|product)$"),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_rank_trajectories(filters, entity_type))
