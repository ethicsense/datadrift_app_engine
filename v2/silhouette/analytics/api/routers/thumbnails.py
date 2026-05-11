from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService, apply_core_aliases

router = APIRouter(prefix="/api/thumbnails", tags=["thumbnails"])


@router.get("/snapshots")
def thumbnail_snapshots(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_thumbnail_snapshots(filters))


@router.get("/records")
def thumbnail_records(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    snapshot_id: str | None = Query(default=None, alias="snapshot_id"),
    start_snapshot_id: str | None = Query(default=None, alias="start_snapshot_id"),
    end_snapshot_id: str | None = Query(default=None, alias="end_snapshot_id"),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_thumbnail_records(filters, snapshot_id, start_snapshot_id, end_snapshot_id))
