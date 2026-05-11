from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters, parse_dashboard_filters
from analytics.api.service import DashboardService, apply_core_aliases

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/meta/datasets")
def list_datasets(service: DashboardService = Depends(get_dashboard_service)) -> dict:
    return service.get_datasets()


@router.get("/filters")
def get_filters(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_filters(filters)


@router.get("/meta/schema/inventory")
def get_schema_inventory(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_schema_inventory(filters))


@router.get("/meta/schema/explorer-rules")
def get_schema_explorer_rules(service: DashboardService = Depends(get_dashboard_service)) -> dict:
    return service.get_schema_explorer_rules()


@router.get("/meta/schema/diff")
def get_schema_diff(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    scope: str = Query(default="raw", pattern="^(raw|normalized)$"),
    compare_source_dataset: str | None = Query(default=None),
    current_source_dataset: str | None = Query(default=None),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_schema_diff(
        filters,
        scope=scope,
        compare_source_dataset=compare_source_dataset,
        current_source_dataset=current_source_dataset,
    )


@router.get("/meta/schema/pair-samples")
def get_schema_pair_samples(
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    sample_size: int = Query(default=5, ge=1, le=20),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_schema_pair_samples(filters, sample_size=sample_size))


@router.get("/meta/schema/field-profile")
def get_field_profile(
    field_name: str = Query(..., min_length=1),
    scope: str = Query(default="raw", pattern="^(raw|normalized)$"),
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return apply_core_aliases(service.get_field_profile(filters, field_name=field_name, scope=scope))


@router.get("/meta/schema/candidate-insights")
def get_candidate_insights(
    field_name: str = Query(..., min_length=1),
    scope: str = Query(default="raw", pattern="^(raw|normalized)$"),
    filters: DashboardFilters = Depends(parse_dashboard_filters),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return service.get_candidate_insights(filters, field_name=field_name, scope=scope)
