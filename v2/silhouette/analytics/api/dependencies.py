from __future__ import annotations

from functools import lru_cache

from analytics.api.repository import AnalyticsRepository
from analytics.api.service import DashboardService
from analytics.api.settings import get_settings


@lru_cache(maxsize=1)
def get_dashboard_service() -> DashboardService:
    settings = get_settings()
    repository = AnalyticsRepository(settings.output_dir, settings.datasets_root)
    return DashboardService(repository)
