"""Quick calibration helper for discount-effects analysis."""

from __future__ import annotations

import json
import time
from typing import Any

from analytics.api.dependencies import get_dashboard_service
from analytics.api.filters import DashboardFilters


SCENARIOS: list[dict[str, Any]] = [
    {"label": "default", "kwargs": {}},
    {"label": "loose_threshold_3pp", "kwargs": {"velocity_threshold": 3.0}},
    {"label": "tight_threshold_7pp", "kwargs": {"velocity_threshold": 7.0}},
    {
        "label": "wider_control",
        "kwargs": {"control_velocity_threshold": 5.0},
    },
    {
        "label": "small_control_min",
        "kwargs": {"min_control_samples": 2},
    },
    {
        "label": "longer_post_window",
        "kwargs": {"post_window": 21},
    },
]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    events = payload.get("events") or []
    confident_abnormal = [
        event["abnormalRankDelta"]
        for event in events
        if event.get("abnormalRankDelta") is not None and not event.get("lowConfidence")
    ]
    return {
        "summary": summary,
        "scatter_count": len(payload.get("effectScatter") or []),
        "abnormal_min": min(confident_abnormal) if confident_abnormal else None,
        "abnormal_p25": _percentile(confident_abnormal, 25),
        "abnormal_p50": _percentile(confident_abnormal, 50),
        "abnormal_p75": _percentile(confident_abnormal, 75),
        "abnormal_max": max(confident_abnormal) if confident_abnormal else None,
    }


def main() -> None:
    service = get_dashboard_service()
    filters = DashboardFilters()
    results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        kwargs = scenario["kwargs"]
        started = time.perf_counter()
        payload = service.get_discount_effects(filters, **kwargs)
        elapsed = round(time.perf_counter() - started, 3)
        profile = summarize(payload)
        results.append({
            "label": scenario["label"],
            "kwargs": kwargs,
            "elapsed_seconds": elapsed,
            **profile,
        })
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
