import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Dataset
from app.services.plan_runner import dataset_out_dir


router = APIRouter(prefix="/status", tags=["status"])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/datasets")
def dataset_status_bulk(
    ids: list[str] | None = Query(
        None, description="상태를 조회할 dataset_id 목록 (없으면 전체)"
    ),
    db: Session = Depends(get_db),
):
    """
    프론트(UI)용 통합 상태 엔드포인트.
    - dataset_id 기준으로 현재 실행 중인 작업(EDA/Drift)을 한 번에 반환합니다.
    - Drift가 내부적으로 EDA를 수행하는 경우도 drift task로 노출됩니다.
    """
    q = db.query(Dataset)
    if ids:
        q = q.filter(Dataset.id.in_(ids))
    datasets = q.all()
    dataset_ids = {d.id for d in datasets}

    by_dataset: dict[str, Any] = {}
    for d in datasets:
        out_dir = dataset_out_dir(d.id)
        index_path = Path(out_dir) / "artifact_index.json"
        by_dataset[d.id] = {
            "dataset_id": d.id,
            "has_running_tasks": False,
            "running_tasks": [],
            "cache_status": {"eda": index_path.exists()},
            "refreshed_at": _utc_now_iso(),
        }

    # EDA task/status
    # - EDA 자체 실행( /eda/{id} )은 eda router 내부 registry + status file로 추적됩니다.
    try:
        from app.routers import eda as eda_router

        for dataset_id in list(dataset_ids):
            task = getattr(eda_router, "_EDA_TASKS", {}).get(dataset_id)
            is_running = bool(task and not task.done())
            status_payload = {}
            try:
                status_payload = _safe_read_json(eda_router._status_path(dataset_id))
            except Exception:
                status_payload = {}

            running = is_running or status_payload.get("state") == "running"
            if not running:
                continue

            by_dataset[dataset_id]["has_running_tasks"] = True
            by_dataset[dataset_id]["running_tasks"].append(
                {
                    "task_id": f"eda:{dataset_id}",
                    "task_type": "eda",
                    "progress": None,
                    "state": status_payload.get("state") or ("running" if is_running else "unknown"),
                    "started_at": status_payload.get("started_at"),
                }
            )
    except Exception:
        # status endpoint는 best-effort (UI용)
        pass

    # Drift task/status
    # - Drift는 pair(base,target) 단위로 실행되며, 양쪽 dataset 카드에 동일 task를 표시합니다.
    try:
        from app.routers import drift as drift_router

        drift_tasks: dict[str, Any] = getattr(drift_router, "_DRIFT_TASKS", {})
        for key, task in list(drift_tasks.items()):
            # key format: "{base_id}__{target_id}"
            try:
                base_id, target_id = key.split("__", 1)
            except Exception:
                continue

            # 대상 dataset이 아니면 skip
            if ids and (base_id not in dataset_ids) and (target_id not in dataset_ids):
                continue

            is_running = bool(task and not task.done())
            out_dir = dataset_out_dir(f"drift_{base_id}_{target_id}")
            status_payload = {}
            try:
                status_payload = _safe_read_json(drift_router._status_path(out_dir))
            except Exception:
                status_payload = {}

            running = is_running or status_payload.get("state") == "running"
            if not running:
                continue

            drift_task_obj = {
                "task_id": f"drift:{base_id}:{target_id}",
                "task_type": "drift",
                "progress": None,
                "state": status_payload.get("state") or ("running" if is_running else "unknown"),
                "started_at": status_payload.get("started_at"),
                "base_id": base_id,
                "target_id": target_id,
                "used_cached_eda": status_payload.get("used_cached_eda"),
                # used_cached_eda == False 이면 drift plan에 EDA step이 포함됩니다.
                "eda_included": (status_payload.get("used_cached_eda") is False),
            }

            for dataset_id in [base_id, target_id]:
                if dataset_id not in by_dataset:
                    continue
                by_dataset[dataset_id]["has_running_tasks"] = True
                by_dataset[dataset_id]["running_tasks"].append(drift_task_obj)
    except Exception:
        pass

    return {"by_dataset": by_dataset}

