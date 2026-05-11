import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Dataset
from app.services.plan_runner import dataset_out_dir, drift_pair_plan_and_cache_mode, run_plan
from driftstudio_spec import Plan

router = APIRouter(prefix="/drift", tags=["drift"])

_DRIFT_TASKS: dict[str, asyncio.Task] = {}


def _pair_key(base_id: str, target_id: str) -> str:
    return f"{base_id}__{target_id}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(out_dir: str) -> Path:
    return Path(out_dir) / "drift.status.json"


def _write_status(out_dir: str, payload: dict[str, Any]) -> None:
    p = _status_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json

    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_plan_and_upsert(
    *,
    base_id: str,
    target_id: str,
    base_path: str,
    target_path: str,
    out_dir: str,
    force: bool,
    plan: Plan,
) -> dict[str, Any]:
    """
    background thread에서 실행되는 동기 함수
    """
    db = SessionLocal()
    try:
        run_plan(plan, force=force)
        report_pdf = str(Path(out_dir) / "report.pdf")
        return {
            "report": {"pdf": report_pdf if Path(report_pdf).exists() else None},
        }
    finally:
        db.close()


async def _run_drift_job(
    *,
    key: str,
    base_id: str,
    target_id: str,
    base_path: str,
    target_path: str,
    out_dir: str,
    force: bool,
    plan: Plan,
    used_cached_eda: bool,
) -> None:
    try:
        _write_status(
            out_dir,
            {
                "state": "running",
                "base_id": base_id,
                "target_id": target_id,
                "started_at": _utc_now_iso(),
                "force": force,
                "plan": plan.model_dump(),
                "used_cached_eda": used_cached_eda,
            },
        )
        await asyncio.to_thread(
            _run_plan_and_upsert,
            base_id=base_id,
            target_id=target_id,
            base_path=base_path,
            target_path=target_path,
            out_dir=out_dir,
            force=force,
            plan=plan,
        )
        _write_status(
            out_dir,
            {
                "state": "completed",
                "base_id": base_id,
                "target_id": target_id,
                "completed_at": _utc_now_iso(),
                "plan": plan.model_dump(),
                "used_cached_eda": used_cached_eda,
            },
        )
    except Exception as e:
        _write_status(
            out_dir,
            {
                "state": "failed",
                "base_id": base_id,
                "target_id": target_id,
                "failed_at": _utc_now_iso(),
                "error": str(e),
                "plan": plan.model_dump(),
                "used_cached_eda": used_cached_eda,
            },
        )
        raise
    finally:
        t = _DRIFT_TASKS.get(key)
        if t and t.done():
            _DRIFT_TASKS.pop(key, None)



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DriftRequest(BaseModel):
    base_id: str
    target_id: str
    force: bool = False


@router.post("/v2")
async def drift_v2(req: DriftRequest, db: Session = Depends(get_db)):
    """
    Drift 실행(동기).

    정책:
    - ZIP + ddoc.yaml 스펙 기반 데이터셋만 허용
    - 실행은 vendored ddoc 플러그인(PythonExecutor) 기반
    - 비동기 작업/WS/진행률은 제거(필요 시 추후 재도입)
    """
    base = db.query(Dataset).filter(Dataset.id == req.base_id).first()
    target = db.query(Dataset).filter(Dataset.id == req.target_id).first()
    if not base or not target:
        raise HTTPException(status_code=404, detail="Dataset not found")

    key = _pair_key(base.id, target.id)

    # 이미 실행 중이면 중복 실행 방지
    existing_task = _DRIFT_TASKS.get(key)
    if existing_task and not existing_task.done():
        return {
            "cached": False,
            "status": "running",
            "base_id": base.id,
            "target_id": target.id,
        }

    # 캐시(artifact_index) 재사용
    if not req.force:
        out_dir = dataset_out_dir(f"drift_{base.id}_{target.id}")
        index_path = Path(out_dir) / "artifact_index.json"
        if index_path.exists():
            status_payload: dict[str, Any] = {}
            try:
                p = _status_path(out_dir)
                if p.exists():
                    import json

                    status_payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                status_payload = {}

            plan_name = None
            try:
                plan_name = (status_payload.get("plan") or {}).get("name")
            except Exception:
                plan_name = None

            report_pdf = str(Path(out_dir) / "report.pdf")

            return {
                "cached": True,
                "base_id": base.id,
                "target_id": target.id,
                "run_id": f"drift_{base.id}_{target.id}",
                "plan_name": plan_name or "cached",
                "used_cached_eda": bool(status_payload.get("used_cached_eda")) if status_payload else None,
                "report": {
                    "pdf": report_pdf if Path(report_pdf).exists() else None,
                },
            }

    out_dir = dataset_out_dir(f"drift_{base.id}_{target.id}")
    plan, used_cached_eda = drift_pair_plan_and_cache_mode(
        base_id=base.id,
        target_id=target.id,
        base_path=base.dvc_path,
        target_path=target.dvc_path,
        out_dir=out_dir,
        force=req.force,
    )

    # 비동기 job 시작(즉시 반환)
    task = asyncio.create_task(
        _run_drift_job(
            key=key,
            base_id=base.id,
            target_id=target.id,
            base_path=base.dvc_path,
            target_path=target.dvc_path,
            out_dir=out_dir,
            force=req.force,
            plan=plan,
            used_cached_eda=used_cached_eda,
        )
    )
    _DRIFT_TASKS[key] = task
    return {
        "cached": False,
        "status": "started",
        "base_id": base.id,
        "target_id": target.id,
        "run_id": f"drift_{base.id}_{target.id}",
        "plan_name": plan.name,
        "used_cached_eda": used_cached_eda,
    }


@router.get("/{base_id}/{target_id}/status")
def drift_status(base_id: str, target_id: str, db: Session = Depends(get_db)):
    base = db.query(Dataset).filter(Dataset.id == base_id).first()
    target = db.query(Dataset).filter(Dataset.id == target_id).first()
    if not base or not target:
        raise HTTPException(status_code=404, detail="Dataset not found")

    key = _pair_key(base_id, target_id)
    task = _DRIFT_TASKS.get(key)
    is_running = bool(task and not task.done())

    out_dir = dataset_out_dir(f"drift_{base_id}_{target_id}")
    index_path = Path(out_dir) / "artifact_index.json"
    status_payload: dict[str, Any] = {}
    try:
        p = _status_path(out_dir)
        if p.exists():
            import json

            status_payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        status_payload = {}

    return {
        "base_id": base_id,
        "target_id": target_id,
        "has_running_tasks": is_running or status_payload.get("state") == "running",
        "running_tasks": (
            [
                {
                    "task_id": f"drift:{base_id}:{target_id}",
                    "task_type": "drift",
                    "progress": None,
                    "state": status_payload.get("state") or ("running" if is_running else "unknown"),
                    "started_at": status_payload.get("started_at"),
                }
            ]
            if (is_running or status_payload.get("state") == "running")
            else []
        ),
        "cache_status": {"drift": index_path.exists()},
        "status": status_payload,
    }


