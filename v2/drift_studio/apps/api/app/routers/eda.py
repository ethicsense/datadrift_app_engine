import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Dataset
from app.services.plan_runner import build_default_plan_for_dataset, dataset_out_dir, run_plan


router = APIRouter(prefix="/eda", tags=["eda"])

_EDA_TASKS: dict[str, asyncio.Task] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(dataset_id: str) -> Path:
    return Path(dataset_out_dir(dataset_id)) / "eda.status.json"


def _write_status(dataset_id: str, payload: dict[str, Any]) -> None:
    p = _status_path(dataset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json

    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run_eda_job(dataset_id: str, *, force: bool) -> None:
    """
    - 길게 도는 EDA를 background task로 실행
    - 완료 후 DB에 결과 upsert + status 파일 업데이트
    """
    try:
        _write_status(
            dataset_id,
            {"state": "running", "dataset_id": dataset_id, "started_at": _utc_now_iso(), "force": force},
        )
        # plan 구성/실행은 blocking이므로 thread로 넘김
        await asyncio.to_thread(_run_plan_and_upsert, dataset_id, force)
        _write_status(
            dataset_id,
            {"state": "completed", "dataset_id": dataset_id, "completed_at": _utc_now_iso()},
        )
    except Exception as e:
        _write_status(
            dataset_id,
            {"state": "failed", "dataset_id": dataset_id, "failed_at": _utc_now_iso(), "error": str(e)},
        )
        raise
    finally:
        # task registry 정리
        t = _EDA_TASKS.get(dataset_id)
        if t and t.done():
            _EDA_TASKS.pop(dataset_id, None)


def _run_plan_and_upsert(dataset_id: str, force: bool) -> None:
    """
    background thread에서 실행되는 동기 함수
    """
    # DB 세션은 thread-safe하게 새로 생성
    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise ValueError("Dataset not found")

        out_dir = dataset_out_dir(ds.id)
        plan = build_default_plan_for_dataset(
            modality="auto",
            base_path=ds.dvc_path,
            target_path=None,
            out_dir=out_dir,
            pdf=False,
        )
        run_plan(plan, force=force)
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{dataset_id}")
async def eda(
    dataset_id: str,
    force: bool = Query(False, description="강제 재분석 여부"),
    db: Session = Depends(get_db),
):
    """
    EDA 결과를 반환합니다.

    정책:
    - ZIP + `ddoc.yaml` 스펙 기반 데이터셋만 허용
    - 내장 분석 로직 제거 → 플러그인(모달리티 분석기) 필수
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 이미 실행 중이면 중복 실행 방지
    existing_task = _EDA_TASKS.get(dataset_id)
    if existing_task and not existing_task.done():
        return {
            "dataset_id": dataset_id,
            "cached": False,
            "status": "running",
        }

    if not force:
        out_dir = dataset_out_dir(ds.id)
        index_path = Path(out_dir) / "artifact_index.json"
        if index_path.exists():
            return {"cached": True, "dataset_id": dataset_id, "run_id": ds.id}

    # 캐시가 없으면 background job 시작(즉시 반환)
    task = asyncio.create_task(_run_eda_job(dataset_id, force=force))
    _EDA_TASKS[dataset_id] = task
    return {
        "dataset_id": dataset_id,
        "cached": False,
        "status": "started",
        "run_id": ds.id,
    }


@router.get("/{dataset_id}/status")
def eda_status(dataset_id: str, db: Session = Depends(get_db)):
    """
    UI용 상태 엔드포인트.
    - dataset_id 기준으로 EDA가 실행 중인지 / 캐시가 존재하는지 반환합니다.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task = _EDA_TASKS.get(dataset_id)
    is_running = bool(task and not task.done())

    status_payload: dict[str, Any] = {}
    try:
        p = _status_path(dataset_id)
        if p.exists():
            import json

            status_payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        status_payload = {}

    out_dir = dataset_out_dir(ds.id)
    index_path = Path(out_dir) / "artifact_index.json"

    return {
        "dataset_id": dataset_id,
        "has_running_tasks": is_running or status_payload.get("state") == "running",
        "running_tasks": (
            [
                {
                    "task_id": f"eda:{dataset_id}",
                    "task_type": "eda",
                    "progress": None,
                    "state": status_payload.get("state") or ("running" if is_running else "unknown"),
                    "started_at": status_payload.get("started_at"),
                }
            ]
            if (is_running or status_payload.get("state") == "running")
            else []
        ),
        "cache_status": {"eda": index_path.exists()},
        "status": status_payload,
    }
