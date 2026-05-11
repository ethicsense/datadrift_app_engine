import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Dataset
from app.services.plan_runner import (
    build_default_plan_for_dataset,
    dataset_out_dir,
    drift_pair_plan_and_cache_mode,
    run_plan,
)

router = APIRouter(prefix="/report", tags=["report"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/eda/{dataset_id}")
def generate_eda_route(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    out_dir = dataset_out_dir(ds.id)
    plan = build_default_plan_for_dataset(
        modality="auto",
        base_path=ds.dvc_path,
        target_path=None,
        out_dir=out_dir,
        pdf=True,
    )

    try:
        result = run_plan(plan, force=False)
    except ValueError as e:
        # 메타 누락/불일치 등: 처리 불가
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    pdf = str(Path(out_dir) / "report.pdf")
    # 캐시만 있고 PDF가 비어 있는 등 예외 케이스: 한 번 더 강제 실행
    if not os.path.exists(pdf):
        try:
            result = run_plan(plan, force=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report regeneration failed: {e}") from e

    if os.path.exists(pdf):
        return {"pdf": pdf}

    report_error = None
    if isinstance(result, dict):
        report_payload = result.get("report")
        if isinstance(report_payload, dict):
            report_error = report_payload.get("error")

    html = str(Path(out_dir) / "report.html")
    if report_error:
        detail = f"PDF report generation failed: {report_error}"
    elif os.path.exists(html):
        detail = (
            "PDF report generation failed: HTML report was generated but PDF conversion failed. "
            "Check WeasyPrint and required system libraries (cairo/pango/gdk-pixbuf) on the API runtime."
        )
    else:
        detail = "PDF report generation failed before any report file was created."
    raise HTTPException(status_code=500, detail=detail)


@router.get("/drift/{base_id}/{target_id}")
def generate_drift_report_route(
    base_id: str,
    target_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
):
    """
    드리프트 비교 PDF를 동기 생성(또는 캐시 활용)하고 서버 경로를 반환.
    EDA 단일 리포트 `/report/eda/{id}`와 동일한 재시도·오류 처리.
    """
    base = db.query(Dataset).filter(Dataset.id == base_id).first()
    target = db.query(Dataset).filter(Dataset.id == target_id).first()
    if not base or not target:
        raise HTTPException(status_code=404, detail="Dataset not found")

    out_dir = dataset_out_dir(f"drift_{base.id}_{target.id}")
    plan, _used_cached = drift_pair_plan_and_cache_mode(
        base_id=base.id,
        target_id=target.id,
        base_path=base.dvc_path,
        target_path=target.dvc_path,
        out_dir=out_dir,
        force=force,
    )

    try:
        result = run_plan(plan, force=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    pdf = str(Path(out_dir) / "report.pdf")
    if not os.path.exists(pdf):
        try:
            result = run_plan(plan, force=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report regeneration failed: {e}") from e

    if os.path.exists(pdf):
        return {"pdf": pdf}

    report_error = None
    if isinstance(result, dict):
        report_payload = result.get("report")
        if isinstance(report_payload, dict):
            report_error = report_payload.get("error")

    html = str(Path(out_dir) / "report.html")
    if report_error:
        detail = f"PDF report generation failed: {report_error}"
    elif os.path.exists(html):
        detail = (
            "PDF report generation failed: HTML report was generated but PDF conversion failed. "
            "Check WeasyPrint and required system libraries (cairo/pango/gdk-pixbuf) on the API runtime."
        )
    else:
        detail = "PDF report generation failed before any report file was created."
    raise HTTPException(status_code=500, detail=detail)


@router.get("/download")
def download_report(path: str):
    safe_path = Path(path).resolve()
    runs_root = Path("storage") / "runs"
    try:
        safe_path.relative_to(runs_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid report path") from exc

    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    suffix = safe_path.suffix.lower()
    media_type = "application/pdf" if suffix == ".pdf" else "application/octet-stream"
    return FileResponse(
        str(safe_path),
        filename=os.path.basename(safe_path),
        media_type=media_type,
    )