from fastapi import APIRouter, UploadFile, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import Dataset
from ..services.dataset_service import create_dataset, delete_dataset

from app.schemas import DatasetSchema

    
router = APIRouter(prefix="/datasets", tags=["datasets"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload")
async def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
    try:
        dataset = create_dataset(db, file)
        return dataset
    except ValueError as e:
        # 메타 누락/불일치 등: 처리 불가(400)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=list[DatasetSchema])
@router.get("/", response_model=list[DatasetSchema])
def list_datasets(db: Session = Depends(get_db)):
    items = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return items  # Pydantic이 자동 변환 + sanitize


@router.get("/{dataset_id}", response_model=DatasetSchema)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@router.get("/{dataset_id}/ddoc")
def get_ddoc_yaml(
    dataset_id: str,
    max_chars: int = Query(200_000, ge=1, le=2_000_000, description="ddoc.yaml 최대 반환 문자 수"),
    db: Session = Depends(get_db),
):
    """
    ddoc.yaml 원문(및 파싱된 meta)을 반환합니다.
    - ZIP 구조 확인/규약 확인용
    """
    from pathlib import Path

    from driftstudio_runtime.infer import load_dataset_meta_or_raise

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    meta_path = Path(ds.dvc_path) / "ddoc.yaml"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="ddoc.yaml not found")

    raw = meta_path.read_text(encoding="utf-8", errors="replace")
    clipped = raw[:max_chars]
    truncated = len(raw) > max_chars

    try:
        meta = load_dataset_meta_or_raise(ds.dvc_path).model_dump()
    except Exception as e:
        meta = {"error": str(e)}

    return {
        "path": str(meta_path),
        "raw": clipped,
        "truncated": truncated,
        "meta": meta,
    }


@router.delete("/{dataset_id}")
def remove_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """
    데이터셋과 관련된 모든 데이터를 삭제합니다.
    - 관련 EDA 결과
    - 관련 Drift 결과 (base 또는 target으로 사용된 경우)
    - 파일 시스템의 데이터셋 폴더
    - Dataset DB 레코드
    """
    result = delete_dataset(db, dataset_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result
