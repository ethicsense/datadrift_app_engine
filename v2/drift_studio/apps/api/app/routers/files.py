from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.dvc_service import BASE_DATA_DIR

router = APIRouter(prefix="/files", tags=["files"])

@router.get("/raw")
def get_raw_file(path: str):
    safe_path = Path(path).resolve()
    allowed_roots = [Path("storage") / "runs", Path(BASE_DATA_DIR)]
    try:
        if not any(safe_path.is_relative_to(root.resolve()) for root in allowed_roots):
            raise HTTPException(status_code=400, detail="Invalid file path")
    except AttributeError:
        resolved_roots = [root.resolve() for root in allowed_roots]
        if not any(str(safe_path).startswith(str(root)) for root in resolved_roots):
            raise HTTPException(status_code=400, detail="Invalid file path")

    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(safe_path), filename=safe_path.name)