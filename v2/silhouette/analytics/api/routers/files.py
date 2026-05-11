from __future__ import annotations

from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from analytics.api.image_paths import guess_image_media_type, resolve_workspace_file
from analytics.api.settings import get_settings

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/product-image")
def product_image(path: str = Query(..., description="로컬 이미지 파일 절대 경로(또는 워크스페이스 기준 상대 경로)")) -> FileResponse:
    settings = get_settings()
    decoded = unquote(path)
    resolved = resolve_workspace_file(decoded, settings)
    if resolved is None:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없거나 허용된 경로가 아닙니다.")
    return FileResponse(resolved, media_type=guess_image_media_type(resolved))
