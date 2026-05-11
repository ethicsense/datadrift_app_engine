from __future__ import annotations

import os
from pathlib import Path

from analytics.api.settings import ApiSettings


def resolve_workspace_file(candidate: str | None, settings: ApiSettings) -> Path | None:
    """
    크롤 이미지 등 로컬 파일을 API로 노출할 때, 워크스페이스(또는 허용 루트) 아래인지 검증한다.
    """
    if not candidate or not isinstance(candidate, str):
        return None
    raw = candidate.strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (settings.workspace_root / path).resolve()
        else:
            path = path.resolve()
    except OSError:
        return None
    if not path.is_file():
        return None
    roots = settings.effective_image_roots()
    for root in roots:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    return None


def guess_image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"
