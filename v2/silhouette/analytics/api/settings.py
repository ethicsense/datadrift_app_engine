from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    output_dir: Path = Path(os.getenv("SILHOUETTE_OUTPUT_DIR", "output/analytics"))
    datasets_root: Path = Path(os.getenv("SILHOUETTE_DATASETS_ROOT", "output"))
    # 이미지 프리뷰 허용 루트 계산에 사용(미설정 시 현재 작업 디렉터리).
    workspace_root: Path = Path(os.getenv("SILHOUETTE_WORKSPACE_ROOT", ".")).resolve()
    app_title: str = "Silhouette Visualization API"
    allow_origins: tuple[str, ...] = ("*",)

    def effective_image_roots(self) -> tuple[Path, ...]:
        extra = os.getenv("SILHOUETTE_IMAGE_ROOTS", "")
        parsed = [
            Path(part.strip()).expanduser().resolve()
            for part in extra.replace(",", ";").split(";")
            if part.strip()
        ]
        roots: list[Path] = []
        for p in parsed:
            if p.exists():
                roots.append(p)
        for p in (self.workspace_root, self.datasets_root.resolve(), self.output_dir.resolve().parent):
            if p.exists():
                roots.append(p)
        seen: set[str] = set()
        unique: list[Path] = []
        for r in roots:
            key = str(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return tuple(unique)


def get_settings() -> ApiSettings:
    return ApiSettings()
