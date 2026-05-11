from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ArtifactPaths:
    """
    산출물 경로 규약 (artifact index 중심).
    """

    root: Path

    @property
    def artifact_index(self) -> Path:
        return self.root / "artifact_index.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def report_html(self) -> Path:
        return self.root / "report.html"

    @property
    def report_pdf(self) -> Path:
        return self.root / "report.pdf"


class ArtifactPayloadInline(BaseModel):
    mode: Literal["inline"] = "inline"
    data: Any


class ArtifactPayloadRef(BaseModel):
    mode: Literal["ref"] = "ref"
    uri: str = Field(..., description="artifact payload path relative to run root")
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


ArtifactPayload = Union[ArtifactPayloadInline, ArtifactPayloadRef]


class ArtifactEntry(BaseModel):
    id: str = Field(..., description="stable artifact id")
    type: str = Field(..., description="artifact type name, e.g. drift.overall_score.v1")
    artifact_schema_version: str = Field("1", description="artifact payload schema version")
    title: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None
    payload: ArtifactPayload


class ArtifactIndex(BaseModel):
    schema_version: str = Field("1", description="artifact index schema version")
    generated_at: Optional[str] = None
    producer: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None
    artifacts: list[ArtifactEntry] = Field(default_factory=list)


