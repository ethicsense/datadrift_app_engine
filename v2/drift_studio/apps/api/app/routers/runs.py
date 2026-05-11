from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.services.plan_runner import dataset_out_dir
from driftstudio_spec import ArtifactIndex

router = APIRouter(prefix="/runs", tags=["runs"])


def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_index(run_id: str) -> tuple[Path, ArtifactIndex]:
    out_dir = Path(dataset_out_dir(run_id))
    index_path = out_dir / "artifact_index.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="artifact_index not found")
    return out_dir, ArtifactIndex.parse_obj(_read_json(index_path))


def _resolve_payload_path(out_dir: Path, uri: str) -> Path:
    candidate = (out_dir / uri).resolve()
    root = out_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact uri") from exc
    return candidate


@router.get("/{run_id}/artifact_index")
def get_artifact_index(run_id: str):
    _, index = _load_index(run_id)
    return JSONResponse(index.dict())


@router.get("/{run_id}/artifacts/{artifact_id}")
def get_artifact_payload(run_id: str, artifact_id: str):
    out_dir, index = _load_index(run_id)
    artifact = next((a for a in index.artifacts if a.id == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    payload = artifact.payload
    if getattr(payload, "mode", None) == "inline":
        return {"id": artifact.id, "type": artifact.type, "data": payload.data}
    if getattr(payload, "mode", None) == "ref":
        path = _resolve_payload_path(out_dir, payload.uri)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact payload not found")
        return FileResponse(path, filename=path.name)

    raise HTTPException(status_code=400, detail="Unsupported artifact payload")
