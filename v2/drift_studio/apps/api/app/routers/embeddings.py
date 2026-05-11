from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.plan_runner import dataset_out_dir
from ddoc.core.embedding_store import load_embeddings
from driftstudio_spec import ArtifactIndex

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/{dataset_id}")
def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifact_index(dataset_id: str) -> ArtifactIndex:
    index_path = Path(dataset_out_dir(dataset_id)) / "artifact_index.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="artifact_index not found")
    return ArtifactIndex.parse_obj(_read_json(index_path))


def _find_embedding_index(index: ArtifactIndex, dataset_id: str) -> dict[str, Any]:
    artifact = next((a for a in index.artifacts if a.type == "embedding.index.v1"), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Embedding index not found")

    payload = artifact.payload
    if getattr(payload, "mode", None) == "inline":
        return payload.data
    if getattr(payload, "mode", None) == "ref":
        return _read_json(Path(dataset_out_dir(dataset_id)) / payload.uri)
    raise HTTPException(status_code=400, detail="Unsupported embedding payload")


def read_embeddings(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
):
    index = _load_artifact_index(dataset_id)
    embedding_index = _find_embedding_index(index, dataset_id)
    if not embedding_index or not embedding_index.get("path"):
        raise HTTPException(status_code=404, detail="Embedding index not found")

    vectors, ids, labels = load_embeddings(
        embedding_index=embedding_index,
        offset=offset,
        limit=limit,
    )

    points = []
    for i, vec in enumerate(vectors):
        item = {"id": ids[i], "vector": vec.tolist()}
        if labels:
            item["label"] = labels[i]
        points.append(item)

    return {
        "dataset_id": dataset_id,
        "embedding_index": embedding_index,
        "offset": offset,
        "limit": limit,
        "count": len(points),
        "points": points,
    }
