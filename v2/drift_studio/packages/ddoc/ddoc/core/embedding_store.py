from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


def _project_root(project_root: Optional[str] = None) -> Path:
    return Path(project_root).resolve() if project_root else Path.cwd().resolve()


def embeddings_root(project_root: Optional[str] = None) -> Path:
    return _project_root(project_root) / "storage" / "embeddings"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _try_write_parquet(path: Path, ids: list[str], vectors: np.ndarray, labels: Optional[list[str]] = None) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = {
            "id": pa.array(ids, type=pa.string()),
            "vector": pa.array(vectors.tolist()),
        }
        if labels:
            fields["label"] = pa.array(labels, type=pa.string())
        table = pa.table(fields)
        pq.write_table(table, path, compression="zstd")
        return True
    except Exception:
        return False


def save_embeddings(
    *,
    modality: str,
    data_hash: str,
    embeddings: Dict[str, Iterable[float]],
    labels: Optional[Dict[str, str]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    if not embeddings:
        return {
            "storage": "file",
            "path": None,
            "format": None,
            "dtype": None,
            "dim": None,
            "count": 0,
        }

    root = embeddings_root(project_root) / modality
    _ensure_dir(root)

    ids = list(embeddings.keys())
    vectors = np.array([np.asarray(embeddings[k], dtype=np.float32) for k in ids])
    dim = int(vectors.shape[1]) if vectors.ndim == 2 else int(vectors.shape[0])
    count = int(vectors.shape[0]) if vectors.ndim == 2 else 1
    label_list = [labels.get(k) for k in ids] if labels else None

    parquet_path = root / f"{data_hash}.parquet"
    fmt = "parquet"
    if not _try_write_parquet(parquet_path, ids, vectors, label_list):
        npz_path = root / f"{data_hash}.npz"
        np.savez_compressed(npz_path, ids=np.array(ids), vectors=vectors, labels=np.array(label_list) if label_list else None)
        parquet_path = npz_path
        fmt = "npz"

    project_root_path = _project_root(project_root)
    rel_path = str(parquet_path.relative_to(project_root_path))
    return {
        "storage": "file",
        "path": rel_path,
        "format": fmt,
        "dtype": "float32",
        "dim": dim,
        "count": count,
    }


def load_embeddings(
    *,
    embedding_index: Dict[str, Any],
    project_root: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, list[str], Optional[list[str]]]:
    if not embedding_index or not embedding_index.get("path"):
        return np.empty((0, 0), dtype=np.float32), [], None

    root = _project_root(project_root)
    path = root / embedding_index["path"]
    fmt = embedding_index.get("format")

    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            ids = table.column("id").to_pylist()
            vectors = np.array(table.column("vector").to_pylist(), dtype=np.float32)
            labels = table.column("label").to_pylist() if "label" in table.column_names else None
        except Exception:
            return np.empty((0, 0), dtype=np.float32), [], None
    elif fmt == "npz":
        data = np.load(path, allow_pickle=True)
        ids = data["ids"].astype(str).tolist()
        vectors = np.array(data["vectors"], dtype=np.float32)
        labels_arr = data["labels"] if "labels" in data else None
        if labels_arr is None or (getattr(labels_arr, "shape", ()) == () and labels_arr.item() is None):
            labels = None
        else:
            labels = labels_arr.astype(str).tolist()
    else:
        return np.empty((0, 0), dtype=np.float32), [], None

    total = len(ids)
    start = max(offset, 0)
    end = total if limit is None else min(total, start + max(limit, 0))
    return vectors[start:end], ids[start:end], labels[start:end] if labels else None
