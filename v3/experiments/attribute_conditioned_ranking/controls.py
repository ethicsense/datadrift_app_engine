"""Turn one image embedding into attribute-controlled embeddings. No model training."""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / max(norm, 1e-12)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def fingerprint(image_embeddings: np.ndarray, grouped_prompts: list[np.ndarray]) -> np.ndarray:
    """Score each image against the mean vector of every prompt group."""
    columns = [l2_normalize(np.mean(group, axis=0)) for group in grouped_prompts]
    text = np.stack(columns, axis=0)
    return l2_normalize(np.asarray(image_embeddings, dtype=np.float32) @ text.T)


def subspace_basis(prompt_matrix: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal attribute directions, shape (dim, rank), from centered prompts."""
    prompts = np.asarray(prompt_matrix, dtype=np.float32)
    centered = prompts - prompts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    used = max(1, min(int(rank), vh.shape[0]))
    return vh[:used].T.astype(np.float32)


def project(image_embeddings: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return l2_normalize(np.asarray(image_embeddings, dtype=np.float32) @ basis)


def remove_subspace(image_embeddings: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Subtract the orthogonal projection onto `basis` (dim, rank)."""
    images = np.asarray(image_embeddings, dtype=np.float32)
    coeff = images @ basis
    return images - coeff @ basis.T


def mix(global_embeddings: np.ndarray, attribute_embeddings: np.ndarray, alpha: float) -> np.ndarray:
    """Concatenate scaled unit vectors so their dot product is the α mixture."""
    weight = float(np.clip(alpha, 0.0, 1.0))
    global_unit = l2_normalize(global_embeddings)
    attribute_unit = l2_normalize(attribute_embeddings)
    mixed = np.concatenate(
        [
            np.sqrt(1.0 - weight) * global_unit,
            np.sqrt(weight) * attribute_unit,
        ],
        axis=1,
    )
    return l2_normalize(mixed)


def data_centroid_basis(
    image_embeddings: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    classes: list[str],
    rank: int,
    *,
    min_count: int = 8,
) -> np.ndarray:
    """SVD of train class centroids after removing their mean. Labels never include eval-only colors."""
    images = np.asarray(image_embeddings, dtype=np.float32)
    centroids: list[np.ndarray] = []
    for label in classes:
        idx = train_mask & (labels == label)
        if int(idx.sum()) < min_count:
            continue
        centroids.append(images[idx].mean(axis=0))
    if len(centroids) < 2:
        raise ValueError("need at least two train classes to build a data direction")
    matrix = np.stack(centroids, axis=0)
    return subspace_basis(matrix, min(rank, len(centroids) - 1))
