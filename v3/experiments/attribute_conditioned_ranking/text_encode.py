"""Encode frozen prompt groups with the existing FashionSigLIP text tower."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ATTR_DIR = HERE.parent / "attribute_embedding"
if str(ATTR_DIR) not in sys.path:
    sys.path.insert(0, str(ATTR_DIR))

from representations import load_model  # noqa: E402


def encode_prompt_groups(
    groups: dict[str, list[str]],
    *,
    model_id: str,
    device: str = "auto",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    import torch

    model, _preprocess, tokenizer, resolved = load_model(model_id, device)
    names = list(groups)
    phrases = [phrase for name in names for phrase in groups[name]]
    offsets: list[tuple[str, int, int]] = []
    cursor = 0
    for name in names:
        count = len(groups[name])
        offsets.append((name, cursor, cursor + count))
        cursor += count
    encoded: list[np.ndarray] = []
    for start in range(0, len(phrases), batch_size):
        tokens = tokenizer(phrases[start : start + batch_size]).to(resolved)
        with torch.inference_mode():
            vectors = model.encode_text(tokens, normalize=True)
        encoded.append(vectors.detach().float().cpu().numpy())
    matrix = np.concatenate(encoded, axis=0).astype(np.float32)
    return {name: matrix[begin:end] for name, begin, end in offsets}
