from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameStats:
    brightness: float
    contrast: float
    sharpness: float


def _to_gray_np(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    arr = np.asarray(g, dtype=np.float32) / 255.0
    return arr


def _sharpness_laplacian_var(gray: np.ndarray) -> float:
    # simple laplacian (no scipy): L = -4c + n+s+e+w
    c = gray[1:-1, 1:-1]
    n = gray[:-2, 1:-1]
    s = gray[2:, 1:-1]
    w = gray[1:-1, :-2]
    e = gray[1:-1, 2:]
    lap = -4.0 * c + n + s + w + e
    return float(np.var(lap))


def analyze_frame(path: Path) -> Dict[str, float]:
    with Image.open(path) as img:
        gray = _to_gray_np(img)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = _sharpness_laplacian_var(gray) if gray.shape[0] >= 3 and gray.shape[1] >= 3 else 0.0
    return {"brightness": brightness, "contrast": contrast, "sharpness": sharpness}


def motion_score(frames_gray: list[np.ndarray]) -> float:
    """
    Simple temporal motion proxy: mean absolute diff between consecutive frames.
    """
    if len(frames_gray) < 2:
        return 0.0
    diffs = []
    for a, b in zip(frames_gray, frames_gray[1:]):
        # ensure same shape
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h <= 0 or w <= 0:
            continue
        diffs.append(float(np.mean(np.abs(a[:h, :w] - b[:h, :w]))))
    return float(np.mean(diffs)) if diffs else 0.0

