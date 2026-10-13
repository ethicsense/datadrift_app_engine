"""sRGB, CIELAB, and CIEDE2000 helpers. Evaluation only."""

from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    c = np.asarray(channel, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_u8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float RGB in 0–255, shape (..., 3), to CIELAB."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = _srgb_to_linear(arr)
    xyz = linear @ _RGB_TO_XYZ.T
    ratio = xyz / _D65
    delta = 6 / 29
    f = np.where(ratio > delta**3, np.cbrt(ratio), ratio / (3 * delta**2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000. Broadcasts over leading dimensions; last axis is Lab."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    kL = kC = kH = 1.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = 0.5 * (C1 + C2)
    c7 = Cbar**7
    G = 0.5 * (1.0 - np.sqrt(c7 / (c7 + 25**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2 - h1
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lbar = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1 + h2
    far = np.abs(h1 - h2) > 180.0
    hbar = np.where(far & (hsum < 360.0), hsum + 360.0, hsum)
    hbar = np.where(far & (hsum >= 360.0), hsum - 360.0, hbar)
    hbar = np.where((C1p * C2p) == 0, hsum, hbar)
    hbar = (0.5 * hbar) % 360.0
    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbar))
        + 0.32 * np.cos(np.radians(3.0 * hbar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbar - 63.0))
    )
    dtheta = 30.0 * np.exp(-(((hbar - 275.0) / 25.0) ** 2))
    cbar7 = Cbarp**7
    Rc = 2.0 * np.sqrt(cbar7 / (cbar7 + 25**7))
    Sl = 1.0 + (0.015 * (Lbar - 50.0) ** 2) / np.sqrt(20.0 + (Lbar - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc
    return np.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


def pairwise_delta_e00(lab: np.ndarray) -> np.ndarray:
    """Symmetric pairwise ΔE00 for a (n, 3) Lab matrix."""
    lab = np.asarray(lab, dtype=np.float64)
    n = lab.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    chunk = 128
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = delta_e00(lab[start:stop, None, :], lab[None, :, :])
    return out
