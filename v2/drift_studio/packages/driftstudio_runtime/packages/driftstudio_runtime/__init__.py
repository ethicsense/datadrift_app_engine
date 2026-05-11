"""
driftstudio_runtime

API와 CLI가 공통으로 사용하는 최소 런타임 구현(EDA/Drift 중심).
"""

from .registry import Modality, ModalityPluginStatus, detect_installed_modalities
from .runner import RuntimeRunner

__all__ = [
    "Modality",
    "ModalityPluginStatus",
    "detect_installed_modalities",
    "RuntimeRunner",
]

