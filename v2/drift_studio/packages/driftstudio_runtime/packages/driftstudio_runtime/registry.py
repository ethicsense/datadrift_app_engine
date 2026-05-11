from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.metadata
from typing import Optional


class Modality(str, Enum):
    vision = "vision"
    text = "text"
    audio = "audio"
    timeseries = "timeseries"


@dataclass(frozen=True)
class ModalityPluginStatus:
    modality: Modality
    package: str
    available: bool
    error: Optional[str] = None


_DDOC_ENTRYPOINTS: dict[Modality, str] = {
    Modality.vision: "ddoc_vision",
    Modality.text: "ddoc_text",
    Modality.audio: "ddoc_audio",
    Modality.timeseries: "ddoc_timeseries",
}


def detect_installed_modalities() -> list[ModalityPluginStatus]:
    """
    설치된 ddoc 플러그인(엔트리포인트)을 탐지합니다.

    - v2에서는 ddoc(v1) 플러그인을 vendoring 후, python import 실행(PythonExecutor)로 호출합니다.
    - 따라서 설치 여부는 `importlib.metadata.entry_points(group="ddoc")`로 판정합니다.
    """
    results: list[ModalityPluginStatus] = []
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            selected = eps.select(group="ddoc")
        else:
            selected = eps.get("ddoc", [])
        installed = {ep.name for ep in selected}
    except Exception as e:
        # entrypoints 조회 실패(환경 문제)
        return [
            ModalityPluginStatus(modality=m, package=name, available=False, error=str(e))
            for m, name in _DDOC_ENTRYPOINTS.items()
        ]

    for modality, name in _DDOC_ENTRYPOINTS.items():
        try:
            available = name in installed
            results.append(ModalityPluginStatus(modality=modality, package=name, available=available))
        except Exception as e:
            results.append(ModalityPluginStatus(modality=modality, package=name, available=False, error=str(e)))
    return results


