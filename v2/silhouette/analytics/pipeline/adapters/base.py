from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class AdapterContext:
    source_dataset: str
    platform: str
    snapshot_date: str
    snapshot_time: str
    source_path: str


class ChannelAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def detect(self, summary_payload: Dict[str, Any]) -> bool:
        ...

    def infer_schema_version(self, summary_payload: Dict[str, Any]) -> str:
        ...

    def load_product_detail(self, session_dir: Path, product_id: str) -> Dict[str, Any]:
        ...

    def build_extension_payload(self, *, summary_payload: Dict[str, Any], product_payload: Dict[str, Any]) -> Dict[str, Any]:
        ...
