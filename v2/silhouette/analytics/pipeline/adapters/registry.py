from __future__ import annotations

from typing import Any, Dict, Iterable, List

from analytics.pipeline.adapters.base import ChannelAdapter


class AdapterRegistry:
    def __init__(self, adapters: Iterable[ChannelAdapter]) -> None:
        self._adapters: List[ChannelAdapter] = list(adapters)

    def select(self, summary_payload: Dict[str, Any]) -> ChannelAdapter:
        for adapter in self._adapters:
            if adapter.detect(summary_payload):
                return adapter
        if not self._adapters:
            raise RuntimeError("No channel adapter is registered.")
        return self._adapters[0]
