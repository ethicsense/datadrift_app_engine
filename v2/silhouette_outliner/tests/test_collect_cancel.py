from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from silhouette_outliner.collector import collect_all
from silhouette_outliner.config import AppConfig
from silhouette_outliner.exceptions import CollectCancelled


def test_collect_all_raises_when_cancelled_before_first_target(tmp_path) -> None:
    config = AppConfig.defaults()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(CollectCancelled):
        collect_all(config, run_dir, should_cancel=lambda: True)


def test_collect_all_checks_cancel_between_targets(tmp_path) -> None:
    config = AppConfig.defaults()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    with patch("silhouette_outliner.collector.collect_target") as mock_target:
        mock_target.return_value = MagicMock(
            ok=True,
            error=None,
            source="client-api",
            to_dict=lambda: {"ok": True},
        )
        with pytest.raises(CollectCancelled):
            collect_all(config, run_dir, should_cancel=should_cancel)
        assert mock_target.call_count == 1
