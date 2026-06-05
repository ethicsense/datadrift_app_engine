from __future__ import annotations

from silhouette_outliner.config import default_collect_config_path, load_config


def test_default_collect_config_matches_full_preset() -> None:
    path = default_collect_config_path()
    assert path.is_file(), path
    config = load_config(path)
    assert [window.id for window in config.ranking_windows] == ["1d", "1w", "1m"]
    assert config.expands_demographics() is True
    assert config.age_rankings_window is not None
    assert len(config.targets()) > 100
