from __future__ import annotations

from pathlib import Path

from silhouette_outliner.config import default_collect_config_path
from silhouette_outliner.runtime_paths import bundled_config_path, configs_dir, is_frozen


def test_bundled_config_path_points_to_periodic_multag() -> None:
    path = bundled_config_path("periodic-multag.json")
    assert path.name == "periodic-multag.json"
    assert path.parent == configs_dir()
    assert path.is_file()


def test_default_collect_config_matches_preset_file() -> None:
    assert default_collect_config_path() == bundled_config_path("periodic-multag.json")


def test_is_frozen_false_in_dev() -> None:
    assert is_frozen() is False
