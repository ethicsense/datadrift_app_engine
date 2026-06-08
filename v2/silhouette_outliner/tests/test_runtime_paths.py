from __future__ import annotations

import os

from silhouette_outliner.config import default_collect_config_path
from silhouette_outliner.runtime_paths import (
    bundled_browsers_dir,
    bundled_config_path,
    configs_dir,
    configure_playwright_browsers,
    configure_tls,
    is_frozen,
)


def test_bundled_config_path_points_to_periodic_multag() -> None:
    path = bundled_config_path("periodic-multag.json")
    assert path.name == "periodic-multag.json"
    assert path.parent == configs_dir()
    assert path.is_file()


def test_default_collect_config_matches_preset_file() -> None:
    assert default_collect_config_path() == bundled_config_path("periodic-multag.json")


def test_is_frozen_false_in_dev() -> None:
    assert is_frozen() is False


def test_configure_playwright_browsers_is_noop_in_dev() -> None:
    before = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    configure_playwright_browsers()
    after = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    # Dev runs must not override the developer's default Playwright cache.
    assert before == after
    assert bundled_browsers_dir() is None


def test_configure_tls_is_noop_in_dev() -> None:
    before = os.environ.get("SSL_CERT_FILE")
    configure_tls()
    after = os.environ.get("SSL_CERT_FILE")
    # Dev runs already have working CA paths; must not be mutated.
    assert before == after
