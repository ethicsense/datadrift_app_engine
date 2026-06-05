from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR_NAME = "SilhouetteOutliner"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def user_data_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / _APP_DIR_NAME
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / _APP_DIR_NAME
    else:
        base = Path.home() / ".local" / "share" / _APP_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def runs_root() -> Path:
    path = user_data_root() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configs_dir() -> Path:
    return bundle_root() / "configs"


def bundled_config_path(filename: str) -> Path:
    return configs_dir() / filename


def configure_playwright_browsers() -> None:
    if not is_frozen():
        return
    browser_dir = bundle_root() / "playwright-browsers"
    if browser_dir.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_dir)
