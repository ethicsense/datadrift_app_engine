from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

_APP_DIR_NAME = "SilhouetteOutliner"
_BROWSERS_DIR_NAME = "playwright-browsers"


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


def _browser_dir_candidates() -> Iterator[Path]:
    """Locations where the bundled Playwright browsers may live (frozen build)."""
    yield bundle_root() / _BROWSERS_DIR_NAME
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        yield exe_dir / _BROWSERS_DIR_NAME
        # macOS .app: Contents/MacOS/<exe> -> Contents/Frameworks/playwright-browsers
        yield exe_dir.parent / "Frameworks" / _BROWSERS_DIR_NAME
        yield exe_dir.parent / "Resources" / _BROWSERS_DIR_NAME


def bundled_browsers_dir() -> Path | None:
    for candidate in _browser_dir_candidates():
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def configure_playwright_browsers() -> None:
    """Point Playwright at the Chromium bundled inside the packaged app.

    Does nothing in dev runs, where the developer's standard Playwright
    installation (default cache) is used as-is.
    """
    if not is_frozen():
        return
    bundled = bundled_browsers_dir()
    if bundled is not None:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(bundled)


def configure_tls() -> None:
    """Make stdlib urllib trust certifi's CA bundle in the packaged app.

    Frozen builds (PyInstaller) do not ship the OpenSSL default CA paths, so
    HTTPS via urllib fails with CERTIFICATE_VERIFY_FAILED. That silently forces
    the slow Playwright fallback for every page. Pointing SSL_CERT_FILE at the
    bundled certifi cacert.pem restores the fast client-api collection path.

    No-op in dev runs, where the interpreter already has working CA paths.
    """
    if not is_frozen():
        return
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except Exception:
        return
    cacert = certifi.where()
    if cacert and Path(cacert).is_file():
        os.environ["SSL_CERT_FILE"] = cacert
        os.environ.setdefault("SSL_CERT_DIR", str(Path(cacert).parent))
