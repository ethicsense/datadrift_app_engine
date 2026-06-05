# PyInstaller spec for Silhouette Outliner GUI demo packages.
# Build from project root:
#   pyinstaller packaging/silhouette_outliner.spec

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

SPEC_FILE = Path(__file__).resolve()
ROOT = SPEC_FILE.parent.parent
SRC = ROOT / "src"
ENTRY = SRC / "silhouette_outliner" / "gui" / "app.py"
CONFIGS = ROOT / "configs"
TEMPLATES = SRC / "silhouette_outliner" / "templates"
PLAYWRIGHT_BROWSERS = ROOT / "packaging" / "playwright-browsers"

datas: list[tuple[str, str]] = [
    (str(CONFIGS), "configs"),
    (str(TEMPLATES), os.path.join("silhouette_outliner", "templates")),
]
if PLAYWRIGHT_BROWSERS.is_dir():
    datas.append((str(PLAYWRIGHT_BROWSERS), "playwright-browsers"))

hiddenimports = collect_submodules("silhouette_outliner")
for pkg in ("PySide6", "playwright", "jinja2"):
    try:
        _pkg_datas, _pkg_binaries, _pkg_hidden = collect_all(pkg)
        datas.extend(_pkg_datas)
        hiddenimports.extend(_pkg_hidden)
    except Exception:
        hiddenimports.append(pkg)

block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SilhouetteOutliner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SilhouetteOutliner",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Silhouette Outliner.app",
        icon=None,
        bundle_identifier="com.datadrift.silhouette-outliner",
    )
