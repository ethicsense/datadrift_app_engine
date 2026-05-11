"""ddoc CLI command registration (minimal).

의도:
- drift_studio(v2) 구동에 필요한 최소 기능만 유지합니다.
- 남기는 명령: analyze, plugin
  (그 외 init/add/snapshot/exp/legacy/vis 등은 제거 대상으로 전환)
"""

from __future__ import annotations

import typer

from .analyze import analyze_drift_command, analyze_eda_command
from .plugin import plugin_info_command, plugin_list_command

analyze_app = typer.Typer(help="Analysis commands (eda, drift)")
plugin_app = typer.Typer(help="Plugin management commands (list, info)")


def register(app: typer.Typer) -> None:
    """Register minimal command set to the root app."""
    analyze_app.command("eda", help="Run EDA for a dataset (ddoc.yaml required)")(analyze_eda_command)
    analyze_app.command("drift", help="Detect drift between two datasets (ddoc.yaml required)")(analyze_drift_command)
    app.add_typer(analyze_app, name="analyze")

    plugin_app.command("list", help="List installed ddoc plugins (entry points)")(plugin_list_command)
    plugin_app.command("info", help="Show details for an installed plugin")(plugin_info_command)
    app.add_typer(plugin_app, name="plugin")


__all__ = ["register"]

