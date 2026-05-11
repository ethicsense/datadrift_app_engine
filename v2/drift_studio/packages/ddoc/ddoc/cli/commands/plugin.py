"""Plugin management commands"""
from __future__ import annotations

from typing import Optional

import typer
from rich import print

from .utils import _pretty


def plugin_list_command():
    """
    List all installed plugins (without loading heavy dependencies).
    
    This command uses entry point metadata only, avoiding expensive imports
    like PyTorch and scikit-learn for fast execution.
    """
    print("[bold cyan]🔌 Installed Plugins:[/bold cyan]")

    # Use importlib.metadata to read entry points without loading plugins
    import importlib.metadata

    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            entry_points = eps.select(group="ddoc")
        else:
            entry_points = eps.get("ddoc", [])
    except Exception as e:
        print(f"[red]❌ Failed to read plugins: {e}[/red]")
        return
    
    # Convert to list for compatibility with older Python versions
    plugins = list(entry_points) if hasattr(entry_points, '__iter__') else entry_points
    
    if plugins:
        print(f"\n[bold]Found {len(plugins)} plugins:[/bold]")
        print("-" * 60)
        print(f"{'Name':<24} {'Target':<30} {'Dist':<20}")
        print("-" * 60)
        
        for ep in plugins:
            name = ep.name
            target = getattr(ep, "value", "") or ""
            dist = "unknown"
            try:
                # Python 3.10+ EntryPoint has .dist sometimes (not guaranteed)
                d = getattr(ep, "dist", None)
                if d is not None:
                    dist = f"{d.metadata['Name']}=={d.version}"
            except Exception:
                dist = "unknown"

            if len(target) > 29:
                target = target[:26] + "..."

            if len(dist) > 19:
                dist = dist[:16] + "..."

            print(f"{name:<24} {target:<30} {dist:<20}")
        
        print("-" * 60)
        print("\n[dim]💡 Tip: Use 'ddoc plugin info <name>' for details[/dim]")
    else:
        print("  No plugins installed.")


def plugin_info_command(
    plugin_name: Optional[str] = typer.Argument(None, help="Plugin entry point name (e.g., ddoc_text)")
):
    """
    Show detailed information about plugins.
    
    Examples:
        ddoc plugin info
        ddoc plugin info ddoc_vision
    """
    print("[bold magenta]🔍 Plugin Information[/bold magenta]")
    
    import importlib.metadata

    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            entry_points = list(eps.select(group="ddoc"))
        else:
            entry_points = list(eps.get("ddoc", []))
    except Exception as e:
        print(f"[red]❌ Failed to read plugins: {e}[/red]")
        return

    if not entry_points:
        print("[yellow]No plugins installed.[/yellow]")
        return

    if not plugin_name:
        plugin_list_command()
        return

    ep = next((p for p in entry_points if p.name == plugin_name), None)
    if not ep:
        print(f"[red]❌ Plugin '{plugin_name}' not found.[/red]")
        return

    payload = {
        "name": ep.name,
        "target": getattr(ep, "value", None),
        "group": "ddoc",
    }
    try:
        d = getattr(ep, "dist", None)
        if d is not None:
            payload["distribution"] = {
                "name": d.metadata.get("Name"),
                "version": getattr(d, "version", None),
            }
    except Exception:
        pass

    print(_pretty(payload))

