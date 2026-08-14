"""Shared matplotlib helpers for commerce notebooks."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from IPython.display import Markdown, display
from matplotlib import font_manager


def chart_note(md: str) -> None:
    """Render interpretation text in the same output stream as the following plot."""
    display(Markdown(md.strip()))


def configure_matplotlib() -> str:
    """Set readable defaults and pick an available CJK-capable font."""
    candidates = [
        "AppleGothic",
        "Apple SD Gothic Neo",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")

    plt.rcParams.update(
        {
            "figure.figsize": (12, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 120,
            "axes.unicode_minus": False,
            "font.family": chosen,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.2,
        }
    )
    return chosen


def format_date_axis(ax, locator: str = "auto") -> None:
    """Ensure datetime x-axis shows readable tick labels."""
    if locator == "month":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif locator == "year":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        locator_obj = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator_obj)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator_obj))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")


def finish_figure(fig, axes=None, date_axis: bool = False, locator: str = "auto") -> None:
    """Apply shared layout polish and optionally format date axes."""
    if axes is None:
        axes = fig.axes
    elif not isinstance(axes, Iterable):
        axes = [axes]
    axes = list(axes)
    if date_axis:
        # For shared-x stacks, only the bottom axis needs visible labels.
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("")
        if axes:
            format_date_axis(axes[-1], locator=locator)
            axes[-1].set_xlabel("date")
    fig.tight_layout()
    plt.show()


def plot_metric_stack(
    series_map: dict[str, pd.Series],
    *,
    title_map: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    rolling: int = 7,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot multiple daily metrics with raw + rolling mean for readability."""
    title_map = title_map or {}
    colors = colors or {}
    n = len(series_map)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=figsize or (13, max(2.6 * n, 6)),
        sharex=True,
        constrained_layout=False,
    )
    if n == 1:
        axes = [axes]
    for ax, (name, series) in zip(axes, series_map.items()):
        s = series.dropna().sort_index()
        color = colors.get(name, None)
        ax.plot(s.index, s.values, color=color, alpha=0.25, lw=0.8, label="daily")
        if rolling and len(s) >= rolling:
            smooth = s.rolling(rolling, min_periods=max(1, rolling // 2)).mean()
            ax.plot(
                smooth.index,
                smooth.values,
                color=color,
                lw=1.8,
                label=f"{rolling}d MA",
            )
        ax.set_title(title_map.get(name, name))
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    finish_figure(fig, axes, date_axis=True)


def style_area(ax, title: str, ylabel: str = "share") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
