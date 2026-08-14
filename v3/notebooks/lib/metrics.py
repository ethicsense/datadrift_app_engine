"""Commerce KPI aggregation and mix helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def commerce_daily_metrics(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    order_col: str | None = None,
    qty_col: str | None = None,
    customer_col: str | None = None,
    cancel_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate line-level commerce data to daily KPI series."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()
    grouped = work.groupby(date_col, dropna=True)

    out = pd.DataFrame(
        {
            "gmv": grouped[revenue_col].sum(),
            "lines": grouped.size(),
        }
    )
    if qty_col and qty_col in work.columns:
        out["units"] = grouped[qty_col].sum()
    if order_col and order_col in work.columns:
        out["orders"] = grouped[order_col].nunique()
        out["aov"] = out["gmv"] / out["orders"].replace(0, np.nan)
    else:
        out["aov"] = np.nan
    if customer_col and customer_col in work.columns:
        out["unique_customers"] = grouped[customer_col].nunique()
    if cancel_col and cancel_col in work.columns:
        out["cancel_rate"] = grouped[cancel_col].mean()
    out["asp"] = out["gmv"] / out.get("units", out["lines"]).replace(0, np.nan)
    out.index.name = "date"
    return out.sort_index()


def mix_shares(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    value_col: str,
    freq: str = "MS",
    top_n: int = 8,
) -> pd.DataFrame:
    """Period x group value shares. Long-tail groups are collapsed to Other."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["period"] = work[date_col].dt.to_period(freq[0] if freq in {"MS", "M", "W", "D"} else "M").dt.to_timestamp()
    totals = work.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    keep = set(totals.head(top_n).index)
    work[group_col] = work[group_col].where(work[group_col].isin(keep), "Other")
    pivot = (
        work.groupby(["period", group_col])[value_col]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    shares = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)
    return shares


def entity_churn(
    df: pd.DataFrame,
    date_col: str,
    entity_col: str,
    baseline_mask: pd.Series,
    current_mask: pd.Series,
) -> dict[str, float]:
    base = set(df.loc[baseline_mask, entity_col].dropna().astype(str).unique())
    cur = set(df.loc[current_mask, entity_col].dropna().astype(str).unique())
    retained = base & cur
    new = cur - base
    dropped = base - cur
    return {
        "baseline_entities": float(len(base)),
        "current_entities": float(len(cur)),
        "retained": float(len(retained)),
        "new": float(len(new)),
        "dropped": float(len(dropped)),
        "new_ratio": float(len(new) / max(len(cur), 1)),
        "dropped_ratio": float(len(dropped) / max(len(base), 1)),
        "jaccard": float(len(retained) / max(len(base | cur), 1)),
    }
