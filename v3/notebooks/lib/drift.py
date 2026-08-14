"""Distribution and mix drift metrics for commerce time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = e_counts / max(e_counts.sum(), 1) + eps
    a_perc = a_counts / max(a_counts.sum(), 1) + eps
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def ks_wasserstein(expected: pd.Series, actual: pd.Series) -> dict[str, float]:
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        return {"ks_stat": np.nan, "ks_pvalue": np.nan, "wasserstein": np.nan}
    ks = stats.ks_2samp(expected, actual)
    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(stats.wasserstein_distance(expected, actual)),
    }


def total_variation(p: pd.Series, q: pd.Series) -> float:
    keys = sorted(set(p.index.astype(str)) | set(q.index.astype(str)))
    p = p.reindex(keys, fill_value=0).astype(float)
    q = q.reindex(keys, fill_value=0).astype(float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    return float(0.5 * np.abs(p - q).sum())


def js_divergence(p: pd.Series, q: pd.Series, eps: float = 1e-12) -> float:
    keys = sorted(set(p.index.astype(str)) | set(q.index.astype(str)))
    p = p.reindex(keys, fill_value=0).astype(float)
    q = q.reindex(keys, fill_value=0).astype(float)
    p = p / max(p.sum(), eps) + eps
    q = q / max(q.sum(), eps) + eps
    m = 0.5 * (p + q)
    return float(0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m))


def compare_windows(
    series: pd.Series,
    baseline: slice | tuple,
    current: slice | tuple,
    mix_base: pd.Series | None = None,
    mix_cur: pd.Series | None = None,
) -> dict[str, float]:
    if isinstance(baseline, tuple):
        base = series.loc[baseline[0] : baseline[1]]
        cur = series.loc[current[0] : current[1]]
    else:
        base = series[baseline]
        cur = series[current]
    out = {
        "baseline_mean": float(base.mean()) if len(base) else np.nan,
        "current_mean": float(cur.mean()) if len(cur) else np.nan,
        "mean_rel_change": float((cur.mean() - base.mean()) / (abs(base.mean()) + 1e-9)) if len(base) and len(cur) else np.nan,
        "psi": psi(base, cur),
        **{f"dist_{k}": v for k, v in ks_wasserstein(base, cur).items()},
    }
    if mix_base is not None and mix_cur is not None:
        out["mix_tv"] = total_variation(mix_base, mix_cur)
        out["mix_js"] = js_divergence(mix_base, mix_cur)
    return out


def rolling_drift(series: pd.Series, window: int = 30, step: int = 7) -> pd.DataFrame:
    """Compare each rolling window against the first baseline window."""
    series = series.dropna().sort_index()
    if len(series) < window * 2:
        return pd.DataFrame()
    baseline = series.iloc[:window]
    rows = []
    for start in range(window, len(series) - window + 1, step):
        cur = series.iloc[start : start + window]
        row = {
            "date": series.index[start + window - 1],
            "psi": psi(baseline, cur),
            "ks_stat": ks_wasserstein(baseline, cur)["ks_stat"],
            "mean_rel_change": float((cur.mean() - baseline.mean()) / (abs(baseline.mean()) + 1e-9)),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def revenue_decomposition(
    df: pd.DataFrame,
    group_col: str,
    qty_col: str,
    revenue_col: str,
    baseline_mask: pd.Series,
    current_mask: pd.Series,
) -> pd.DataFrame:
    """Laspeyres-style volume / price / mix decomposition of GMV change.

    GMV = sum_i qty_i * price_i
    ΔGMV ≈ volume(at base mix/price) + mix(at base price) + price(at current qty)
    """
    def _agg(mask: pd.Series) -> pd.DataFrame:
        g = df.loc[mask].groupby(group_col).agg(qty=(qty_col, "sum"), gmv=(revenue_col, "sum"))
        g["price"] = g["gmv"] / g["qty"].replace(0, np.nan)
        g["share"] = g["qty"] / g["qty"].sum()
        return g

    base = _agg(baseline_mask)
    cur = _agg(current_mask)
    idx = base.index.union(cur.index)
    base = base.reindex(idx).fillna(0)
    cur = cur.reindex(idx).fillna(0)
    p0, q0, s0 = base["price"], base["qty"], base["share"]
    p1, q1, s1 = cur["price"], cur["qty"], cur["share"]
    q0_total, q1_total = q0.sum(), q1.sum()
    volume = (q1_total - q0_total) * (s0 * p0).sum()
    mix = q1_total * ((s1 - s0) * p0).sum()
    price = (q1 * (p1 - p0)).sum()
    actual = cur["gmv"].sum() - base["gmv"].sum()
    return pd.DataFrame(
        {
            "component": ["volume", "mix", "price", "actual_delta", "explained"],
            "value": [volume, mix, price, actual, volume + mix + price],
        }
    )
