"""Shared market-state features and synthetic relative-time alignment.

Public-commerce dates mapped onto 2026 survey days are synthetic. They must
not be read as contemporaneous events or as a causal identification strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .drift import js_divergence

SILHOUETTE_RUNS = Path("/Users/bhc/dev/datadrift/v2/silhouette_v2/runs")
DEFAULT_FAVORITA_ANCHOR = "2016-05-21"
DEFAULT_M5_ANCHOR = "2015-05-21"
ALIGNMENT_METHOD = "relative_elapsed_from_seasonal_anchor"
HORIZON_DAYS = 7


def entropy_from_shares(shares: dict[str, float] | None) -> float:
    if not shares:
        return np.nan
    p = np.array([float(v) for v in shares.values() if v is not None and float(v) > 0.0], dtype=float)
    if p.size == 0:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def hhi_from_shares(shares: dict[str, float] | None) -> float:
    if not shares:
        return np.nan
    p = np.array([float(v) for v in shares.values() if v is not None], dtype=float)
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p / total
    return float((p ** 2).sum())


def mix_js(prev: dict[str, float] | None, curr: dict[str, float] | None) -> float:
    if not prev or not curr:
        return np.nan
    return float(js_divergence(pd.Series(prev, dtype=float), pd.Series(curr, dtype=float)))


def profile_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "table": name,
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "null_rate": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def load_survey_dates(runs_root: Path = SILHOUETTE_RUNS) -> list[str]:
    manifest = runs_root / "analytics" / "drift_timeline" / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        dates = [str(d) for d in (payload.get("dates") or [])]
        if dates:
            return dates
    days = sorted((runs_root / "survey_days").glob("20??-??-??"))
    return [p.name for p in days if p.is_dir()]


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload


def _source_metrics(src: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    src = src or {}
    cat = src.get("category_share") or {}
    price = src.get("price_mode") or {}
    disc = src.get("discount_mode") or {}
    feat = src.get("feature_share") or {}
    shares = dict(cat.get("shares") or {})
    feat_shares = dict(feat.get("shares") or {})
    top = (cat.get("top") or [{}])
    top1_share = top[0].get("share") if top else None
    return {
        f"{prefix}_status": src.get("status"),
        f"{prefix}_n": src.get("n"),
        f"{prefix}_top1": cat.get("top1"),
        f"{prefix}_top1_share": top1_share,
        f"{prefix}_cat_entropy": entropy_from_shares(shares),
        f"{prefix}_cat_hhi": hhi_from_shares(shares),
        f"{prefix}_price_median": price.get("median"),
        f"{prefix}_price_mode_mass": price.get("mode_mass"),
        f"{prefix}_discount_median": disc.get("median"),
        f"{prefix}_discount_mode_mass": disc.get("mode_mass"),
        f"{prefix}_feature_entropy": entropy_from_shares(feat_shares),
        f"_{prefix}_cat_shares": shares,
    }


def _rank_energy_summary(products: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not products:
        return {
            f"{prefix}_rank_energy_n": 0,
            f"{prefix}_rank_energy_1d_mean": np.nan,
            f"{prefix}_rank_energy_sustained_mean": np.nan,
            f"{prefix}_momentum_span_mean": np.nan,
        }
    e1d, sustained, span = [], [], []
    for row in products:
        energy = row.get("rank_energy") or {}
        if energy.get("1d") is not None:
            e1d.append(float(energy["1d"]))
        if row.get("sustained_rank_energy") is not None:
            sustained.append(float(row["sustained_rank_energy"]))
        if row.get("momentum_span") is not None:
            span.append(float(row["momentum_span"]))
    return {
        f"{prefix}_rank_energy_n": int(len(products)),
        f"{prefix}_rank_energy_1d_mean": float(np.mean(e1d)) if e1d else np.nan,
        f"{prefix}_rank_energy_sustained_mean": float(np.mean(sustained)) if sustained else np.nan,
        f"{prefix}_momentum_span_mean": float(np.mean(span)) if span else np.nan,
    }


def load_musinsa_rank_kpis(runs_root: Path, day: str) -> dict[str, Any]:
    path = runs_root / "survey_days" / day / "channels" / "musinsa" / "active" / "analysis.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return _rank_energy_summary([], "musinsa")
    products = ((payload.get("cross_window") or {}).get("products")) or []
    out = _rank_energy_summary(products, "musinsa")
    kpis = payload.get("kpis") or {}
    out["musinsa_kpi_item_count"] = kpis.get("item_count")
    out["musinsa_kpi_median_price"] = kpis.get("median_price")
    out["musinsa_kpi_avg_discount"] = kpis.get("avg_discount_rate")
    out["musinsa_kpi_discount_pct"] = kpis.get("discount_application_pct")
    return out


def load_cm29_kpis(runs_root: Path, day: str) -> dict[str, Any]:
    path = runs_root / "survey_days" / day / "channels" / "29cm" / "active" / "channel_analysis.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"cm29_kpi_median_price": np.nan, "cm29_unique_products": np.nan}
    kpis = payload.get("kpis") or {}
    headline = payload.get("headline") or {}
    return {
        "cm29_kpi_median_price": kpis.get("median_price"),
        "cm29_kpi_brand_count": kpis.get("brand_count"),
        "cm29_unique_products": headline.get("unique_product_count"),
    }


def load_metric_point_summaries(runs_root: Path, day: str) -> dict[str, Any]:
    path = runs_root / "survey_days" / day / "fused" / "intelligence" / "drift_metric_points.json"
    payload = _read_json(path)
    rows = (payload or {}).get("rows") if isinstance(payload, dict) else None
    out: dict[str, Any] = {
        "musinsa_ranking_turnover": np.nan,
        "cm29_ranking_turnover": np.nan,
        "musinsa_review_drift": np.nan,
        "cm29_review_drift": np.nan,
        "metric_point_count": 0,
    }
    if not rows:
        return out
    out["metric_point_count"] = len(rows)
    frame = pd.DataFrame(rows)

    def _pick(metric: str, source: str, col: str = "current_value") -> float:
        sub = frame[(frame["metric"] == metric) & (frame["source"].astype(str) == source)]
        if sub.empty or col not in sub.columns:
            return np.nan
        return float(pd.to_numeric(sub[col], errors="coerce").median())

    out["musinsa_ranking_turnover"] = _pick("ranking_turnover", "musinsa")
    out["cm29_ranking_turnover"] = _pick("ranking_turnover", "29cm")
    out["musinsa_review_drift"] = _pick("review_count_distribution", "musinsa", "score")
    out["cm29_review_drift"] = _pick("review_count_distribution", "29cm", "score")
    return out


def load_silhouette_state(runs_root: Path = SILHOUETTE_RUNS) -> pd.DataFrame:
    dates = load_survey_dates(runs_root)
    visual_dir = runs_root / "survey_timeline" / "visual" / "diagnostics"
    rows: list[dict[str, Any]] = []
    prev_shares: dict[str, dict[str, float]] = {"musinsa": {}, "29cm": {}}
    prev_date: pd.Timestamp | None = None
    for day in dates:
        panel = _read_json(runs_root / "survey_days" / day / "fused" / "intelligence" / "numeric_panel.json")
        sources = (panel or {}).get("sources") if isinstance(panel, dict) else {}
        sources = sources or {}
        run_ts = pd.Timestamp(day)
        row: dict[str, Any] = {
            "run_date": run_ts,
            "weekday": int(run_ts.dayofweek),
            "month": int(run_ts.month),
            "gap_days": np.nan if prev_date is None else float((run_ts - prev_date).days),
            "panel_status": (panel or {}).get("status") if isinstance(panel, dict) else None,
        }
        mus = _source_metrics(sources.get("musinsa"), "musinsa")
        cm = _source_metrics(sources.get("29cm"), "cm29")
        mus_shares = mus.pop("_musinsa_cat_shares")
        cm_shares = cm.pop("_cm29_cat_shares")
        mus["musinsa_mix_js"] = mix_js(prev_shares.get("musinsa"), mus_shares)
        cm["cm29_mix_js"] = mix_js(prev_shares.get("29cm"), cm_shares)
        row.update(mus)
        row.update(cm)
        row["_musinsa_cat_shares"] = mus_shares
        row["_cm29_cat_shares"] = cm_shares
        if mus_shares:
            prev_shares["musinsa"] = mus_shares
        if cm_shares:
            prev_shares["29cm"] = cm_shares
        vis = _read_json(visual_dir / f"{day}.json") or {}
        vis = vis if isinstance(vis, dict) else {}
        contrast = vis.get("channel_contrast") or []
        js_vals = [float(item.get("js") or 0.0) for item in contrast if isinstance(item, dict)]
        nvr = vis.get("new_vs_retained") or {}
        new_n = float(nvr.get("new_count") or 0.0)
        ret_n = float(nvr.get("retained_count") or 0.0)
        churn_n = float(nvr.get("churned_count") or 0.0)
        row.update(
            {
                "visual_js_mean": float(np.mean(js_vals)) if js_vals else np.nan,
                "visual_js_max": float(np.max(js_vals)) if js_vals else np.nan,
                "visual_new_rate": new_n / max(new_n + ret_n, 1.0),
                "visual_churn_rate": churn_n / max(churn_n + ret_n, 1.0),
                "visual_mean_nn_new": nvr.get("mean_nn_new"),
            }
        )
        row.update(load_musinsa_rank_kpis(runs_root, day))
        row.update(load_cm29_kpis(runs_root, day))
        row.update(load_metric_point_summaries(runs_root, day))
        rows.append(row)
        prev_date = run_ts
    out = pd.DataFrame(rows).sort_values("run_date").reset_index(drop=True)
    out["elapsed_days"] = (out["run_date"] - out["run_date"].iloc[0]).dt.days.astype(float)
    # Prefer KPI price/discount when numeric_panel is empty.
    if "musinsa_kpi_median_price" in out.columns:
        out["musinsa_price_median"] = out["musinsa_price_median"].fillna(out["musinsa_kpi_median_price"])
    if "musinsa_kpi_avg_discount" in out.columns:
        out["musinsa_discount_median"] = out["musinsa_discount_median"].fillna(out["musinsa_kpi_avg_discount"])
    if "cm29_kpi_median_price" in out.columns:
        out["cm29_price_median"] = out["cm29_price_median"].fillna(out["cm29_kpi_median_price"])
    drift = load_drift_headlines(runs_root)
    if not drift.empty:
        out = out.merge(drift, on="run_date", how="left")
    return out


def load_drift_headlines(runs_root: Path = SILHOUETTE_RUNS) -> pd.DataFrame:
    path = runs_root / "analytics" / "drift_timeline" / "daily_scores.json"
    if not path.is_file():
        return pd.DataFrame()
    payload = json.loads(path.read_text())
    raw = payload.get("rows") if isinstance(payload, dict) else payload
    df = pd.DataFrame(raw or [])
    if df.empty:
        return df
    df["run_date"] = pd.to_datetime(df["run_date"])
    if "chart_role" in df.columns:
        df = df[df["chart_role"].fillna("headline") == "headline"]
    if "coverage" in df.columns:
        overall = df[df["coverage"].fillna("overall") == "overall"]
        df = overall if not overall.empty else df
    parts = []
    for axis, source in (("statistical", "musinsa"), ("text", "all"), ("visual", "all")):
        sub = df[df["drift_axis"] == axis]
        if sub.empty:
            continue
        if "source" in sub.columns and (sub["source"].astype(str) == source).any():
            sub = sub[sub["source"].astype(str) == source]
        grp = sub.groupby("run_date", dropna=False)["score"].median().rename(f"drift_{axis}")
        parts.append(grp)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).reset_index()
    return out


def _ensure_date_index(df: pd.DataFrame, value_name: str | None = None) -> pd.DataFrame:
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"])
        work = work.set_index("date")
    work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    if value_name and work.shape[1] == 1:
        work = work.rename(columns={work.columns[0]: value_name})
    return work


def mix_entropy_series(mix: pd.DataFrame) -> pd.Series:
    work = mix.copy()
    work.index = pd.to_datetime(work.index)
    shares = work.div(work.sum(axis=1).replace(0, np.nan), axis=0)
    ent = shares.apply(lambda row: entropy_from_shares(row.dropna().to_dict()), axis=1)
    ent.name = "mix_entropy"
    return ent


def mix_js_series(mix: pd.DataFrame) -> pd.Series:
    work = mix.copy()
    work.index = pd.to_datetime(work.index)
    shares = work.div(work.sum(axis=1).replace(0, np.nan), axis=0)
    vals: list[float] = []
    prev = None
    for _, row in shares.iterrows():
        curr = row.dropna().to_dict()
        vals.append(mix_js(prev, curr) if prev is not None else np.nan)
        prev = curr
    return pd.Series(vals, index=shares.index, name="mix_js")


def load_public_state(
    favorita: dict[str, pd.DataFrame], m5: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fav_total = _ensure_date_index(favorita["daily_total"], "sales")
    fav_promo = _ensure_date_index(favorita["daily_promo_rate"], "promo_rate")
    tx_key = "tx_daily" if "tx_daily" in favorita else "transactions_daily"
    fav_tx = _ensure_date_index(favorita[tx_key])
    if "transactions" not in fav_tx.columns:
        fav_tx = fav_tx.rename(columns={fav_tx.columns[0]: "transactions"})
    fav_family = _ensure_date_index(favorita["daily_family"])
    fav_store = _ensure_date_index(favorita["daily_store"]) if "daily_store" in favorita else None
    fav = pd.DataFrame(index=fav_total.index)
    fav["sales"] = fav_total["sales"]
    fav["sales_growth"] = np.log1p(fav["sales"]).diff()
    fav["sales_vol_7"] = fav["sales_growth"].rolling(7, min_periods=3).std()
    fav["promo_rate"] = fav_promo.reindex(fav.index)["promo_rate"]
    fav["transactions"] = fav_tx.reindex(fav.index)["transactions"]
    fav["family_entropy"] = mix_entropy_series(fav_family).reindex(fav.index)
    fav["family_mix_js"] = mix_js_series(fav_family).reindex(fav.index)
    fav["family_hhi"] = fav_family.div(fav_family.sum(axis=1).replace(0, np.nan), axis=0).pow(2).sum(axis=1)
    if fav_store is not None:
        fav["store_entropy"] = mix_entropy_series(fav_store).reindex(fav.index)

    m5_total = _ensure_date_index(m5["daily_total"], "units")
    m5_cat = _ensure_date_index(m5["daily_cat"])
    m5_state = _ensure_date_index(m5["daily_state"])
    m5_df = pd.DataFrame(index=m5_total.index)
    m5_df["units"] = m5_total["units"]
    m5_df["units_growth"] = np.log1p(m5_df["units"]).diff()
    m5_df["units_vol_7"] = m5_df["units_growth"].rolling(7, min_periods=3).std()
    m5_df["cat_entropy"] = mix_entropy_series(m5_cat).reindex(m5_df.index)
    m5_df["cat_mix_js"] = mix_js_series(m5_cat).reindex(m5_df.index)
    m5_df["state_entropy"] = mix_entropy_series(m5_state).reindex(m5_df.index)
    m5_df["state_mix_js"] = mix_js_series(m5_state).reindex(m5_df.index)
    return fav, m5_df


def asof_lookup(public: pd.DataFrame, source_date: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, dict[str, Any]]:
    hist = public.loc[:source_date]
    if hist.empty:
        return pd.NaT, {col: np.nan for col in public.columns}
    used = hist.index[-1]
    return used, hist.iloc[-1].to_dict()


def align_public(
    survey: pd.DataFrame,
    public: pd.DataFrame,
    *,
    prefix: str,
    anchor: str,
    offset_days: int = 0,
) -> pd.DataFrame:
    t0 = pd.Timestamp(survey["run_date"].iloc[0])
    anchor_ts = pd.Timestamp(anchor) + pd.Timedelta(days=int(offset_days))
    rows = []
    for rec in survey.itertuples(index=False):
        elapsed = int((pd.Timestamp(rec.run_date) - t0).days)
        requested = anchor_ts + pd.Timedelta(days=elapsed)
        used, values = asof_lookup(public, requested)
        row = {
            "run_date": pd.Timestamp(rec.run_date),
            f"source_date_{prefix}": used,
            f"requested_source_date_{prefix}": requested,
            f"{prefix}_asof_lag_days": np.nan if pd.isna(used) else float((requested - used).days),
        }
        for key, val in values.items():
            row[f"{prefix}_{key}"] = val
        rows.append(row)
    out = pd.DataFrame(rows)
    out["alignment_offset"] = int(offset_days)
    out["alignment_method"] = ALIGNMENT_METHOD
    out["synthetic_alignment"] = True
    return out


def build_joint_panel(
    silhouette: pd.DataFrame,
    favorita_state: pd.DataFrame,
    m5_state: pd.DataFrame,
    *,
    favorita_anchor: str = DEFAULT_FAVORITA_ANCHOR,
    m5_anchor: str = DEFAULT_M5_ANCHOR,
    offset_days: int = 0,
) -> pd.DataFrame:
    fav_aln = align_public(
        silhouette, favorita_state, prefix="fav", anchor=favorita_anchor, offset_days=offset_days
    )
    m5_aln = align_public(silhouette, m5_state, prefix="m5", anchor=m5_anchor, offset_days=offset_days)
    joint = silhouette.merge(fav_aln, on="run_date", how="left")
    overlap = {"alignment_offset", "alignment_method", "synthetic_alignment"}
    m5_keep = [c for c in m5_aln.columns if c != "run_date" and c not in overlap]
    joint = joint.merge(m5_aln[["run_date", *m5_keep]], on="run_date", how="left")
    joint["synthetic_date"] = joint["run_date"]
    joint["alignment_offset"] = int(offset_days)
    joint["alignment_method"] = ALIGNMENT_METHOD
    joint["synthetic_alignment"] = True
    joint["favorita_anchor"] = str(pd.Timestamp(favorita_anchor).date())
    joint["m5_anchor"] = str(pd.Timestamp(m5_anchor).date())
    return joint.sort_values("run_date").reset_index(drop=True)


def permute_public_block(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle public-feature rows while keeping survey dates. Destroys temporal alignment."""
    work = panel.copy()
    public_cols = [c for c in work.columns if c.startswith("fav_") or c.startswith("m5_")]
    if not public_cols:
        return work
    order = rng.permutation(len(work))
    work[public_cols] = work.iloc[order][public_cols].to_numpy()
    work["alignment_method"] = "permutation_placebo"
    work["synthetic_alignment"] = True
    return work


def attach_horizon_targets(panel: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    work = panel.sort_values("run_date").reset_index(drop=True)
    dates = work["run_date"].tolist()
    shares = work["_musinsa_cat_shares"] if "_musinsa_cat_shares" in work.columns else None
    top1 = work.get("musinsa_top1_share")
    turnover = work.get("musinsa_ranking_turnover")
    drift_stat = work.get("drift_statistical", pd.Series(np.nan, index=work.index))
    sales = work.get("fav_sales")
    units = work.get("m5_units")
    y_mix, y_top1, y_turn, y_drift, y_sales, y_units, target_dates = [], [], [], [], [], [], []
    for i, t in enumerate(dates):
        horizon = t + pd.Timedelta(days=horizon_days)
        later = [j for j, d in enumerate(dates) if d >= horizon]
        if not later:
            y_mix.append(np.nan)
            y_top1.append(np.nan)
            y_turn.append(np.nan)
            y_drift.append(np.nan)
            y_sales.append(np.nan)
            y_units.append(np.nan)
            target_dates.append(pd.NaT)
            continue
        j = later[0]
        if shares is not None:
            y_mix.append(mix_js(shares.iloc[i], shares.iloc[j]))
        else:
            y_mix.append(np.nan)
        if top1 is not None and pd.notna(top1.iloc[j]) and pd.notna(top1.iloc[i]):
            y_top1.append(float(top1.iloc[j] - top1.iloc[i]))
        else:
            y_top1.append(np.nan)
        y_turn.append(turnover.iloc[j] if turnover is not None else np.nan)
        y_drift.append(drift_stat.iloc[j] if drift_stat is not None else np.nan)
        if sales is not None and pd.notna(sales.iloc[i]) and pd.notna(sales.iloc[j]) and float(sales.iloc[i]) > 0:
            y_sales.append(float(np.log1p(sales.iloc[j]) - np.log1p(sales.iloc[i])))
        else:
            y_sales.append(np.nan)
        if units is not None and pd.notna(units.iloc[i]) and pd.notna(units.iloc[j]) and float(units.iloc[i]) > 0:
            y_units.append(float(np.log1p(units.iloc[j]) - np.log1p(units.iloc[i])))
        else:
            y_units.append(np.nan)
        target_dates.append(dates[j])
    work["horizon_days"] = horizon_days
    work["target_date"] = target_dates
    work["y_future_rank_mix_change"] = y_mix
    work["y_future_top1_delta"] = y_top1
    work["y_future_ranking_turnover"] = y_turn
    work["y_future_drift_score"] = y_drift
    work["y_future_sales_growth"] = y_sales
    work["y_future_units_growth"] = y_units
    # Mix JS is sparse early; ranking turnover at the horizon is the fallback mix-change proxy.
    work["y_rank_mix_proxy"] = work["y_future_rank_mix_change"].fillna(work["y_future_ranking_turnover"])
    work["label_ready"] = work["target_date"].notna()
    return work


SILHOUETTE_FEATURES = [
    "weekday",
    "month",
    "gap_days",
    "elapsed_days",
    "musinsa_n",
    "musinsa_top1_share",
    "musinsa_cat_entropy",
    "musinsa_cat_hhi",
    "musinsa_mix_js",
    "musinsa_price_median",
    "musinsa_discount_median",
    "musinsa_rank_energy_1d_mean",
    "musinsa_rank_energy_sustained_mean",
    "musinsa_momentum_span_mean",
    "musinsa_ranking_turnover",
    "musinsa_review_drift",
    "cm29_n",
    "cm29_top1_share",
    "cm29_cat_entropy",
    "cm29_mix_js",
    "cm29_price_median",
    "cm29_ranking_turnover",
    "cm29_review_drift",
    "visual_js_mean",
    "visual_new_rate",
    "visual_churn_rate",
    "visual_mean_nn_new",
    "drift_statistical",
    "drift_text",
    "drift_visual",
]

PUBLIC_FEATURES = [
    "fav_sales_growth",
    "fav_sales_vol_7",
    "fav_promo_rate",
    "fav_family_entropy",
    "fav_family_mix_js",
    "fav_family_hhi",
    "fav_store_entropy",
    "m5_units_growth",
    "m5_units_vol_7",
    "m5_cat_entropy",
    "m5_cat_mix_js",
    "m5_state_entropy",
]

TARGETS = [
    "y_rank_mix_proxy",
    "y_future_drift_score",
    "y_future_sales_growth",
]


def feature_matrix(panel: pd.DataFrame, include_public: bool = True) -> tuple[pd.DataFrame, list[str]]:
    work = panel.copy()
    cols = list(SILHOUETTE_FEATURES)
    if include_public:
        cols = cols + list(PUBLIC_FEATURES)
    present = [c for c in cols if c in work.columns]
    x = work[present].apply(pd.to_numeric, errors="coerce")
    return x, present


def rolling_origin_indices(n: int, min_train: int = 12) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for i in range(min_train, n):
        yield np.arange(0, i), np.array([i])


SILHOUETTE_COLUMN_GLOSSARY: list[dict[str, str]] = [
    {"column": "run_date", "group": "달력", "meaning": "직접 수집 관측일 (KST 기준 survey day)", "source": "manifest / survey_days", "note": "42개 비연속 날짜"},
    {"column": "weekday", "group": "달력", "meaning": "요일 (월=0 … 일=6)", "source": "run_date", "note": "nunique=6이면 일요일이 없는 수집 패턴"},
    {"column": "month", "group": "달력", "meaning": "월 (5–8)", "source": "run_date", "note": ""},
    {"column": "gap_days", "group": "달력", "meaning": "직전 관측일과의 실제 간격(일)", "source": "run_date diff", "note": "첫 날은 결측"},
    {"column": "elapsed_days", "group": "달력", "meaning": "첫 관측일(2026-05-21)부터의 경과 일수. 공개 데이터 상대시간 정렬에 사용", "source": "run_date", "note": ""},
    {"column": "panel_status", "group": "품질", "meaning": "numeric_panel 전체 상태 (ok / insufficient_n)", "source": "numeric_panel.json", "note": "한쪽 소스만 살아도 ok가 될 수 있음"},
    {"column": "musinsa_status", "group": "품질", "meaning": "무신사 numeric_panel 상태", "source": "numeric_panel.json", "note": "ok는 2026-07-21부터 19일"},
    {"column": "cm29_status", "group": "품질", "meaning": "29CM numeric_panel 상태", "source": "numeric_panel.json", "note": "ok는 2026-06-16부터"},
    {"column": "musinsa_n", "group": "규모", "meaning": "무신사 numeric_panel 표본 수 (카테고리 share 분모)", "source": "numeric_panel.json", "note": "0이면 카테고리 믹스 컬럼도 비어 있음"},
    {"column": "cm29_n", "group": "규모", "meaning": "29CM numeric_panel 표본 수", "source": "numeric_panel.json", "note": ""},
    {"column": "musinsa_top1", "group": "구성", "meaning": "무신사 1위 카테고리 이름", "source": "numeric_panel.category_share", "note": "panel 부족 구간 결측"},
    {"column": "musinsa_top1_share", "group": "구성", "meaning": "무신사 1위 카테고리 점유율 (0–1)", "source": "numeric_panel.category_share", "note": ""},
    {"column": "musinsa_cat_entropy", "group": "구성", "meaning": "무신사 카테고리 share 엔트로피. 높을수록 구성이 분산", "source": "category_share.shares", "note": ""},
    {"column": "musinsa_cat_hhi", "group": "구성", "meaning": "무신사 카테고리 HHI (점유율 제곱합). 높을수록 소수 카테고리 집중", "source": "category_share.shares", "note": ""},
    {"column": "musinsa_mix_js", "group": "구성", "meaning": "직전 유효 관측 대비 무신사 카테고리 믹스 Jensen–Shannon. 구성 교체 강도", "source": "consecutive category shares", "note": "첫 유효일·panel 부족 구간 결측"},
    {"column": "cm29_top1", "group": "구성", "meaning": "29CM 1위 카테고리 이름", "source": "numeric_panel.category_share", "note": "관측 구간에서는 거의 항상 같은 라벨일 수 있음"},
    {"column": "cm29_top1_share", "group": "구성", "meaning": "29CM 1위 카테고리 점유율", "source": "numeric_panel.category_share", "note": ""},
    {"column": "cm29_cat_entropy", "group": "구성", "meaning": "29CM 카테고리 share 엔트로피", "source": "category_share.shares", "note": ""},
    {"column": "cm29_cat_hhi", "group": "구성", "meaning": "29CM 카테고리 HHI", "source": "category_share.shares", "note": ""},
    {"column": "cm29_mix_js", "group": "구성", "meaning": "직전 유효 관측 대비 29CM 카테고리 믹스 JS", "source": "consecutive category shares", "note": ""},
    {"column": "musinsa_price_median", "group": "가격·할인", "meaning": "무신사 가격 중앙값. panel이 비면 KPI median으로 보완", "source": "numeric_panel.price_mode / analysis.kpis", "note": ""},
    {"column": "musinsa_price_mode_mass", "group": "가격·할인", "meaning": "무신사 가격 최빈 구간에 모인 비중", "source": "numeric_panel.price_mode", "note": "n=0이어도 mode_mass=0으로 채워질 수 있음"},
    {"column": "musinsa_discount_median", "group": "가격·할인", "meaning": "무신사 할인율 중앙값(또는 KPI 평균 할인). 단위는 원본 필드 그대로 (%)", "source": "numeric_panel.discount_mode / analysis.kpis", "note": ""},
    {"column": "musinsa_discount_mode_mass", "group": "가격·할인", "meaning": "무신사 할인 최빈 구간 비중", "source": "numeric_panel.discount_mode", "note": ""},
    {"column": "musinsa_feature_entropy", "group": "시각 속성", "meaning": "무신사 패션 피처(색/실루엣 등) share 엔트로피", "source": "numeric_panel.feature_share", "note": "카테고리 panel과 달리 조기부터 채워짐"},
    {"column": "cm29_price_median", "group": "가격·할인", "meaning": "29CM 가격 중앙값. panel이 비면 channel KPI로 보완", "source": "numeric_panel / channel_analysis.kpis", "note": ""},
    {"column": "cm29_price_mode_mass", "group": "가격·할인", "meaning": "29CM 가격 최빈 구간 비중", "source": "numeric_panel.price_mode", "note": ""},
    {"column": "cm29_discount_median", "group": "가격·할인", "meaning": "29CM 할인율 중앙값", "source": "numeric_panel.discount_mode", "note": "29CM panel이 할인을 안 내리면 전 구간 결측"},
    {"column": "cm29_discount_mode_mass", "group": "가격·할인", "meaning": "29CM 할인 최빈 구간 비중", "source": "numeric_panel.discount_mode", "note": "값이 있어도 nunique=1이면 상수(정보량 없음)"},
    {"column": "cm29_feature_entropy", "group": "시각 속성", "meaning": "29CM 패션 피처 share 엔트로피", "source": "numeric_panel.feature_share", "note": ""},
    {"column": "musinsa_rank_energy_n", "group": "순위", "meaning": "무신사 교차윈도우 조인 상품 수 (rank_energy 집계 분모)", "source": "analysis.json cross_window.products", "note": ""},
    {"column": "musinsa_rank_energy_1d_mean", "group": "순위", "meaning": "일간 창 rank_energy=(limit+1−rank)/limit 의 조인 상품 평균. 1에 가까울수록 일간 상위", "source": "analysis.json rank_energy.1d", "note": "조인 집합이 1d 랭킹 전체를 거의 포함하면 평균이 상수(~0.505)가 됨. 변별력 없음"},
    {"column": "musinsa_rank_energy_sustained_mean", "group": "순위", "meaning": "sustained_rank_energy 평균. 여러 윈도우 rank_energy를 √days 가중 합산한 누적 강도", "source": "analysis.json sustained_rank_energy", "note": "1d 평균과 달리 날짜마다 변함"},
    {"column": "musinsa_momentum_span_mean", "group": "순위", "meaning": "momentum_span=rank_energy(1d)−rank_energy(1m) 평균. +면 최근 펄스가 강함", "source": "analysis.json momentum_span", "note": ""},
    {"column": "musinsa_kpi_item_count", "group": "규모", "meaning": "무신사 리포트 KPI 아이템 수 (보통 차트 limit)", "source": "analysis.json kpis", "note": "nunique=1이면 고정 리포트 크기"},
    {"column": "musinsa_kpi_median_price", "group": "가격·할인", "meaning": "무신사 리포트 KPI 가격 중앙값 (panel 보완 원천)", "source": "analysis.json kpis", "note": ""},
    {"column": "musinsa_kpi_avg_discount", "group": "가격·할인", "meaning": "무신사 리포트 평균 할인율", "source": "analysis.json kpis", "note": ""},
    {"column": "musinsa_kpi_discount_pct", "group": "가격·할인", "meaning": "무신사 할인 적용 상품 비율", "source": "analysis.json kpis", "note": ""},
    {"column": "cm29_kpi_median_price", "group": "가격·할인", "meaning": "29CM 채널 KPI 가격 중앙값", "source": "channel_analysis.json kpis", "note": "29CM 수집 시작 전 결측"},
    {"column": "cm29_kpi_brand_count", "group": "규모", "meaning": "29CM 채널 KPI 브랜드 수", "source": "channel_analysis.json kpis", "note": ""},
    {"column": "cm29_unique_products", "group": "규모", "meaning": "29CM 헤드라인 unique product 수", "source": "channel_analysis.json headline", "note": ""},
    {"column": "musinsa_ranking_turnover", "group": "순위", "meaning": "무신사 순위 집합 교체율 (1−Jaccard). 높을수록 랭킹 멤버가 바뀜", "source": "drift_metric_points ranking_turnover", "note": "첫날은 baseline 부족으로 결측"},
    {"column": "cm29_ranking_turnover", "group": "순위", "meaning": "29CM 순위 집합 교체율", "source": "drift_metric_points ranking_turnover", "note": "29CM 수집 전·이벤트 없는 날 결측"},
    {"column": "musinsa_review_drift", "group": "리뷰", "meaning": "무신사 리뷰 수 분포 드리프트 점수 (review_count_distribution score)", "source": "drift_metric_points", "note": "nunique가 작으면 거의 on/off 알람"},
    {"column": "cm29_review_drift", "group": "리뷰", "meaning": "29CM 리뷰 수 분포 드리프트 점수", "source": "drift_metric_points", "note": ""},
    {"column": "metric_point_count", "group": "품질", "meaning": "그날 drift_metric_points 행 수 (디텍터 산출량)", "source": "drift_metric_points.json", "note": ""},
    {"column": "visual_js_mean", "group": "비주얼", "meaning": "채널 대비 visual embedding 분포 JS 평균", "source": "visual/diagnostics channel_contrast", "note": "contrast가 빈 날 결측"},
    {"column": "visual_js_max", "group": "비주얼", "meaning": "채널 대비 visual JS 최댓값", "source": "visual/diagnostics channel_contrast", "note": ""},
    {"column": "visual_new_rate", "group": "비주얼", "meaning": "신규 상품 비율 new/(new+retained)", "source": "visual/diagnostics new_vs_retained", "note": "insufficient여도 0으로 떨어질 수 있음"},
    {"column": "visual_churn_rate", "group": "비주얼", "meaning": "이탈 상품 비율 churned/(churned+retained)", "source": "visual/diagnostics new_vs_retained", "note": ""},
    {"column": "visual_mean_nn_new", "group": "비주얼", "meaning": "신규 상품의 최근접 이웃 거리 평균. 클수록 기존 구색과 시각적으로 멂", "source": "visual/diagnostics new_vs_retained", "note": "첫날 결측 가능"},
    {"column": "drift_statistical", "group": "드리프트", "meaning": "통계 축 headline drift score (무신사 overall 중앙값)", "source": "daily_scores.json", "note": "ranking_turnover 등과 일부 겹칠 수 있음"},
    {"column": "drift_text", "group": "드리프트", "meaning": "텍스트 축 headline drift score (source=all)", "source": "daily_scores.json", "note": ""},
    {"column": "drift_visual", "group": "드리프트", "meaning": "비주얼 축 headline drift score (source=all)", "source": "daily_scores.json", "note": ""},
]


def silhouette_column_catalog(profile: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog = pd.DataFrame(SILHOUETTE_COLUMN_GLOSSARY)
    if profile is None or profile.empty:
        return catalog
    keep = ["column", "dtype", "non_null", "null_rate", "nunique"]
    stats = profile.loc[:, [c for c in keep if c in profile.columns]]
    return stats.merge(catalog, on="column", how="left")
