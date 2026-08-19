"""Generate notebooks/04_synthetic_joint_market_state_experiment.ipynb. Not a runtime dependency."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).with_name("04_synthetic_joint_market_state_experiment.ipynb")


def md(source: str) -> dict:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> dict:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (v3/.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = []

    cells.append(
        md(
            """# 합성 시장상태 모델 실험

**날짜:** 2026-08-18  
**실행 환경:** `v3/.venv` (Python 3.12)  
**헬퍼:** `lib/joint_market_state.py`, `lib/kaggle_data.py`, `lib/drift.py`

이 노트북은 직접 수집한 42개 관측일(무신사·29CM)과 Favorita·M5 공개 수요 시계열을 **상대시간으로 합성 정렬**한 뒤, 전처리 형상·모델 입출력·상위 잠재 상관을 관측한다.

> **합성 정렬이지 인과 검증이 아니다.** Favorita(2013–2017)와 M5(2011–2016)의 날짜를 2026 수집일에 맞춘 것은 파이프라인·상관 민감도 샌드박스다. 같은 현실의 동시 변화를 증명하지 않는다. 모든 결합 행은 `synthetic_alignment=true`를 가진다.
"""
        )
    )

    cells.append(
        md(
            """## 0. 환경

카테고리 이름을 억지로 매핑하지 않는다. 결합하는 것은 성장률·변동성·집중도·mix drift처럼 도메인 공통인 상위 시장상태 지표뿐이다.
"""
        )
    )

    cells.append(
        code(
            """
from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

NB_ROOT = Path.cwd()
if not (NB_ROOT / "lib").exists():
    NB_ROOT = Path.cwd() / "notebooks"
sys.path.insert(0, str(NB_ROOT))

from lib.kaggle_data import load_favorita_panels, load_m5_panels
from lib.joint_market_state import (
    ALIGNMENT_METHOD,
    DEFAULT_FAVORITA_ANCHOR,
    DEFAULT_M5_ANCHOR,
    HORIZON_DAYS,
    PUBLIC_FEATURES,
    SILHOUETTE_FEATURES,
    SILHOUETTE_RUNS,
    TARGETS,
    attach_horizon_targets,
    build_joint_panel,
    feature_matrix,
    load_public_state,
    load_silhouette_state,
    load_survey_dates,
    permute_public_block,
    profile_frame,
    rolling_origin_indices,
    silhouette_column_catalog,
)
from lib.plotting import configure_matplotlib, finish_figure, format_date_axis

font = configure_matplotlib()
pd.set_option("display.max_columns", 24)
pd.set_option("display.width", 140)
print("NB_ROOT", NB_ROOT)
print("Python", sys.version.split()[0], "| font:", font)
print("RUNS_ROOT", SILHOUETTE_RUNS, "exists=", SILHOUETTE_RUNS.is_dir())
print("survey dates", len(load_survey_dates()), "| horizon", HORIZON_DAYS, "d")
print("alignment", ALIGNMENT_METHOD, "| fav_anchor", DEFAULT_FAVORITA_ANCHOR, "| m5_anchor", DEFAULT_M5_ANCHOR)

try:
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("sklearn", sklearn.__version__)
except Exception as exc:
    sklearn = None
    print("sklearn import failed:", type(exc).__name__, exc)

try:
    import lightgbm as lgb

    print("lightgbm", lgb.__version__, "(nonlinear champion candidate)")
    HAS_LGB = True
except Exception:
    print("lightgbm not installed → HistGradientBoostingRegressor")
    HAS_LGB = False
"""
        )
    )

    cells.append(
        md(
            """## 1. 세 데이터층을 동일한 시장상태 계약으로 변환

직접 수집층은 `numeric_panel.json`(카테고리 share/entropy, 가격·할인), `analysis.json`(rank_energy), `drift_metric_points.json`(ranking turnover·review drift), visual diagnostics(신규·잔존·churn), `daily_scores.json`(statistical/text/visual headline)을 쓴다. **fused_signals.json(수십 MB)은 읽지 않는다.**
"""
        )
    )

    cells.append(
        code(
            """
silhouette = load_silhouette_state(SILHOUETTE_RUNS)
print("silhouette shape", silhouette.shape)
print("run_date span", silhouette["run_date"].min().date(), "→", silhouette["run_date"].max().date())
print("elapsed_days max", float(silhouette["elapsed_days"].max()), "| n_dates", silhouette["run_date"].nunique())
print("panel_status")
display(silhouette["panel_status"].value_counts(dropna=False).to_frame("n"))
print("musinsa_status / cm29_status")
display(pd.crosstab(silhouette["musinsa_status"].fillna("NA"), silhouette["cm29_status"].fillna("NA")))
print("numeric_panel musinsa_n>0 days:", int((pd.to_numeric(silhouette["musinsa_n"], errors="coerce").fillna(0) > 0).sum()))
print("numeric_panel cm29_n>0 days:", int((pd.to_numeric(silhouette["cm29_n"], errors="coerce").fillna(0) > 0).sum()))
print("rank_energy days:", int(silhouette["musinsa_rank_energy_1d_mean"].notna().sum()))
print("ranking_turnover musinsa days:", int(silhouette["musinsa_ranking_turnover"].notna().sum()))

drop_preview = [c for c in silhouette.columns if c.startswith("_")]
sil_profile = profile_frame(silhouette.drop(columns=drop_preview, errors="ignore"), "silhouette")
print("컬럼 카탈로그 (형상 + 뜻). null_rate·nunique는 이 실행에서 관측된 값이다.")
display(silhouette_column_catalog(sil_profile))
print("상수·결측 주의")
display(
    silhouette_column_catalog(sil_profile).query("nunique <= 2 or null_rate >= 0.5")[
        ["column", "nunique", "null_rate", "meaning", "note"]
    ]
)
print("sample rows (provenance-relevant columns)")
display(
    silhouette[
        [
            "run_date",
            "gap_days",
            "musinsa_status",
            "musinsa_n",
            "musinsa_rank_energy_1d_mean",
            "musinsa_ranking_turnover",
            "musinsa_cat_entropy",
            "cm29_n",
            "cm29_ranking_turnover",
            "visual_new_rate",
            "drift_statistical",
            "drift_text",
            "drift_visual",
        ]
    ].head(8)
)
"""
        )
    )

    cells.append(
        md(
            """Favorita는 일별 `sales`·`promo_rate`·family/store mix entropy·JS, M5는 `units`·category/state mix entropy·JS·변동성이다. 패밀리명과 패션 카테고리를 1:1로 잇지 않는다.
"""
        )
    )

    cells.append(
        code(
            """
fav_raw = load_favorita_panels()
m5_raw = load_m5_panels()
print("Favorita panels")
for k, df in fav_raw.items():
    print(f"  {k:18s} {tuple(df.shape)} index={df.index.name} cols={list(df.columns)[:6]}")
print("M5 panels")
for k, df in m5_raw.items():
    print(f"  {k:18s} {tuple(df.shape)} index={df.index.name} cols={list(df.columns)[:6]}")

fav_state, m5_state = load_public_state(fav_raw, m5_raw)
print("\\nFavorita state", fav_state.shape, fav_state.index.min().date(), "→", fav_state.index.max().date())
print("M5 state", m5_state.shape, m5_state.index.min().date(), "→", m5_state.index.max().date())
display(profile_frame(fav_state.reset_index(), "favorita_state").head(20))
display(profile_frame(m5_state.reset_index(), "m5_state"))
print("Favorita sample")
display(fav_state.head(3))
print("M5 sample")
display(m5_state.head(3))
"""
        )
    )

    cells.append(
        md(
            """## 2. 상대시간 합성 정렬과 provenance

수집일의 **실제 간격(`elapsed_days`)**을 보존한다. 첫 관측일 2026-05-21을 Favorita 2016-05-21, M5 2015-05-21 계절 앵커에 대응시킨 뒤 as-of lookup한다. 결합 테이블은 `synthetic_date`, `source_date_fav`, `source_date_m5`, `alignment_offset`, `alignment_method`, `synthetic_alignment=true`를 유지한다.
"""
        )
    )

    cells.append(
        code(
            """
print("BEFORE align  silhouette", silhouette.shape, "favorita_state", fav_state.shape, "m5_state", m5_state.shape)
joint = build_joint_panel(silhouette, fav_state, m5_state, offset_days=0)
print("AFTER align   joint", joint.shape)
assert bool(joint["synthetic_alignment"].all())
print("alignment_method", joint["alignment_method"].unique().tolist())
print("alignment_offset", joint["alignment_offset"].unique().tolist())
print("favorita_anchor", joint["favorita_anchor"].iloc[0], "| m5_anchor", joint["m5_anchor"].iloc[0])

prov = joint[
    [
        "synthetic_date",
        "run_date",
        "elapsed_days",
        "source_date_fav",
        "requested_source_date_fav",
        "fav_asof_lag_days",
        "source_date_m5",
        "requested_source_date_m5",
        "m5_asof_lag_days",
        "alignment_offset",
        "alignment_method",
        "synthetic_alignment",
    ]
]
print("provenance head")
display(prov.head(8))
print("provenance tail")
display(prov.tail(4))
print("fav asof lag describe")
display(joint["fav_asof_lag_days"].describe().to_frame("fav_asof_lag_days"))
print("public block null rates")
pub_cols = [c for c in joint.columns if c.startswith("fav_") or c.startswith("m5_")]
display(profile_frame(joint[pub_cols], "joint_public"))

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(joint["synthetic_date"], joint["musinsa_rank_energy_1d_mean"], marker="o", ms=3)
axes[0].set_title("직접 수집 · musinsa rank_energy(1d) mean (2026)")
axes[1].plot(joint["synthetic_date"], joint["fav_sales_growth"], marker="o", ms=3, color="C1")
axes[1].set_title("합성 정렬 Favorita sales growth (source year ≈ 2016)")
axes[2].plot(joint["synthetic_date"], joint["m5_units_growth"], marker="o", ms=3, color="C2")
axes[2].set_title("합성 정렬 M5 units growth (source year ≈ 2015)")
for ax in axes:
    ax.set_ylabel("value")
finish_figure(fig, axes, date_axis=True)
"""
        )
    )

    cells.append(
        md(
            """## 3. 타깃 · 시간순 split · 3계층 모델

- Persistence: 직전 관측값 반복 (최소 기준선)
- ElasticNet: 방향·계수 해석이 가능한 선형 결합. Imputer·Scaler는 **해당 fold의 train에만** fit
- HistGradientBoosting: 비선형·결측 허용 비교 (LightGBM 미설치 시)
- 별도 기술 분석: PCA 1–2요인 (예측 파이프라인의 PCA와 분리)

타깃은 `y_rank_mix_proxy`(가능하면 t→t+7 카테고리 mix JS, 아니면 미래 ranking turnover), `y_future_drift_score`, 합성 샌드박스 `y_future_sales_growth`로 분리한다. 최근 7일 안쪽은 label incomplete다.
"""
        )
    )

    cells.append(
        code(
            """
labeled = attach_horizon_targets(joint, horizon_days=HORIZON_DAYS)
print("labeled shape", labeled.shape, "| label_ready", int(labeled["label_ready"].sum()))
for col in TARGETS:
    print(f"  {col:28s} non-null {int(labeled[col].notna().sum())}  null_rate={labeled[col].isna().mean():.2f}")

x_all, feat_names = feature_matrix(labeled, include_public=True)
print("\\nX shape", x_all.shape, "n_features", len(feat_names))
print("feature names:", feat_names)
display(profile_frame(x_all, "X"))
print("one input row (raw features, last labeled)")
ready = labeled[labeled["label_ready"]].reset_index(drop=True)
x_ready, _ = feature_matrix(ready, include_public=True)
display(x_ready.iloc[[-1]].T.rename(columns={x_ready.index[-1]: "x_last"}))
print("corresponding y")
display(ready.iloc[[-1]][["run_date", "target_date", *TARGETS]].T)

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(labeled["run_date"], labeled["y_rank_mix_proxy"], marker="o", ms=3, label="y_rank_mix_proxy")
ax.plot(labeled["run_date"], labeled["y_future_drift_score"], marker="o", ms=3, label="y_future_drift_score")
ax.plot(labeled["run_date"], labeled["y_future_sales_growth"], marker="o", ms=3, label="y_future_sales_growth")
ax.legend(frameon=False)
ax.set_title("Horizon targets on synthetic dates (NaN = incomplete label)")
finish_figure(fig, ax, date_axis=True)
"""
        )
    )

    cells.append(
        code(
            """
def _spearman(a, b) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(a[mask], b[mask])
    return float(rho)


def _dir_acc(y_true, y_pred) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def persistence_predict(y: pd.Series, test_i: int) -> float:
    prev = y.iloc[:test_i].dropna()
    return float(prev.iloc[-1]) if len(prev) else np.nan


def fit_predict_fold(model_name: str, x_train, y_train, x_test):
    y_train = pd.Series(y_train).astype(float)
    mask = y_train.notna()
    if int(mask.sum()) < 8:
        return np.array([np.nan] * len(x_test)), None, "insufficient_train_y"
    x_tr, y_tr = x_train.loc[mask], y_train.loc[mask]
    keep = [c for c in x_tr.columns if x_tr[c].notna().any()]
    if not keep:
        return np.array([np.nan] * len(x_test)), None, "all_nan_features"
    x_tr, x_te = x_tr[keep], x_test.reindex(columns=keep)
    if model_name == "elasticnet":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.2, l1_ratio=0.4, max_iter=8000)),
            ]
        )
        pipe.fit(x_tr, y_tr)
        return pipe.predict(x_te), pipe, "ok"
    if model_name == "hgb":
        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.08,
            max_iter=120,
            min_samples_leaf=4,
            random_state=42,
        )
        model.fit(x_tr.to_numpy(), y_tr.to_numpy())
        return model.predict(x_te.to_numpy()), model, "ok"
    raise ValueError(model_name)


def walk_forward(panel: pd.DataFrame, target: str, include_public: bool, min_train: int = 12) -> dict:
    work = panel.reset_index(drop=True)
    x, cols = feature_matrix(work, include_public=include_public)
    y = pd.to_numeric(work[target], errors="coerce")
    records = []
    last_io = None
    models_last = {}
    for train_idx, test_idx in rolling_origin_indices(len(work), min_train=min_train):
        i = int(test_idx[0])
        if not np.isfinite(y.iloc[i]):
            continue
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        row = {
            "run_date": work.loc[i, "run_date"],
            "y_true": float(y.iloc[i]),
            "pred_persistence": persistence_predict(y, i),
        }
        for name in ("elasticnet", "hgb"):
            try:
                pred, fitted, status = fit_predict_fold(name, x_tr, y_tr, x_te)
                row[f"pred_{name}"] = float(pred[0]) if status == "ok" else np.nan
                row[f"status_{name}"] = status
                models_last[name] = fitted
            except Exception as exc:
                row[f"pred_{name}"] = np.nan
                row[f"status_{name}"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
        last_io = {
            "target": target,
            "include_public": include_public,
            "test_date": str(work.loc[i, "run_date"].date()),
            "x_raw": x_te.iloc[0].to_dict(),
            "y_true": row["y_true"],
            "preds": {k: row[k] for k in row if k.startswith("pred_")},
        }
        pipe = models_last.get("elasticnet")
        if pipe is not None:
            used = [c for c in x_te.columns if c in getattr(pipe, "feature_names_in_", x_te.columns)]
            if not used:
                used = list(x_te.columns)
            x_one = x_te[used]
            try:
                transformed = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(x_one))[0]
                last_io["x_transformed_elasticnet"] = {
                    c: float(v) for c, v in zip(used, transformed) if np.isfinite(v)
                }
                coef = pipe.named_steps["model"].coef_
                last_io["elasticnet_top_coef"] = (
                    pd.Series(coef, index=used).abs().sort_values(ascending=False).head(8).to_dict()
                )
            except Exception as exc:
                last_io["x_transformed_elasticnet"] = {"error": f"{type(exc).__name__}: {exc}"}
    pred_df = pd.DataFrame(records)
    summary_rows = []
    if pred_df.empty:
        return {"preds": pred_df, "metrics": pd.DataFrame(), "last_io": last_io, "n_eval": 0}
    y_true = pred_df["y_true"].to_numpy(dtype=float)
    base_mae = mean_absolute_error(y_true, pred_df["pred_persistence"].to_numpy(dtype=float)) if pred_df["pred_persistence"].notna().all() else np.nan
    for name in ("persistence", "elasticnet", "hgb"):
        y_hat = pred_df[f"pred_{name}"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "target": target,
                "include_public": include_public,
                "model": name,
                "n_eval": int(np.isfinite(y_hat).sum()),
                "mae": float(mean_absolute_error(y_true[np.isfinite(y_hat)], y_hat[np.isfinite(y_hat)]))
                if np.isfinite(y_hat).sum()
                else np.nan,
                "mae_vs_persistence": np.nan,
                "dir_acc": _dir_acc(y_true, y_hat),
                "spearman": _spearman(y_true, y_hat),
            }
        )
    metrics = pd.DataFrame(summary_rows)
    pers = float(metrics.loc[metrics["model"] == "persistence", "mae"].iloc[0])
    metrics["mae_vs_persistence"] = metrics["mae"] - pers
    return {"preds": pred_df, "metrics": metrics, "last_io": last_io, "n_eval": len(pred_df)}


if sklearn is None:
    print("sklearn missing — cannot train models")
    model_tables = []
    io_log = []
else:
    model_tables = []
    io_log = []
    for target in TARGETS:
        for include_public in (False, True):
            result = walk_forward(labeled, target, include_public=include_public, min_train=12)
            print(f"\\n=== {target} | public={include_public} | n_eval={result['n_eval']} ===")
            display(result["metrics"])
            model_tables.append(result["metrics"])
            io_log.append(result["last_io"])
            if result["last_io"]:
                print("last fold I/O")
                io = result["last_io"]
                print("  test_date", io["test_date"], "y_true", io["y_true"], "preds", io["preds"])
                trans = pd.Series(io.get("x_transformed_elasticnet") or {}).head(12)
                display(trans.to_frame("x_transformed"))
                display(pd.Series(io.get("elasticnet_top_coef") or {}, name="|coef|").to_frame())

    metrics_all = pd.concat(model_tables, ignore_index=True)
    print("\\nall walk-forward metrics")
    display(metrics_all)
"""
        )
    )

    cells.append(
        md(
            """## 4. 상위 상관 · PCA · lead-lag · ablation

동시 Pearson/Spearman, weekday·promo 통제 부분상관, 관측 시차 -7~+7. PCA loading은 **수요·프로모션·구성교체 후보 국면**으로만 이름 붙이며 인과가 아니다. Ablation은 직접 수집 변수만 vs 합성 공개 변수 추가의 예측 lift와 불안정성을 같이 본다.
"""
        )
    )

    cells.append(
        code(
            """
pair_left = [
    "musinsa_rank_energy_1d_mean",
    "musinsa_ranking_turnover",
    "musinsa_mix_js",
    "musinsa_cat_entropy",
    "visual_new_rate",
    "drift_statistical",
    "drift_text",
]
pair_right = [
    "fav_sales_growth",
    "fav_promo_rate",
    "fav_family_entropy",
    "fav_family_mix_js",
    "m5_units_growth",
    "m5_cat_entropy",
]


def corr_table(df: pd.DataFrame, lefts, rights) -> pd.DataFrame:
    rows = []
    for a in lefts:
        for b in rights:
            if a not in df.columns or b not in df.columns:
                continue
            sub = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 6 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                pear = spear = np.nan
            else:
                pear = float(sub[a].corr(sub[b], method="pearson"))
                spear = float(sub[a].corr(sub[b], method="spearman"))
            rows.append({"left": a, "right": b, "n": int(len(sub)), "pearson": pear, "spearman": spear})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


simult = corr_table(labeled, pair_left, pair_right)
print("simultaneous correlation (synthetic alignment)")
display(simult.head(16))


def partial_spearman(df, a, b, controls):
    ctrl = [c for c in controls if c not in (a, b)]
    cols = [a, b, *ctrl]
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sklearn is None or len(work) < 8 or not ctrl:
        return np.nan, len(work)
    if work[a].nunique() < 2 or work[b].nunique() < 2:
        return np.nan, len(work)
    lr = LinearRegression()
    z = work[ctrl].to_numpy()
    ra = work[a].to_numpy() - lr.fit(z, work[a]).predict(z)
    rb = work[b].to_numpy() - lr.fit(z, work[b]).predict(z)
    return _spearman(ra, rb), len(work)


controls = [c for c in ("weekday", "month", "fav_promo_rate") if c in labeled.columns]
partial_rows = []
for rec in simult.head(10).itertuples():
    rho, n = partial_spearman(labeled, rec.left, rec.right, controls)
    partial_rows.append({"left": rec.left, "right": rec.right, "spearman": rec.spearman, "partial_spearman": rho, "n": n, "controls": ",".join(controls)})
print("partial Spearman (calendar + promo)")
display(pd.DataFrame(partial_rows))


def lag_corr(df, a, b, lags=range(-7, 8)) -> pd.DataFrame:
    rows = []
    sa = pd.to_numeric(df[a], errors="coerce")
    sb = pd.to_numeric(df[b], errors="coerce")
    for lag in lags:
        if lag < 0:
            x, y = sa.iloc[-lag:], sb.iloc[:lag]
        elif lag > 0:
            x, y = sa.iloc[:-lag], sb.iloc[lag:]
        else:
            x, y = sa, sb
        aligned = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1).dropna()
        aligned.columns = ["x", "y"]
        rows.append({"lag": lag, "n": int(len(aligned)), "spearman": _spearman(aligned["x"].to_numpy(), aligned["y"].to_numpy())})
    return pd.DataFrame(rows)


focus_pairs = [
    ("musinsa_ranking_turnover", "fav_sales_growth"),
    ("musinsa_rank_energy_1d_mean", "fav_family_mix_js"),
    ("drift_statistical", "fav_promo_rate"),
    ("visual_new_rate", "m5_cat_entropy"),
]
lag_frames = []
for a, b in focus_pairs:
    if a in labeled.columns and b in labeled.columns:
        lc = lag_corr(labeled, a, b)
        lc["pair"] = f"{a} vs {b}"
        lag_frames.append(lc)
        print(f"lead-lag {a} vs {b} (lag>0 ⇒ right leads)")
        display(lc)

if lag_frames:
    fig, ax = plt.subplots(figsize=(12, 4))
    for frame in lag_frames:
        ax.plot(frame["lag"], frame["spearman"], marker="o", label=frame["pair"].iloc[0][:42])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("observation lag")
    ax.set_ylabel("Spearman")
    ax.set_title("Cross-correlation on observation index (not calendar days)")
    ax.legend(frameon=False, fontsize=8)
    finish_figure(fig, ax)
"""
        )
    )

    cells.append(
        code(
            """
pca_cols = [c for c in SILHOUETTE_FEATURES + PUBLIC_FEATURES if c in labeled.columns]
pca_x = labeled[pca_cols].apply(pd.to_numeric, errors="coerce")
if sklearn is None:
    print("skip PCA")
    pca_loadings = pd.DataFrame()
else:
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    z = sc.fit_transform(imp.fit_transform(pca_x))
    pca = PCA(n_components=min(4, z.shape[1], z.shape[0] - 1), random_state=42)
    scores = pca.fit_transform(z)
    print("PCA explained variance ratio", np.round(pca.explained_variance_ratio_, 3))
    pca_loadings = pd.DataFrame(pca.components_[:2].T, index=pca_cols, columns=["pc1", "pc2"])
    print("PC1 loadings (abs top 10) — common co-movement, not a named cause")
    display(pca_loadings.reindex(pca_loadings["pc1"].abs().sort_values(ascending=False).head(10).index))
    print("PC2 loadings (abs top 10)")
    display(pca_loadings.reindex(pca_loadings["pc2"].abs().sort_values(ascending=False).head(10).index))

    def name_factor(load: pd.Series) -> str:
        keys = load.abs().sort_values(ascending=False).head(5).index.tolist()
        blob = " ".join(keys)
        tags = []
        if any(k.startswith("fav_") or k.startswith("m5_") for k in keys):
            tags.append("수요/공개변동성")
        if "promo" in blob:
            tags.append("프로모션")
        if any(s in blob for s in ("mix", "entropy", "turnover", "rank_energy", "hhi")):
            tags.append("구성교체")
        return " · ".join(tags) or "혼합 공통변동"

    print("PC1 candidate label:", name_factor(pca_loadings["pc1"]))
    print("PC2 candidate label:", name_factor(pca_loadings["pc2"]))
    score_df = pd.DataFrame(scores[:, :2], columns=["pc1", "pc2"])
    score_df["synthetic_date"] = labeled["synthetic_date"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(score_df["synthetic_date"], score_df["pc1"], label="PC1")
    ax.plot(score_df["synthetic_date"], score_df["pc2"], label="PC2")
    ax.set_title("PCA factor scores on synthetic dates (descriptive, full-sample)")
    ax.legend(frameon=False)
    finish_figure(fig, ax, date_axis=True)

print("\\nAblation: silhouette-only vs +public (from walk-forward table)")
if sklearn is not None and "metrics_all" in globals():
    ab = metrics_all.copy()
    display(ab.sort_values(["target", "model", "include_public"]))
    lift_rows = []
    for (target, model), g in ab.groupby(["target", "model"]):
        g = g.set_index("include_public")
        if False in g.index and True in g.index:
            lift_rows.append(
                {
                    "target": target,
                    "model": model,
                    "mae_silhouette_only": float(g.loc[False, "mae"]),
                    "mae_plus_public": float(g.loc[True, "mae"]),
                    "mae_lift": float(g.loc[False, "mae"] - g.loc[True, "mae"]),
                    "spearman_silhouette_only": float(g.loc[False, "spearman"]),
                    "spearman_plus_public": float(g.loc[True, "spearman"]),
                    "n_eval": int(g.loc[True, "n_eval"]),
                }
            )
    lift = pd.DataFrame(lift_rows)
    print("public-feature lift (positive mae_lift = public helped)")
    display(lift)
else:
    lift = pd.DataFrame()
    print("ablation skipped")
"""
        )
    )

    cells.append(
        md(
            """## 5. Offset · 순열 placebo

공개 시계열을 ±30/±60일 밀고, 같은 공개 블록을 날짜와 무관하게 순열한다. 선택한 임시 정렬에서만 커지는 상관·예측 lift인지 본다.
"""
        )
    )

    cells.append(
        code(
            """
FOCUS_LEFT = "musinsa_ranking_turnover"
FOCUS_RIGHT = "fav_sales_growth"


def focus_spearman(panel: pd.DataFrame) -> dict:
    work = attach_horizon_targets(panel)
    sub = work[[FOCUS_LEFT, FOCUS_RIGHT]].apply(pd.to_numeric, errors="coerce").dropna()
    rho = _spearman(sub[FOCUS_LEFT].to_numpy(), sub[FOCUS_RIGHT].to_numpy()) if len(sub) else np.nan
    out = {"n": int(len(sub)), "spearman_turnover_vs_sales_growth": rho}
    if sklearn is not None:
        wf = walk_forward(work, "y_rank_mix_proxy", include_public=True, min_train=12)
        m = wf["metrics"]
        if not m.empty:
            out["mae_hgb"] = float(m.loc[m["model"] == "hgb", "mae"].iloc[0])
            out["mae_persistence"] = float(m.loc[m["model"] == "persistence", "mae"].iloc[0])
            out["n_eval"] = int(m.loc[m["model"] == "hgb", "n_eval"].iloc[0])
    return out


placebo_rows = []
base_stats = focus_spearman(joint)
placebo_rows.append({"variant": "anchor_offset_0", "offset_days": 0, **base_stats})
for off in (-60, -30, 30, 60):
    try:
        alt = build_joint_panel(silhouette, fav_state, m5_state, offset_days=off)
        stats_off = focus_spearman(alt)
        placebo_rows.append({"variant": f"anchor_offset_{off}", "offset_days": off, **stats_off})
    except Exception as exc:
        placebo_rows.append({"variant": f"anchor_offset_{off}", "offset_days": off, "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()

rng = np.random.default_rng(42)
for i in range(5):
    perm = permute_public_block(joint, rng)
    stats_p = focus_spearman(perm)
    placebo_rows.append({"variant": f"permutation_{i}", "offset_days": np.nan, **stats_p})

placebo = pd.DataFrame(placebo_rows)
print("placebo comparison")
display(placebo)

obs = placebo.loc[placebo["variant"] == "anchor_offset_0", "spearman_turnover_vs_sales_growth"]
perm_vals = placebo.loc[placebo["variant"].astype(str).str.startswith("permutation_"), "spearman_turnover_vs_sales_growth"]
print("observed spearman", float(obs.iloc[0]) if len(obs) else np.nan)
print("permutation mean/std", float(perm_vals.mean()) if len(perm_vals) else np.nan, float(perm_vals.std()) if len(perm_vals) else np.nan)

fig, ax = plt.subplots(figsize=(11, 4))
lab = placebo["variant"]
ax.bar(range(len(placebo)), placebo["spearman_turnover_vs_sales_growth"].fillna(0), color=["C0" if v == "anchor_offset_0" else "C7" for v in lab])
ax.set_xticks(range(len(placebo)))
ax.set_xticklabels(lab, rotation=40, ha="right")
ax.axhline(0, color="0.5", lw=0.8)
ax.set_ylabel("Spearman")
ax.set_title(f"{FOCUS_LEFT} vs {FOCUS_RIGHT} under offset/permutation placebo")
finish_figure(fig, ax)
"""
        )
    )

    cells.append(
        md(
            """## 6. 파이프라인 검증 vs 현실 인과에 추가로 필요한 것

아래 셀은 이 실행에서 **실제로 관측된** 숫자만 모아 두 덩어리로 나눈다.

1. **파이프라인 검증 결과** — shape, 유효 표본, 모델 MAE, loading/lag, placebo 민감도
2. **현실 인과 검증에 추가로 필요한 것** — 동기간 매출·프로모션·재고 정답. 이 노트북만으로는 한국 패션 시장의 원인을 주장할 수 없다.
"""
        )
    )

    cells.append(
        code(
            """
print("=" * 72)
print("A. 파이프라인 검증 결과 (이 실행에서 관측)")
print("=" * 72)
print("silhouette days:", int(silhouette["run_date"].nunique()), "shape", silhouette.shape)
print("joint shape:", joint.shape, "| synthetic_alignment all true:", bool(joint["synthetic_alignment"].all()))
print("label_ready:", int(labeled["label_ready"].sum()), "/", len(labeled))
for col in TARGETS:
    print(f"  {col}: n={int(labeled[col].notna().sum())}")
print("X features:", len(feat_names), "X shape on labeled:", x_all.shape)
print("numeric_panel musinsa ok days:", int((silhouette["musinsa_status"] == "ok").sum()), "/ 42  ← early days are insufficient_n, not hidden")
print("rank_energy coverage:", int(silhouette["musinsa_rank_energy_1d_mean"].notna().sum()), "/ 42")
if sklearn is not None and "metrics_all" in globals():
    print("walk-forward metrics:")
    display(metrics_all)
    if "lift" in globals() and not lift.empty:
        print("ablation lift:")
        display(lift)
if "simult" in globals():
    print("top |spearman| pairs:")
    display(simult.head(8))
if "placebo" in globals():
    print("placebo table:")
    display(placebo)
    obs_rho = placebo.loc[placebo["variant"] == "anchor_offset_0", "spearman_turnover_vs_sales_growth"]
    perm_rho = placebo.loc[placebo["variant"].astype(str).str.startswith("permutation_"), "spearman_turnover_vs_sales_growth"]
    if len(obs_rho) and len(perm_rho.dropna()):
        print(
            "placebo note: observed Spearman",
            round(float(obs_rho.iloc[0]), 3),
            "vs permutation mean",
            round(float(perm_rho.mean()), 3),
        )

print()
print("=" * 72)
print("B. 현실 인과 검증에 추가로 필요한 동기간 데이터")
print("=" * 72)
print("- 2026 관측일과 겹치는 한국 패션 채널의 실제 매출/단위/프로모션 (동기간 GMV)")
print("- 순위 변동의 교란항: 수집 커버리지 변화, 카테고리 리뉴얼, 광고 집행")
print("- 합성 앵커(2016 Favorita / 2015 M5)를 대체할 동일 달력의 외부 수요")
print("- 이 노트북의 상관·예측 lift는 상대시간 샌드박스 민감도이지 인과 효과 크기가 아니다")
print("done")
"""
        )
    )

    nb["cells"] = cells
    nbf.write(nb, NB_PATH)
    print("wrote", NB_PATH, "cells", len(cells))


if __name__ == "__main__":
    build()
