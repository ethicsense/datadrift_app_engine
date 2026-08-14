"""Kaggle commerce dataset download + panel builders.

Caches aggregated panels under notebooks/data/*/panels/ so notebooks stay light.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DATA_DIR, _ensure_dir

KAGGLE_BIN = Path(sys.executable).with_name("kaggle")


def _run_kaggle(args: list[str]) -> None:
    cmd = [str(KAGGLE_BIN), *args]
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_favorita(force: bool = False) -> Path:
    dest = _ensure_dir(DATA_DIR / "favorita")
    marker = dest / "train.csv"
    if marker.exists() and not force:
        return dest
    _run_kaggle(
        [
            "competitions",
            "download",
            "-c",
            "store-sales-time-series-forecasting",
            "-p",
            str(dest),
        ]
    )
    zips = list(dest.glob("*.zip"))
    if zips:
        import zipfile

        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(dest)
        zips[0].unlink(missing_ok=True)
    return dest


def download_instacart(force: bool = False) -> Path:
    dest = _ensure_dir(DATA_DIR / "instacart")
    marker = dest / "orders.csv"
    if marker.exists() and not force:
        return dest
    _run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            "yasserh/instacart-online-grocery-basket-analysis-dataset",
            "-p",
            str(dest),
            "--unzip",
        ]
    )
    return dest


_M5_OFFICIAL_FILES = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def download_m5(force: bool = False) -> Path:
    """Download official M5 via kagglehub (preferred), then Kaggle CLI.

    Requires competition Rules acceptance. Falls back to parquet mirror only if
    both official paths fail.
    """
    dest = _ensure_dir(DATA_DIR / "m5")
    marker = dest / "sales_train_evaluation.csv"
    if marker.exists() and not force:
        return dest

    # 1) kagglehub competition download
    try:
        import kagglehub

        cache_path = Path(kagglehub.competition_download("m5-forecasting-accuracy"))
        print(f"kagglehub M5 cache: {cache_path}")
        for name in _M5_OFFICIAL_FILES:
            src = cache_path / name
            if src.exists():
                _link_or_copy(src, dest / name)
        if (dest / "sales_train_evaluation.csv").exists():
            return dest
    except Exception as e:
        print(f"kagglehub M5 download failed ({e}); trying kaggle CLI")

    # 2) kaggle CLI
    try:
        _run_kaggle(
            ["competitions", "download", "-c", "m5-forecasting-accuracy", "-p", str(dest)]
        )
        zips = list(dest.glob("*.zip"))
        if zips:
            import zipfile

            with zipfile.ZipFile(zips[0]) as zf:
                zf.extractall(dest)
            zips[0].unlink(missing_ok=True)
        if (dest / "sales_train_evaluation.csv").exists():
            return dest
    except subprocess.CalledProcessError as e:
        print(f"official M5 CLI download failed ({e}); using parquet mirror")

    # 3) parquet mirror fallback
    _run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            "marcozanotti/m5-competition-dataset-parquet",
            "-p",
            str(dest),
            "--unzip",
        ]
    )
    return dest


def download_olist_kaggle(force: bool = False) -> Path:
    dest = _ensure_dir(DATA_DIR / "olist_kaggle")
    marker = dest / "olist_orders_dataset.csv"
    if marker.exists() and not force:
        return dest
    _run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            "olistbr/brazilian-ecommerce",
            "-p",
            str(dest),
            "--unzip",
        ]
    )
    return dest


def download_kaggle_all(force: bool = False) -> dict[str, Path]:
    return {
        "olist_kaggle": download_olist_kaggle(force=force),
        "favorita": download_favorita(force=force),
        "instacart": download_instacart(force=force),
        "m5": download_m5(force=force),
    }


def _parse_m5_unique_id(uid: str) -> tuple[str, str, str, str, str]:
    parts = str(uid).split("_")
    # FOODS_1_001_CA_1
    cat = parts[0]
    dept = f"{parts[0]}_{parts[1]}"
    item = f"{parts[0]}_{parts[1]}_{parts[2]}"
    state = parts[3]
    store = f"{parts[3]}_{parts[4]}"
    return cat, dept, item, state, store


def _m5_panel_paths(panel_dir: Path) -> dict[str, Path]:
    return {
        "daily_total": panel_dir / "daily_total.parquet",
        "daily_cat": panel_dir / "daily_cat.parquet",
        "daily_state": panel_dir / "daily_state.parquet",
        "series_meta": panel_dir / "series_meta.parquet",
    }


def _build_m5_panels_from_official(root: Path, out: dict[str, Path]) -> dict[str, Path]:
    """Wide CSV (sales_train_evaluation + calendar) → daily panels."""
    print("building M5 panels from official sales_train_evaluation.csv")
    calendar = pd.read_csv(root / "calendar.csv", usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    d_to_date = calendar.set_index("d")["date"]

    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = pd.read_csv(root / "sales_train_evaluation.csv")
    dcols = [c for c in sales.columns if c.startswith("d_")]
    print(f"official M5 series={len(sales)} days={len(dcols)}")

    meta = sales[meta_cols].rename(
        columns={
            "id": "unique_id",
            "item_id": "item",
            "dept_id": "dept",
            "cat_id": "cat",
            "store_id": "store",
            "state_id": "state",
        }
    )
    meta["unique_id"] = meta["unique_id"].str.replace("_evaluation$", "", regex=True)
    meta.to_parquet(out["series_meta"], index=False)

    dates = d_to_date.reindex(dcols).to_numpy()

    total = pd.DataFrame({"date": dates, "units": sales[dcols].sum(axis=0).to_numpy()})
    total = total.set_index("date").sort_index()
    total.to_parquet(out["daily_total"])

    cat = sales.groupby("cat_id", sort=True)[dcols].sum().T
    cat.index = dates
    cat.index.name = "date"
    cat = cat.sort_index()
    cat.columns.name = None
    cat.to_parquet(out["daily_cat"])

    state = sales.groupby("state_id", sort=True)[dcols].sum().T
    state.index = dates
    state.index.name = "date"
    state = state.sort_index()
    state.columns.name = None
    state.to_parquet(out["daily_state"])
    return out


def _build_m5_panels_from_parquet(root: Path, out: dict[str, Path]) -> dict[str, Path]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    print("building M5 panels from parquet mirror")
    train = root / "m5_train.parquet"
    if not train.exists():
        raise FileNotFoundError(train)

    pf = pq.ParquetFile(train)
    uids: set[str] = set()
    for i in range(pf.num_row_groups):
        uniq = pc.unique(pf.read_row_group(i, columns=["unique_id"]).column(0)).to_pylist()
        uids.update(uniq)
    meta = pd.DataFrame({"unique_id": sorted(uids)})
    parsed = meta["unique_id"].map(_parse_m5_unique_id)
    meta[["cat", "dept", "item", "state", "store"]] = pd.DataFrame(parsed.tolist(), index=meta.index)
    meta.to_parquet(out["series_meta"], index=False)
    uid_map = meta.set_index("unique_id")[["cat", "state", "store"]]

    total_acc: dict[pd.Timestamp, float] = {}
    cat_acc: dict[tuple[pd.Timestamp, str], float] = {}
    state_acc: dict[tuple[pd.Timestamp, str], float] = {}

    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.join(uid_map, on="unique_id")
        g_total = df.groupby("ds", sort=False)["y"].sum()
        for ds, val in g_total.items():
            total_acc[ds] = total_acc.get(ds, 0.0) + float(val)
        g_cat = df.groupby(["ds", "cat"], sort=False)["y"].sum()
        for (ds, cat), val in g_cat.items():
            cat_acc[(ds, cat)] = cat_acc.get((ds, cat), 0.0) + float(val)
        g_state = df.groupby(["ds", "state"], sort=False)["y"].sum()
        for (ds, state), val in g_state.items():
            state_acc[(ds, state)] = state_acc.get((ds, state), 0.0) + float(val)
        print(f"m5 row_group {i+1}/{pf.num_row_groups}")

    (
        pd.Series(total_acc, name="units").sort_index().rename_axis("date").to_frame().to_parquet(out["daily_total"])
    )
    (
        pd.DataFrame([{"date": k[0], "cat": k[1], "units": v} for k, v in cat_acc.items()])
        .pivot(index="date", columns="cat", values="units")
        .fillna(0)
        .sort_index()
        .to_parquet(out["daily_cat"])
    )
    (
        pd.DataFrame([{"date": k[0], "state": k[1], "units": v} for k, v in state_acc.items()])
        .pivot(index="date", columns="state", values="units")
        .fillna(0)
        .sort_index()
        .to_parquet(out["daily_state"])
    )
    return out


def build_m5_panels(force: bool = False) -> dict[str, Path]:
    root = download_m5()
    panel_dir = _ensure_dir(root / "panels")
    out = _m5_panel_paths(panel_dir)
    source_marker = panel_dir / "source.txt"
    official = (root / "sales_train_evaluation.csv").exists()
    desired_source = "official" if official else "parquet_mirror"

    if (
        all(p.exists() for p in out.values())
        and source_marker.exists()
        and source_marker.read_text().strip() == desired_source
        and not force
    ):
        return out

    if official:
        _build_m5_panels_from_official(root, out)
    else:
        _build_m5_panels_from_parquet(root, out)
    source_marker.write_text(desired_source + "\n", encoding="utf-8")
    return out


def build_favorita_panels(force: bool = False) -> dict[str, Path]:
    root = download_favorita()
    panel_dir = _ensure_dir(root / "panels")
    out = {
        "daily_total": panel_dir / "daily_total.parquet",
        "daily_family": panel_dir / "daily_family.parquet",
        "daily_store": panel_dir / "daily_store.parquet",
        "daily_promo_rate": panel_dir / "daily_promo_rate.parquet",
        "tx_daily": panel_dir / "transactions_daily.parquet",
    }
    if all(p.exists() for p in out.values()) and not force:
        return out

    stores = pd.read_csv(root / "stores.csv")
    tx = pd.read_csv(root / "transactions.csv", parse_dates=["date"])
    tx.groupby("date", as_index=True)["transactions"].sum().sort_index().to_frame().to_parquet(
        out["tx_daily"]
    )

    total_acc: dict[pd.Timestamp, float] = {}
    family_acc: dict[tuple[pd.Timestamp, str], float] = {}
    store_acc: dict[tuple[pd.Timestamp, int], float] = {}
    promo_num: dict[pd.Timestamp, float] = {}
    promo_den: dict[pd.Timestamp, float] = {}

    for chunk in pd.read_csv(root / "train.csv", chunksize=500_000, parse_dates=["date"]):
        g = chunk.groupby("date")["sales"].sum()
        for ds, val in g.items():
            total_acc[ds] = total_acc.get(ds, 0.0) + float(val)
        gf = chunk.groupby(["date", "family"])["sales"].sum()
        for key, val in gf.items():
            family_acc[key] = family_acc.get(key, 0.0) + float(val)
        gs = chunk.groupby(["date", "store_nbr"])["sales"].sum()
        for key, val in gs.items():
            store_acc[key] = store_acc.get(key, 0.0) + float(val)
        gp = chunk.groupby("date").agg(promo=("onpromotion", "sum"), n=("onpromotion", "size"))
        for ds, row in gp.iterrows():
            promo_num[ds] = promo_num.get(ds, 0.0) + float(row["promo"])
            promo_den[ds] = promo_den.get(ds, 0.0) + float(row["n"])

    pd.Series(total_acc, name="sales").sort_index().rename_axis("date").to_frame().to_parquet(
        out["daily_total"]
    )
    fam = (
        pd.DataFrame([{"date": k[0], "family": k[1], "sales": v} for k, v in family_acc.items()])
        .pivot(index="date", columns="family", values="sales")
        .fillna(0)
        .sort_index()
    )
    fam.to_parquet(out["daily_family"])

    # keep top 12 stores by total sales for mix viz
    store_df = pd.DataFrame(
        [{"date": k[0], "store_nbr": k[1], "sales": v} for k, v in store_acc.items()]
    )
    top_stores = (
        store_df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(12).index
    )
    store_df["store"] = store_df["store_nbr"].where(store_df["store_nbr"].isin(top_stores), other=-1)
    store_pivot = (
        store_df.groupby(["date", "store"])["sales"].sum().unstack(fill_value=0).sort_index()
    )
    store_pivot.columns = [f"store_{int(c)}" if c != -1 else "Other" for c in store_pivot.columns]
    store_pivot.to_parquet(out["daily_store"])

    promo = pd.Series(
        {ds: promo_num[ds] / max(promo_den[ds], 1.0) for ds in promo_den},
        name="promo_rate",
    ).sort_index()
    promo.rename_axis("date").to_frame().to_parquet(out["daily_promo_rate"])

    # attach store meta for later joins if needed
    stores.to_parquet(panel_dir / "stores.parquet", index=False)
    return out


def build_instacart_panels(force: bool = False) -> dict[str, Path]:
    root = download_instacart()
    panel_dir = _ensure_dir(root / "panels")
    out = {
        "by_order_number": panel_dir / "by_order_number.parquet",
        "by_dow_hour": panel_dir / "by_dow_hour.parquet",
        "dept_mix_by_ordernum": panel_dir / "dept_mix_by_ordernum.parquet",
        "reorder_by_ordernum": panel_dir / "reorder_by_ordernum.parquet",
    }
    if all(p.exists() for p in out.values()) and not force:
        return out

    orders = pd.read_csv(root / "orders.csv")
    products = pd.read_csv(root / "products.csv")
    departments = pd.read_csv(root / "departments.csv")
    prod = products.merge(departments, on="department_id", how="left")

    # order-level calendar proxies
    by_n = (
        orders.groupby("order_number")
        .agg(
            orders=("order_id", "size"),
            users=("user_id", "nunique"),
            avg_days_since=("days_since_prior_order", "mean"),
        )
        .sort_index()
    )
    by_n.to_parquet(out["by_order_number"])

    by_cal = (
        orders.groupby(["order_dow", "order_hour_of_day"])
        .size()
        .rename("orders")
        .reset_index()
        .pivot(index="order_hour_of_day", columns="order_dow", values="orders")
        .fillna(0)
    )
    by_cal.to_parquet(out["by_dow_hour"])

    # department mix / reorder along order_number using prior basket (chunked)
    # join order_id -> order_number, product_id -> department
    order_map = orders.set_index("order_id")["order_number"]
    dept_map = prod.set_index("product_id")["department"]

    mix_acc: dict[tuple[int, str], float] = {}
    reorder_num: dict[int, float] = {}
    reorder_den: dict[int, float] = {}

    for chunk in pd.read_csv(root / "order_products__prior.csv", chunksize=1_000_000):
        chunk["order_number"] = chunk["order_id"].map(order_map)
        chunk["department"] = chunk["product_id"].map(dept_map)
        chunk = chunk.dropna(subset=["order_number", "department"])
        chunk["order_number"] = chunk["order_number"].astype(int)
        # bucket high order numbers
        chunk["order_bucket"] = chunk["order_number"].clip(upper=50)
        g = chunk.groupby(["order_bucket", "department"]).size()
        for key, val in g.items():
            mix_acc[key] = mix_acc.get(key, 0.0) + float(val)
        rg = chunk.groupby("order_bucket").agg(
            reorders=("reordered", "sum"), n=("reordered", "size")
        )
        for ob, row in rg.iterrows():
            reorder_num[ob] = reorder_num.get(ob, 0.0) + float(row["reorders"])
            reorder_den[ob] = reorder_den.get(ob, 0.0) + float(row["n"])
        print("instacart chunk processed", chunk.shape[0])

    mix = (
        pd.DataFrame(
            [{"order_bucket": k[0], "department": k[1], "lines": v} for k, v in mix_acc.items()]
        )
        .pivot(index="order_bucket", columns="department", values="lines")
        .fillna(0)
        .sort_index()
    )
    mix_share = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0)
    mix_share.to_parquet(out["dept_mix_by_ordernum"])

    reorder = pd.Series(
        {k: reorder_num[k] / max(reorder_den[k], 1.0) for k in reorder_den},
        name="reorder_rate",
    ).sort_index()
    reorder.rename_axis("order_bucket").to_frame().to_parquet(out["reorder_by_ordernum"])
    return out


def build_all_kaggle_panels(force: bool = False) -> dict[str, dict[str, Path]]:
    download_kaggle_all(force=force)
    return {
        "m5": build_m5_panels(force=force),
        "favorita": build_favorita_panels(force=force),
        "instacart": build_instacart_panels(force=force),
    }


def load_m5_panels() -> dict[str, pd.DataFrame]:
    paths = build_m5_panels()
    return {k: pd.read_parquet(v) for k, v in paths.items()}


def load_favorita_panels() -> dict[str, pd.DataFrame]:
    paths = build_favorita_panels()
    return {k: pd.read_parquet(v) for k, v in paths.items()}


def load_instacart_panels() -> dict[str, pd.DataFrame]:
    paths = build_instacart_panels()
    return {k: pd.read_parquet(v) for k, v in paths.items()}
