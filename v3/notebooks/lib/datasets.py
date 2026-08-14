"""Open-source commerce dataset catalog, download, and loaders."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
USER_AGENT = "datadrift-commerce-notebook/0.1"

DATASET_CATALOG = [
    {
        "id": "uci_online_retail_ii",
        "name": "UCI Online Retail II",
        "grain": "invoice line (주문 라인)",
        "period": "2009-12 ~ 2011-12",
        "license": "CC BY 4.0",
        "why": "거래 원장 시계열. GMV/취소/국가 믹스/SKU 카탈로그 드리프트에 적합.",
        "source": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "access": "직접 다운로드",
    },
    {
        "id": "olist",
        "name": "Olist Brazilian E-Commerce",
        "grain": "order + order_item + product + payment (관계형)",
        "period": "2016-09 ~ 2018-10",
        "license": "CC BY-NC-SA 4.0",
        "why": "주문 상태/배송 SLA/카테고리/결제수단 등 운영 시계열과 믹스 시프트.",
        "source": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "access": "GitHub 미러 (원본은 Kaggle)",
    },
    {
        "id": "superstore",
        "name": "Tableau Sample Superstore",
        "grain": "order line",
        "period": "2014 ~ 2017 (샘플)",
        "license": "Tableau 샘플 (교육용)",
        "why": "카테고리/세그먼트/지역 믹스가 깔끔해 구성 변화 시각화에 좋음.",
        "source": "https://community.tableau.com (Sample Superstore)",
        "access": "공개 gist CSV",
    },
    {
        "id": "uci_online_retail",
        "name": "UCI Online Retail (1년)",
        "grain": "invoice line",
        "period": "2010-12 ~ 2011-12",
        "license": "CC BY 4.0",
        "why": "Online Retail II의 부분집합. II를 쓰면 별도 다운로드 불필요.",
        "source": "https://archive.ics.uci.edu/dataset/352/online+retail",
        "access": "참고용 (이번 실험에서는 II 사용)",
    },
    {
        "id": "m5",
        "name": "M5 Walmart Forecasting",
        "grain": "SKU x store x day",
        "period": "2011-01 ~ 2016-06",
        "license": "대회 데이터 (Academic / Non-Commercial)",
        "why": "계층 시계열(점포-카테고리-SKU) 수요·믹스 드리프트 벤치마크.",
        "source": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "access": "kagglehub.competition_download (Rules 수락 필요)",
    },
    {
        "id": "instacart",
        "name": "Instacart Market Basket 2017",
        "grain": "order + order_product",
        "period": "상대 시간(dow/hour, days_since_prior)",
        "license": "CC0-1.0 (Kaggle 미러)",
        "why": "재구매/장바구니 구성 드리프트. 절대 날짜 대신 order_number 축.",
        "source": "https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset",
        "access": "Kaggle API (access_token)",
    },
    {
        "id": "favorita",
        "name": "Corporación Favorita Store Sales",
        "grain": "store x family x day",
        "period": "2013-01 ~ 2017-08",
        "license": "대회 데이터",
        "why": "프로모션/휴일/유가 공변량과 점포·카테고리 수요 시계열.",
        "source": "https://www.kaggle.com/competitions/store-sales-time-series-forecasting",
        "access": "Kaggle API (Rules 수락된 계정)",
    },
]


def _get(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_uci_online_retail_ii(force: bool = False) -> Path:
    dest_dir = _ensure_dir(DATA_DIR / "uci_online_retail_ii")
    xlsx = dest_dir / "online_retail_II.xlsx"
    if xlsx.exists() and not force:
        return xlsx
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    raw = _get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not members:
            raise FileNotFoundError(f"xlsx not in zip: {zf.namelist()}")
        with zf.open(members[0]) as src, xlsx.open("wb") as dst:
            dst.write(src.read())
    return xlsx


def download_olist(force: bool = False) -> Path:
    dest_dir = _ensure_dir(DATA_DIR / "olist")
    base = "https://raw.githubusercontent.com/Kaaykun/OlistAnalysis/master/data/csv/"
    files = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_products_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv",
    ]
    for name in files:
        path = dest_dir / name
        if path.exists() and not force:
            continue
        path.write_bytes(_get(base + name))
    return dest_dir


def download_superstore(force: bool = False) -> Path:
    dest_dir = _ensure_dir(DATA_DIR / "superstore")
    csv_path = dest_dir / "superstore.csv"
    if csv_path.exists() and not force:
        return csv_path
    url = "https://gist.githubusercontent.com/nnbphuong/38db511db14542f3ba9ef16e69d3814c/raw/Superstore.csv"
    csv_path.write_bytes(_get(url))
    return csv_path


def download_all(force: bool = False) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "uci_online_retail_ii": download_uci_online_retail_ii(force=force),
        "olist": download_olist(force=force),
        "superstore": download_superstore(force=force),
    }


def load_uci_online_retail_ii() -> pd.DataFrame:
    xlsx = download_uci_online_retail_ii()
    frames = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
    df = pd.concat(frames.values(), ignore_index=True)
    df = df.rename(
        columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
        }
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancel"] = df["InvoiceNo"].str.startswith("C") | (df["Quantity"] < 0)
    df["HasCustomer"] = df["Customer ID"].notna() if "Customer ID" in df.columns else df["CustomerID"].notna()
    if "Customer ID" in df.columns and "CustomerID" not in df.columns:
        df["CustomerID"] = df["Customer ID"]
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    return df


def load_olist() -> dict[str, pd.DataFrame]:
    dest = download_olist()
    tables = {
        "orders": pd.read_csv(dest / "olist_orders_dataset.csv"),
        "items": pd.read_csv(dest / "olist_order_items_dataset.csv"),
        "payments": pd.read_csv(dest / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(dest / "olist_order_reviews_dataset.csv"),
        "products": pd.read_csv(dest / "olist_products_dataset.csv"),
        "customers": pd.read_csv(dest / "olist_customers_dataset.csv"),
        "sellers": pd.read_csv(dest / "olist_sellers_dataset.csv"),
        "categories": pd.read_csv(dest / "product_category_name_translation.csv"),
    }
    ts_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        tables["orders"][col] = pd.to_datetime(tables["orders"][col], errors="coerce")
    tables["items"]["shipping_limit_date"] = pd.to_datetime(
        tables["items"]["shipping_limit_date"], errors="coerce"
    )
    tables["reviews"]["review_creation_date"] = pd.to_datetime(
        tables["reviews"]["review_creation_date"], errors="coerce"
    )
    return tables


def load_superstore() -> pd.DataFrame:
    csv_path = download_superstore()
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df["Date"] = df["Order Date"].dt.normalize()
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    return df
