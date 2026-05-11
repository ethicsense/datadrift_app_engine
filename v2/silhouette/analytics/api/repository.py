from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class AnalyticsRepository:
    def __init__(self, output_dir: Path, datasets_root: Path) -> None:
        self.output_dir = output_dir
        self.datasets_root = datasets_root

    def list_datasets(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "analytics",
                "label": self.output_dir.name,
                "path": str(self.output_dir),
                "is_default": True,
            }
        ]

    def resolve_dataset_dir(self, dataset: str | None) -> Path:
        return self.output_dir

    def _read_duckdb_table(self, dataset_dir: Path, table_name: str) -> pd.DataFrame:
        db_path = dataset_dir / "analytics.duckdb"
        if not db_path.exists():
            return pd.DataFrame()
        try:
            import duckdb

            with duckdb.connect(str(db_path), read_only=True) as con:
                return con.execute(f"SELECT * FROM {table_name}").df()
        except Exception:
            return pd.DataFrame()

    def load_fact_snapshots(self, dataset: str | None) -> pd.DataFrame:
        dataset_dir = self.resolve_dataset_dir(dataset)
        duckdb_df = self._read_duckdb_table(dataset_dir, "fact_snapshots")
        if not duckdb_df.empty:
            duckdb_df["crawl_datetime"] = pd.to_datetime(duckdb_df["crawl_datetime"], errors="coerce")
            return duckdb_df
        parquet_path = dataset_dir / "fact_snapshots.parquet"
        csv_path = dataset_dir / "fact_snapshots.csv"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        else:
            raise FileNotFoundError(f"분석 결과 파일이 없습니다: {parquet_path} 또는 {csv_path}")
        df["crawl_datetime"] = pd.to_datetime(df["crawl_datetime"], errors="coerce")
        return df

    def load_json(self, dataset: str | None, filename: str) -> dict[str, Any]:
        dataset_dir = self.resolve_dataset_dir(dataset)
        path = dataset_dir / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_schema_artifact(self, dataset: str | None, filename: str) -> dict[str, Any]:
        payload = self.load_json(dataset, filename)
        return payload if isinstance(payload, dict) else {}

    def save_json(self, dataset: str | None, filename: str, payload: dict[str, Any]) -> None:
        dataset_dir = self.resolve_dataset_dir(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = dataset_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_table(self, dataset: str | None, filename: str, table_name: str | None = None) -> pd.DataFrame:
        dataset_dir = self.resolve_dataset_dir(dataset)
        if table_name:
            duckdb_df = self._read_duckdb_table(dataset_dir, table_name)
            if not duckdb_df.empty:
                return duckdb_df
        path = dataset_dir / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_raw_snapshot_products(self, dataset: str | None) -> pd.DataFrame:
        return self.load_table(dataset, "raw_snapshot_products.parquet", "raw_snapshot_products")
