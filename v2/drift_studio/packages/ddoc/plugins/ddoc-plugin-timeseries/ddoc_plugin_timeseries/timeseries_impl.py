"""
Time Series Analysis Plugin Implementation for ddoc

Provides hookimpl for:
- eda_run: Time series attribute analysis
- drift_detect: Drift detection between baseline and current time series datasets
"""
import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import warnings

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(func):
        return func

try:
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tools.sm_exceptions import InterpolationWarning
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")


class DOCTimeSeriesPlugin:
    """Time Series Analysis Plugin for ddoc"""

    def _histogram(self, values: list[float], bins: int = 20, max_samples: int = 2000) -> dict[str, Any] | None:
        if not values:
            return None
        counts, edges = np.histogram(values, bins=bins)
        samples = values
        if len(values) > max_samples:
            idx = np.random.choice(len(values), size=max_samples, replace=False)
            samples = [values[i] for i in idx]
        return {"bins": edges.tolist(), "counts": counts.tolist(), "samples": samples}
    
    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Load and validate ddoc.yaml from dataset directory.

        drift_studio(v2)에서는 driftstudio_spec 스키마를 사용합니다:
          modality: timeseries
          data:
            csv: <relpath>.csv
            timestamp_column: <col>
            numeric_columns: [...]
            categorical_columns: [...]

        이 플러그인은 과거 스키마(top-level csv_file 등)도 일부 호환합니다.
        """
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}")

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # driftstudio_spec 스타일 우선
        modality = raw.get("modality")
        if modality != "timeseries":
            raise ValueError(f"Dataset {dataset_path} is not configured as timeseries modality")

        data = raw.get("data") or {}

        # v2 spec fields
        csv_rel = data.get("csv") or data.get("path")
        ts_col = data.get("timestamp_column") or raw.get("timestamp_column")
        numeric_cols = data.get("numeric_columns") or raw.get("numeric_columns") or []
        categorical_cols = data.get("categorical_columns") or raw.get("categorical_columns") or []

        # legacy fields fallback
        if not csv_rel:
            csv_rel = raw.get("csv_file")
        if not ts_col:
            ts_col = raw.get("timestamp_column")

        if not csv_rel:
            raise ValueError("ddoc.yaml must specify data.csv (or legacy csv_file)")
        if not ts_col:
            raise ValueError("ddoc.yaml must specify data.timestamp_column (or legacy timestamp_column)")

        return {
            "modality": "timeseries",
            "csv": csv_rel,
            "timestamp_column": ts_col,
            "numeric_columns": numeric_cols or [],
            "categorical_columns": categorical_cols or [],
        }

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _relative_diff(self, base: float, cur: float, eps: float = 1e-10) -> float:
        denom = abs(base) + eps
        return float(abs(cur - base) / denom)

    def _categorical_distribution_distance(
        self,
        base_freq: Dict[str, int],
        cur_freq: Dict[str, int],
    ) -> float:
        # Total Variation distance on empirical category distributions [0, 1]
        base_total = float(sum(base_freq.values()) or 1.0)
        cur_total = float(sum(cur_freq.values()) or 1.0)
        keys = set(base_freq.keys()) | set(cur_freq.keys())
        tv = 0.0
        for key in keys:
            p = float(base_freq.get(key, 0)) / base_total
            q = float(cur_freq.get(key, 0)) / cur_total
            tv += abs(p - q)
        return float(0.5 * tv)
    
    def _analyze_numeric_series(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate numeric time series metrics (column-level EDA profile)."""
        if series.empty or series.isna().all():
            return {}
        
        total_count = int(len(series))
        missing_count = int(series.isna().sum())
        series_clean = series.dropna()
        if series_clean.empty:
            return {}

        q1 = float(series_clean.quantile(0.25))
        median = float(series_clean.quantile(0.5))
        q3 = float(series_clean.quantile(0.75))
        iqr = float(q3 - q1)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series_clean < lower) | (series_clean > upper)).sum())
        
        metrics = {
            'count': int(len(series_clean)),
            'missing_count': missing_count,
            'missing_ratio': float(missing_count / max(total_count, 1)),
            'mean': float(series_clean.mean()),
            'std': float(series_clean.std()),
            'min': float(series_clean.min()),
            'q25': q1,
            'median': median,
            'q75': q3,
            'max': float(series_clean.max()),
            'iqr': iqr,
            'outlier_count': outlier_count,
            'outlier_ratio': float(outlier_count / max(int(len(series_clean)), 1)),
            'zero_ratio': float((series_clean == 0).sum() / max(int(len(series_clean)), 1)),
            'variance': float(series_clean.var()),
            'skewness': float(stats.skew(series_clean)),
            'kurtosis': float(stats.kurtosis(series_clean))
        }
        
        # Trend, seasonality, residual (if enough data)
        if len(series_clean) >= 24:  # Minimum for decomposition
            try:
                decomposition = seasonal_decompose(series_clean, model='additive', period=min(12, len(series_clean)//2))
                metrics['trend_strength'] = float(np.var(decomposition.trend.dropna()) / (np.var(series_clean) + 1e-10))
                metrics['seasonal_strength'] = float(np.var(decomposition.seasonal.dropna()) / (np.var(series_clean) + 1e-10))
                metrics['residual_strength'] = float(np.var(decomposition.resid.dropna()) / (np.var(series_clean) + 1e-10))
            except:
                pass
        
        # Stationarity tests
        try:
            adf_result = adfuller(series_clean)
            metrics['adf_statistic'] = float(adf_result[0])
            metrics['adf_pvalue'] = float(adf_result[1])
            # json.dump는 numpy.bool_를 직렬화하지 못하므로 파이썬 bool로 캐스팅
            metrics['is_stationary_adf'] = bool(adf_result[1] < 0.05)
        except:
            pass
        
        try:
            # statsmodels KPSS는 데이터 특성에 따라 InterpolationWarning이 매우 자주 발생할 수 있어
            # 로그를 오염시킵니다. 분석 자체는 유효하므로 warning을 suppress 합니다.
            with warnings.catch_warnings():
                try:
                    warnings.simplefilter("ignore", InterpolationWarning)
                except Exception:
                    warnings.simplefilter("ignore")
                kpss_result = kpss(series_clean, regression='c')
            metrics['kpss_statistic'] = float(kpss_result[0])
            metrics['kpss_pvalue'] = float(kpss_result[1])
            metrics['is_stationary_kpss'] = bool(kpss_result[1] > 0.05)
        except:
            pass
        
        return metrics
    
    def _analyze_categorical_series(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate categorical time series metrics"""
        if series.empty:
            return {}

        total_count = int(len(series))
        missing_count = int(series.isna().sum())
        series_clean = series.dropna()
        if series_clean.empty:
            return {
                "count": 0,
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / max(total_count, 1)),
                "frequencies": {},
                "entropy": 0.0,
                "unique_count": 0,
                "top": None,
                "top_ratio": 0.0,
            }

        value_counts = series_clean.value_counts()
        frequencies = value_counts.to_dict()
        
        # Entropy
        probs = value_counts / len(series_clean)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        top = str(value_counts.index[0]) if len(value_counts) else None
        top_ratio = float(value_counts.iloc[0] / max(int(len(series_clean)), 1)) if len(value_counts) else 0.0
        
        return {
            "count": int(len(series_clean)),
            "missing_count": missing_count,
            "missing_ratio": float(missing_count / max(total_count, 1)),
            'frequencies': {str(k): int(v) for k, v in frequencies.items()},
            'entropy': float(entropy),
            'unique_count': int(len(value_counts)),
            "top": top,
            "top_ratio": top_ratio,
        }
    
    @hookimpl
    def eda_run(self, snapshot_id, data_path, data_hash, output_path, invalidate_cache=False):
        """Run EDA for time series datasets"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        input_path = Path(data_path)
        output_path = Path(output_path)
        
        print(f"🚀 Time Series EDA Analysis Started")
        print(f"=" * 80)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'snapshot_id': snapshot_id,
            'data_hash': data_hash,
            'modality': 'timeseries'
        }
        
        # Find time series datasets.
        # - Prefer dataset at root (input_path/ddoc.yaml)
        # - Also support nested datasets under input_path/*
        ts_datasets = []
        root_yaml = input_path / "ddoc.yaml"
        if root_yaml.exists():
            try:
                config = self._load_ddoc_yaml(input_path)
                ts_datasets.append((input_path, config))
            except Exception as e:
                print(f"⚠️ Root dataset skipped: {e}")

        for item in input_path.iterdir():
            if item.is_dir():
                yaml_path = item / "ddoc.yaml"
                if yaml_path.exists():
                    try:
                        config = self._load_ddoc_yaml(item)
                        ts_datasets.append((item, config))
                    except Exception as e:
                        print(f"⚠️ Skipping {item}: {e}")
        
        if not ts_datasets:
            print("⚠️ No time series datasets found")
            return None
        
        # Load cache
        attr_cache = {}
        if not invalidate_cache:
            attr_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_timeseries"
            )
            if attr_cache_data:
                attr_cache = attr_cache_data
        
        # Process each dataset
        all_attributes = {}
        flat_stats: Dict[str, float] = {}
        column_distributions: Dict[str, Any] = {}
        
        for dataset_path, config in ts_datasets:
            print(f"\n📊 Processing dataset: {dataset_path.name}")
            
            csv_file = dataset_path / config["csv"]
            if not csv_file.exists():
                raise ValueError(f"CSV file not found: {csv_file} (from ddoc.yaml data.csv)")
            
            timestamp_col = config["timestamp_column"]
            numeric_cols = config.get("numeric_columns", []) or []
            categorical_cols = config.get("categorical_columns", []) or []
            
            try:
                df = pd.read_csv(csv_file)
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values(timestamp_col)
            except Exception as e:
                raise ValueError(f"Error loading CSV '{csv_file}': {e}") from e
            
            # Analyze each column
            for col in numeric_cols:
                if col in df.columns:
                    series = df[col]
                    col_key = col
                    analyzed = self._analyze_numeric_series(series)
                    analyzed["_column"] = col
                    analyzed["_dataset"] = dataset_path.name
                    analyzed["_type"] = "numeric"
                    all_attributes[col_key] = analyzed
                    raw_vals = [float(v) for v in series.dropna().tolist() if isinstance(v, (int, float, np.number))]
                    hist = self._histogram(raw_vals)
                    if hist:
                        column_distributions[col_key] = {
                            "type": "numeric",
                            "histogram": hist,
                        }
                    for metric_key, metric_val in analyzed.items():
                        if isinstance(metric_val, (int, float)):
                            flat_stats[f"{col}.{metric_key}"] = float(metric_val)
            
            for col in categorical_cols:
                if col in df.columns:
                    series = df[col]
                    col_key = col
                    analyzed = self._analyze_categorical_series(series)
                    analyzed["_column"] = col
                    analyzed["_dataset"] = dataset_path.name
                    analyzed["_type"] = "categorical"
                    all_attributes[col_key] = analyzed
                    freqs = analyzed.get("frequencies") or {}
                    if isinstance(freqs, dict) and freqs:
                        column_distributions[col_key] = {
                            "type": "categorical",
                            "frequencies": {str(k): int(v) for k, v in freqs.items()},
                        }
                    entropy_val = analyzed.get("entropy")
                    unique_val = analyzed.get("unique_count")
                    if isinstance(entropy_val, (int, float)):
                        flat_stats[f"{col}.entropy"] = float(entropy_val)
                    if isinstance(unique_val, (int, float)):
                        flat_stats[f"{col}.unique_count"] = float(unique_val)
        
        # Save cache
        if all_attributes:
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_timeseries",
                data=all_attributes
            )
        else:
            raise ValueError("No time series columns were analyzed. Check ddoc.yaml numeric_columns/categorical_columns and CSV headers.")
        
        metrics['num_series'] = len(all_attributes)
        # Evidently-style: per-column distributions instead of "distribution of column-level metrics"
        distributions_attributes: Dict[str, Any] = column_distributions
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Also persist detailed attributes to the run directory for inspection/debugging
        attrs_file = output_path / "attributes.json"
        with open(attrs_file, "w") as f:
            json.dump(all_attributes, f, indent=2)
        
        print(f"\n✅ Time Series Analysis Complete")
        
        return {
            "status": "success",
            "modality": "timeseries",
            "series_analyzed": len(all_attributes),
            "metrics_file": str(metrics_file),
            "summary": metrics,
            "stats": flat_stats,
            "attributes": all_attributes,
            "distributions_attributes": distributions_attributes or None,
        }
    
    @hookimpl
    def drift_detect(
        self,
        snapshot_id_ref: str,
        snapshot_id_cur: str,
        data_path_ref: str,
        data_path_cur: str,
        data_hash_ref: str,
        data_hash_cur: str,
        detector: str,
        cfg: Dict[str, Any],
        output_path: str
    ) -> Optional[Dict[str, Any]]:
        """Detect drift between two time series snapshots"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        def _compute_attributes_from_dir(dataset_dir: Path) -> Dict[str, Any]:
            # ddoc.yaml 스키마 기반으로 CSV를 읽고 EDA와 동일한 attribute dict를 생성
            config = self._load_ddoc_yaml(dataset_dir)
            csv_file = dataset_dir / config["csv"]
            if not csv_file.exists():
                raise ValueError(f"CSV file not found: {csv_file} (from ddoc.yaml data.csv)")
            timestamp_col = config["timestamp_column"]
            numeric_cols = config.get("numeric_columns", []) or []
            categorical_cols = config.get("categorical_columns", []) or []

            df = pd.read_csv(csv_file)
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)

            attrs: Dict[str, Any] = {}
            for col in numeric_cols:
                if col in df.columns:
                    attrs[col] = self._analyze_numeric_series(df[col])
            for col in categorical_cols:
                if col in df.columns:
                    attrs[col] = self._analyze_categorical_series(df[col])
            if not attrs:
                raise ValueError(
                    "No time series columns were analyzed. Check ddoc.yaml numeric_columns/categorical_columns and CSV headers."
                )
            return attrs

        # Load caches (깨졌거나 없으면 None)
        baseline_attr = cfg.get("baseline_cache") or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="attributes_timeseries"
        )
        
        current_attr = cfg.get("current_cache") or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="attributes_timeseries"
        )
        
        # 캐시가 없으면 on-the-fly로 계산(EDA 선행 없이도 drift 가능하게)
        if not baseline_attr:
            baseline_attr = _compute_attributes_from_dir(Path(data_path_ref))
        if not current_attr:
            current_attr = _compute_attributes_from_dir(Path(data_path_cur))

        # 스키마 불일치/고아 컬럼(orphan) 경고 로그
        try:
            bk = set(baseline_attr.keys())
            ck = set(current_attr.keys())
            only_base = sorted(bk - ck)
            only_cur = sorted(ck - bk)
            inter = sorted(bk & ck)
            if only_base:
                print(f"⚠️ [timeseries drift] Columns only in BASE (orphans): {len(only_base)} -> {only_base[:20]}")
            if only_cur:
                print(f"⚠️ [timeseries drift] Columns only in TARGET (orphans): {len(only_cur)} -> {only_cur[:20]}")
            if not inter:
                print("⚠️ [timeseries drift] No common columns between base/target. overall_score will be 0.0.")
        except Exception:
            # 경고 로깅은 best-effort
            pass
        
        drift_metrics = {
            'modality': 'timeseries',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Calculate drift for each column/metric
        attribute_drifts: Dict[str, float] = {}
        column_drifts: Dict[str, Any] = {}
        column_scores: list[float] = []
        for key in sorted(set(baseline_attr.keys()) & set(current_attr.keys())):
            baseline = baseline_attr[key]
            current = current_attr[key]

            metric_diffs: Dict[str, float] = {}
            col_type = (baseline.get("_type") or current.get("_type") or "").lower()

            if col_type == "categorical" or ("frequencies" in baseline and "frequencies" in current):
                base_entropy = self._safe_float(baseline.get("entropy"))
                cur_entropy = self._safe_float(current.get("entropy"))
                if base_entropy is not None and cur_entropy is not None:
                    metric_diffs["entropy"] = self._relative_diff(base_entropy, cur_entropy)

                base_unique = self._safe_float(baseline.get("unique_count"))
                cur_unique = self._safe_float(current.get("unique_count"))
                if base_unique is not None and cur_unique is not None:
                    metric_diffs["unique_count"] = self._relative_diff(base_unique, cur_unique)

                base_freq = baseline.get("frequencies") or {}
                cur_freq = current.get("frequencies") or {}
                if isinstance(base_freq, dict) and isinstance(cur_freq, dict):
                    metric_diffs["frequency_tv"] = self._categorical_distribution_distance(base_freq, cur_freq)
            else:
                for metric in [
                    "mean",
                    "variance",
                    "skewness",
                    "kurtosis",
                    "trend_strength",
                    "seasonal_strength",
                    "residual_strength",
                    "adf_pvalue",
                    "kpss_pvalue",
                ]:
                    base_val = self._safe_float(baseline.get(metric))
                    cur_val = self._safe_float(current.get(metric))
                    if base_val is None or cur_val is None:
                        continue
                    if metric.endswith("_pvalue"):
                        diff = float(abs(cur_val - base_val))
                    else:
                        diff = self._relative_diff(base_val, cur_val)
                    metric_diffs[metric] = diff

            for metric, score in metric_diffs.items():
                attribute_drifts[f"{key}.{metric}"] = float(score)

            if metric_diffs:
                col_score = float(np.mean(list(metric_diffs.values())))
                column_drifts[key] = {
                    "column": key,
                    "type": col_type or "numeric",
                    "score": col_score,
                    "metrics": metric_diffs,
                }
                attribute_drifts[f"{key}.overall"] = col_score
                column_scores.append(col_score)

        drift_metrics["attribute_drifts"] = attribute_drifts
        drift_metrics["column_drifts"] = column_drifts
        drift_metrics["overall_score"] = float(np.mean(column_scores)) if column_scores else 0.0
        
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        
        return drift_metrics

