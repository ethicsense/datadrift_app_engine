"""
MLflow Log Analysis Plugin for ddoc.

- eda_run: parse MLflow file-store logs and return run/params/metrics/artifacts summary
- drift_detect: compare two MLflow log datasets (aggregate + matched pairs)
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import base64
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:

    def hookimpl(func):
        return func


_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except Exception:
        return v


def _parse_metrics_file(path: Path) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            ts = int(parts[0])
            value = float(parts[1])
            step = int(parts[2])
        except Exception:
            continue
        series.append({"timestamp": ts, "value": value, "step": step})
    series.sort(key=lambda item: item.get("step", 0))
    return series


def _select_overlay_metrics(metric_keys: Iterable[str], limit: int = 8) -> list[str]:
    def _priority(name: str) -> int:
        lowered = name.lower()
        if lowered.startswith("metrics/metrics"):
            return 0
        if lowered.startswith("metrics/val"):
            return 1
        if lowered.startswith("metrics/train"):
            return 2
        if lowered.startswith("metrics/lr"):
            return 3
        return 4

    keys = sorted(set(metric_keys), key=lambda k: (_priority(k), k))
    return keys[:limit]


def _distribution_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    vals = list(values)
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def _final_metric(series: list[dict[str, Any]]) -> Optional[float]:
    if not series:
        return None
    last = max(series, key=lambda item: item.get("step", 0))
    return last.get("value")


def _curve_mse(base: list[dict[str, Any]], target: list[dict[str, Any]]) -> Optional[float]:
    if not base or not target:
        return None
    base_map = {item["step"]: item["value"] for item in base if "step" in item and "value" in item}
    target_map = {item["step"]: item["value"] for item in target if "step" in item and "value" in item}
    common_steps = sorted(set(base_map) & set(target_map))
    if not common_steps:
        return None
    diffs = [(base_map[s] - target_map[s]) ** 2 for s in common_steps]
    return sum(diffs) / len(diffs)


@dataclass
class RunRecord:
    run_id: str
    run_name: str
    experiment_id: str
    user_id: Optional[str]
    status: Optional[int]
    start_time: Optional[int]
    end_time: Optional[int]
    artifact_uri: Optional[str]
    tags: dict[str, Any]
    run_dir: Path


def _scan_tracking_dir(
    tracking_dir: Path,
    *,
    default_experiment_id: Optional[str] = None,
    default_run_id: Optional[str] = None,
) -> tuple[list[dict[str, Any]], list[RunRecord], dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]]:
    experiments: list[dict[str, Any]] = []
    runs: list[RunRecord] = []
    params_index: dict[str, dict[str, Any]] = {}
    metrics_index: dict[str, dict[str, list[dict[str, Any]]]] = {}
    artifacts_index: dict[str, list[dict[str, Any]]] = {}

    for exp_dir in sorted(p for p in tracking_dir.iterdir() if p.is_dir()):
        exp_meta_path = exp_dir / "meta.yaml"
        if not exp_meta_path.exists():
            continue
        exp_meta = _read_yaml(exp_meta_path)
        experiment_id = str(exp_meta.get("experiment_id") or exp_dir.name)
        if default_experiment_id and experiment_id != str(default_experiment_id):
            continue
        experiments.append(
            {
                "experiment_id": experiment_id,
                "name": exp_meta.get("name"),
                "artifact_location": exp_meta.get("artifact_location"),
                "creation_time": exp_meta.get("creation_time"),
                "last_update_time": exp_meta.get("last_update_time"),
                "lifecycle_stage": exp_meta.get("lifecycle_stage"),
            }
        )

        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            run_meta_path = run_dir / "meta.yaml"
            if not run_meta_path.exists():
                continue
            run_meta = _read_yaml(run_meta_path)
            run_id = str(run_meta.get("run_id") or run_dir.name)
            if default_run_id and run_id != str(default_run_id):
                continue

            tags_dir = run_dir / "tags"
            tags: dict[str, Any] = {}
            if tags_dir.exists():
                for tag_file in tags_dir.iterdir():
                    if tag_file.is_file():
                        tags[tag_file.name] = _read_text(tag_file)
            run_name = str(run_meta.get("run_name") or tags.get("mlflow.runName") or run_id)

            record = RunRecord(
                run_id=run_id,
                run_name=run_name,
                experiment_id=str(run_meta.get("experiment_id") or experiment_id),
                user_id=run_meta.get("user_id") or tags.get("mlflow.user"),
                status=run_meta.get("status"),
                start_time=run_meta.get("start_time"),
                end_time=run_meta.get("end_time"),
                artifact_uri=run_meta.get("artifact_uri"),
                tags=tags,
                run_dir=run_dir,
            )
            runs.append(record)

            params_dir = run_dir / "params"
            params: dict[str, Any] = {}
            if params_dir.exists():
                for param_file in params_dir.iterdir():
                    if param_file.is_file():
                        params[param_file.name] = _parse_scalar(_read_text(param_file))
            params_index[run_id] = params

            metrics_dir = run_dir / "metrics"
            metric_map: dict[str, list[dict[str, Any]]] = {}
            if metrics_dir.exists():
                for metric_file in metrics_dir.rglob("*"):
                    if not metric_file.is_file():
                        continue
                    rel_name = metric_file.relative_to(metrics_dir).as_posix()
                    series = _parse_metrics_file(metric_file)
                    if series:
                        metric_map[f"metrics/{rel_name}"] = series
            metrics_index[run_id] = metric_map

            artifacts_dir = run_dir / "artifacts"
            artifacts: list[dict[str, Any]] = []
            if artifacts_dir.exists():
                for artifact in artifacts_dir.rglob("*"):
                    if not artifact.is_file():
                        continue
                    rel_path = artifact.relative_to(artifacts_dir).as_posix()
                    ext = artifact.suffix.lower()
                    artifacts.append(
                        {
                            "path": rel_path,
                            "size_bytes": artifact.stat().st_size,
                            "ext": ext,
                            "kind": "image" if ext in _IMAGE_EXT else "file",
                        }
                    )
            artifacts_index[run_id] = artifacts

    runs.sort(key=lambda r: r.start_time or 0, reverse=True)
    return experiments, runs, params_index, metrics_index, artifacts_index


def _summary_from_runs(
    experiments: list[dict[str, Any]],
    runs: list[RunRecord],
    metrics_index: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    start_times = [r.start_time for r in runs if r.start_time]
    end_times = [r.end_time for r in runs if r.end_time]
    earliest_run_id = None
    if runs:
        earliest_run = min(runs, key=lambda r: r.start_time if r.start_time is not None else float("inf"))
        earliest_run_id = earliest_run.run_id if earliest_run.start_time is not None else None
    metric_keys: list[str] = []
    for metrics in metrics_index.values():
        metric_keys.extend(metrics.keys())
    overlay_defaults = _select_overlay_metrics(metric_keys, limit=8)
    return {
        "total_runs": len(runs),
        "total_experiments": len(experiments),
        "earliest_start_time": min(start_times) if start_times else None,
        "latest_end_time": max(end_times) if end_times else None,
        "metric_count": len(set(metric_keys)),
        "overlay_defaults": overlay_defaults,
        "earliest_run_id": earliest_run_id,
        "latest_run_id": runs[0].run_id if runs else None,
    }


def _is_loss_metric(name: str) -> bool:
    lowered = name.lower()
    return "loss" in lowered or "/loss" in lowered


def _best_value(series: list[dict[str, Any]], *, prefer_min: bool) -> tuple[Optional[float], Optional[int]]:
    if not series:
        return None, None
    best = None
    for item in series:
        value = item.get("value")
        if value is None:
            continue
        if best is None:
            best = value
        else:
            cur_val = best
            if prefer_min and value < cur_val:
                best = value
            if not prefer_min and value > cur_val:
                best = value
    if best is None:
        return None, None
    return best


def _performance_summary(
    runs: list[RunRecord],
    metrics_index: dict[str, dict[str, list[dict[str, Any]]]],
    overlay_defaults: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_keys = set(overlay_defaults)
    for metrics in metrics_index.values():
        for key in metrics.keys():
            if _is_loss_metric(key):
                metric_keys.add(key)
    for metric in sorted(metric_keys):
        best_value = None
        prefer_min = _is_loss_metric(metric)
        for run in runs:
            series = metrics_index.get(run.run_id, {}).get(metric, [])
            value = _best_value(series, prefer_min=prefer_min)
            if value is None:
                continue
            if best_value is None:
                best_value = value
            else:
                if prefer_min and value < best_value:
                    best_value = value
                if not prefer_min and value > best_value:
                    best_value = value
        if best_value is not None:
            summary[f"best.{metric}"] = best_value

    return summary


def _build_mlflow_guide(tracking_dir: Path) -> dict[str, Any]:
    return {
        "tracking_dir": str(tracking_dir),
        "command": f"mlflow ui --backend-store-uri \"{tracking_dir}\"",
        "note": "로컬에서 위 명령으로 MLflow UI를 실행할 수 있습니다.",
    }


def _run_signature(run: RunRecord, params: dict[str, Any], keys: list[str]) -> str:
    items = [f"{k}={params.get(k)}" for k in keys if k in params]
    base = f"{run.run_name}|{';'.join(items)}"
    return sha1(base.encode("utf-8")).hexdigest()[:12]


def _resolve_tracking_dir(root: Path, tracking_dir_raw: str) -> Path:
    raw = tracking_dir_raw.strip() if tracking_dir_raw else ""
    if not raw or raw.lower() == "auto":
        found = _find_tracking_dir(root)
        return found or (root / "mlruns")
    return (root / raw).resolve()


def _find_tracking_dir(root: Path, max_depth: int = 4) -> Path | None:
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        current = Path(dirpath)
        if current.name not in {"mlruns", "mlflow"}:
            continue
        for child in current.iterdir():
            if child.is_dir() and (child / "meta.yaml").exists():
                return current
    return None


def _select_preview_image(
    runs: list[RunRecord],
    artifacts_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    preferred = ("val_batch", "pred")
    sorted_runs = sorted(runs, key=lambda r: r.start_time or 0, reverse=True)
    for run in sorted_runs:
        artifacts = artifacts_index.get(run.run_id, [])
        for artifact in artifacts:
            path = artifact.get("path") or ""
            lower = path.lower()
            if any(key in lower for key in preferred) and lower.endswith(tuple(_IMAGE_EXT)):
                return {"run": run, "path": path}
        for artifact in artifacts:
            path = artifact.get("path") or ""
            if path.lower().endswith(tuple(_IMAGE_EXT)):
                return {"run": run, "path": path}
    return None


class DOCMlflowLogPlugin:
    DDOC_HOOKSPEC_MIN = "1.0.0"
    DDOC_HOOKSPEC_MAX = "2.0.0"

    @hookimpl
    def eda_run(
        self,
        snapshot_id: str,
        data_path: str,
        data_hash: str,
        output_path: str,
        invalidate_cache: bool = False,
    ) -> dict[str, Any]:
        root = Path(data_path)
        meta_path = root / "ddoc.yaml"
        meta = _read_yaml(meta_path)
        data = meta.get("data") or {}
        tracking_dir = _resolve_tracking_dir(root, str(data.get("tracking_dir") or "auto"))
        default_experiment_id = data.get("default_experiment_id")
        default_run_id = data.get("default_run_id")

        experiments, runs, params_index, metrics_index, artifacts_index = _scan_tracking_dir(
            tracking_dir,
            default_experiment_id=default_experiment_id,
            default_run_id=default_run_id,
        )

        run_entries = [
            {
                "run_id": r.run_id,
                "run_name": r.run_name,
                "experiment_id": r.experiment_id,
                "user_id": r.user_id,
                "status": r.status,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "artifact_uri": r.artifact_uri,
                "tags": r.tags,
            }
            for r in runs
        ]

        meta_summary = _summary_from_runs(experiments, runs, metrics_index)
        performance_summary = _performance_summary(
            runs, metrics_index, meta_summary.get("overlay_defaults") or []
        )
        preview_payload = None
        preview = _select_preview_image(runs, artifacts_index)
        if preview:
            run = preview["run"]
            rel_path = preview["path"]
            img_path = run.run_dir / "artifacts" / rel_path
            try:
                raw = img_path.read_bytes()
                preview_payload = {
                    "run_id": run.run_id,
                    "path": rel_path,
                    "mime": "image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png",
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            except Exception:
                preview_payload = None
        return {
            "summary": performance_summary,
            "trainlog_meta_summary": meta_summary,
            "trainlog_runs": run_entries,
            "trainlog_params": params_index,
            "trainlog_metrics": metrics_index,
            "trainlog_artifacts": artifacts_index,
            "trainlog_mlflow_ui": _build_mlflow_guide(tracking_dir),
            "trainlog_preview_image": preview_payload,
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
        output_path: str,
    ) -> dict[str, Any]:
        base_root = Path(data_path_ref)
        target_root = Path(data_path_cur)

        base_meta = _read_yaml(base_root / "ddoc.yaml")
        target_meta = _read_yaml(target_root / "ddoc.yaml")
        base_tracking = _resolve_tracking_dir(
            base_root, str((base_meta.get("data") or {}).get("tracking_dir") or "auto")
        )
        target_tracking = _resolve_tracking_dir(
            target_root, str((target_meta.get("data") or {}).get("tracking_dir") or "auto")
        )

        base_exp, base_runs, base_params, base_metrics, _ = _scan_tracking_dir(base_tracking)
        target_exp, target_runs, target_params, target_metrics, _ = _scan_tracking_dir(target_tracking)

        def _final_metrics(metrics_index: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, float]]:
            result: dict[str, dict[str, float]] = {}
            for run_id, metrics in metrics_index.items():
                final_vals: dict[str, float] = {}
                for key, series in metrics.items():
                    value = _final_metric(series)
                    if value is not None:
                        final_vals[key] = value
                result[run_id] = final_vals
            return result

        base_final = _final_metrics(base_metrics)
        target_final = _final_metrics(target_metrics)

        aggregate: dict[str, Any] = {
            "base": {"runs": len(base_runs), "experiments": len(base_exp)},
            "target": {"runs": len(target_runs), "experiments": len(target_exp)},
            "metrics": {},
            "params": {},
        }

        all_metric_keys = sorted(
            set(k for m in base_final.values() for k in m.keys())
            | set(k for m in target_final.values() for k in m.keys())
        )
        for metric in all_metric_keys:
            base_vals = [vals.get(metric) for vals in base_final.values() if metric in vals]
            target_vals = [vals.get(metric) for vals in target_final.values() if metric in vals]
            base_stats = _distribution_stats([v for v in base_vals if v is not None])
            target_stats = _distribution_stats([v for v in target_vals if v is not None])
            if base_stats.get("n") and target_stats.get("n"):
                std_sum = (base_stats.get("std") or 0) + (target_stats.get("std") or 0)
                normalized = (
                    abs((base_stats.get("mean") or 0) - (target_stats.get("mean") or 0)) / (std_sum or 1e-9)
                )
            else:
                normalized = None
            aggregate["metrics"][metric] = {
                "base": base_stats,
                "target": target_stats,
                "delta_mean": (target_stats.get("mean") or 0) - (base_stats.get("mean") or 0)
                if base_stats.get("n") and target_stats.get("n")
                else None,
                "normalized_delta": normalized,
            }

        param_keys = sorted(
            set(k for p in base_params.values() for k in p.keys())
            | set(k for p in target_params.values() for k in p.keys())
        )
        changed_keys: list[str] = []
        for key in param_keys:
            base_set = {str(p.get(key)) for p in base_params.values() if key in p}
            target_set = {str(p.get(key)) for p in target_params.values() if key in p}
            if base_set != target_set:
                changed_keys.append(key)
        aggregate["params"] = {
            "total_keys": len(param_keys),
            "changed_keys": changed_keys[:30],
            "changed_ratio": len(changed_keys) / len(param_keys) if param_keys else 0.0,
        }

        key_fields = ["model", "data", "imgsz", "batch", "lr0", "optimizer", "seed"]
        base_signatures: dict[str, list[RunRecord]] = {}
        target_signatures: dict[str, list[RunRecord]] = {}
        for run in base_runs:
            sig = _run_signature(run, base_params.get(run.run_id, {}), key_fields)
            base_signatures.setdefault(sig, []).append(run)
        for run in target_runs:
            sig = _run_signature(run, target_params.get(run.run_id, {}), key_fields)
            target_signatures.setdefault(sig, []).append(run)

        pairs: list[dict[str, Any]] = []
        for sig, base_list in base_signatures.items():
            target_list = target_signatures.get(sig)
            if not target_list:
                continue
            base_sorted = sorted(base_list, key=lambda r: r.start_time or 0, reverse=True)
            target_sorted = sorted(target_list, key=lambda r: r.start_time or 0, reverse=True)
            for idx in range(min(len(base_sorted), len(target_sorted))):
                base_run = base_sorted[idx]
                target_run = target_sorted[idx]
                base_metrics_map = base_metrics.get(base_run.run_id, {})
                target_metrics_map = target_metrics.get(target_run.run_id, {})
                common_metrics = sorted(set(base_metrics_map) & set(target_metrics_map))
                deltas: dict[str, Any] = {}
                curve_mse: dict[str, Any] = {}
                for metric in common_metrics:
                    base_final_val = _final_metric(base_metrics_map.get(metric, []))
                    target_final_val = _final_metric(target_metrics_map.get(metric, []))
                    if base_final_val is not None and target_final_val is not None:
                        deltas[metric] = target_final_val - base_final_val
                    mse = _curve_mse(base_metrics_map.get(metric, []), target_metrics_map.get(metric, []))
                    if mse is not None:
                        curve_mse[metric] = mse
                pairs.append(
                    {
                        "signature": sig,
                        "base_run_id": base_run.run_id,
                        "target_run_id": target_run.run_id,
                        "base_run_name": base_run.run_name,
                        "target_run_name": target_run.run_name,
                        "delta_final_metrics": deltas,
                        "curve_mse": curve_mse,
                    }
                )

        return {
            "trainlog_drift_aggregate": aggregate,
            "trainlog_drift_pairs": pairs,
        }
