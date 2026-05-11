from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:

    def hookimpl(func):
        return func


from .midi_metrics import compute_js_divergence, compute_midi_metrics, compute_psi
from .midi_parser import MidiParseError, parse_midi_bytes


class DOCAudioMidiPlugin:
    """
    audio_midi plugin (symbolic features from MIDI + JSON label metadata)
    - EDA: per-file metrics + label distribution summary
    - Drift: PSI for numeric metrics + JS divergence for categorical labels
    """

    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if config.get("modality") != "audio_midi":
            raise ValueError(f"Dataset {dataset_path} is not configured as audio_midi modality")
        return config

    def _iter_midi_files(self, root: Path) -> list[Path]:
        exts = {".mid", ".midi"}
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    def _label_path_for(self, midi_path: Path) -> Path:
        return midi_path.with_suffix(".json")

    def _load_label(self, json_path: Path) -> Optional[dict[str, Any]]:
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _extract_label_fields(self, label: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Extract common fields from the loop dataset JSON schema (best-effort).
        """
        if not isinstance(label, dict):
            return {}
        ds = label.get("dataSet") if isinstance(label.get("dataSet"), dict) else label
        loop_info = ds.get("loopInfo") if isinstance(ds.get("loopInfo"), dict) else {}

        # keep values compact (strings/numbers only)
        return {
            "genre": loop_info.get("genre"),
            "bpm": loop_info.get("bpm"),
            "musicStyle": loop_info.get("musicStyle"),
            "songForm": loop_info.get("songForm"),
            "scale": loop_info.get("scale"),
            "InstrumentName": loop_info.get("InstrumentName"),
            "InstrumentType": ds.get("InstrumentType"),
            "classification": ds.get("classification"),
            "loopType": ds.get("loopType"),
            "barCount": ds.get("barCount"),
            "beatCount": loop_info.get("beatCount"),
            "isMain": ds.get("isMain"),
        }

    def _histogram(self, values: list[float], bins: int = 20, max_samples: int = 2000) -> dict[str, Any] | None:
        if not values:
            return None
        counts, edges = np.histogram(values, bins=bins)
        samples = values
        if len(values) > max_samples:
            idx = np.random.choice(len(values), size=max_samples, replace=False)
            samples = [values[i] for i in idx]
        return {"bins": edges.tolist(), "counts": counts.tolist(), "samples": samples}

    @hookimpl
    def eda_run(self, snapshot_id, data_path, data_hash, output_path, invalidate_cache=False):
        from ddoc.core.cache_service import get_cache_service

        cache_service = get_cache_service()
        dataset_root = Path(data_path)
        out_root = Path(output_path)
        out_root.mkdir(parents=True, exist_ok=True)

        config = self._load_ddoc_yaml(dataset_root)
        data_dir = config.get("data", {}).get("data_dir", ".")
        # dataset_root를 절대 경로로 변환하여 일관성 유지
        dataset_root_abs = Path(dataset_root).resolve()
        midi_root = (dataset_root_abs / data_dir).resolve()
        if not midi_root.exists() or not midi_root.is_dir():
            raise ValueError(f"audio_midi data_dir not found: {data_dir}")

        # cache
        cache_type = "attributes_audio_midi"
        cached: dict[str, Any] = {}
        if not invalidate_cache:
            cached = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id, data_hash=data_hash, cache_type=cache_type
            ) or {}

        results: dict[str, Any] = {}
        label_counters: dict[str, Counter] = {
            "genre": Counter(),
            "musicStyle": Counter(),
            "songForm": Counter(),
            "scale": Counter(),
            "InstrumentName": Counter(),
            "InstrumentType": Counter(),
            "classification": Counter(),
            "loopType": Counter(),
            "isMain": Counter(),
        }
        bpm_vals: list[float] = []

        midi_files = self._iter_midi_files(midi_root)
        for midi_file in midi_files:
            # midi_file과 dataset_root 모두 절대 경로로 변환하여 relative_to 사용
            midi_file_abs = Path(midi_file).resolve()
            rel = str(midi_file_abs.relative_to(dataset_root_abs))
            if rel in cached:
                results[rel] = cached[rel]
                continue

            json_path = self._label_path_for(midi_file)
            label = self._load_label(json_path) if json_path.exists() else None
            label_fields = self._extract_label_fields(label)

            try:
                parsed = parse_midi_bytes(midi_file.read_bytes())
                midi_metrics = compute_midi_metrics(parsed)
            except (MidiParseError, OSError) as e:
                midi_metrics = {"error": f"midi_parse_failed: {e}"}

            row = {"midi": midi_metrics, "label": label_fields}
            results[rel] = row

            # update label stats
            for k, counter in label_counters.items():
                v = label_fields.get(k)
                if v is None or v == "":
                    continue
                counter[str(v)] += 1
            bpm = label_fields.get("bpm")
            try:
                if bpm is not None:
                    bpm_vals.append(float(bpm))
            except Exception:
                pass

        # save cache (store per-file result)
        cache_service.save_analysis_cache(
            snapshot_id=snapshot_id, data_hash=data_hash, cache_type=cache_type, data=results
        )

        # summarize numeric midi metrics
        def _collect(metric_key: str) -> np.ndarray:
            vals = []
            for v in results.values():
                mm = (v or {}).get("midi") or {}
                if isinstance(mm, dict) and metric_key in mm and isinstance(mm[metric_key], (int, float)):
                    vals.append(float(mm[metric_key]))
            return np.array(vals, dtype=np.float64)

        notes_per_sec = _collect("notes_per_sec")
        pitch_range = _collect("pitch_range")
        vel_mean = _collect("velocity_mean")

        summary = {
            "timestamp": None,
            "snapshot_id": snapshot_id,
            "data_hash": data_hash,
            "modality": "audio_midi",
            "num_files": len(results),
            "num_midi_files": len(midi_files),
            "bpm_mean": float(np.mean(bpm_vals)) if bpm_vals else None,
            "bpm_std": float(np.std(bpm_vals)) if bpm_vals else None,
            "notes_per_sec_mean": float(np.mean(notes_per_sec)) if notes_per_sec.size else None,
            "pitch_range_mean": float(np.mean(pitch_range)) if pitch_range.size else None,
            "velocity_mean_mean": float(np.mean(vel_mean)) if vel_mean.size else None,
            "label_distributions": {k: dict(c.most_common(30)) for k, c in label_counters.items()},
        }

        distributions = {}
        if bpm_vals:
            hist = self._histogram([float(x) for x in bpm_vals])
            if hist:
                distributions["bpm"] = hist
        if notes_per_sec.size:
            hist = self._histogram(notes_per_sec.tolist())
            if hist:
                distributions["notes_per_sec"] = hist
        if pitch_range.size:
            hist = self._histogram(pitch_range.tolist())
            if hist:
                distributions["pitch_range"] = hist
        if vel_mean.size:
            hist = self._histogram(vel_mean.tolist())
            if hist:
                distributions["velocity_mean"] = hist

        metrics_file = out_root / "metrics.json"
        metrics_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "status": "success",
            "modality": "audio_midi",
            "files_analyzed": len(results),
            "metrics_file": str(metrics_file),
            "summary": summary,
            "distributions": distributions or None,
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
    ) -> Optional[Dict[str, Any]]:
        from ddoc.core.cache_service import get_cache_service

        cache_service = get_cache_service()
        out_root = Path(output_path)
        out_root.mkdir(parents=True, exist_ok=True)

        cache_type = "attributes_audio_midi"
        ref = cfg.get("baseline_cache") or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref, data_hash=data_hash_ref, cache_type=cache_type
        )
        cur = cfg.get("current_cache") or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur, data_hash=data_hash_cur, cache_type=cache_type
        )
        if not ref or not cur:
            return None

        def _num(arr: dict[str, Any], key: str) -> np.ndarray:
            vals = []
            for v in (arr or {}).values():
                mm = (v or {}).get("midi") or {}
                if isinstance(mm, dict) and key in mm and isinstance(mm[key], (int, float)):
                    vals.append(float(mm[key]))
            return np.array(vals, dtype=np.float64)

        # numeric drift (PSI)
        numeric_keys = ["notes_per_sec", "pitch_range", "velocity_mean", "polyphony_bucket_mean"]
        numeric_drift: dict[str, float] = {}
        for k in numeric_keys:
            r = _num(ref, k)
            c = _num(cur, k)
            numeric_drift[k] = compute_psi(r, c) if (r.size and c.size) else 0.0

        # categorical drift (JS) from labels
        def _cat(arr: dict[str, Any], key: str) -> Counter:
            c = Counter()
            for v in (arr or {}).values():
                lb = (v or {}).get("label") or {}
                if isinstance(lb, dict):
                    vv = lb.get(key)
                    if vv is None or vv == "":
                        continue
                    c[str(vv)] += 1
            return c

        cat_keys = ["genre", "musicStyle", "songForm", "scale", "InstrumentName", "loopType"]
        cat_drift: dict[str, float] = {}
        for k in cat_keys:
            cat_drift[k] = compute_js_divergence(_cat(ref, k), _cat(cur, k))

        overall = float(
            np.mean(list(numeric_drift.values()) + list(cat_drift.values()))
            if (numeric_drift or cat_drift)
            else 0.0
        )

        metrics = {
            "modality": "audio_midi",
            "timestamp": None,
            "numeric_drift": numeric_drift,
            "categorical_drift": cat_drift,
            "overall_score": overall,
        }

        metrics_file = out_root / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

