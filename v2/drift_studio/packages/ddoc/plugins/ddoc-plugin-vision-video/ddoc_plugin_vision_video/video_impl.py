from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:

    def hookimpl(func):
        return func


from .ffmpeg_utils import extract_frames, probe_video
from .frame_metrics import analyze_frame, motion_score


class DDOCVisionVideoPlugin:
    """
    vision_video plugin:
    - Use ffprobe for metadata
    - Use ffmpeg to sample frames
    - Compute frame quality (brightness/contrast/sharpness) + temporal motion proxy
    """

    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if config.get("modality") != "vision_video":
            raise ValueError(f"Dataset {dataset_path} is not configured as vision_video modality")
        return config

    def _iter_videos(self, root: Path) -> list[Path]:
        exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

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
        video_root = (dataset_root_abs / data_dir).resolve()
        if not video_root.exists() or not video_root.is_dir():
            raise ValueError(f"vision_video data_dir not found: {data_dir}")

        sample_fps = float(config.get("data", {}).get("sample_fps", 1.0) or 1.0)
        max_frames = int(config.get("data", {}).get("max_frames_per_video", 64) or 64)
        sample_fps = max(0.1, min(sample_fps, 10.0))
        max_frames = max(1, min(max_frames, 500))

        # CacheService는 JSON 캐시 타입으로 summary/file_metadata/attributes_* 만 허용합니다.
        # video는 per-video summary를 저장하므로 attributes 네임스페이스를 사용합니다.
        cache_type = "attributes_vision_video"
        cached: dict[str, Any] = {}
        if not invalidate_cache:
            cached = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id, data_hash=data_hash, cache_type=cache_type
            ) or {}

        summaries: dict[str, Any] = {}
        videos = self._iter_videos(video_root)
        tmp_root = out_root / "_frames_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

        for vp in videos:
            # vp와 dataset_root 모두 절대 경로로 변환하여 relative_to 사용
            vp_abs = Path(vp).resolve()
            rel = str(vp_abs.relative_to(dataset_root_abs))
            if rel in cached:
                summaries[rel] = cached[rel]
                continue

            meta = probe_video(vp)
            frames_dir = tmp_root / rel.replace("/", "_")
            frames = extract_frames(video_path=vp, out_dir=frames_dir, sample_fps=sample_fps, max_frames=max_frames)

            frame_stats = []
            gray_frames = []
            for fp in frames:
                st = analyze_frame(fp)
                frame_stats.append(st)
                # reuse grayscale for motion proxy
                try:
                    from PIL import Image
                    import numpy as _np

                    with Image.open(fp) as img:
                        g = _np.asarray(img.convert("L"), dtype=_np.float32) / 255.0
                    gray_frames.append(g)
                except Exception:
                    pass

            def _avg(key: str) -> Optional[float]:
                vals = [x.get(key) for x in frame_stats if isinstance(x.get(key), (int, float))]
                return float(np.mean(vals)) if vals else None

            summary = {
                "video_meta": {
                    "duration_sec": meta.duration_sec,
                    "fps": meta.fps,
                    "width": meta.width,
                    "height": meta.height,
                    "codec": meta.codec,
                },
                "sampling": {"sample_fps": sample_fps, "max_frames": max_frames, "frames_extracted": len(frames)},
                "frame_quality": {
                    "brightness_mean": _avg("brightness"),
                    "contrast_mean": _avg("contrast"),
                    "sharpness_mean": _avg("sharpness"),
                },
                "temporal": {"motion_score": motion_score(gray_frames)},
            }
            summaries[rel] = summary

        cache_service.save_analysis_cache(
            snapshot_id=snapshot_id, data_hash=data_hash, cache_type=cache_type, data=summaries
        )

        metrics = {
            "snapshot_id": snapshot_id,
            "data_hash": data_hash,
            "modality": "vision_video",
            "num_videos": len(videos),
            "sample_fps": sample_fps,
            "max_frames_per_video": max_frames,
        }
        metrics_file = out_root / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        # cleanup tmp frames
        shutil.rmtree(tmp_root, ignore_errors=True)

        return {
            "status": "success",
            "modality": "vision_video",
            "files_analyzed": len(videos),
            "metrics_file": str(metrics_file),
            "summary": metrics,
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

        cache_type = "attributes_vision_video"
        ref = cfg.get("baseline_cache") or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref, data_hash=data_hash_ref, cache_type=cache_type
        )
        cur = cfg.get("current_cache") or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur, data_hash=data_hash_cur, cache_type=cache_type
        )
        if not ref or not cur:
            return None

        def _extract(arr: dict[str, Any], path: str, *keys: str) -> Optional[float]:
            d = (arr or {}).get(path) or {}
            for k in keys:
                if not isinstance(d, dict):
                    return None
                d = d.get(k)
            return float(d) if isinstance(d, (int, float)) else None

        # compare distributions across common files
        ref_keys = set(ref.keys())
        cur_keys = set(cur.keys())
        common = sorted(ref_keys & cur_keys)

        def _collect(keys: list[str]) -> np.ndarray:
            vals = []
            for p in common:
                v = _extract(cur, p, *keys)
                r = _extract(ref, p, *keys)
                # store pairwise differences? We instead collect ref and cur separately below
                _ = r
                if v is not None:
                    vals.append(v)
            return np.array(vals, dtype=np.float64)

        def _collect_ref(keys: list[str]) -> np.ndarray:
            vals = []
            for p in common:
                v = _extract(ref, p, *keys)
                if v is not None:
                    vals.append(v)
            return np.array(vals, dtype=np.float64)

        def _psi(r: np.ndarray, c: np.ndarray, bins: int = 10) -> float:
            r = r[np.isfinite(r)]
            c = c[np.isfinite(c)]
            if r.size == 0 or c.size == 0:
                return 0.0
            qs = np.linspace(0.0, 1.0, bins + 1)
            edges = np.quantile(r, qs)
            edges[0] = -np.inf
            edges[-1] = np.inf
            rh, _ = np.histogram(r, bins=edges)
            ch, _ = np.histogram(c, bins=edges)
            rp = rh / max(1, rh.sum())
            cp = ch / max(1, ch.sum())
            eps = 1e-6
            rp = np.clip(rp, eps, 1.0)
            cp = np.clip(cp, eps, 1.0)
            return float(np.sum((cp - rp) * np.log(cp / rp)))

        metrics = {}
        for name, kpath in [
            ("duration_sec", ["video_meta", "duration_sec"]),
            ("fps", ["video_meta", "fps"]),
            ("brightness_mean", ["frame_quality", "brightness_mean"]),
            ("sharpness_mean", ["frame_quality", "sharpness_mean"]),
            ("motion_score", ["temporal", "motion_score"]),
        ]:
            r = _collect_ref(kpath)
            c = _collect(kpath)
            metrics[name] = _psi(r, c)

        overall = float(np.mean(list(metrics.values()))) if metrics else 0.0
        out = {"modality": "vision_video", "overall_score": overall, "metric_psi": metrics}
        (out_root / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

