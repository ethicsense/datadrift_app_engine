from __future__ import annotations

import hashlib
import random
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from driftstudio_spec import (
    ArtifactEntry,
    ArtifactIndex,
    ArtifactPaths,
    ArtifactPayloadInline,
    ArtifactPayloadRef,
    Plan,
    ReportFormat,
    StepType,
)

from driftstudio_runtime.executors.python_executor import PythonExecutor
from driftstudio_runtime.infer import infer_dtype, infer_modality, validate_dataset_dir_or_raise
from driftstudio_runtime.zip_resolver import extract_zip_dataset


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _sample_indices(total: int, n: int, *, seed: int) -> list[int]:
    if n >= total:
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(range(total), n)


def _pca_projection(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
        return coords.astype(np.float32)
    except Exception:
        return np.empty((0, 2), dtype=np.float32)


def _histogram_pair(ref_values: list[float], cur_values: list[float], *, bins: int = 20) -> dict[str, Any] | None:
    if not ref_values or not cur_values:
        return None
    ref_arr = np.asarray(ref_values, dtype=np.float64)
    cur_arr = np.asarray(cur_values, dtype=np.float64)
    combined = np.concatenate([ref_arr, cur_arr])
    if combined.size == 0:
        return None
    edges = np.histogram_bin_edges(combined, bins=bins)
    ref_counts, _ = np.histogram(ref_arr, bins=edges)
    cur_counts, _ = np.histogram(cur_arr, bins=edges)
    return {
        "base": {"bins": edges.tolist(), "counts": ref_counts.tolist()},
        "target": {"bins": edges.tolist(), "counts": cur_counts.tolist()},
    }


def _read_index(path: Path) -> ArtifactIndex:
    payload = _read_json(path)
    return ArtifactIndex.parse_obj(payload)


def _write_index(path: Path, index: ArtifactIndex) -> None:
    _write_json(path, index.dict())


def _artifact_payload_for_json(root: Path, artifact_id: str, payload: Any) -> ArtifactPayloadRef:
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_id(artifact_id)}.json"
    payload_path = artifacts_dir / filename
    _write_json(payload_path, payload)
    rel_uri = f"artifacts/{filename}"
    size_bytes = payload_path.stat().st_size if payload_path.exists() else None
    return ArtifactPayloadRef(uri=rel_uri, content_type="application/json", size_bytes=size_bytes)


def _find_artifact(index: ArtifactIndex, *, artifact_type: str, meta_match: dict[str, Any] | None = None) -> ArtifactEntry | None:
    for artifact in index.artifacts:
        if artifact.type != artifact_type:
            continue
        if meta_match:
            meta = artifact.meta or {}
            if any(meta.get(k) != v for k, v in meta_match.items()):
                continue
        return artifact
    return None


def _load_artifact_payload(root: Path, artifact: ArtifactEntry) -> Any:
    if isinstance(artifact.payload, ArtifactPayloadInline):
        return artifact.payload.data
    if isinstance(artifact.payload, ArtifactPayloadRef):
        return _read_json(root / artifact.payload.uri)
    return None


def _add_artifact(
    *,
    index: ArtifactIndex,
    artifact_id: str,
    artifact_type: str,
    payload: ArtifactPayloadInline | ArtifactPayloadRef,
    title: str | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    entry = ArtifactEntry(
        id=artifact_id,
        type=artifact_type,
        title=title,
        tags=tags or [],
        meta=meta,
        payload=payload,
    )
    index.artifacts.append(entry)


class RuntimeRunner:
    """
    최소 Runner.

    - 캐시: 동일 산출물 파일이 존재하면 재계산을 생략(옵션 확장 가능)
    - 저장: 표준 artifact 경로(`ArtifactPaths`)에 json/html/pdf 기록
    """

    def __init__(self, *, force: bool = False):
        self.force = force

    def run(self, plan: Plan) -> dict[str, Any]:
        # 입력은 zip 또는 extracted dir을 허용(내부 처리 시 extracted dir로 정규화)
        plan = self._normalize_plan_inputs(plan)

        # modality auto 지원 (ddoc.yaml 기반 추론; 없으면 예외)
        if plan.modality in ["auto", "", None]:
            inferred = infer_modality(plan.base_path or plan.target_path or "")
            plan = plan.model_copy(update={"modality": inferred})

        def _snapshot_id_for(path: str) -> str:
            # dataset identity로 절대경로를 사용(프로세스/러너 내부 최소 구현)
            try:
                p = str(Path(path).resolve())
            except Exception:
                p = str(path)
            return "path:" + hashlib.sha1(p.encode("utf-8")).hexdigest()[:16]

        def _fingerprint_dir(root: str) -> str:
            """
            데이터셋 버전을 구분하기 위한 경량 fingerprint.
            - 파일 내용까지 읽지 않고 (rel_path, size, mtime) 기반으로 해시 생성
            - drift/eda 캐시 충돌(unknown) 방지를 위해 사용
            """
            h = hashlib.sha256()
            root_p = Path(root)
            for dirpath, dirnames, filenames in os.walk(root_p):
                dirnames[:] = [d for d in dirnames if d not in {"__MACOSX"} and not d.startswith(".")]
                for fn in sorted(filenames):
                    if fn in {".DS_Store", "Thumbs.db"} or fn.startswith("._"):
                        continue
                    fp = Path(dirpath) / fn
                    try:
                        rel = fp.relative_to(root_p).as_posix()
                        st = fp.stat()
                    except Exception:
                        continue
                    h.update(rel.encode("utf-8"))
                    h.update(str(st.st_size).encode("utf-8"))
                    h.update(str(int(st.st_mtime)).encode("utf-8"))
            return h.hexdigest()

        base_sid = _snapshot_id_for(plan.base_path) if plan.base_path else "base"
        target_sid = _snapshot_id_for(plan.target_path) if plan.target_path else "target"
        base_hash = _fingerprint_dir(plan.base_path) if plan.base_path else "unknown"
        target_hash = _fingerprint_dir(plan.target_path) if plan.target_path else "unknown"

        artifacts = ArtifactPaths(Path(plan.out_dir))
        results: dict[str, Any] = {
            "plan": plan.model_dump(),
            "artifact_index": str(artifacts.artifact_index),
        }

        index: ArtifactIndex
        if artifacts.artifact_index.exists() and not self.force:
            index = _read_index(artifacts.artifact_index)
        else:
            index = ArtifactIndex(
                schema_version="1",
                generated_at=_now_iso(),
                producer={"plan": plan.name, "modality": plan.modality},
                context={
                    "modality": plan.modality,
                    "base_path": plan.base_path,
                    "target_path": plan.target_path,
                },
                artifacts=[],
            )

        executor = PythonExecutor()

        for step in plan.steps:
            if step.type == StepType.eda:
                which = step.params.get("which")
                if which not in ["base", "target"]:
                    raise ValueError(f"EDA step requires params.which in ['base','target'] (got {which})")
                in_path = plan.base_path if which == "base" else plan.target_path
                if not in_path:
                    raise ValueError(f"Missing {which}_path for EDA")
                cached = _find_artifact(
                    index,
                    artifact_type="eda.raw.v1",
                    meta_match={"step_id": step.id},
                )
                if cached and not self.force:
                    eda_res = _load_artifact_payload(Path(plan.out_dir), cached)
                else:
                    comp_out = str(Path(plan.out_dir) / f"eda.{which}")
                    eda_res = executor.run_eda(
                        modality=plan.modality,
                        data_path=in_path,
                        output_path=comp_out,
                        invalidate_cache=self.force,
                        snapshot_id=base_sid if which == "base" else target_sid,
                        data_hash=base_hash if which == "base" else target_hash,
                    )
                    raw_id = f"{step.id}.eda.raw.v1"
                    raw_payload = _artifact_payload_for_json(Path(plan.out_dir), raw_id, eda_res)
                    _add_artifact(
                        index=index,
                        artifact_id=raw_id,
                        artifact_type="eda.raw.v1",
                        payload=raw_payload,
                        title=f"EDA Raw ({which})",
                        meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                    )

                    summary = eda_res.get("summary") if isinstance(eda_res, dict) else None
                    if summary:
                        _add_artifact(
                            index=index,
                            artifact_id=f"{step.id}.eda.summary.v1",
                            artifact_type="eda.summary.v1",
                            payload=ArtifactPayloadInline(data=summary),
                            title=f"EDA Summary ({which})",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    stats = eda_res.get("stats") if isinstance(eda_res, dict) else None
                    if stats:
                        stats_id = f"{step.id}.eda.metrics.numeric.v1"
                        stats_payload = _artifact_payload_for_json(Path(plan.out_dir), stats_id, stats)
                        _add_artifact(
                            index=index,
                            artifact_id=stats_id,
                            artifact_type="eda.metrics.numeric.v1",
                            payload=stats_payload,
                            title=f"EDA Metrics ({which})",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    distributions = None
                    if isinstance(eda_res, dict):
                        distributions_basic = eda_res.get("distributions_basic")
                        distributions_attributes = eda_res.get("distributions_attributes")
                        if distributions_basic:
                            basic_id = f"{step.id}.eda.distributions.basic.v1"
                            basic_payload = _artifact_payload_for_json(
                                Path(plan.out_dir),
                                basic_id,
                                {"distributions": distributions_basic},
                            )
                            _add_artifact(
                                index=index,
                                artifact_id=basic_id,
                                artifact_type="eda.distributions.basic.v1",
                                payload=basic_payload,
                                title=f"EDA Basic Distributions ({which})",
                                meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                            )
                        if distributions_attributes:
                            attr_id = f"{step.id}.eda.distributions.attributes.v1"
                            attr_payload = _artifact_payload_for_json(
                                Path(plan.out_dir),
                                attr_id,
                                {"distributions": distributions_attributes},
                            )
                            _add_artifact(
                                index=index,
                                artifact_id=attr_id,
                                artifact_type="eda.distributions.attributes.v1",
                                payload=attr_payload,
                                title=f"EDA Attribute Distributions ({which})",
                                meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                            )
                        distributions = eda_res.get("distributions") or (summary or {}).get("distributions")
                        label_distributions = (summary or {}).get("label_distributions")
                        if distributions or label_distributions:
                            payload = {
                                "distributions": distributions,
                                "label_distributions": label_distributions,
                            }
                            dist_id = f"{step.id}.eda.distributions.v1"
                            dist_payload = _artifact_payload_for_json(Path(plan.out_dir), dist_id, payload)
                            _add_artifact(
                                index=index,
                                artifact_id=dist_id,
                                artifact_type="eda.distributions.v1",
                                payload=dist_payload,
                                title=f"EDA Distributions ({which})",
                                meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                            )

                    embedding_index = None
                    if isinstance(eda_res, dict):
                        embedding_index = eda_res.get("embedding_index")
                        if not embedding_index and stats:
                            embedding_index = stats.get("embedding_index")
                    if embedding_index:
                        _add_artifact(
                            index=index,
                            artifact_id=f"{step.id}.embedding.index.v1",
                            artifact_type="embedding.index.v1",
                            payload=ArtifactPayloadInline(data=embedding_index),
                            title=f"Embedding Index ({which})",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    embedding_projection = None
                    if isinstance(eda_res, dict):
                        embedding_projection = eda_res.get("embedding_projection")
                        if not embedding_projection and stats:
                            embedding_projection = stats.get("embedding_projection")
                    if embedding_projection:
                        proj_id = f"{step.id}.embedding.projection.2d.v1"
                        proj_payload = _artifact_payload_for_json(Path(plan.out_dir), proj_id, embedding_projection)
                        _add_artifact(
                            index=index,
                            artifact_id=proj_id,
                            artifact_type="embedding.projection.2d.v1",
                            payload=proj_payload,
                            title=f"Embedding Projection ({which})",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    embedding_clustering = None
                    if isinstance(eda_res, dict):
                        embedding_clustering = eda_res.get("embedding_clustering")
                        if not embedding_clustering and stats:
                            embedding_clustering = stats.get("embedding_clustering")
                    if embedding_clustering:
                        cluster_id = f"{step.id}.embedding.clustering.v1"
                        cluster_payload = _artifact_payload_for_json(Path(plan.out_dir), cluster_id, embedding_clustering)
                        _add_artifact(
                            index=index,
                            artifact_id=cluster_id,
                            artifact_type="embedding.clustering.v1",
                            payload=cluster_payload,
                            title=f"Embedding Clustering ({which})",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    trainlog_runs = eda_res.get("trainlog_runs")
                    trainlog_params = eda_res.get("trainlog_params")
                    trainlog_metrics = eda_res.get("trainlog_metrics")
                    trainlog_artifacts = eda_res.get("trainlog_artifacts")
                    trainlog_guide = eda_res.get("trainlog_mlflow_ui")
                    trainlog_meta_summary = eda_res.get("trainlog_meta_summary")
                    trainlog_preview_image = eda_res.get("trainlog_preview_image")
                    has_trainlog = any(
                        [
                            trainlog_runs,
                            trainlog_params,
                            trainlog_metrics,
                            trainlog_artifacts,
                            trainlog_guide,
                            trainlog_meta_summary,
                            trainlog_preview_image,
                        ]
                    )
                    if trainlog_meta_summary:
                        summary_id = f"{step.id}.trainlog.mlflow.summary.v1"
                        _add_artifact(
                            index=index,
                            artifact_id=summary_id,
                            artifact_type="trainlog.mlflow.summary.v1",
                            payload=ArtifactPayloadInline(data=trainlog_meta_summary),
                            title="MLflow Summary",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )
                    if trainlog_runs:
                        run_id = f"{step.id}.trainlog.mlflow.runs.index.v1"
                        run_payload = _artifact_payload_for_json(Path(plan.out_dir), run_id, trainlog_runs)
                        _add_artifact(
                            index=index,
                            artifact_id=run_id,
                            artifact_type="trainlog.mlflow.runs.index.v1",
                            payload=run_payload,
                            title="MLflow Runs",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )
                    if trainlog_params:
                        params_id = f"{step.id}.trainlog.mlflow.params.index.v1"
                        params_payload = _artifact_payload_for_json(Path(plan.out_dir), params_id, trainlog_params)
                        _add_artifact(
                            index=index,
                            artifact_id=params_id,
                            artifact_type="trainlog.mlflow.params.index.v1",
                            payload=params_payload,
                            title="MLflow Params",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    if trainlog_metrics:
                        metrics_id = f"{step.id}.trainlog.mlflow.metrics.index.v1"
                        metrics_payload = _artifact_payload_for_json(Path(plan.out_dir), metrics_id, trainlog_metrics)
                        _add_artifact(
                            index=index,
                            artifact_id=metrics_id,
                            artifact_type="trainlog.mlflow.metrics.index.v1",
                            payload=metrics_payload,
                            title="MLflow Metrics",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    if trainlog_artifacts:
                        artifacts_id = f"{step.id}.trainlog.mlflow.artifacts.index.v1"
                        artifacts_payload = _artifact_payload_for_json(Path(plan.out_dir), artifacts_id, trainlog_artifacts)
                        _add_artifact(
                            index=index,
                            artifact_id=artifacts_id,
                            artifact_type="trainlog.mlflow.artifacts.index.v1",
                            payload=artifacts_payload,
                            title="MLflow Artifacts",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

                    if trainlog_guide:
                        guide_id = f"{step.id}.trainlog.mlflow.mlflow_ui.guide.v1"
                        _add_artifact(
                            index=index,
                            artifact_id=guide_id,
                            artifact_type="trainlog.mlflow.mlflow_ui.guide.v1",
                            payload=ArtifactPayloadInline(data=trainlog_guide),
                            title="MLflow UI Guide",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )
                    if trainlog_preview_image:
                        preview_id = f"{step.id}.trainlog.mlflow.preview.image.v1"
                        _add_artifact(
                            index=index,
                            artifact_id=preview_id,
                            artifact_type="trainlog.mlflow.preview.image.v1",
                            payload=ArtifactPayloadInline(data=trainlog_preview_image),
                            title="MLflow Preview Image",
                            meta={"analysis_type": "eda", "which": which, "step_id": step.id},
                        )

            elif step.type == StepType.drift:
                if not plan.base_path or not plan.target_path:
                    raise ValueError("Missing base_path/target_path for drift")
                cached = _find_artifact(
                    index,
                    artifact_type="drift.raw.v1",
                    meta_match={"step_id": step.id},
                )
                if cached and not self.force:
                    drift_res = _load_artifact_payload(Path(plan.out_dir), cached)
                else:
                    comp_out = str(Path(plan.out_dir) / "drift")
                    drift_res = executor.run_drift(
                        modality=plan.modality,
                        data_path_ref=plan.base_path,
                        data_path_cur=plan.target_path,
                        output_path=comp_out,
                        detector="mmd",
                        cfg={},
                        snapshot_id_ref=base_sid,
                        snapshot_id_cur=target_sid,
                        data_hash_ref=base_hash,
                        data_hash_cur=target_hash,
                    )
                    raw_id = f"{step.id}.drift.raw.v1"
                    raw_payload = _artifact_payload_for_json(Path(plan.out_dir), raw_id, drift_res)
                    _add_artifact(
                        index=index,
                        artifact_id=raw_id,
                        artifact_type="drift.raw.v1",
                        payload=raw_payload,
                        title="Drift Raw",
                        meta={"analysis_type": "drift", "step_id": step.id},
                    )

                    if isinstance(drift_res, dict):
                        overall_score = drift_res.get("overall_score")
                        if overall_score is None:
                            derived_status = "UNKNOWN"
                        elif overall_score >= 1.0:
                            derived_status = "CRITICAL"
                        elif overall_score >= 0.7:
                            derived_status = "WARNING"
                        else:
                            derived_status = "NORMAL"

                        status_payload = {
                            "status": derived_status,
                            "overall_score": overall_score,
                            "modality": drift_res.get("modality"),
                        }
                        if any(v is not None for v in status_payload.values()):
                            _add_artifact(
                                index=index,
                                artifact_id=f"{step.id}.drift.status.v1",
                                artifact_type="drift.status.v1",
                                payload=ArtifactPayloadInline(data=status_payload),
                                title="Drift Status",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

                        overall_score = drift_res.get("overall_score")
                        if overall_score is not None:
                            _add_artifact(
                                index=index,
                                artifact_id=f"{step.id}.drift.overall_score.v1",
                                artifact_type="drift.overall_score.v1",
                                payload=ArtifactPayloadInline(data={"overall_score": overall_score}),
                                title="Overall Score",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

                        attribute_drifts = drift_res.get("attribute_drifts")
                        if attribute_drifts:
                            attr_id = f"{step.id}.drift.attribute_drifts.v1"
                            attr_payload = _artifact_payload_for_json(
                                Path(plan.out_dir), attr_id, {"attribute_drifts": attribute_drifts}
                            )
                            _add_artifact(
                                index=index,
                                artifact_id=attr_id,
                                artifact_type="drift.attribute_drifts.v1",
                                payload=attr_payload,
                                title="Attribute Drifts",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

                        attribute_values_ref = drift_res.get("attribute_values_ref")
                        attribute_values_cur = drift_res.get("attribute_values_cur")
                        if isinstance(attribute_values_ref, dict) and isinstance(attribute_values_cur, dict):
                            metrics: dict[str, Any] = {}
                            method = drift_res.get("attribute_drift_method") or "psi"
                            for key in sorted(set(attribute_values_ref.keys()) & set(attribute_values_cur.keys())):
                                ref_vals = [
                                    v for v in (attribute_values_ref.get(key) or []) if isinstance(v, (int, float))
                                ]
                                cur_vals = [
                                    v for v in (attribute_values_cur.get(key) or []) if isinstance(v, (int, float))
                                ]
                                hist_pair = _histogram_pair(ref_vals, cur_vals, bins=20)
                                if not hist_pair:
                                    continue
                                metrics[key] = {
                                    "base": hist_pair["base"],
                                    "target": hist_pair["target"],
                                    "score": attribute_drifts.get(key) if isinstance(attribute_drifts, dict) else None,
                                    "method": method,
                                }
                            if metrics:
                                dist_id = f"{step.id}.drift.attribute_distributions.v1"
                                dist_payload = _artifact_payload_for_json(
                                    Path(plan.out_dir),
                                    dist_id,
                                    {"metrics": metrics},
                                )
                                _add_artifact(
                                    index=index,
                                    artifact_id=dist_id,
                                    artifact_type="drift.attribute_distributions.v1",
                                    payload=dist_payload,
                                    title="Attribute Distributions",
                                    meta={"analysis_type": "drift", "step_id": step.id},
                                )

                        embedding_summary = {
                            "embedding_drift": drift_res.get("embedding_drift"),
                            "embedding_drift_detailed": drift_res.get("embedding_drift_detailed"),
                        }
                        if embedding_summary["embedding_drift"] is not None or embedding_summary["embedding_drift_detailed"]:
                            embed_id = f"{step.id}.drift.embedding.summary.v1"
                            embed_payload = _artifact_payload_for_json(Path(plan.out_dir), embed_id, embedding_summary)
                            _add_artifact(
                                index=index,
                                artifact_id=embed_id,
                                artifact_type="drift.embedding.summary.v1",
                                payload=embed_payload,
                                title="Embedding Drift",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

                        embedding_index_ref = drift_res.get("embedding_index_ref")
                        embedding_index_cur = drift_res.get("embedding_index_cur")
                        if embedding_index_ref and embedding_index_cur:
                            try:
                                from ddoc.core.embedding_store import load_embeddings  # type: ignore

                                ref_vectors, _, _ = load_embeddings(embedding_index=embedding_index_ref)
                                cur_vectors, _, _ = load_embeddings(embedding_index=embedding_index_cur)
                                if ref_vectors.size and cur_vectors.size:
                                    base_count = int(ref_vectors.shape[0])
                                    target_count = int(cur_vectors.shape[0])
                                    n = min(2000, base_count, target_count)
                                    seed_source = (
                                        f"{embedding_index_ref.get('path')}:{embedding_index_cur.get('path')}:"
                                        f"{drift_res.get('data_hash_ref')}:{drift_res.get('data_hash_cur')}"
                                    )
                                    seed = _stable_seed(seed_source)
                                    ref_idx = _sample_indices(base_count, n, seed=seed)
                                    cur_idx = _sample_indices(target_count, n, seed=seed + 1)
                                    ref_sample = ref_vectors[ref_idx]
                                    cur_sample = cur_vectors[cur_idx]
                                    merged = np.vstack([ref_sample, cur_sample])
                                    coords = _pca_projection(merged)
                                    if coords.size:
                                        points = []
                                        for i in range(n):
                                            points.append(
                                                {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "split": "base"}
                                            )
                                        for i in range(n):
                                            j = n + i
                                            points.append(
                                                {"x": float(coords[j, 0]), "y": float(coords[j, 1]), "split": "target"}
                                            )
                                        proj_payload = {
                                            "method": "pca",
                                            "points": points,
                                            "sampling": {
                                                "cap": 2000,
                                                "n": n,
                                                "base_count": base_count,
                                                "target_count": target_count,
                                                "seed": seed,
                                                "strategy": "fixed_seed_random",
                                            },
                                        }
                                        proj_id = f"{step.id}.drift.embedding.projection.2d.v1"
                                        proj_ref = _artifact_payload_for_json(Path(plan.out_dir), proj_id, proj_payload)
                                        _add_artifact(
                                            index=index,
                                            artifact_id=proj_id,
                                            artifact_type="drift.embedding.projection.2d.v1",
                                            payload=proj_ref,
                                            title="Embedding Projection (Drift)",
                                            meta={"analysis_type": "drift", "step_id": step.id},
                                        )
                            except Exception:
                                pass

                        files_payload = {
                            "files_added": drift_res.get("files_added"),
                            "files_removed": drift_res.get("files_removed"),
                            "files_common": drift_res.get("files_common"),
                        }
                        if any(v is not None for v in files_payload.values()):
                            file_id = f"{step.id}.drift.file_changes.v1"
                            file_payload = _artifact_payload_for_json(Path(plan.out_dir), file_id, files_payload)
                            _add_artifact(
                                index=index,
                                artifact_id=file_id,
                                artifact_type="drift.file_changes.v1",
                                payload=file_payload,
                                title="File Changes",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

                        trainlog_drift_aggregate = drift_res.get("trainlog_drift_aggregate")
                        if trainlog_drift_aggregate:
                            agg_id = f"{step.id}.trainlog.mlflow.drift.aggregate.v1"
                            agg_payload = _artifact_payload_for_json(Path(plan.out_dir), agg_id, trainlog_drift_aggregate)
                            _add_artifact(
                                index=index,
                                artifact_id=agg_id,
                                artifact_type="trainlog.mlflow.drift.aggregate.v1",
                                payload=agg_payload,
                                title="MLflow Drift Aggregate",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

                        trainlog_drift_pairs = drift_res.get("trainlog_drift_pairs")
                        if trainlog_drift_pairs:
                            pairs_id = f"{step.id}.trainlog.mlflow.drift.matched_pairs.v1"
                            pairs_payload = _artifact_payload_for_json(Path(plan.out_dir), pairs_id, trainlog_drift_pairs)
                            _add_artifact(
                                index=index,
                                artifact_id=pairs_id,
                                artifact_type="trainlog.mlflow.drift.matched_pairs.v1",
                                payload=pairs_payload,
                                title="MLflow Drift Matched Pairs",
                                meta={"analysis_type": "drift", "step_id": step.id},
                            )

            elif step.type == StepType.report:
                # 리포트는 driftstudio_reports가 제공 (없으면 json만 남김)
                results["artifacts"] = {
                    "html": str(artifacts.report_html),
                    "pdf": str(artifacts.report_pdf),
                }

            else:
                raise ValueError(f"Unsupported step type: {step.type}")

        # 실제 리포트 렌더 (선택)
        try:
            from driftstudio_reports import render_report  # type: ignore

            eda_base_artifact = _find_artifact(
                index,
                artifact_type="eda.raw.v1",
                meta_match={"analysis_type": "eda", "which": "base"},
            )
            eda_target_artifact = _find_artifact(
                index,
                artifact_type="eda.raw.v1",
                meta_match={"analysis_type": "eda", "which": "target"},
            )
            drift_artifact = _find_artifact(
                index,
                artifact_type="drift.raw.v1",
                meta_match={"analysis_type": "drift"},
            )
            drift_attr_dist_artifact = _find_artifact(
                index,
                artifact_type="drift.attribute_distributions.v1",
                meta_match={"analysis_type": "drift"},
            )
            drift_proj_artifact = _find_artifact(
                index,
                artifact_type="drift.embedding.projection.2d.v1",
                meta_match={"analysis_type": "drift"},
            )
            eda_base = _load_artifact_payload(Path(plan.out_dir), eda_base_artifact) if eda_base_artifact else None
            eda_target = _load_artifact_payload(Path(plan.out_dir), eda_target_artifact) if eda_target_artifact else None
            drift_json = _load_artifact_payload(Path(plan.out_dir), drift_artifact) if drift_artifact else None
            drift_attr_dist = (
                _load_artifact_payload(Path(plan.out_dir), drift_attr_dist_artifact)
                if drift_attr_dist_artifact
                else None
            )
            drift_proj = _load_artifact_payload(Path(plan.out_dir), drift_proj_artifact) if drift_proj_artifact else None
            rendered = render_report(
                out_dir=Path(plan.out_dir),
                plan=plan,
                eda_base=eda_base,
                eda_target=eda_target,
                drift=drift_json,
                drift_attribute_distributions=drift_attr_dist,
                drift_embedding_projection=drift_proj,
                formats=plan.report_formats,
            )
            results["report"] = rendered
        except Exception as e:
            results["report"] = {"error": str(e)}

        _write_index(artifacts.artifact_index, index)

        return results

    def _normalize_plan_inputs(self, plan: Plan) -> Plan:
        """
        - 입력이 .zip이면 out_dir 아래로 압축 해제 후 extracted dir 경로로 치환
        - 입력이 디렉토리면 ddoc.yaml 검증(없으면 예외)
        """
        out_dir = Path(plan.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_path = plan.base_path
        target_path = plan.target_path

        def _prep(p: str | None, label: str) -> str | None:
            if not p:
                return None
            pp = Path(p)
            if pp.is_file() and pp.suffix.lower() == ".zip":
                dest = out_dir / "_inputs" / f"{label}_extracted"
                extracted = extract_zip_dataset(str(pp), dest=str(dest))
                validate_dataset_dir_or_raise(extracted)
                return extracted
            if pp.is_dir():
                validate_dataset_dir_or_raise(pp)
                return str(pp)
            raise ValueError(f"입력 경로는 .zip 또는 압축 해제된 디렉토리여야 합니다: {p}")

        base_norm = _prep(base_path, "base")
        target_norm = _prep(target_path, "target")

        return plan.model_copy(update={"base_path": base_norm, "target_path": target_norm})

def _infer_dtype(path: str) -> str:
    # (호환용) 구버전 함수명. 새 로직은 infer_dtype 사용.
    return infer_dtype(path)


