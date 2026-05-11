from __future__ import annotations

import json
from pathlib import Path

from driftstudio_runtime import RuntimeRunner
from driftstudio_runtime.infer import infer_modality
from driftstudio_spec import ArtifactIndex, Plan, ReportFormat, Step, StepType


def dataset_out_dir(dataset_id: str) -> str:
    """
    API 서버 관점의 기본 산출물 루트.
    - workspace 기능은 제거했지만, 결과 파일은 서버 로컬에 저장되어야 함
    """
    return str(Path("storage") / "runs" / dataset_id)


def build_default_plan_for_dataset(
    *,
    modality: str,
    base_path: str,
    target_path: str | None,
    out_dir: str,
    pdf: bool = False,
) -> Plan:
    # modality="auto" 지원
    if modality in ["auto", "", None]:
        modality = infer_modality(base_path)
    if target_path:
        formats = [ReportFormat.html] + ([ReportFormat.pdf] if pdf else [])
        return Plan.default_eda_drift_report(
            modality=modality,
            base_path=base_path,
            target_path=target_path,
            out_dir=out_dir,
            report_formats=formats,
        )

    # 단일 EDA 플랜 (최소)
    return Plan(
        name="eda-only",
        modality=modality,
        base_path=base_path,
        out_dir=out_dir,
        report_formats=[ReportFormat.html] + ([ReportFormat.pdf] if pdf else []),
        steps=[
            # base만 수행
            # (runner에서 which 파라미터를 요구하므로 동일 규약 사용)
            Step(type=StepType.eda, id="eda_base", params={"which": "base"}),
            Step(type=StepType.report, id="report", params={}),
        ],
    )


def run_plan(plan: Plan, *, force: bool = False) -> dict:
    runner = RuntimeRunner(force=force)
    return runner.run(plan)


def has_eda_raw_artifact_for_dataset(dataset_id: str) -> bool:
    """데이터셋 단일 EDA 산출물(eda.raw.v1)이 artifact_index에 있는지."""
    out = Path(dataset_out_dir(dataset_id)) / "artifact_index.json"
    if not out.exists():
        return False
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
        index = ArtifactIndex.parse_obj(payload)
        return any(a.type == "eda.raw.v1" for a in index.artifacts)
    except Exception:
        return False


def drift_pair_plan_and_cache_mode(
    *,
    base_id: str,
    target_id: str,
    base_path: str,
    target_path: str,
    out_dir: str,
    force: bool = False,
) -> tuple[Plan, bool]:
    """
    Drift 비교용 플랜과 EDA 재사용 여부.
    force=True이면 전체 EDA+drift 파이프라인(캐시된 EDA 무시).
    """
    used_cached_eda = bool(
        has_eda_raw_artifact_for_dataset(base_id)
        and has_eda_raw_artifact_for_dataset(target_id)
        and not force
    )
    if used_cached_eda:
        plan = Plan.default_drift_report(
            modality="auto",
            base_path=base_path,
            target_path=target_path,
            out_dir=out_dir,
            report_formats=[ReportFormat.pdf],
        )
    else:
        plan = Plan.default_eda_drift_report(
            modality="auto",
            base_path=base_path,
            target_path=target_path,
            out_dir=out_dir,
            report_formats=[ReportFormat.pdf],
        )
    return plan, used_cached_eda


