from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    ingest = "ingest"
    eda = "eda"
    drift = "drift"
    report = "report"


class ReportFormat(str, Enum):
    html = "html"
    pdf = "pdf"


class Step(BaseModel):
    type: StepType
    id: str = Field(..., description="step id (stable key)")
    params: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """
    최소 실행 플랜.

    v2 목표: EDA → Drift → Report 중심 (workspace/git/dvc 제외)
    """

    name: str = "default"
    modality: str = Field(..., description="vision/text/audio/timeseries 등")

    # inputs
    base_path: Optional[str] = None
    target_path: Optional[str] = None

    # outputs
    out_dir: str = Field(..., description="artifact root directory")
    report_formats: list[ReportFormat] = Field(default_factory=lambda: [ReportFormat.html])

    steps: list[Step] = Field(default_factory=list)

    @staticmethod
    def default_eda_drift_report(
        *,
        modality: str,
        base_path: str,
        target_path: str,
        out_dir: str,
        report_formats: Optional[list[ReportFormat]] = None,
    ) -> "Plan":
        return Plan(
            name="eda-drift-report",
            modality=modality,
            base_path=base_path,
            target_path=target_path,
            out_dir=out_dir,
            report_formats=report_formats or [ReportFormat.html],
            steps=[
                Step(type=StepType.eda, id="eda_base", params={"which": "base"}),
                Step(type=StepType.eda, id="eda_target", params={"which": "target"}),
                Step(type=StepType.drift, id="drift", params={}),
                Step(type=StepType.report, id="report", params={}),
            ],
        )

    @staticmethod
    def default_drift_report(
        *,
        modality: str,
        base_path: str,
        target_path: str,
        out_dir: str,
        report_formats: Optional[list[ReportFormat]] = None,
    ) -> "Plan":
        """
        Drift + Report만 실행하는 플랜.

        - 이미 base/target EDA 결과가 캐시/DB 등에 존재하는 경우 drift 플랜에서
          EDA를 재실행하지 않도록 사용합니다.
        """
        return Plan(
            name="drift-report",
            modality=modality,
            base_path=base_path,
            target_path=target_path,
            out_dir=out_dir,
            report_formats=report_formats or [ReportFormat.html],
            steps=[
                Step(type=StepType.drift, id="drift", params={}),
                Step(type=StepType.report, id="report", params={}),
            ],
        )


