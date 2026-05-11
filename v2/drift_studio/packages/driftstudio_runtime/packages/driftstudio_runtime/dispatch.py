from __future__ import annotations

from typing import Any, Optional

from driftstudio_runtime.executors.python_executor import PythonExecutor


# NOTE:
# - 2차 리팩터링에서는 v1 ddoc를 vendoring(`packages/ddoc`)하고,
#   pluggy entrypoints 기반 플러그인을 PythonExecutor로 호출합니다.
# - runner는 직접 PythonExecutor를 쓰고 있고, 이 파일은 (호환용) 얇은 위임만 유지합니다.


def try_component_eda_run(
    *,
    modality: str,
    data_path: str,
    output_path: str,
    invalidate_cache: bool = False,
) -> Optional[dict[str, Any]]:
    try:
        return PythonExecutor().run_eda(
            modality=modality,
            data_path=data_path,
            output_path=output_path,
            invalidate_cache=invalidate_cache,
        )
    except Exception:
        return None


def try_component_drift_detect(
    *,
    modality: str,
    data_path_ref: str,
    data_path_cur: str,
    output_path: str,
    detector: str = "mmd",
    cfg: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    try:
        return PythonExecutor().run_drift(
            modality=modality,
            data_path_ref=data_path_ref,
            data_path_cur=data_path_cur,
            output_path=output_path,
            detector=detector,
            cfg=cfg or {},
        )
    except Exception:
        return None


