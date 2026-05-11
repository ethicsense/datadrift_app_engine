from __future__ import annotations

from typing import Any, Protocol


class Executor(Protocol):
    """
    하이브리드 실행기 경계.

    - PythonExecutor: 같은 프로세스에서 python import로 실행
    - (future) SubprocessExecutor: ddoc CLI를 subprocess로 호출
    """

    def run_eda(self, *, modality: str, data_path: str, output_path: str, invalidate_cache: bool) -> dict[str, Any]:
        ...

    def run_drift(
        self,
        *,
        modality: str,
        data_path_ref: str,
        data_path_cur: str,
        output_path: str,
        detector: str,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        ...


