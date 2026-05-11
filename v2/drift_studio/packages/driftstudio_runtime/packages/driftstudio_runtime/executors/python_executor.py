from __future__ import annotations

from typing import Any


class PythonExecutor:
    """
    ddoc를 python import로 직접 호출하는 실행기.
    """

    def run_eda(
        self,
        *,
        modality: str,
        data_path: str,
        output_path: str,
        invalidate_cache: bool,
        snapshot_id: str = "local",
        data_hash: str = "unknown",
    ) -> dict[str, Any]:
        try:
            from ddoc.core.analysis_facade import run_eda  # type: ignore
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            raise ValueError(
                f"ddoc 실행에 필요한 의존성이 없습니다: {missing}. "
                "ddoc 및 해당 모달리티 플러그인의 의존성을 설치하세요."
            ) from e
        except Exception as e:
            raise ValueError(
                "ddoc 패키지를 import할 수 없습니다. packages/ddoc가 설치되어 있는지 확인하세요."
            ) from e

        return run_eda(
            modality=modality,
            data_path=data_path,
            output_path=output_path,
            invalidate_cache=invalidate_cache,
            snapshot_id=snapshot_id,
            data_hash=data_hash,
        )

    def run_drift(
        self,
        *,
        modality: str,
        data_path_ref: str,
        data_path_cur: str,
        output_path: str,
        detector: str,
        cfg: dict[str, Any],
        snapshot_id_ref: str = "base",
        snapshot_id_cur: str = "target",
        data_hash_ref: str = "unknown",
        data_hash_cur: str = "unknown",
    ) -> dict[str, Any]:
        try:
            from ddoc.core.analysis_facade import run_drift  # type: ignore
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            raise ValueError(
                f"ddoc 실행에 필요한 의존성이 없습니다: {missing}. "
                "ddoc 및 해당 모달리티 플러그인의 의존성을 설치하세요."
            ) from e
        except Exception as e:
            raise ValueError(
                "ddoc 패키지를 import할 수 없습니다. packages/ddoc가 설치되어 있는지 확인하세요."
            ) from e

        return run_drift(
            modality=modality,
            data_path_ref=data_path_ref,
            data_path_cur=data_path_cur,
            output_path=output_path,
            detector=detector,
            cfg=cfg,
            snapshot_id_ref=snapshot_id_ref,
            snapshot_id_cur=snapshot_id_cur,
            data_hash_ref=data_hash_ref,
            data_hash_cur=data_hash_cur,
        )


