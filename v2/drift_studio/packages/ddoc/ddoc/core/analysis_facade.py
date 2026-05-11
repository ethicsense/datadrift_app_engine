from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ddoc.core.plugins import get_plugin_manager


# drift_studio(v2) modality → ddoc plugin entrypoint name
_MODALITY_TO_PROVIDER: dict[str, str] = {
    "vision_image": "ddoc_vision_image",
    "vision_video": "ddoc_vision_video",
    "text": "ddoc_text",
    "audio_wave": "ddoc_audio_wave",
    "audio_midi": "ddoc_audio_midi",
    "timeseries": "ddoc_timeseries",
    "mlflow_log": "ddoc_mlflow_log",
}


def _provider_for(modality: str) -> str:
    m = (modality or "").strip().lower()
    if m not in _MODALITY_TO_PROVIDER:
        raise ValueError(f"Unsupported modality for ddoc execution: {modality}")
    return _MODALITY_TO_PROVIDER[m]


def run_eda(
    *,
    modality: str,
    data_path: str,
    output_path: str,
    invalidate_cache: bool = False,
    snapshot_id: str = "local",
    data_hash: str = "unknown",
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """
    drift_studio에서 ddoc 플러그인 기반 EDA 실행을 위한 얇은 파사드.

    - ddoc의 pluggy PluginManager(call_hook)를 사용
    - 모달리티별 provider(entrypoint name)를 지정해 단일 플러그인만 실행
    """
    provider_name = provider or _provider_for(modality)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    pmgr = get_plugin_manager()
    res = pmgr.call_hook(
        "eda_run",
        provider=provider_name,
        first_non_none=True,
        snapshot_id=snapshot_id,
        data_path=data_path,
        data_hash=data_hash,
        output_path=output_path,
        invalidate_cache=invalidate_cache,
    )
    if res is None:
        raise ValueError(
            f"No EDA plugin returned a result (provider={provider_name}, modality={modality}). "
            "Plugin 설치/로드 상태를 확인하세요."
        )
    if isinstance(res, dict) and res.get("status") == "error":
        raise ValueError(str(res.get("error") or res))
    return res


def run_drift(
    *,
    modality: str,
    data_path_ref: str,
    data_path_cur: str,
    output_path: str,
    detector: str = "mmd",
    cfg: Optional[dict[str, Any]] = None,
    snapshot_id_ref: str = "base",
    snapshot_id_cur: str = "target",
    data_hash_ref: str = "unknown",
    data_hash_cur: str = "unknown",
    provider: Optional[str] = None,
) -> dict[str, Any]:
    provider_name = provider or _provider_for(modality)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    pmgr = get_plugin_manager()
    res = pmgr.call_hook(
        "drift_detect",
        provider=provider_name,
        first_non_none=True,
        snapshot_id_ref=snapshot_id_ref,
        snapshot_id_cur=snapshot_id_cur,
        data_path_ref=data_path_ref,
        data_path_cur=data_path_cur,
        data_hash_ref=data_hash_ref,
        data_hash_cur=data_hash_cur,
        detector=detector,
        cfg=cfg or {},
        output_path=output_path,
    )
    if res is None:
        raise ValueError(
            f"No drift plugin returned a result (provider={provider_name}, modality={modality}). "
            "Plugin 설치/로드 상태를 확인하세요."
        )
    if isinstance(res, dict) and res.get("status") == "error":
        raise ValueError(str(res.get("error") or res))
    return res


