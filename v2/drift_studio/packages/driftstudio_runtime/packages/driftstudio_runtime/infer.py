from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import yaml

from driftstudio_spec import DatasetMeta
from pydantic import TypeAdapter

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
_AUDIO_MIDI_EXT = {".mid", ".midi"}
_AUDIO_WAVE_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_AUDIO_LABEL_EXT = {".json"}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def load_dataset_meta_or_raise(dataset_dir: str | Path) -> DatasetMeta:
    """
    강제 규약:
    - 데이터셋은 압축 해제된 디렉토리 형태로 처리되며, 루트에 `ddoc.yaml`이 반드시 존재해야 합니다.
    - `ddoc.yaml`이 없거나 스키마 불일치/값 불일치면 예외를 발생시켜 **처리되지 않게** 합니다.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError("dataset_dir는 압축 해제된 디렉토리여야 합니다(파일 경로 불가)")

    meta_path = root / "ddoc.yaml"
    if not meta_path.exists():
        raise ValueError("메타파일 ddoc.yaml이 없습니다(루트에 필수)")

    try:
        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"ddoc.yaml 파싱 실패: {e}")

    try:
        # DatasetMeta는 discriminated union 타입(alias)이므로 TypeAdapter로 검증합니다.
        return TypeAdapter(DatasetMeta).validate_python(raw)
    except Exception as e:
        raise ValueError(f"ddoc.yaml 스키마 검증 실패: {e}")


def validate_dataset_dir_or_raise(dataset_dir: str | Path) -> DatasetMeta:
    """
    메타 + 실제 파일 구성 검증.
    - 누락/불일치 시 예외 발생(= fallback 없이 처리 중단)
    """
    root = Path(dataset_dir)
    meta = load_dataset_meta_or_raise(root)

    modality = getattr(meta, "modality")
    data = getattr(meta, "data")

    if modality in ["text", "timeseries"]:
        csv_rel = getattr(data, "csv")
        csv_path = (root / csv_rel).resolve()
        if not csv_path.exists() or not csv_path.is_file():
            raise ValueError(f"{modality}: CSV 파일을 찾을 수 없습니다: {csv_rel}")
        if csv_path.suffix.lower() != ".csv":
            raise ValueError(f"{modality}: CSV 포맷만 지원합니다: {csv_rel}")

    if modality == "text":
        # columns는 스키마에서 이미 non-empty 검증
        pass

    if modality == "timeseries":
        # timestamp/numeric/categorical는 스키마에서 기본 검증
        pass

    if modality == "audio_wave":
        data_dir = (root / getattr(data, "data_dir")).resolve()
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"audio_wave: data_dir을 찾을 수 없습니다: {getattr(data, 'data_dir')}")
        wave_files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO_WAVE_EXT]
        if not wave_files:
            raise ValueError("audio_wave: 파형 오디오 파일이 1개 이상 필요합니다(wav/mp3/flac/ogg/m4a/aac)")

    if modality == "audio_midi":
        data_dir = (root / getattr(data, "data_dir")).resolve()
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"audio_midi: data_dir을 찾을 수 없습니다: {getattr(data, 'data_dir')}")
        midi_files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO_MIDI_EXT]
        if not midi_files:
            raise ValueError("audio_midi: .mid/.midi 파일이 1개 이상 필요합니다")
        json_files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO_LABEL_EXT]
        if not json_files:
            raise ValueError("audio_midi: 라벨(.json) 파일이 1개 이상 필요합니다")

        # MIDI 파일마다 매칭되는 JSON 라벨이 있는지 확인(동일 경로+파일명 stem 기준)
        def _stem_rel(p: Path) -> str:
            rel = p.relative_to(data_dir)
            return rel.with_suffix("").as_posix()

        midi_stems = {_stem_rel(p) for p in midi_files}
        json_stems = {_stem_rel(p) for p in json_files}
        missing = sorted(midi_stems - json_stems)
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"audio_midi: MIDI와 매칭되는 라벨(.json)이 없는 파일이 있습니다 (예: {sample}{' ...' if len(missing) > 5 else ''})"
            )

    if modality == "vision_video":
        data_dir = (root / getattr(data, "data_dir")).resolve()
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"vision_video: data_dir을 찾을 수 없습니다: {getattr(data, 'data_dir')}")
        files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXT]
        if not files:
            raise ValueError("vision_video: 지원 비디오 파일이 1개 이상 필요합니다(mp4/mov/avi/mkv/webm/m4v)")

    if modality == "vision_image":
        data_dir = (root / getattr(data, "data_dir")).resolve()
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"vision_image: data_dir을 찾을 수 없습니다: {getattr(data, 'data_dir')}")
        files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT]
        if not files:
            raise ValueError("vision_image: 이미지 파일이 1개 이상 필요합니다")

    if modality == "mlflow_log":
        tracking_dir_raw = str(getattr(data, "tracking_dir") or "").strip()
        tracking_dir = _resolve_mlflow_tracking_dir(root, tracking_dir_raw)
        if not tracking_dir.exists() or not tracking_dir.is_dir():
            raise ValueError(f"mlflow_log: tracking_dir을 찾을 수 없습니다: {tracking_dir_raw}")
        experiment_dirs = [p for p in tracking_dir.iterdir() if p.is_dir()]
        experiment_meta = [p for p in experiment_dirs if (p / "meta.yaml").exists()]
        if not experiment_meta:
            raise ValueError("mlflow_log: experiment meta.yaml을 찾을 수 없습니다")
        has_run_meta = False
        for exp_dir in experiment_meta:
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if (run_dir / "meta.yaml").exists():
                    has_run_meta = True
                    break
            if has_run_meta:
                break
        if not has_run_meta:
            raise ValueError("mlflow_log: run meta.yaml을 포함한 run 디렉토리가 필요합니다")

    return meta


def _resolve_mlflow_tracking_dir(root: Path, raw: str) -> Path:
    if not raw or raw.strip().lower() == "auto":
        return _find_mlflow_tracking_dir(root) or (root / "mlruns")
    return (root / raw).resolve()


def _find_mlflow_tracking_dir(root: Path, max_depth: int = 4) -> Path | None:
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        current = Path(dirpath)
        if current.name not in {"mlruns", "mlflow"}:
            continue
        has_experiment_meta = False
        for child in current.iterdir():
            if child.is_dir() and (child / "meta.yaml").exists():
                has_experiment_meta = True
                break
        if has_experiment_meta:
            return current
    return None


def infer_modality(dataset_dir: str | Path) -> str:
    """휴리스틱 금지: ddoc.yaml 기반으로만 판정."""
    meta = load_dataset_meta_or_raise(dataset_dir)
    return getattr(meta, "modality")


def infer_dtype(path: str) -> str:
    """
    v2 규약에서는 입력이 항상 zip(업로드) → extracted dir(처리)로 정리됩니다.
    런타임 내부에서는 주로 CSV 실행을 위해 확장자 판정만 사용합니다.
    """
    p = Path(path)
    if p.is_dir():
        return "dir"
    return p.suffix.lower().lstrip(".") or "unknown"


