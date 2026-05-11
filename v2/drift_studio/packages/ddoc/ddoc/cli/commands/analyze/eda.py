"""EDA analysis command (minimal, drift_studio v2 style)"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from ..utils import _pretty, dataset_identity, infer_modality, resolve_dataset_input


def analyze_eda_command(
    dataset: str = typer.Argument(..., help="Dataset directory or .zip (must contain ddoc.yaml)"),
    out_dir: str = typer.Option("analysis", "--out", help="Output directory (results + temp extracted inputs)"),
    invalidate_cache: bool = typer.Option(False, "--invalidate-cache", help="Invalidate existing cache before analysis"),
):
    """
    EDA 실행(플러그인 기반) + 결과 캐시 저장.

    - 입력은 '개별 데이터셋 디렉토리' 또는 '.zip'을 지원합니다.
    - 모달리티는 ddoc.yaml의 modality로 결정됩니다.
    """
    out_root = Path(out_dir)
    dataset_dir = resolve_dataset_input(dataset, work_dir=out_root)

    modality = infer_modality(dataset_dir)
    snapshot_id, data_hash = dataset_identity(dataset_dir)

    output_path = out_root / "eda" / dataset_dir.name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[cyan]📊 EDA[/cyan] modality={modality} dataset={dataset_dir} out={output_path}")

    from ddoc.core.analysis_facade import run_eda

    res = run_eda(
        modality=modality,
        data_path=str(dataset_dir),
        output_path=str(output_path),
        invalidate_cache=invalidate_cache,
        snapshot_id=snapshot_id,
        data_hash=data_hash,
    )

    print(_pretty(res))
