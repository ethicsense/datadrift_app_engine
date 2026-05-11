"""Drift analysis command (minimal, drift_studio v2 style)"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from ..utils import _pretty, dataset_identity, infer_modality, resolve_dataset_input


def analyze_drift_command(
    baseline: str = typer.Argument(..., help="Baseline dataset dir or .zip (ddoc.yaml required)"),
    current: str = typer.Argument(..., help="Current dataset dir or .zip (ddoc.yaml required)"),
    out_dir: str = typer.Option("analysis", "--out", help="Output directory (results + temp extracted inputs)"),
    detector: str = typer.Option("mmd", "--detector", help="Drift detector method"),
):
    """
    두 데이터셋 간 drift를 계산합니다(플러그인 기반).

    - 입력은 디렉토리 또는 .zip
    - 두 데이터셋의 modality는 동일해야 합니다.
    """
    out_root = Path(out_dir)
    base_dir = resolve_dataset_input(baseline, work_dir=out_root)
    cur_dir = resolve_dataset_input(current, work_dir=out_root)

    base_mod = infer_modality(base_dir)
    cur_mod = infer_modality(cur_dir)
    if base_mod != cur_mod:
        raise ValueError(f"modality mismatch: baseline={base_mod}, current={cur_mod}")

    snapshot_id_ref, data_hash_ref = dataset_identity(base_dir)
    snapshot_id_cur, data_hash_cur = dataset_identity(cur_dir)

    output_path = out_root / "drift" / f"{base_dir.name}__{cur_dir.name}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[cyan]🔍 Drift[/cyan] modality={base_mod} baseline={base_dir} current={cur_dir} out={output_path}")

    from ddoc.core.analysis_facade import run_drift

    res = run_drift(
        modality=base_mod,
        data_path_ref=str(base_dir),
        data_path_cur=str(cur_dir),
        output_path=str(output_path),
        detector=detector,
        cfg={},
        snapshot_id_ref=snapshot_id_ref,
        snapshot_id_cur=snapshot_id_cur,
        data_hash_ref=data_hash_ref,
        data_hash_cur=data_hash_cur,
    )

    print(_pretty(res))
