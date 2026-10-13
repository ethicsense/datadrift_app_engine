"""Merge SAM3-ABCD seed+supplement embeddings into cohort and score cluster quality."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument(
        "--seed-run",
        type=Path,
        default=REPO_ROOT
        / "artifacts/garment_segmentation/attribute_cluster_eval-20261002/pipeline/preprocess/preprocess-abcd-sam3-450",
    )
    p.add_argument(
        "--supp-run",
        type=Path,
        default=REPO_ROOT
        / "artifacts/garment_segmentation/attribute_cluster_eval-20261002/pipeline/preprocess/preprocess-abcd-sam3-supp211",
    )
    p.add_argument(
        "--expand-run",
        type=Path,
        default=None,
        help="Optional third preprocess run (e.g. expand293) to merge into bank",
    )
    p.add_argument(
        "--extra-runs",
        type=Path,
        nargs="*",
        default=[],
        help="Additional preprocess runs to merge (e.g. topup49)",
    )
    return p.parse_args()


def _index(records_path: Path) -> dict[str, int]:
    rows = read_jsonl(records_path)
    return {str(row["item_id"]): i for i, row in enumerate(rows)}


def main() -> None:
    args = parse_args()
    root = args.cohort_dir
    cohort = read_json(root / "cohort.json")
    item_ids = [str(row["item_id"]) for row in cohort["items"]]
    n = len(item_ids)
    pos = {iid: i for i, iid in enumerate(item_ids)}

    runs = [args.seed_run, args.supp_run]
    if args.expand_run is not None:
        runs.append(args.expand_run)
    runs.extend(args.extra_runs or [])
    for path in runs:
        if not (path / "embeddings" / "A.npy").exists():
            raise FileNotFoundError(f"missing embeddings in {path}")

    out = root / "embeddings" / "sam3_abcd"
    out.mkdir(parents=True, exist_ok=True)
    arms = ("A", "B", "C", "D")
    mats = {arm: np.full((n, 768), np.nan, dtype=np.float32) for arm in arms}

    for run in runs:
        idx = _index(run / "dataset" / "records.jsonl")
        for arm in arms:
            src = np.load(run / "embeddings" / f"{arm}.npy")
            for iid, j in idx.items():
                if iid in pos:
                    mats[arm][pos[iid]] = src[j]

    for arm, mat in mats.items():
        np.save(out / f"{arm}.npy", mat)
    coverage = {arm: int(np.isfinite(mat).all(axis=1).sum()) for arm, mat in mats.items()}
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "n": n,
        "coverage": coverage,
        "seed_run": str(args.seed_run),
        "supp_run": str(args.supp_run),
        "expand_run": str(args.expand_run) if args.expand_run else None,
        "extra_runs": [str(p) for p in (args.extra_runs or [])],
        "bank": "sam3_abcd",
        "note": "single bank: FashionSigLIP embeddings under SAM3-derived A/B/C/D preprocess",
    }
    write_json(out / "summary.json", summary)
    write_json(root / "embeddings" / "item_ids.json", {"item_ids": item_ids, "n": n})
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # run metrics for this bank only
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            str(HERE / "cluster_quality.py"),
            "--cohort-dir",
            str(root),
            "--banks",
            "sam3_abcd",
            "--bootstrap",
            "2000",
        ]
    )


if __name__ == "__main__":
    main()
