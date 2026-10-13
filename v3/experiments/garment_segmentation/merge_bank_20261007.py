"""Merge seed sam3_abcd (20261002 order) + add1k preprocess into 20261007 bank."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DST = REPO / "artifacts/garment_segmentation/attribute_cluster_eval-20261007"
SRC = REPO / "artifacts/garment_segmentation/attribute_cluster_eval-20261002"


def _index(records_path: Path) -> dict[str, int]:
    rows = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return {str(r["item_id"]): i for i, r in enumerate(rows)}


def main() -> None:
    cohort = json.loads((DST / "cohort.json").read_text(encoding="utf-8"))
    item_ids = [str(r["item_id"]) for r in cohort["items"]]
    n = len(item_ids)
    pos = {iid: i for i, iid in enumerate(item_ids)}

    seed_ids = json.loads((SRC / "embeddings" / "item_ids.json").read_text())["item_ids"]
    arms = ("A", "B", "C", "D")
    mats = {arm: np.full((n, 768), np.nan, dtype=np.float32) for arm in arms}
    for arm in arms:
        seed = np.load(SRC / "embeddings" / "sam3_abcd" / f"{arm}.npy")
        for j, iid in enumerate(seed_ids):
            if iid in pos:
                mats[arm][pos[iid]] = seed[j]

    add_run = DST / "pipeline" / "preprocess" / "preprocess-abcd-sam3-add1k"
    if not (add_run / "embeddings" / "A.npy").is_file():
        # alternate run id pattern
        cands = sorted((DST / "pipeline" / "preprocess").glob("preprocess-abcd*"))
        if not cands:
            raise FileNotFoundError(f"missing add1k preprocess under {DST}/pipeline/preprocess")
        add_run = cands[-1]
    rec = add_run / "dataset" / "records.jsonl"
    idx = _index(rec)
    for arm in arms:
        src = np.load(add_run / "embeddings" / f"{arm}.npy")
        for iid, j in idx.items():
            if iid in pos and j < len(src):
                mats[arm][pos[iid]] = src[j]

    out = DST / "embeddings" / "sam3_abcd"
    out.mkdir(parents=True, exist_ok=True)
    (DST / "embeddings" / "item_ids.json").write_text(
        json.dumps({"item_ids": item_ids, "n": n}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coverage = {}
    for arm in arms:
        np.save(out / f"{arm}.npy", mats[arm])
        coverage[arm] = int(np.isfinite(mats[arm]).all(1).sum())
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "n": n,
        "coverage": coverage,
        "seed_hub": str(SRC),
        "add1k_run": str(add_run),
        "bank": "sam3_abcd",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
