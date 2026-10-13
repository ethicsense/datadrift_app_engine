"""Merge add2k chunk labels into attribute_cluster_eval-20261007/labels.jsonl."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "artifacts/garment_segmentation/attribute_cluster_eval-20261007"
ALLOWED = {
    "pattern": {"solid", "stripe", "check", "graphic_logo", "other", "uncertain"},
    "sleeve": {"sleeveless", "short", "long", "uncertain"},
    "neck": {"round", "vneck", "collar", "hood", "turtleneck", "uncertain"},
    "color": {"white_ivory", "black", "gray", "navy_blue", "beige_brown", "chromatic", "uncertain"},
    "fit": {"slim", "regular", "oversized", "uncertain"},
}


def main() -> None:
    cohort = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))
    expected_new = set(cohort.get("new_item_ids") or [])
    merged: dict[str, dict] = {}
    errors: list[str] = []
    chunks = ROOT / "label_chunks_add2k"
    for path in sorted(chunks.glob("labels_chunk_*.jsonl")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row["item_id"])
            labs = row.get("labels")
            if not isinstance(labs, dict):
                labs = {a: row.get(a) for a in ALLOWED}
                row = {"item_id": iid, "labels": labs, **{k: v for k, v in row.items() if k not in ALLOWED and k != "item_id"}}
            for a, allowed in ALLOWED.items():
                if labs.get(a) not in allowed:
                    errors.append(f"{path.name}:{i} {a}={labs.get(a)}")
            row["labeler"] = "agent"
            row.setdefault("labeled_at", datetime.now(UTC).isoformat())
            merged[iid] = row

    all_labels: dict[str, dict] = {}
    for line in (ROOT / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            all_labels[str(r["item_id"])] = r
    hp = ROOT / "human_labels_review50.jsonl"
    if hp.is_file():
        for line in hp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = str(r["item_id"])
            base = all_labels.get(iid, {"item_id": iid, "labeler": "agent", "labels": {}})
            base["labels"] = {**base.get("labels", {}), **(r.get("labels") or {})}
            all_labels[iid] = base
    all_labels.update(merged)
    (ROOT / "labels.jsonl").write_text(
        "\n".join(json.dumps(all_labels[i], ensure_ascii=False) for i in sorted(all_labels)) + "\n",
        encoding="utf-8",
    )
    missing = sorted(expected_new - set(merged))
    summary = {
        "new_merged": len(merged),
        "expected_new": len(expected_new),
        "missing": len(missing),
        "missing_ids_head": missing[:30],
        "errors": errors[:40],
        "error_count": len(errors),
        "total_labels": len(all_labels),
    }
    (chunks / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
