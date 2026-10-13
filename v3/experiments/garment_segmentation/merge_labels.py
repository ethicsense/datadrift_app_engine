"""Merge chunk label JSONL files into labels.jsonl and rebuild review HTML."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

ALLOWED = {
    "pattern": {"solid", "stripe", "check", "graphic_logo", "other", "uncertain"},
    "sleeve": {"sleeveless", "short", "long", "uncertain"},
    "neck": {"round", "vneck", "collar", "hood", "turtleneck", "uncertain"},
    "color": {
        "white_ivory",
        "black",
        "gray",
        "navy_blue",
        "beige_brown",
        "chromatic",
        "uncertain",
    },
    "fit": {"slim", "regular", "oversized", "uncertain"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument("--rebuild-review", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.cohort_dir
    cohort = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    expected = {row["item_id"] for row in cohort["items"]}
    chunks_dir = root / "label_chunks"
    merged: dict[str, dict] = {}
    errors: list[str] = []
    for path in sorted(chunks_dir.glob("labels_chunk_*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row.get("item_id") or "")
            labels = row.get("labels") or {}
            if iid not in expected:
                errors.append(f"{path.name}:{line_no} unknown item_id {iid}")
                continue
            for attr, allowed in ALLOWED.items():
                val = labels.get(attr)
                if val not in allowed:
                    errors.append(f"{path.name}:{line_no} bad {attr}={val!r}")
            row["labeler"] = "agent"
            row.setdefault("labeled_at", datetime.now(UTC).isoformat())
            merged[iid] = row
    out = root / "labels.jsonl"
    lines = [json.dumps(merged[iid], ensure_ascii=False) for iid in sorted(merged)]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    missing = sorted(expected - set(merged))
    summary = {
        "merged": len(merged),
        "expected": len(expected),
        "missing": len(missing),
        "missing_ids_head": missing[:20],
        "errors": errors[:50],
        "error_count": len(errors),
    }
    (root / "labels_merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.rebuild_review:
        sys.path.insert(0, str(HERE))
        from build_label_review import main as build_review

        # rebuild with defaults pointing at same cohort dir
        import build_label_review as blr

        sys.argv = ["build_label_review.py", "--cohort-dir", str(root)]
        blr.main()


if __name__ == "__main__":
    main()
