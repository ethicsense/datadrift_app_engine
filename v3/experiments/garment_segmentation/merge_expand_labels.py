from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / "artifacts/garment_segmentation/attribute_cluster_eval-20261002"
ALLOWED = {
    "pattern": {"solid","stripe","check","graphic_logo","other","uncertain"},
    "sleeve": {"sleeveless","short","long","uncertain"},
    "neck": {"round","vneck","collar","hood","turtleneck","uncertain"},
    "color": {"white_ivory","black","gray","navy_blue","beige_brown","chromatic","uncertain"},
    "fit": {"slim","regular","oversized","uncertain"},
}
def main():
    cohort = json.loads((ROOT/"cohort.json").read_text())
    expected = set(cohort["new_item_ids"])
    merged = {}
    errors = []
    for path in sorted((ROOT/"label_chunks_expand").glob("labels_chunk_*.jsonl")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row["item_id"]
            labs = row.get("labels") or {}
            for a, allowed in ALLOWED.items():
                if labs.get(a) not in allowed:
                    errors.append(f"{path.name}:{i} {a}={labs.get(a)}")
            row["labeler"] = "agent"
            row.setdefault("labeled_at", datetime.now(UTC).isoformat())
            merged[iid] = row
    missing = sorted(expected - set(merged))
    # merge into labels.jsonl keeping old
    all_labels = {}
    for line in (ROOT/"labels.jsonl").read_text().splitlines():
        if line.strip():
            r=json.loads(line); all_labels[r["item_id"]]=r
    # human overrides
    hp = ROOT/"human_labels_review50.jsonl"
    if hp.exists():
        for line in hp.read_text().splitlines():
            if line.strip():
                r=json.loads(line)
                base=all_labels.get(r["item_id"], {"item_id":r["item_id"],"labeler":"agent","labels":{}})
                base["labels"]={**base.get("labels",{}), **r.get("labels",{})}
                all_labels[r["item_id"]]=base
    all_labels.update(merged)
    (ROOT/"labels.jsonl").write_text("\n".join(json.dumps(all_labels[i], ensure_ascii=False) for i in sorted(all_labels))+"\n")
    summary={"new_merged":len(merged),"expected_new":len(expected),"missing":missing,"errors":errors[:20],"error_count":len(errors),"total_labels":len(all_labels)}
    (ROOT/"label_chunks_expand/merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
