"""Compute Cohen's κ between agent and human attribute labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["item_id"])] = dict(row.get("labels") or {})
    return out


def cohen_kappa(y1: list[str], y2: list[str]) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    agree = sum(a == b for a, b in zip(y1, y2)) / n
    c1 = Counter(y1)
    c2 = Counter(y2)
    labels = set(c1) | set(c2)
    pe = sum((c1[lab] / n) * (c2[lab] / n) for lab in labels)
    if pe >= 1.0:
        return 1.0 if agree == 1.0 else float("nan")
    return (agree - pe) / (1.0 - pe)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", type=Path, required=True)
    p.add_argument("--human", type=Path, required=True)
    p.add_argument(
        "--attrs",
        nargs="*",
        default=["pattern", "sleeve", "neck", "color", "fit"],
    )
    args = p.parse_args()
    agent = load_labels(args.agent)
    human = load_labels(args.human)
    ids = sorted(set(agent) & set(human))
    print(f"paired_items={len(ids)}")
    for attr in args.attrs:
        pairs = [
            (agent[i].get(attr), human[i].get(attr))
            for i in ids
            if agent[i].get(attr) and human[i].get(attr)
        ]
        if not pairs:
            print(f"{attr}: no paired labels")
            continue
        y1 = [a for a, _ in pairs]
        y2 = [b for _, b in pairs]
        agree = sum(a == b for a, b in pairs) / len(pairs)
        kappa = cohen_kappa(y1, y2)
        print(f"{attr}: n={len(pairs)} agree={agree:.3f} kappa={kappa:.3f}")


if __name__ == "__main__":
    main()
