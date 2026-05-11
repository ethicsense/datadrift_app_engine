#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data").resolve()
    if not data_dir.exists():
        print(f"[preflight] data directory not found: {data_dir}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"[preflight] path is not a directory: {data_dir}", file=sys.stderr)
        return 1

    ranking_files = list(data_dir.rglob("ranking_summary.json"))
    if not ranking_files:
        print(
            "[preflight] no ranking_summary.json found under data directory. "
            "Please insert source snapshot data first.",
            file=sys.stderr,
        )
        return 1

    print(f"[preflight] data directory ready: {data_dir}")
    print(f"[preflight] detected ranking snapshots: {len(ranking_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
