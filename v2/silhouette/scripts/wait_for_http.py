#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import urllib.request


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wait_for_http.py <url> [timeout_seconds]", file=sys.stderr)
        return 2

    url = sys.argv[1]
    timeout_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    start = time.time()

    while time.time() - start < timeout_seconds:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"[wait] ready: {url}")
            return 0
        except Exception:
            time.sleep(1)

    print(f"[wait] timeout waiting for: {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
