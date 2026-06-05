#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "오류: 프로젝트 루트에 .venv가 없습니다. 먼저 가상환경을 생성하세요." >&2
  echo "  python -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -e \".[dev,gui]\"" >&2
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"
exec python -m silhouette_outliner.gui
