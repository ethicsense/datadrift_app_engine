#!/usr/bin/env bash
# 프로젝트 표준 가상환경(silhouette_venv)으로 pytest 실행.
# 사용: ./scripts/run_tests.sh
#       ./scripts/run_tests.sh -v tests/test_brand_style_embedding.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT}/silhouette_venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "오류: ${VENV_PY} 가 없습니다." >&2
  echo "  python3 -m venv silhouette_venv" >&2
  echo "  source silhouette_venv/bin/activate && pip install -r requirements.txt pytest" >&2
  exit 1
fi

exec "$VENV_PY" -m pytest "${ROOT}/tests" "$@"
