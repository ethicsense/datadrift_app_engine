#!/usr/bin/env bash
# DD Diagnosis 웹앱 실행 (가상환경이 있으면 자동 활성화)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
elif [[ -f "${ROOT}/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/venv/bin/activate"
fi

exec python "${ROOT}/run_webapp.py" "$@"
