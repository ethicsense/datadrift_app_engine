#!/usr/bin/env bash
# DD Diagnosis: Python 가상환경 생성 + 의존성 설치
#
# 사용법:
#   ./setup_env.sh
#   SKIP_CLIP=1 ./setup_env.sh    # CLIP(git) 설치 생략 (빠른 검증용)
#   PYTHON=python3.12 ./setup_env.sh   # 특정 Python으로 venv 생성

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${ROOT}/.venv"
PYTHON_CMD="${PYTHON:-python3}"

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "오류: '${PYTHON_CMD}' 를 찾을 수 없습니다. PYTHON= 경로를 지정하세요." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo ">>> 가상환경 생성: ${VENV_DIR}"
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
else
  echo ">>> 기존 가상환경 사용: ${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo ">>> pip 업그레이드"
python -m pip install --upgrade pip setuptools wheel

echo ">>> requirements.txt 설치"
pip install -r "${ROOT}/requirements.txt"

if [[ "${SKIP_CLIP:-}" == "1" ]]; then
  echo ">>> SKIP_CLIP=1 — OpenAI CLIP 설치를 건너뜁니다."
else
  echo ">>> OpenAI CLIP 설치 (git)"
  pip install "git+https://github.com/openai/CLIP.git"
fi

echo ""
echo "완료. 활성화:"
echo "  source ${VENV_DIR}/bin/activate"
echo "실행:"
echo "  ./run_webapp.sh"
echo "  # 또는: python run_webapp.py --port 5555"
