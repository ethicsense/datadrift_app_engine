#!/bin/bash

# 로컬 테스트용 스크립트
# shebang을 유지하면서 로컬에서 실행할 때 사용

# 현재 환경의 Python 경로 확인
PYTHON_PATH=$(which python)

echo "🐍 Python 경로: $PYTHON_PATH"
echo "🚀 DDoc 로컬 실행 중..."

# main.py를 현재 Python으로 실행
$PYTHON_PATH main.py "$@" 