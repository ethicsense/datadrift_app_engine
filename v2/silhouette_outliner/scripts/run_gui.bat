@echo off
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist ".venv\Scripts\activate.bat" (
  echo 오류: 프로젝트 루트에 .venv가 없습니다. 먼저 가상환경을 생성하세요.
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -e ".[dev,gui]"
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m silhouette_outliner.gui
