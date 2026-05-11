# ddoc/cli/main.py
from __future__ import annotations
import logging
import sys

import click
import typer
from dotenv import load_dotenv

from importlib.metadata import version as get_version, metadata as get_metadata, PackageNotFoundError

from ddoc.cli import commands as core_commands
from ddoc.core.plugins import get_plugin_manager

# ------------------------------------------------------
# 📦 pyproject.toml 메타 정보 읽기
# ------------------------------------------------------
try:
    APP_VERSION = get_version("ddoc")
except PackageNotFoundError:
    APP_VERSION = "0.0.0"

try:
    meta = get_metadata("ddoc")
    DESCRIPTION = meta.get("Summary", "ddoc: data drift doctor")
except Exception:
    DESCRIPTION = "ddoc: data drift doctor"

RELEASE_DATE = ""         # 여전히 config에 있다면 별도 유지
DDOC_HUB_URL = ""         # 필요 시 상수 처리
ASCII_LOGO = r"""
=======================================
 _____    ____     ___     ____ 
|  __ \  |  _ \   / _ \   / ___| 
| |  | | | | | | | | | | |    
| |__| | | |_| | | |_| | | |___ 
|_____/  |____/   \___/   \____| 

Data Drift Doctor (ddoc)
Korea Electronics Technology Institute
=======================================
"""

# ------------------------------------------------------
# 🎨 로고 표시 함수
# ------------------------------------------------------
def show_logo():
    if ASCII_LOGO:
        click.echo(ASCII_LOGO)

# ------------------------------------------------------
# ⚙️ 공통 초기화 함수
# ------------------------------------------------------
def init_app(debug: bool = False, load_plugins: bool = True):
    load_dotenv()
    #click.echo("✅ .env 환경변수 로드됨")
    
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.debug("📋 로깅이 초기화되었습니다.")

    if debug:
        click.echo("🔬 디버그 모드 활성화: 상세 로그 (DEBUG 레벨)가 출력")
    
    # Only load plugins if needed (for performance)
    if load_plugins:
        get_plugin_manager()
        logging.debug("🔌 플러그인 매니저 로드됨.")

# ------------------------------------------------------
# 📘 메타 정보 출력
# ------------------------------------------------------
def print_meta_info(is_show_logo=True, full=False):
    if is_show_logo:
        show_logo()
        
    click.echo(f"🔖 Version       : {APP_VERSION}")
    if RELEASE_DATE:
        click.echo(f"📅 Release Date  : {RELEASE_DATE}")
    
    if full:
        click.echo(f"📘 Description   : {DESCRIPTION}")
        if DDOC_HUB_URL:
            click.echo(f"🌐 Hub URL       : {DDOC_HUB_URL}")
    raise typer.Exit()

# ------------------------------------------------------
# 🧭 Typer 앱 정의
# ------------------------------------------------------
app = typer.Typer(
    help=DESCRIPTION,
    add_completion=False,
)

@app.callback(invoke_without_command=True)
def _bootstrap(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        help="Show version info and exit.",
        is_eager=True,
        callback=lambda v: print_meta_info(is_show_logo=False, full=False) if v else None,
    ),
    about: bool = typer.Option(
        None,
        "--about",
        help="Show full app meta info and exit.",
        is_eager=True,
        callback=lambda a: print_meta_info(is_show_logo=True, full=True) if a else None,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    # help 요청은 플러그인 로딩 금지 (무거운 의존성 import 회피)
    is_help_request = "--help" in sys.argv or "-h" in sys.argv

    # 최소 CLI: 분석만 플러그인이 필요합니다(plugin list/info는 entry point만 읽음)
    load_plugins = (ctx.invoked_subcommand == "analyze") and (not is_help_request)

    init_app(debug=debug, load_plugins=load_plugins)

    if ctx.invoked_subcommand is None:
        show_logo()
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

# ------------------------------------------------------
# 🔗 명령어 등록
# ------------------------------------------------------
core_commands.register(app)

# ------------------------------------------------------
# 🚀 엔트리포인트
# ------------------------------------------------------
def main():
    try:
        
        app()
    except typer.Exit:
        raise
    except Exception as e:
        logging.exception("❌ 처리되지 않은 예외 발생:")
        click.echo(f"❌ 에러: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    main()