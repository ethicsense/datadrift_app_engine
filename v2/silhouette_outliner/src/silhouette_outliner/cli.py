from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from .analyze import (
    analyze_dataset,
    brand_portfolio_report_filename,
    category_report_filename,
    customer_signals_report_filename,
)
from .collector import collect_all
from .customer_signals import collect_customer_signals
from .config import AppConfig, load_config
from .discover import discover_target
from .exceptions import CollectCancelled
from .normalize import normalize_collections
from .report import render_report, write_json

_LOG = logging.getLogger("silhouette_outliner")
_CLI_LOGGING_CONFIGURED = False


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise CollectCancelled("사용자가 분석을 중단했습니다.")


def _configure_cli_logging() -> None:
    """stderr로 진행 로그를보냅니다(Jupyter 커널 로그·터미널에서 확인 가능)."""
    global _CLI_LOGGING_CONFIGURED
    if _CLI_LOGGING_CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[silhouette-outliner] %(message)s"))
    _LOG.addHandler(handler)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False
    _CLI_LOGGING_CONFIGURED = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fashion ranking snapshots and render a static HTML report."
    )
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="Run one-shot collection and report generation.")
    collect_parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path.")
    collect_parser.add_argument("--out", type=Path, default=Path("runs"), help="Output root directory.")
    collect_parser.add_argument("--run-name", default=None, help="Optional run directory name.")

    discover_parser = subparsers.add_parser("discover", help="Inspect the first configured target.")
    discover_parser.add_argument("--config", type=Path, default=None, help="Optional JSON config path.")
    discover_parser.add_argument("--out", type=Path, default=Path("runs"), help="Output root directory.")

    subparsers.add_parser("sample-config", help="Print the default JSON config.")

    args = parser.parse_args()
    if args.command == "collect":
        run_collect(args.config, args.out, args.run_name)
    elif args.command == "discover":
        run_discover(args.config, args.out)
    elif args.command == "sample-config":
        print(json.dumps(AppConfig.defaults().to_dict(), ensure_ascii=False, indent=2))
    else:
        parser.print_help()


def _format_elapsed(seconds: float) -> str:
    total = max(0.0, seconds)
    if total < 60:
        return f"{total:.1f}초"
    minutes = int(total // 60)
    rem = total % 60
    if minutes < 60:
        return f"{minutes}분 {rem:.1f}초" if rem >= 0.05 else f"{minutes}분"
    hours = minutes // 60
    minutes = minutes % 60
    if minutes:
        return f"{hours}시간 {minutes}분 {rem:.1f}초"
    return f"{hours}시간 {rem:.1f}초"


def run_collect(
    config_path: Path | None,
    output_root: Path,
    run_name: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    _configure_cli_logging()
    started_at = time.perf_counter()
    _LOG.info("collect 시작")
    _raise_if_cancelled(should_cancel)
    config = load_config(config_path)
    cfg_desc = str(config_path.resolve()) if config_path else "(기본 설정)"
    _LOG.info("설정 로드 완료: %s", cfg_desc)

    run_dir = output_root / (run_name or _run_name())
    run_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("실행 디렉터리 준비: %s", run_dir.resolve())

    write_json(run_dir / "config.json", config.to_dict())
    _LOG.info("설정 사본 저장: %s", (run_dir / "config.json").resolve())

    targets = config.targets()
    _LOG.info("원시 데이터 수집 단계 (%d개 대상)", len(targets))
    collections = collect_all(config, run_dir, should_cancel=should_cancel)
    _raise_if_cancelled(should_cancel)

    if config.collect_customer_signals:
        _LOG.info("고객 신호 수집 중 (콘텐츠판 · 검색어 랭킹)…")
        customer_signals = collect_customer_signals(config)
    else:
        customer_signals = None
    _raise_if_cancelled(should_cancel)

    _LOG.info("정규화 중…")
    dataset = normalize_collections(collections)
    if customer_signals is not None:
        dataset.customer_signals = customer_signals
    _LOG.info("정규화 완료: 랭킹 항목 %d건", len(dataset.items))

    _LOG.info("분석 중…")
    analysis = analyze_dataset(dataset)
    _LOG.info("분석 완료")
    _raise_if_cancelled(should_cancel)

    write_json(run_dir / "normalized.json", dataset.to_dict())
    _LOG.info("중간 결과 저장: %s", (run_dir / "normalized.json").resolve())
    write_json(run_dir / "analysis.json", analysis)
    _LOG.info("중간 결과 저장: %s", (run_dir / "analysis.json").resolve())

    _LOG.info("HTML 리포트 렌더링 중…")
    _raise_if_cancelled(should_cancel)
    report_path = render_report(dataset, analysis, run_dir / "report.html")
    primary_category_code = analysis.get("meta", {}).get("primary_category_code")
    category_reports = analysis.get("category_reports", {})
    if isinstance(category_reports, dict):
        for category_code, category_analysis in category_reports.items():
            if not isinstance(category_analysis, dict):
                continue
            # Primary category is the top-level report.html; skip the duplicate.
            if category_code == primary_category_code:
                continue
            render_report(
                dataset,
                category_analysis,
                run_dir / category_report_filename(category_code, primary_category_code=primary_category_code),
            )
    customer_signals_report = analysis.get("customer_signals_report")
    if isinstance(customer_signals_report, dict):
        cs_path = run_dir / customer_signals_report_filename()
        render_report(dataset, customer_signals_report, cs_path)
        _LOG.info("고객 신호 리포트: %s", cs_path.resolve())
    brand_report = analysis.get("brand_portfolio_report")
    if isinstance(brand_report, dict) and brand_report.get("brand_portfolio", {}).get("has_data"):
        brand_path = run_dir / brand_portfolio_report_filename()
        render_report(dataset, brand_report, brand_path)
        _LOG.info("브랜드 포트폴리오 리포트: %s", brand_path.resolve())
    _LOG.info("리포트 작성 완료: %s", report_path.resolve())
    _LOG.info("총 소요시간: %s", _format_elapsed(time.perf_counter() - started_at))
    print(f"결과 파일 위치 : {report_path.resolve()}")
    return report_path


def run_discover(config_path: Path | None, output_root: Path) -> Path:
    _configure_cli_logging()
    started_at = time.perf_counter()
    _LOG.info("discover 시작")
    config = load_config(config_path)
    cfg_desc = str(config_path.resolve()) if config_path else "(기본 설정)"
    _LOG.info("설정 로드 완료: %s", cfg_desc)

    targets = config.targets()
    if not targets:
        raise SystemExit("No collection targets configured.")
    first = targets[0]
    _LOG.info("첫 번째 대상 탐색: key=%s", first.key)

    run_dir = output_root / f"{_run_name()}_discovery"
    run_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("출력 디렉터리: %s", run_dir.resolve())

    result = discover_target(first, run_dir)
    _LOG.info(
        "탐색 완료 (JSON 응답 후보 %d건, DOM 상품 링크 %d개, 오류 %d건)",
        len(result.responses),
        result.dom_product_count,
        len(result.errors),
    )
    path = run_dir / "discovery.json"
    write_json(path, result.to_dict())
    _LOG.info("결과 저장: %s", path.resolve())
    _LOG.info("총 소요시간: %s", _format_elapsed(time.perf_counter() - started_at))
    print(f"탐색 결과 파일 위치 : {path.resolve()}")
    return path


def _run_name() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
