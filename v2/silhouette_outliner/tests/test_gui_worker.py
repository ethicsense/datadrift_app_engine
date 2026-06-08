from __future__ import annotations

import logging

from silhouette_outliner.gui.worker import QtLogHandler


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="silhouette_outliner",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_prep_logs_do_not_advance_progress() -> None:
    progress: list[float] = []
    handler = QtLogHandler(
        on_log=lambda _msg: None,
        on_progress=lambda value, _msg: progress.append(value),
    )

    handler.emit(_record("collect 시작"))
    handler.emit(_record("설정 사본 저장: /tmp/config.json"))
    handler.emit(_record("원시 데이터 수집 단계 (212개 대상)"))

    assert progress == []


def test_qt_log_handler_parses_collect_progress_from_zero() -> None:
    progress: list[tuple[float, str]] = []
    handler = QtLogHandler(
        on_log=lambda _msg: None,
        on_progress=lambda value, msg: progress.append((value, msg)),
    )

    handler.emit(_record("원시 데이터 수집 단계 (4개 대상)"))
    handler.emit(_record("수집 완료 (1/4) key=foo → 성공, 소스=client-api"))
    handler.emit(_record("수집 중 (2/4) key=bar …"))
    handler.emit(_record("수집 완료 (2/4) key=bar → 성공, 소스=client-api"))
    handler.emit(_record("정규화 중…"))
    handler.emit(_record("리포트 작성 완료: /tmp/report.html"))

    assert progress[0] == (17.5, "수집 완료 (1/4) key=foo → 성공, 소스=client-api")
    assert progress[1] == (35.0, "수집 완료 (2/4) key=bar → 성공, 소스=client-api")
    assert all(progress[i][0] <= progress[i + 1][0] for i in range(len(progress) - 1))
    assert progress[-1] == (100.0, "리포트 작성 완료: /tmp/report.html")
