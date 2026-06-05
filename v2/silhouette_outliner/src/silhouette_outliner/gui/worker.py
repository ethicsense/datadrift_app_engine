from __future__ import annotations

import logging
import re
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..cli import run_collect
from ..config import default_collect_config_path
from ..exceptions import CollectCancelled

_LOG = logging.getLogger("silhouette_outliner")

_COLLECT_DONE_RE = re.compile(r"수집 완료 \((\d+)/(\d+)\)")
_COLLECT_START_RE = re.compile(r"수집 중 \((\d+)/(\d+)\)")
_COLLECT_TOTAL_RE = re.compile(r"원시 데이터 수집 단계 \((\d+)개 대상\)")
_COLLECT_PROGRESS_START = 0.0
_COLLECT_PROGRESS_SPAN = 70.0

_PREP_PROGRESS_PREFIXES = (
    "collect 시작",
    "설정 로드 완료",
    "실행 디렉터리 준비",
    "설정 사본 저장",
)

_STAGE_PROGRESS: dict[str, float] = {
    "고객 신호 수집 중": 72.0,
    "정규화 중": 74.0,
    "정규화 완료": 80.0,
    "분석 중": 82.0,
    "분석 완료": 88.0,
    "HTML 리포트 렌더링 중": 90.0,
    "리포트 작성 완료": 100.0,
    "총 소요시간": 100.0,
}


class QtLogHandler(logging.Handler):
    """Bridge silhouette_outliner logger records to Qt signals."""

    def __init__(
        self,
        on_log: Callable[[str], None],
        on_progress: Callable[[int, str], None],
    ) -> None:
        super().__init__()
        self._on_log = on_log
        self._on_progress = on_progress
        self.setFormatter(logging.Formatter("%(message)s"))
        self._collect_total: int | None = None
        self._last_progress = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        self._on_log(message)
        progress = self._parse_progress(message)
        if progress is not None:
            self._emit_progress(progress, message)

    def _emit_progress(self, value: float, message: str) -> None:
        clamped = round(max(0.0, min(100.0, value)), 1)
        if clamped <= self._last_progress:
            return
        self._last_progress = clamped
        self._on_progress(clamped, message)

    def _collect_progress(self, current: int, total: int, *, completed: bool) -> float | None:
        if total <= 0:
            return None
        index = current if completed else max(0, current - 1)
        ratio = min(1.0, index / total)
        return round(_COLLECT_PROGRESS_START + ratio * _COLLECT_PROGRESS_SPAN, 1)

    def _parse_progress(self, message: str) -> float | None:
        if message.startswith(_PREP_PROGRESS_PREFIXES):
            return None

        for prefix, value in _STAGE_PROGRESS.items():
            if message.startswith(prefix):
                return value

        total_match = _COLLECT_TOTAL_RE.search(message)
        if total_match:
            self._collect_total = int(total_match.group(1))
            return None

        done_match = _COLLECT_DONE_RE.search(message)
        if done_match:
            current = int(done_match.group(1))
            total = int(done_match.group(2)) or self._collect_total or 0
            return self._collect_progress(current, total, completed=True)

        start_match = _COLLECT_START_RE.search(message)
        if start_match:
            current = int(start_match.group(1))
            total = int(start_match.group(2)) or self._collect_total or 0
            return self._collect_progress(current, total, completed=False)

        if message.startswith("중간 결과 저장"):
            return 89.0

        if message.startswith("고객 신호 리포트") or message.startswith("브랜드 포트폴리오 리포트"):
            return 95.0

        return None


class CollectWorker(QObject):
    """Run run_collect in a background thread."""

    log_message = Signal(str)
    progress_changed = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        output_root: Path,
        config_path: Path | None = None,
        run_name: str | None = None,
    ) -> None:
        super().__init__()
        self._output_root = output_root
        self._config_path = config_path if config_path is not None else default_collect_config_path()
        self._run_name = run_name
        self._handler: QtLogHandler | None = None
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        self._cancel_event.clear()
        self._handler = QtLogHandler(
            on_log=lambda msg: self.log_message.emit(msg),
            on_progress=lambda value, msg: self.progress_changed.emit(value, msg),
        )
        _LOG.addHandler(self._handler)
        try:
            report_path = run_collect(
                self._config_path,
                self._output_root,
                self._run_name,
                should_cancel=self._cancel_event.is_set,
            )
            self.progress_changed.emit(100.0, "분석이 완료되었습니다.")
            self.finished.emit(report_path)
        except CollectCancelled:
            self.log_message.emit("분석이 중단되었습니다.")
            self.cancelled.emit()
        except Exception as exc:
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.log_message.emit(detail)
            self.failed.emit(str(exc))
        finally:
            if self._handler is not None:
                _LOG.removeHandler(self._handler)
                self._handler = None
