from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..runtime_paths import configure_playwright_browsers, configure_tls, runs_root
from .network import check_online
from .worker import CollectWorker

_PLATFORMS: tuple[tuple[str, str, bool], ...] = (
    ("musinsa", "무신사", True),
    ("28cm", "28CM", False),
    ("youtube", "유튜브", False),
    ("instagram", "인스타그램", False),
    ("naver_news", "네이버뉴스", False),
)


def _find_latest_report(runs_root: Path) -> Path | None:
    if not runs_root.is_dir():
        return None

    candidates: list[tuple[float, Path]] = []
    for entry in runs_root.iterdir():
        if not entry.is_dir() or entry.name.endswith("_discovery"):
            continue
        report = entry / "report.html"
        if report.is_file():
            candidates.append((report.stat().st_mtime, report))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Silhouette Outliner")
        self.resize(1024, 640)

        self._runs_root = runs_root()
        self._report_path: Path | None = _find_latest_report(self._runs_root)
        self._worker_thread: QThread | None = None
        self._worker: CollectWorker | None = None
        self._is_running = False
        self._platform_buttons: dict[str, QPushButton] = {}
        self._progress_message = "대기 중"

        self._build_ui()
        self._refresh_report_buttons()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        control_panel = QWidget()
        control_panel.setMinimumWidth(300)
        control_panel.setMaximumWidth(380)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 8, 0)

        platform_group = QGroupBox("플랫폼")
        platform_layout = QGridLayout(platform_group)
        platform_layout.setHorizontalSpacing(8)
        platform_layout.setVerticalSpacing(8)

        self._platform_group = QButtonGroup(self)
        self._platform_group.setExclusive(True)

        for index, (platform_id, label, enabled) in enumerate(_PLATFORMS):
            button = QPushButton(label if enabled else f"{label}\n(준비 중)")
            button.setCheckable(True)
            button.setEnabled(enabled)
            button.setMinimumHeight(52)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if not enabled:
                button.setStyleSheet("color: palette(mid);")
            self._platform_group.addButton(button)
            self._platform_buttons[platform_id] = button
            row = index // 2
            col = index % 2
            platform_layout.addWidget(button, row, col)

        self._platform_buttons["musinsa"].setChecked(True)
        control_layout.addWidget(platform_group)

        run_row = QHBoxLayout()
        self._run_button = QPushButton("분석 실행")
        self._run_button.setMinimumHeight(36)
        self._run_button.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self._run_button)

        self._stop_button = QPushButton("분석 중단")
        self._stop_button.setMinimumHeight(36)
        self._stop_button.setEnabled(False)
        self._stop_button.setStyleSheet(
            """
            QPushButton:enabled {
                color: #b71c1c;
                border: 1px solid #ef9a9a;
            }
            QPushButton:enabled:hover {
                background-color: #ffebee;
            }
            """
        )
        self._stop_button.clicked.connect(self._on_stop_clicked)
        run_row.addWidget(self._stop_button)
        control_layout.addLayout(run_row)

        self._open_report_button = QPushButton("보고서 열기")
        self._open_report_button.setMinimumHeight(36)
        self._open_report_button.clicked.connect(self._on_open_report_clicked)
        control_layout.addWidget(self._open_report_button)

        self._open_folder_button = QPushButton("보고서 폴더 열기")
        self._open_folder_button.setMinimumHeight(36)
        self._open_folder_button.clicked.connect(self._on_open_folder_clicked)
        control_layout.addWidget(self._open_folder_button)

        progress_group = QGroupBox("진행 상황")
        progress_layout = QVBoxLayout(progress_group)
        self._progress_label = QLabel("대기 중")
        self._progress_label.setWordWrap(False)
        label_height = self._progress_label.fontMetrics().height() + 6
        self._progress_label.setFixedHeight(label_height)
        self._progress_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #a5d6a7;
                border-radius: 4px;
                text-align: center;
                background-color: #e8f5e9;
                color: #1b5e20;
            }
            QProgressBar::chunk {
                background-color: #43a047;
                border-radius: 3px;
            }
            """
        )
        self._set_progress_percent(0)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._progress_bar)
        control_layout.addWidget(progress_group)

        control_layout.addStretch()
        splitter.addWidget(control_panel)

        log_group = QGroupBox("실행 로그")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Monospace", 11))
        log_layout.addWidget(self._log_view)
        splitter.addWidget(log_group)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 660])

    def _selected_platform(self) -> tuple[str, str, bool] | None:
        for platform_id, label, enabled in _PLATFORMS:
            button = self._platform_buttons.get(platform_id)
            if button is not None and button.isChecked():
                return platform_id, label, enabled
        return None

    def _set_platforms_enabled(self, enabled: bool) -> None:
        for platform_id, _, available in _PLATFORMS:
            button = self._platform_buttons[platform_id]
            button.setEnabled(enabled and available)

    def _refresh_report_buttons(self) -> None:
        has_report = self._report_path is not None and self._report_path.is_file()
        self._open_report_button.setEnabled(has_report and not self._is_running)
        self._open_folder_button.setEnabled(has_report and not self._is_running)

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_progress_percent(self, value: int | float) -> None:
        clamped = max(0.0, min(100.0, float(value)))
        self._progress_bar.setValue(int(round(clamped * 10)))
        self._progress_bar.setFormat(f"{clamped:.1f}%")

    def _set_progress_message(self, message: str) -> None:
        self._progress_message = message
        self._update_progress_label_elided()

    def _update_progress_label_elided(self) -> None:
        label_width = self._progress_label.width()
        if label_width <= 0:
            label_width = self._progress_label.sizeHint().width() or 280
        elided = self._progress_label.fontMetrics().elidedText(
            self._progress_message,
            Qt.TextElideMode.ElideRight,
            max(label_width, 120),
        )
        self._progress_label.setText(elided)
        self._progress_label.setToolTip(self._progress_message)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_progress_label_elided()

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self._run_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._set_platforms_enabled(not running)
        self._refresh_report_buttons()

    def _on_run_clicked(self) -> None:
        if self._is_running:
            return

        selected = self._selected_platform()
        if selected is None:
            QMessageBox.warning(self, "플랫폼 선택", "분석할 플랫폼을 선택해 주세요.")
            return

        platform_id, label, enabled = selected
        if not enabled or platform_id != "musinsa":
            QMessageBox.information(
                self,
                "준비 중",
                f"{label} 플랫폼은 아직 지원되지 않습니다.",
            )
            return

        if not check_online():
            QMessageBox.warning(
                self,
                "네트워크 오류",
                "연결된 네트워크를 찾을 수 없습니다.",
            )
            return

        self._log_view.clear()
        self._set_progress_percent(0)
        self._set_progress_message("분석을 시작합니다…")
        self._set_running(True)

        self._worker_thread = QThread()
        self._worker = CollectWorker(output_root=self._runs_root)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self._append_log)
        self._worker.progress_changed.connect(self._on_progress_changed)
        self._worker.finished.connect(self._on_collect_finished)
        self._worker.failed.connect(self._on_collect_failed)
        self._worker.cancelled.connect(self._on_collect_cancelled)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)

        self._worker_thread.start()

    def _on_stop_clicked(self) -> None:
        if not self._is_running or self._worker is None:
            return
        self._stop_button.setEnabled(False)
        self._set_progress_message("중단 요청 중… 현재 작업이 끝나면 멈춥니다.")
        self._worker.request_cancel()

    def _on_progress_changed(self, value: float, message: str) -> None:
        self._set_progress_percent(value)
        self._set_progress_message(message)

    def _on_collect_finished(self, report_path: object) -> None:
        path = Path(report_path) if report_path else None
        if path is not None and path.is_file():
            self._report_path = path
        self._set_progress_percent(100)
        self._set_progress_message("분석이 완료되었습니다.")
        self._set_running(False)
        self._refresh_report_buttons()
        QMessageBox.information(
            self,
            "완료",
            f"분석이 완료되었습니다.\n\n보고서: {self._report_path}",
        )

    def _on_collect_failed(self, error_message: str) -> None:
        self._set_progress_message("분석 중 오류가 발생했습니다.")
        self._set_running(False)
        QMessageBox.critical(
            self,
            "오류",
            f"분석 중 오류가 발생했습니다.\n\n{error_message}",
        )

    def _on_collect_cancelled(self) -> None:
        self._set_progress_message("분석이 중단되었습니다.")
        self._set_running(False)
        QMessageBox.information(self, "중단", "분석이 중단되었습니다.")

    def _cleanup_worker(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
            self._worker_thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _on_open_report_clicked(self) -> None:
        if self._report_path is None or not self._report_path.is_file():
            QMessageBox.warning(self, "보고서 없음", "열 수 있는 보고서가 없습니다.")
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_path.resolve())))
        if not opened:
            QMessageBox.warning(
                self,
                "열기 실패",
                f"보고서를 열 수 없습니다.\n\n{self._report_path}",
            )

    def _on_open_folder_clicked(self) -> None:
        if self._report_path is None or not self._report_path.is_file():
            QMessageBox.warning(self, "폴더 없음", "열 수 있는 보고서 폴더가 없습니다.")
            return
        folder = self._report_path.parent.resolve()
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        if not opened:
            QMessageBox.warning(
                self,
                "열기 실패",
                f"폴더를 열 수 없습니다.\n\n{folder}",
            )


def main() -> None:
    configure_tls()
    configure_playwright_browsers()
    app = QApplication(sys.argv)
    app.setApplicationName("Silhouette Outliner")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
