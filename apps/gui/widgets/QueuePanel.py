"""Queue panel: live view of the RenderQueue for the process page.

Rows are driven entirely by the queue's signals — ``queueChanged`` rebuilds
the list, ``jobUpdated`` refreshes a single row — so jobs added while a
render is running appear immediately. Row actions (reorder / remove /
retry / open output) call the queue model; the currently-rendering job
can't be removed or moved.
"""

import os
import subprocess
import sys

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QProgressBar,
    QToolButton,
)

from apps.gui.lib.RenderQueue import (
    QUEUED,
    RENDERING,
    DONE,
    FAILED,
    CANCELLED,
)

_STATUS_COLORS = {
    QUEUED: "#9aa4b2",
    RENDERING: "#22D3EE",
    DONE: "#34d399",
    FAILED: "#f87171",
    CANCELLED: "#fbbf24",
}


class QueuePanel(QWidget):
    def __init__(self, render_queue, parent=None):
        super().__init__(parent)
        self._queue = render_queue

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.empty_label = QLabel("Nothing queued\nAdd videos from the Source card")
        self.empty_label.setProperty("caption", True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.empty_label, 1)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SingleSelection)
        self.list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        lay.addWidget(self.list, 1)

        footer = QHBoxLayout()
        self.counts_label = QLabel("")
        self.counts_label.setProperty("caption", True)
        footer.addWidget(self.counts_label)
        footer.addStretch(1)
        self.clear_finished_btn = QToolButton()
        self.clear_finished_btn.setText("Clear finished")
        self.clear_finished_btn.setProperty("previewMode", True)
        self.clear_finished_btn.clicked.connect(self._queue.clear_finished)
        footer.addWidget(self.clear_finished_btn)
        lay.addLayout(footer)

        self._queue.queueChanged.connect(self.rebuild)
        self._queue.jobUpdated.connect(self._refresh_row)
        self.rebuild()

    # ---- rendering ----

    def rebuild(self):
        self.list.clear()
        for index, job in enumerate(self._queue.jobs):
            item = QListWidgetItem()
            item.setSizeHint(QSize(self.list.viewport().width(), 58))
            self.list.addItem(item)
            row = _JobRow(job, index, self._queue)
            row.openOutputRequested.connect(self._open_output)
            self.list.setItemWidget(item, row)
        self._update_counts()

    def _refresh_row(self, index: int):
        if not (0 <= index < self.list.count()):
            return
        item = self.list.item(index)
        row = self.list.itemWidget(item)
        if row is not None:
            row.refresh()

    def _update_counts(self):
        empty = len(self._queue.jobs) == 0
        self.empty_label.setVisible(empty)
        self.list.setVisible(not empty)
        self.clear_finished_btn.setVisible(not empty)
        c = self._queue.counts()
        parts = []
        if c.get("rendering"):
            parts.append(f"{c['rendering']} rendering")
        if c.get("queued"):
            parts.append(f"{c['queued']} queued")
        if c.get("done"):
            parts.append(f"{c['done']} done")
        if c.get("failed"):
            parts.append(f"{c['failed']} failed")
        if c.get("cancelled"):
            parts.append(f"{c['cancelled']} cancelled")
        self.counts_label.setText(" · ".join(parts) if parts else "Queue is empty")

    def _open_output(self, path: str):
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            return
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])


class _JobRow(QWidget):
    """One queue row: name + summary + status chip, progress bar while
    rendering, hover actions per status."""

    openOutputRequested = Signal(str)

    def __init__(self, job, index: int, queue, parent=None):
        super().__init__(parent)
        self._job = job
        self._index = index
        self._queue = queue

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 4, 4)
        lay.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.title = QLabel(os.path.basename(job.input_file))
        self.title.setToolTip(job.input_file)
        top.addWidget(self.title, 1)

        self.status_chip = QLabel(job.status)
        top.addWidget(self.status_chip)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(2)
        top.addLayout(self.actions)
        lay.addLayout(top)

        self.summary = QLabel(job.summary())
        self.summary.setProperty("caption", True)
        lay.addWidget(self.summary)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        lay.addWidget(self.progress)

        self.refresh()

    def refresh(self):
        job = self._job
        color = _STATUS_COLORS.get(job.status, "#9aa4b2")
        self.status_chip.setText(job.status_detail or job.status)
        self.status_chip.setStyleSheet(
            f"color: {color};"
            "background-color: rgba(255,255,255,0.05);"
            "border-radius: 4px; padding: 1px 8px; font-size: 8pt;"
        )
        self.summary.setText(job.summary())

        is_rendering = job.status == RENDERING
        self.progress.setVisible(is_rendering)
        if is_rendering:
            self.progress.setRange(0, max(job.max_value, 1))
            self.progress.setValue(job.progress)

        self._rebuild_actions()

    def _rebuild_actions(self):
        while self.actions.count():
            item = self.actions.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        job = self._job

        def add_btn(text, tooltip, callback):
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tooltip)
            btn.setProperty("previewMode", True)
            btn.clicked.connect(callback)
            self.actions.addWidget(btn)
            return btn

        if job.status == QUEUED:
            add_btn("↑", "Move up", lambda: self._queue.move(self._index, "up"))
            add_btn("↓", "Move down", lambda: self._queue.move(self._index, "down"))
            add_btn("✕", "Remove", lambda: self._queue.remove(self._index))
        elif job.status in (FAILED, CANCELLED):
            add_btn("↻", "Retry", lambda: self._queue.retry(self._index))
            add_btn("✕", "Remove", lambda: self._queue.remove(self._index))
        elif job.status == DONE:
            add_btn(
                "📁",
                "Open output folder",
                lambda: self.openOutputRequested.emit(job.output_path),
            )
