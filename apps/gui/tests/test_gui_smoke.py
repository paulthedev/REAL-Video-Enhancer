"""
Offscreen GUI smoke test.

Boots the real MainWindow under QT_QPA_PLATFORM=offscreen with the backend
health check stubbed (no subprocess, no network) and asserts the window
constructs, the backend combo reflects the onnx-only state, and the UI
actually painted something.

Requires the generated UI modules in repo-root dist/ (built on demand via
`python build.py`).
"""

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

pytestmark = pytest.mark.gui

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir
    )
)
ENTRY = os.path.join(REPO_ROOT, "apps", "gui", "REAL-Video-Enhancer.py")


def _ensure_ui_modules():
    if os.path.exists(os.path.join(REPO_ROOT, "dist", "mainwindow.py")):
        return
    subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "build.py")],
        check=True,
        cwd=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def gui_entry(tmp_path_factory):
    """The GUI entry point module, loaded like the runtime loads it.

    dist/ must be on sys.path before exec so `from mainwindow import ...`
    resolves; the entry point inserts it itself, but importing it twice
    under different names would double-import Qt pages, so a unique module
    name is used and stubs are applied before the window is constructed.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _ensure_ui_modules()

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    module_name = "rve_gui_entry_smoke_test"
    if module_name in sys.modules:
        yield sys.modules[module_name], app
        return

    # stub the backend health check: no backend subprocess at startup
    from apps.gui.lib import BackendHandler as backend_handler_module

    backend_handler_module.BackendHandler.getAvailableBackends = lambda self: (
        ["onnx"],
        "RVE Backend Version: smoke-test\n"
        "ONNX Runtime Version: smoke-test\n"
        "Available Backends: ['onnx']",
    )

    # stub the download worker: the updater popup must not hit the network
    from apps.gui.lib import QTcustom

    def _fake_run(self):
        self.progress.emit(100)
        self.finished.emit()

    QTcustom.DownloadAndReportToQTThread.run = _fake_run

    loader = importlib.machinery.SourceFileLoader(module_name, ENTRY)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    yield module, app


def test_ui_polish_hides_dead_controls(gui_entry):
    module, app = gui_entry
    module.QMessageBox.question = staticmethod(
        lambda *a, **k: module.QMessageBox.StandardButton.Yes
    )
    window = module.MainWindow()
    try:
        window.show()
        deadline = time.time() + 1.0
        while time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)

        # dead controls from the retired backend/UHD/torch paths
        assert not window.backendComboBox.isVisible()
        assert not window.uhd_mode.isVisible()
        assert not window.pauseRenderButton.isVisible()
        assert not window.killRenderButton.isVisible()

        # empty-state affordances
        assert window.inputFileText.placeholderText()
        assert window.outputFileText.placeholderText()
        assert window.renderOutput.placeholderText()

        # preview with no video must not raise or touch the queue
        window.renderPreview()
    finally:
        window.close()
        app.processEvents()


def test_queue_rows_track_width_and_status(gui_entry):
    module, app = gui_entry
    from apps.gui.lib.RenderQueue import DONE, RenderOptions

    module.QMessageBox.question = staticmethod(
        lambda *a, **k: module.QMessageBox.StandardButton.Yes
    )
    window = module.MainWindow()
    try:
        window.show()
        deadline = time.time() + 0.8
        while time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)

        assert window.settingsTab.gpu_device_index("Auto") == 0
        assert window.settingsTab.gpu_device_index("not-a-real-gpu") == 0

        window.stackedWidget.setCurrentIndex(1)
        for _ in range(3):
            app.processEvents()

        window.renderQueue.add(
            RenderOptions(
                inputFile="/tmp/a_very_long_input_name_that_should_elide.mp4",
                outputPath="/tmp/out.mkv",
                videoWidth=1920,
                videoHeight=1080,
                videoFps=24,
                tilingEnabled=False,
                tilesize="0",
                videoFrameCount=480,
                backend="onnx",
                modelScale=4,
                interpolateTimes=2,
                benchmarkMode=False,
                sloMoMode=False,
                dyanmicScaleOpticalFlow=False,
                ensemble=False,
            )
        )
        app.processEvents()

        panel = window.processTab.queue_panel
        item = panel.list.item(0)
        row = panel.list.itemWidget(item)
        assert item.sizeHint().width() == panel.list.viewport().width()

        window.renderQueue.update_job(
            window.renderQueue.jobs[0], status=DONE, return_code=0
        )
        app.processEvents()
        assert row.status_chip.text() == "Done"
        assert "done" in panel.counts_label.text().lower()
        assert panel.counts_label.isVisible()

        # rows must reflow when the rail changes width
        window.resize(1180, 780)
        for _ in range(3):
            app.processEvents()
        assert item.sizeHint().width() == panel.list.viewport().width()
    finally:
        window.close()
        app.processEvents()


def test_main_window_boots_offscreen(gui_entry):
    module, app = gui_entry

    # closeEvent shows a modal "sure you want to exit?" — answer it so the
    # close path completes without a user
    module.QMessageBox.question = staticmethod(
        lambda *a, **k: module.QMessageBox.StandardButton.Yes
    )

    window = module.MainWindow()
    try:
        window.show()
        # let the startup fade-in and deferred layout settle before grabbing
        deadline = time.time() + 1.5
        while time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)

        # onnx-only: the backend combo carries exactly the one backend
        items = [
            window.backendComboBox.itemText(i)
            for i in range(window.backendComboBox.count())
        ]
        assert items == ["onnx"]

        pixmap = window.grab()
        assert not pixmap.isNull()
        assert pixmap.width() >= 900 and pixmap.height() >= 600

        # the window painted more than a solid fill
        image = pixmap.toImage()
        colors = {
            image.pixel(x, y)
            for x in range(0, image.width(), 37)
            for y in range(0, image.height(), 37)
        }
        assert len(colors) > 4, "window rendered a solid blank frame"
    finally:
        window.close()
        app.processEvents()
