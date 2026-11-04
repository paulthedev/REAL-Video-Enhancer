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
