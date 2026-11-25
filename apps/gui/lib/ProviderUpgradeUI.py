"""Opt-in startup flow for the GPU onnxruntime provider upgrade.

Checks the hardware off the GUI thread, asks before downloading (GPU wheels
can be ~1 GB) and runs the install in a cancellable, non-modal dialog. The
downloaded build is activated on the next launch — the bundled CPU build
keeps working in the meantime and on any failure.
"""

import os

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from apps.gui.util import log


class _CheckThread(QThread):
    result = Signal(object)

    def run(self):
        try:
            from apps.gui.utils.backends.ProviderUpgrade import (
                get_recommended_provider,
                provider_upgrade_status,
            )

            recommendation = get_recommended_provider()
            if (
                recommendation
                and provider_upgrade_status().get("package")
                == recommendation.get("package")
            ):
                recommendation = None  # already active
            self.result.emit(recommendation)
        except Exception as e:
            log(f"Provider upgrade check failed: {e}")
            self.result.emit(None)


class _InstallThread(QThread):
    progress = Signal(str)
    done = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from apps.gui.utils.backends.ProviderUpgrade import (
                install_recommended_provider,
            )

            ok = install_recommended_provider(
                progress=self.progress.emit,
                cancel_check=lambda: self._cancelled,
            )
        except Exception as e:
            log(f"Provider upgrade install failed: {e}")
            ok = False
        self.done.emit(bool(ok))


def _vendor_label(package: str) -> str:
    if "openvino" in package:
        return "Intel GPU (OpenVINO)"
    if "rocm" in package:
        return "AMD GPU (ROCm)"
    return "NVIDIA GPU (CUDA/TensorRT)"


def _finish(parent, dialog, thread, ok):
    dialog.close()
    try:
        from apps.gui.lib.QTcustom import NotificationOverlay

        if ok:
            NotificationOverlay(
                "Faster GPU acceleration installed. Restart the app to use it.",
                parent,
                timeout=5000,
            )
        else:
            NotificationOverlay(
                "GPU acceleration was not installed — continuing on the bundled build.",
                parent,
                timeout=4000,
            )
    except Exception:
        pass
    thread.wait(1000)


def _start_install(parent, recommendation):
    package = recommendation["package"]
    dialog = QProgressDialog(
        f"Downloading {_vendor_label(package)} support…",
        "Cancel",
        0,
        0,
        parent,
    )
    dialog.setWindowTitle("Installing GPU acceleration")
    dialog.setWindowModality(Qt.NonModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    thread = _InstallThread(parent)

    def on_progress(line: str) -> None:
        if line:
            dialog.setLabelText(line[-160:])

    dialog.canceled.connect(thread.cancel)
    thread.progress.connect(on_progress)
    thread.done.connect(lambda ok: _finish(parent, dialog, thread, ok))

    # keep both alive for the lifetime of the install
    parent._provider_install_dialog = dialog
    parent._provider_install_thread = thread
    thread.start()
    dialog.show()


def _on_check_result(parent, recommendation):
    if not recommendation:
        return
    package = recommendation["package"]
    details = _vendor_label(package)
    if recommendation.get("tensorrt"):
        details += " with TensorRT"
    answer = QMessageBox.question(
        parent,
        "Faster GPU acceleration available",
        (
            f"A faster ONNX Runtime build is available for your {details}.\n\n"
            "Install it now? The download runs in the background and the new "
            "build is used after restarting the app. You can keep using the "
            "bundled build until then."
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if answer == QMessageBox.StandardButton.Yes:
        _start_install(parent, recommendation)


def maybe_prompt_provider_upgrade(parent) -> None:
    """Check for a faster provider build and offer to install it."""
    if os.environ.get("RVE_NO_PROVIDER_PROMPT"):
        return
    if QApplication.platformName() == "offscreen":
        return  # test/smoke runs never touch the network
    from apps.gui.constants import HAS_NETWORK_ON_STARTUP

    if not HAS_NETWORK_ON_STARTUP:
        return

    checker = _CheckThread(parent)
    checker.result.connect(lambda recommendation: _on_check_result(parent, recommendation))
    parent._provider_check_thread = checker
    checker.start()
