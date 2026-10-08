import os
from PySide6.QtWidgets import QMainWindow, QMessageBox
from apps.gui.constants import IS_FLATPAK, PLATFORM, CWD, USE_LOCAL_BACKEND
from apps.gui.lib.Updater import ApplicationUpdater
from apps.gui.util import FileHandler, log


class DownloadTab:
    """Backend management page.

    ONNX Runtime ships bundled with the app, so this page no longer installs
    dependencies — it only offers the app uninstaller and shows application
    update state. Models are downloaded on demand from the process page.
    """

    def __init__(
        self,
        parent: QMainWindow,
        backends: list = None,
        skip_info_popup: bool = False,
    ):
        self.parent = parent
        self.backends = backends or []
        self.skip_info_popup = skip_info_popup
        self.applicationUpdater = ApplicationUpdater()

        self.has_enough_space = True
        if IS_FLATPAK:
            minimum_space_required = 7
        else:
            minimum_space_required = 15
        self.parent.low_storage_label.setVisible(False)
        if FileHandler.getFreeSpace() < minimum_space_required:
            self.parent.low_storage_label.setVisible(True)
            self.has_enough_space = False

        # The dependency-install UI is obsolete: ONNX Runtime and ffmpeg are
        # bundled. Hide every legacy installer widget.
        for name in (
            "downloadNCNNBtn", "downloadTorchBtn", "downloadDirectMLBtn",
            "downloadTensorRTBtn", "downloadRecommendedBtn",
            "uninstallNCNNBtn", "uninstallTorchBtn",
            "uninstallDirectMLBtn", "uninstallTensorRTBtn",
            "selectPytorchCustomModel", "selectNCNNCustomModel",
            "pytorchBackendInstallerContainer", "pytorchBackendInstallerContainer_2",
            "pytorchBackendInstallerContainer_4", "directMLBackendInstallerContainer",
            "backend_type_combo", "backend_variant_combo", "loadBackendBtn",
            "pytorch_version", "pytorch_backend",
        ):
            widget = getattr(self.parent, name, None)
            if widget is not None:
                widget.setVisible(False)

        if IS_FLATPAK or USE_LOCAL_BACKEND:
            self.parent.uninstallAppBtn.setDisabled(True)
        else:
            self.parent.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self.parent.ApplicationUpdateContainer.setVisible(False)

    def uninstallApp(self):
        reply = QMessageBox.question(
            self.parent,
            "",
            "Are you sure you want to uninstall?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply == QMessageBox.Yes:  # type: ignore
            os.chdir(HOME_PATH)
            FileHandler().removeFolder(CWD)
            os._exit(0)

