import os
from PySide6.QtWidgets import QMainWindow, QMessageBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from apps.gui.ui.QTcustom import RegularQTPopup, NetworkCheckPopup, remove_combobox_item_by_text
from apps.gui.utils.backends import (
    DownloadDependencies,
    TorchVersion,
    detect_gpu_hardware,
    get_preferred_backend,
    get_torch_backend_family,
)
from apps.gui.lib.Updater import ApplicationUpdater
from apps.gui.constants import IS_FLATPAK, PLATFORM, CWD, USE_LOCAL_BACKEND, HOME_PATH, PLATFORM, IS_FLATPAK, CWD, CPU_ARCH
from apps.gui.lib.GPUDetect import GPUDetect
from apps.gui.util import FileHandler, log


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        backends: list,
        skip_info_popup: bool = False,
    ):
        self.parent = parent
        self.torch_versions:list[TorchVersion] = [version for version in TorchVersion.__subclasses__()]
        self.downloadDeps = DownloadDependencies()
        self.backends = backends
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

        # Hide old separate download buttons
        self.parent.downloadNCNNBtn.setVisible(False)
        self.parent.downloadTorchBtn.setVisible(False)
        self.parent.downloadDirectMLBtn.setVisible(False)
        self.parent.uninstallNCNNBtn.setVisible(False)
        self.parent.uninstallTorchBtn.setVisible(False)
        self.parent.uninstallDirectMLBtn.setVisible(False)
        self.parent.selectPytorchCustomModel.setVisible(False)
        self.parent.selectNCNNCustomModel.setVisible(False)

        # Setup unified backend selector
        self._setup_backend_selector()

        # Platform-specific backend filtering
        self._filter_backends_for_platform()

        if PLATFORM == "darwin":
            if CPU_ARCH == "arm64":
                self.parent.pytorch_backend.clear()
                self.parent.pytorch_backend.addItems(["MPS (Apple Silicon)"])
                self.parent.pytorch_backend.setCurrentText("MPS (Apple Silicon)")
                self.parent.pytorch_backend.setEnabled(False)

        if IS_FLATPAK or USE_LOCAL_BACKEND:
            self.parent.uninstallAppBtn.setDisabled(True)
        else:
            self.parent.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self.parent.ApplicationUpdateContainer.setVisible(False)
        self.QButtonConnect()

    def _setup_backend_selector(self):
        """Setup unified backend selection dropdowns."""
        # Backend type selector (ONNX, PyTorch, NCNN)
        self.parent.backend_type_combo.clear()
        self.parent.backend_type_combo.addItems(["ONNX", "PyTorch", "NCNN"])

        # Backend variant selector (TensorRT, CUDA, ROCm, etc. - only for PyTorch)
        self.parent.backend_variant_combo.clear()
        self.parent.backend_variant_combo.addItems(["CUDA", "ROCm", "XPU", "MPS (Apple Silicon)"])
        self.parent.backend_variant_combo.setEnabled(False)  # Disabled until PyTorch is selected

        # Connect backend type change to update variant options
        self.parent.backend_type_combo.currentIndexChanged.connect(
            self._on_backend_type_changed
        )

        # Load Backend button
        self.parent.loadBackendBtn.clicked.connect(self.load_backend)

    def _on_backend_type_changed(self, index):
        """Update variant dropdown based on selected backend type."""
        backend_type = self.parent.backend_type_combo.currentText()

        if backend_type == "PyTorch":
            self.parent.backend_variant_combo.setEnabled(True)
            # Filter variants based on platform
            self._filter_variant_for_platform()
        else:
            self.parent.backend_variant_combo.setEnabled(False)
            self.parent.backend_variant_combo.clear()
            if backend_type == "ONNX":
                self.parent.backend_variant_combo.addItems(["Auto (Recommended)"])
            elif backend_type == "NCNN":
                self.parent.backend_variant_combo.addItems(["Vulkan (Recommended)"])

    def _filter_variant_for_platform(self):
        """Filter PyTorch variants based on current platform."""
        self.parent.backend_variant_combo.clear()
        variants = []

        if PLATFORM == "linux":
            variants = ["CUDA", "ROCm", "XPU"]
        elif PLATFORM == "win32":
            variants = ["CUDA"]
        elif PLATFORM == "darwin":
            if CPU_ARCH == "arm64":
                variants = ["MPS (Apple Silicon)"]
            else:
                variants = ["CPU"]

        self.parent.backend_variant_combo.addItems(variants)
        if variants:
            self.parent.backend_variant_combo.setCurrentIndex(0)

    def _filter_backends_for_platform(self):
        """Remove incompatible backend variants based on platform."""
        if PLATFORM != "linux":
            remove_combobox_item_by_text(self.parent.pytorch_backend, "ROCm")
        else:
            if CPU_ARCH == "arm64":
                remove_combobox_item_by_text(self.parent.pytorch_backend, "XPU")
                remove_combobox_item_by_text(self.parent.pytorch_backend, "ROCm")
        if IS_FLATPAK:
            remove_combobox_item_by_text(self.parent.pytorch_backend, "XPU")
            remove_combobox_item_by_text(self.parent.pytorch_backend, "ROCm")

    def QButtonConnect(self):
        self.parent.downloadRecommendedBtn.clicked.connect(
            self.installRecommended
        )
        self.parent.pytorch_version.addItems(
            [version.torch_version for version in self.torch_versions]
        )

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

    def load_backend(self):
        """Load the selected backend."""
        backend_type = self.parent.backend_type_combo.currentText()
        backend_variant = self.parent.backend_variant_combo.currentText() if self.parent.backend_variant_combo.isEnabled() else None

        log(f"Loading backend: {backend_type} with variant: {backend_variant}")

        if backend_type == "PyTorch":
            self.download("torch", install=True, pytorch_backend=backend_variant)
        elif backend_type == "ONNX":
            self.download("onnx", install=True)
        elif backend_type == "NCNN":
            self.download("ncnn", install=True)

    def installRecommended(self):
        """Install the recommended backend based on hardware."""
        hardware = detect_gpu_hardware()
        preferred = get_preferred_backend(hardware)

        if preferred == "pytorch":
            family = get_torch_backend_family(hardware)
            self.download("torch", install=True, pytorch_backend=family)
        elif preferred == "onnx":
            self.download("onnx", install=True)
        else:
            self.download("ncnn", install=True)

    def download(self, dep, install: bool = True, pytorch_backend: str = None):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        pytorch_ver: TorchVersion | None = None
        current_pytorch_version = self.parent.pytorch_version.currentText().split()[0]
        current_pytorch_backend = pytorch_backend

        for version in self.torch_versions:
            if version.torch_version == current_pytorch_version:
                pytorch_ver = version
                torchvision_ver = version.torchvision_version

        if not pytorch_ver:
            RegularQTPopup(
                "Please select a valid PyTorch version from the dropdown."
            )
            return

        if current_pytorch_backend:
            if current_pytorch_backend == "CUDA":
                pytorch_backend = pytorch_ver.cuda_version
            elif current_pytorch_backend == "ROCm":
                pytorch_backend = pytorch_ver.rocm_version
            elif current_pytorch_backend == "XPU":
                pytorch_backend = pytorch_ver.xpu_version
            elif current_pytorch_backend == "MPS (Apple Silicon)":
                pytorch_backend = pytorch_ver.mps_version

        if NetworkCheckPopup("https://pypi.org/"):
            return_code = self.downloadDeps.downloadPythonDeps(
                dep,
                pytorch_ver.torch_version,
                torchvision_ver,
                pytorch_backend.lower() if pytorch_backend else "cu126",
                install,
                pytorch_ver.is_nightly
            )
            if return_code == 0 and not self.skip_info_popup:
                RegularQTPopup(
                    "Download Complete\nPlease restart the application to apply changes."
                )
            elif return_code != 0:
                RegularQTPopup("Download Failed!\nPlease check logs for more info.")
