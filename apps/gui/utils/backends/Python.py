"""PyTorch Python interpreter installer.

Downloads a standalone CPython for the detected platform (PLATFORM /
CPU_ARCH) into PYTHON_DIRECTORY and reports whether an installed version
needs updating to PYTHON_VERSION.
"""

import subprocess
import os

from PySide6.QtWidgets import QMessageBox

from apps.gui.constants import (
    CPYTHON_RELEASE_BASE_URL,
    CWD,
    CPU_ARCH,
    PLATFORM,
    PYTHON_DIRECTORY,
    PYTHON_EXECUTABLE_PATH,
    PYTHON_VERSION,
)
from apps.gui.ui.QTcustom import DownloadProgressPopup, needs_network_else_exit
from apps.gui.util import FileHandler, extractTarGZ

from .Dependency import Dependency

class Python(Dependency):
    download_path = os.path.join(CWD, "python", "python.tar.gz")
    installed_path = PYTHON_DIRECTORY
    is_update_available: bool

    def get_download_link(self) -> str:
        link = f"{CPYTHON_RELEASE_BASE_URL}cpython-{PYTHON_VERSION}+20250317-"
        match PLATFORM:
            case "linux":
                link += "x86_64-unknown-linux-gnu-install_only.tar.gz" if CPU_ARCH == "x86_64" else "aarch64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link += "x86_64-pc-windows-msvc-install_only.tar.gz"
            case "darwin":
                link += "x86_64-apple-darwin-install_only.tar.gz" if CPU_ARCH == "x86_64" else "aarch64-apple-darwin-install_only.tar.gz"
        return link

    def download(self):
        needs_network_else_exit()
        download_link = self.get_download_link()
        FileHandler.createDirectory(os.path.dirname(self.download_path))
        DownloadProgressPopup(link = download_link, downloadLocation=self.download_path, title = f"Downloading Python {PYTHON_VERSION}")
        extractTarGZ(self.download_path)

    def get_version(self):
        return subprocess.run([PYTHON_EXECUTABLE_PATH, "--version"], check=True, capture_output=True, text=True).stdout.strip().split(" ")[1]

    def get_if_update_available(self) -> bool:
        try:
            output = self.get_version()
        except subprocess.CalledProcessError: # if python is not found
            self.download()
            return False
        is_update = not output == PYTHON_VERSION
        if is_update:
            reply = QMessageBox.question(
                None,
                "Update Python?",
                "The installed version of Python is older than the current version of RVE. Update?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,  # type: ignore
            )
            if reply == QMessageBox.Yes:  # type: ignore
                self.is_update_available = True
                print("Updating Python")
            else:
                is_update = False
        self.is_update_available = is_update
        return self.is_update_available

    def update_if_updates_available(self):
        if self.is_update_available:
            FileHandler.removeFolder(PYTHON_DIRECTORY)
            self.download()
