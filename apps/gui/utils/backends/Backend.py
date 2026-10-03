"""Shared install/download logic for the REAL-Video-Enhancer backend bundle.

Handles fetching, extracting and version-checking the packaged backend
(rve-backend) tarball into BACKEND_PATH.
"""

import os
import subprocess

from apps.gui.constants import (
    BACKEND_PATH,
    BACKEND_RELEASE_URL_TEMPLATE,
    CWD,
    PYTHON_EXECUTABLE_PATH,
    USE_LOCAL_BACKEND,
)
from apps.gui.ui.QTcustom import DownloadProgressPopup, needs_network_else_exit
from apps.gui.util import (
    FileHandler,
    extractTarGZ,
    log,
    subprocess_popen_without_terminal,
)
from apps.gui.version import backend_dev_version, version

from .Dependency import Dependency


class Backend(Dependency):
    updatable: bool = True
    is_update_available: bool = False
    download_path = os.path.join(CWD, "backend.tar.gz")
    installed_path = BACKEND_PATH

    def get_download_link(self) -> str:
        return BACKEND_RELEASE_URL_TEMPLATE.format(version=version)

    def download(self):
        if USE_LOCAL_BACKEND:
            return
        needs_network_else_exit()
        download_link = self.get_download_link()
        DownloadProgressPopup(
            link=download_link,
            downloadLocation=self.download_path,
            title="Downloading Backend",
        )
        extractTarGZ(self.download_path)

    def get_if_update_available(self) -> bool:
        try:
            process = subprocess_popen_without_terminal(
                [PYTHON_EXECUTABLE_PATH, os.path.join(BACKEND_PATH, "rve-backend.py"), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
                )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
            output = stdout.strip() # this extracts the version number from the output
            log(f"\nBackend Version: {output}\n")
            update_available = not output == backend_dev_version
            self.is_update_available = update_available
            return update_available
        except subprocess.CalledProcessError as e: # if the backend is not found
            log("Backend not found, downloading..." + str(e))
            self.download()
            self.is_update_available = False
            return False

    def update_if_updates_available(self) -> None:
        if self.is_update_available:
            needs_network_else_exit()
            FileHandler.removeFolder(BACKEND_PATH) # remove the old backend directory
            self.download()
