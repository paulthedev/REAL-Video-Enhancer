"""Microsoft Visual C++ Redistributable installer (Windows only).

On Windows it launches the elevated VC_redist.x64.exe installer. It
subclasses Dependency with updatable=False so download() skips the network
guard on other platforms.
"""

import ctypes
import os

from apps.gui.constants import CWD, PLATFORM
from apps.gui.ui.QTcustom import (
    DownloadProgressPopup,
    RegularQTPopup,
    needs_network_else_exit,
)

from .Dependency import Dependency


class VCRedList(Dependency):
    updatable = False
    download_path = os.path.join(CWD, "bin", "VC_redist.x64.exe")
    installed_path = download_path

    def get_download_link(self) -> str:
        return "https://aka.ms/vs/17/release/vc_redist.x64.exe"

    def download(self):
        if PLATFORM == "win32":
            needs_network_else_exit()

            download_link = self.get_download_link()
            DownloadProgressPopup(
                link=download_link,
                downloadLocation=self.download_path,
                title="Downloading VCRedist",
            )

            # Use ShellExecute to properly handle admin elevation
            try:
                result = ctypes.windll.shell32.ShellExecuteW(
                    None,                          # hwnd
                    "runas",                       # operation (runas = run as admin)
                    self.download_path,            # file
                    "/install /norestart /quiet",  # parameters
                    None,                          # directory
                    1                              # show command (1 = normal window)
                )
                if result <= 32:  # Error codes are <= 32
                    RegularQTPopup(
                        "Failed to launch VCRedist installer. Please run it manually."
                    )
            except Exception as e:
                RegularQTPopup(
                    f"Error launching VCRedist installer: {str(e)}\nThe installer will now close."
                )
