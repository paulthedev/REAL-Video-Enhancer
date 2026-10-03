"""FFmpeg installer.

Downloads a platform-specific FFmpeg binary (PLATFORM / CPU_ARCH) and
installs it to FFMPEG_PATH.
"""

import os

from apps.gui.constants import (
    CPU_ARCH,
    CWD,
    FFMPEG_PATH,
    MODELS_RELEASE_BASE_URL,
    PLATFORM,
)
from apps.gui.ui.QTcustom import DownloadProgressPopup, needs_network_else_exit
from apps.gui.util import FileHandler

from .Dependency import Dependency


class FFMpeg(Dependency):
    download_path = os.path.join(CWD, "ffmpeg")
    installed_path = FFMPEG_PATH

    def get_download_link(self) -> str:
        link = MODELS_RELEASE_BASE_URL
        match PLATFORM:
            case "linux":
                link += "ffmpeg" if CPU_ARCH == "x86_64" else "ffmpeg-linux-arm64"
            case "win32":
                link += "ffmpeg.exe" if CPU_ARCH == "x86_64" else "ffmpeg-windows-arm64.exe"
            case "darwin":
                link += "ffmpeg-macos-bin" if CPU_ARCH == "x86_64" else "ffmpeg-macos-arm"
        return link

    def download(self):
        needs_network_else_exit()

        download_link = self.get_download_link()
        DownloadProgressPopup(
            link=download_link,
            downloadLocation=self.download_path,
            title="Downloading FFMpeg",
        )
        FileHandler.createDirectory(os.path.dirname(self.installed_path))
        FileHandler.moveFile(self.download_path, self.installed_path)
        FileHandler.makeExecutable(self.installed_path)
