"""FFmpeg installer.

Downloads FFmpeg from official upstream builds (BtbN for Windows/Linux,
evermeet.cx for macOS — see docs/BUG_TRACKER.md, "Sources") and installs
the binary to FFMPEG_PATH. Upstream ships archives, not bare binaries, so
the download is extracted before install.
"""

import os
import tarfile
import tempfile
import zipfile

from apps.gui.constants import (
    CPU_ARCH,
    CWD,
    FFMPEG_BTBN_RELEASE_URL,
    FFMPEG_EVERMEET_RELEASE_URL,
    FFMPEG_PATH,
    PLATFORM,
)
from apps.gui.lib.QTcustom import DownloadProgressPopup, needs_network_else_exit
from apps.gui.util import FileHandler

from .Dependency import Dependency


class FFMpeg(Dependency):
    download_path = os.path.join(CWD, "ffmpeg")
    installed_path = FFMPEG_PATH

    def get_download_link(self) -> str:
        if PLATFORM == "darwin":
            # zip with a bare ffmpeg binary; x86_64 build (arm64 runs via Rosetta)
            return FFMPEG_EVERMEET_RELEASE_URL
        if PLATFORM == "win32":
            arch = "win64" if CPU_ARCH == "x86_64" else "winarm64"
            return f"{FFMPEG_BTBN_RELEASE_URL}/ffmpeg-master-latest-{arch}-gpl.zip"
        arch = "linux64" if CPU_ARCH == "x86_64" else "linuxarm64"
        return f"{FFMPEG_BTBN_RELEASE_URL}/ffmpeg-master-latest-{arch}-gpl.tar.xz"

    @staticmethod
    def extract_ffmpeg_binary(archive_path: str) -> str:
        """Extract the ffmpeg binary from the downloaded archive to a temp dir."""
        binary_name = "ffmpeg.exe" if PLATFORM == "win32" else "ffmpeg"
        out_dir = tempfile.mkdtemp(prefix="ffmpeg_extract_")
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tar:
                for member in tar.getmembers():
                    # BtbN archives nest the binary under <name>/bin/
                    if member.isfile() and member.name.rsplit("/", 1)[-1] == binary_name:
                        tar.extract(member, out_dir, filter="data")
                        return os.path.join(out_dir, member.name)
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                for name in zf.namelist():
                    if name.rsplit("/", 1)[-1] == binary_name:
                        zf.extract(name, out_dir)
                        return os.path.join(out_dir, name)
        raise FileNotFoundError(f"No {binary_name} found in {archive_path}")

    def download(self):
        needs_network_else_exit()

        download_link = self.get_download_link()
        DownloadProgressPopup(
            link=download_link,
            downloadLocation=self.download_path,
            title="Downloading FFMpeg",
        )
        extracted = self.extract_ffmpeg_binary(self.download_path)
        FileHandler.createDirectory(os.path.dirname(self.installed_path))
        FileHandler.moveFile(extracted, self.installed_path)
        FileHandler.makeExecutable(self.installed_path)
        os.remove(self.download_path)
