"""Download FFmpeg from official upstream builds when no binary is available.

Stdlib only — the backend must not depend on the GUI package. See
apps/backend/constants.py for the download sources.
"""

from ..constants import CPU_ARCH, FFMPEG_BTBN_RELEASE_URL, FFMPEG_EVERMEET_RELEASE_URL, PLATFORM
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


def get_ffmpeg_download_link() -> str:
    if PLATFORM == "darwin":
        # zip with a bare ffmpeg binary; x86_64 build (arm64 runs via Rosetta)
        return FFMPEG_EVERMEET_RELEASE_URL
    if PLATFORM == "win32":
        arch = "win64" if CPU_ARCH == "x86_64" else "winarm64"
        return f"{FFMPEG_BTBN_RELEASE_URL}/ffmpeg-master-latest-{arch}-gpl.zip"
    arch = "linux64" if CPU_ARCH == "x86_64" else "linuxarm64"
    return f"{FFMPEG_BTBN_RELEASE_URL}/ffmpeg-master-latest-{arch}-gpl.tar.xz"


def _extract_ffmpeg_binary(archive_path: str, binary_name: str, out_dir: str) -> str:
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


def download_ffmpeg(cwd: str = os.getcwd()) -> str | None:
    binary_name = "ffmpeg.exe" if PLATFORM == "win32" else "ffmpeg"
    download_path = os.path.join(cwd, binary_name)
    link = get_ffmpeg_download_link()

    archive_fd, archive_path = tempfile.mkstemp(prefix="ffmpeg_dl_")
    os.close(archive_fd)
    try:
        print("Downloading FFMpeg from " + link)
        with urllib.request.urlopen(link, timeout=30) as response:
            with open(archive_path, "wb") as file:
                while True:
                    data = response.read(1024 * 1024)
                    if not data:
                        break
                    file.write(data)

        extracted = _extract_ffmpeg_binary(archive_path, binary_name, os.path.dirname(archive_path))
        shutil.move(extracted, download_path)
        os.chmod(download_path, 0o755)
        print("Download completed: " + download_path)
        return download_path
    except Exception as e:
        print(f"Error downloading FFMpeg: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)
