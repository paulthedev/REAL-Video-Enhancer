import os
import sys
import urllib.request
import platform
from PySide6.QtCore import QDir
def networkCheck(hostname="https://raw.githubusercontent.com") -> bool:
    """
    checks network availability against a url, default url: raw.githubusercontent.com
    """
    try:
        req = urllib.request.Request(
            hostname, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=1):
            pass
        return True
    except Exception:
        return False
HAS_NETWORK_ON_STARTUP = networkCheck()

# GitHub locations for RVE releases (owner/repo is case-insensitive in the API)
GITHUB_REPO = "tntwise/real-video-enhancer"
MODEL_HOSTED_REPO = "TNTwise/real-video-enhancer-models"
APP_URL = f"https://github.com/{GITHUB_REPO}"
MODELS_RELEASE_BASE_URL = (
    f"https://github.com/{MODEL_HOSTED_REPO}/releases/download/models/"
)
RELEASES_LATEST_API_URL = (
    f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
)
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
RELEASE_DOWNLOAD_URL_TEMPLATE = (
    "https://github.com/" + GITHUB_REPO + "/releases/download/{tag}/{file_name}"
)
BACKEND_RELEASE_URL_TEMPLATE = (
    "https://github.com/" + GITHUB_REPO + "/releases/download/RVE-{version}/backend-v{version}.tar.gz"
)

# Official upstream sources (do not re-host these — see docs/BUG_TRACKER.md,
# "Sources — runtime download provenance").
FFMPEG_BTBN_RELEASE_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download"
)
FFMPEG_EVERMEET_RELEASE_URL = "https://evermeet.cx/ffmpeg/getrelease/zip"
PYTHON_BUILD_STANDALONE_TAG = "20250317"
PYTHON_BUILD_STANDALONE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PYTHON_BUILD_STANDALONE_TAG}"
)

LOCKFILE = QDir.tempPath() + "/REAL-Video-Enhancer.lock"



PLATFORM = sys.platform  # win32, darwin, linux
IS_STEAM = "SteamAppId" in os.environ
IS_FLATPAK = "FLATPAK_ID" in os.environ and not IS_STEAM
IS_APPIMAGE = "APPIMAGE" in os.environ

HOME_PATH = os.path.expanduser("~")

IS_COMPILED_OR_FROZEN = hasattr(sys, "frozen")
USE_LOCAL_BACKEND =  os.path.exists(
    os.path.join(os.getcwd(), "backend")
) 

if not USE_LOCAL_BACKEND:
    
    if PLATFORM == "win32":
        CWD = os.path.join(HOME_PATH, "AppData", "Local", "REAL-Video-Enhancer")
    if PLATFORM == "darwin":
        CWD = os.path.join(HOME_PATH, "Library", "REAL-Video-Enhancer")
    if PLATFORM == "linux":
        CWD = os.path.join(
            HOME_PATH, ".local", "share", "REAL-Video-Enhancer"
        )
else:
    CWD = os.getcwd()
    
if IS_FLATPAK:
    CWD = (
        os.path.join(
            HOME_PATH, ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
        )
    )
elif IS_APPIMAGE:
    # The AppImage squashfs is read-only, so point CWD at a writable area.
    CWD = os.path.join(HOME_PATH, ".local", "share", "REAL-Video-Enhancer")

CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()


EXE_NAME = "REAL-Video-Enhancer.exe" if PLATFORM == "win32" else "REAL-Video-Enhancer"
LIBS_NAME = "_internal" if PLATFORM == "win32" else "lib"
# dirs
MODELS_PATH = os.path.join(CWD, "models")
CUSTOM_MODELS_PATH = os.path.join(CWD, "custom_models")
VIDEOS_PATH = (
    os.path.join(HOME_PATH, "Desktop")
    if PLATFORM == "darwin"
    else os.path.join(HOME_PATH, "Videos")
)
BACKEND_PATH = "/app/bin/apps/backend" if IS_FLATPAK else os.path.join(CWD, "apps", "backend")
TEMP_DOWNLOAD_PATH = os.path.join(CWD, "temp")
# exes
FFMPEG_PATH = (
    os.path.join(CWD, "bin", "ffmpeg.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "bin", "ffmpeg")
)
PYTHON_DIRECTORY = os.path.join(CWD, "python")

PYTHON_EXECUTABLE_PATH = (
    os.path.join(CWD, "python", "python", "python.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "python", "python", "bin", "python3")
)
#PYTHON_VERSION = "3.13.2" if PLATFORM != "darwin" else "3.12.9" # sets python version of backend
PYTHON_VERSION = "3.12.9"

EXE_PATH = os.path.join(
    CWD,
    EXE_NAME,
)
LIBS_PATH = os.path.join(
    CWD,
    LIBS_NAME,
)

# is installed
IS_INSTALLED = os.path.isfile(FFMPEG_PATH) and os.path.isfile(PYTHON_EXECUTABLE_PATH)

IMAGE_SHARED_MEMORY_ID = "/image_preview" + str(os.getpid())
PAUSED_STATE_SHARED_MEMORY_ID = "/paused_state" + str(os.getpid())
INPUT_TEXT_FILE = os.path.join(CWD, f"INPUT{os.getpid()}.txt")
if (
    "--swap-flatpak-checks" in sys.argv
):  # swap check down here as to not interfere with directories
    IS_FLATPAK = not IS_FLATPAK
