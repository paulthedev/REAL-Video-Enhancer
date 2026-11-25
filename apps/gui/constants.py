import os
import sys
import urllib.request
import platform
from PySide6.QtCore import QDir
def networkCheck(hostname="https://pypi.org") -> bool:
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
GITHUB_REPO = "paulthedev/REAL-Video-Enhancer"
MODEL_HOSTED_REPO = "paulthedev/real-video-enhancer-models"
APP_URL = f"https://github.com/{GITHUB_REPO}"
MODELS_RELEASE_BASE_URL = (
    f"https://github.com/{MODEL_HOSTED_REPO}/releases/download/models/"
)
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"

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
            HOME_PATH, ".var", "app", "io.github.paulthedv.REAL-Video-Enhancer"
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
if getattr(sys, "frozen", False):
    # the backend ships inside the bundle next to the executable, not in the
    # per-user data dir. cx_Freeze uses apps/backend; PyInstaller uses backend/.
    _exe_root = os.path.dirname(os.path.abspath(sys.executable))
    for _candidate in (
        os.path.join(_exe_root, "apps", "backend"),
        os.path.join(_exe_root, "backend"),
    ):
        if os.path.isdir(_candidate):
            BACKEND_PATH = _candidate
            break
TEMP_DOWNLOAD_PATH = os.path.join(CWD, "temp")
# ffmpeg ships via the imageio-ffmpeg pip package (static per-platform
# build); RVE_FFMPEG_PATH / --ffmpeg_path override it
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = (
        os.path.join(CWD, "bin", "ffmpeg.exe")
        if PLATFORM == "win32"
        else os.path.join(CWD, "bin", "ffmpeg")
    )

PYTHON_EXECUTABLE_PATH = sys.executable  # the backend runs on our own interpreter
if getattr(sys, "frozen", False):
    # the frozen GUI binary doubles as the backend launcher: children spawned
    # from here re-enter the binary and the entry-point dispatch runs the
    # bundled backend instead of a second GUI instance
    os.environ["RVE_RUN_BACKEND"] = "1"
PYTHON_DIRECTORY = os.path.dirname(sys.executable)

EXE_PATH = os.path.join(
    CWD,
    EXE_NAME,
)
LIBS_PATH = os.path.join(
    CWD,
    LIBS_NAME,
)

# is installed
IS_INSTALLED = os.path.isfile(FFMPEG_PATH)

IMAGE_SHARED_MEMORY_ID = "/image_preview" + str(os.getpid())
PAUSED_STATE_SHARED_MEMORY_ID = "/paused_state" + str(os.getpid())
INPUT_TEXT_FILE = os.path.join(CWD, f"INPUT{os.getpid()}.txt")
if (
    "--swap-flatpak-checks" in sys.argv
):  # swap check down here as to not interfere with directories
    IS_FLATPAK = not IS_FLATPAK
