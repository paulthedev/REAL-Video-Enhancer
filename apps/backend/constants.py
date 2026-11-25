"""Backend-side constants: platform info.

Kept separate from the GUI package so the backend stays importable on its
own (apps/backend/utils imports these via ``..constants``). Runtime
binaries (ffmpeg, onnxruntime) ship with the app; there are no download
URLs left here.
"""

import platform
import sys

# an upgraded onnxruntime (downloaded by the GUI into the user data dir)
# must shadow the bundled CPU build — prepend before anything imports it
try:
    from apps.gui.utils.backends.ProviderUpgrade import activate_providers_dir

    activate_providers_dir()
except Exception:
    pass  # GUI package not available (standalone backend runs) — bundled build

PLATFORM = sys.platform
CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()
