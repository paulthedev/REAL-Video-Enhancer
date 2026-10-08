"""Backend-side constants: platform info and upstream download sources.

Kept separate from the GUI package so the backend stays importable on its
own (apps/backend/utils imports these via ``..constants``). Download URLs
live here so download provenance is auditable from one file — see
docs/BUG_TRACKER.md, "Sources — runtime download provenance".
"""

import platform
import sys

PLATFORM = sys.platform
CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()

# Official FFmpeg builds (S1). BtbN publishes rolling "latest" aliases for
# static GPL builds (win/linux, x86_64/arm64); evermeet.cx serves macOS
# x86_64 (arm64 runs via Rosetta).
FFMPEG_BTBN_RELEASE_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download"
)
FFMPEG_EVERMEET_RELEASE_URL = "https://evermeet.cx/ffmpeg/getrelease/zip"
