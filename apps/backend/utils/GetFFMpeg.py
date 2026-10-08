"""Resolve the ffmpeg binary.

The GUI bundles ffmpeg via the imageio-ffmpeg pip package and passes its
path on the command line. This module is the backend-side fallback when no
path is given: it asks imageio-ffmpeg for its bundled build.
"""

import os
import sys


def get_ffmpeg_download_link() -> str:
    """Kept for CLI help/back-compat — there is no download anymore."""
    return "bundled (imageio-ffmpeg)"


def download_ffmpeg(cwd: str = os.getcwd()) -> str | None:
    """Return the path of a usable ffmpeg binary, or None.

    Historically this downloaded ffmpeg from the network; ffmpeg now ships
    with the app via imageio-ffmpeg.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"imageio-ffmpeg not available: {e}", file=sys.stderr)
        return None
