"""
Device tuning and persistent model-tuning cache for PyTorch backends.

Works across all GPU device families the app supports:
  - ``cuda``   -> Nvidia CUDA, AMD ROCm (HIP), Intel XPU via SYCL/CUDA-compat builds
  - ``xpu``    -> Intel oneAPI
  - ``mps`` / ``cpu`` -> tuning is a no-op

What gets cached where:
  - Inductor/Triton compiled kernels and cuDNN/MIOpen benchmark results are
    keyed by (device type, device name, torch/triton versions) and stored under
    ``<appdata>/cache/tuning/``, so the one-time autotune / compile cost of a few
    minutes is paid once per machine+driver instead of every launch.
  - The MIOpen database file itself lives at ``$MIOPEN_USER_DB_PATH`` (default:
    ``~/.miopen/cache/db/<device>.fdb.txt``). Setting the env var to our cache
    dir keeps it inside the app data folder and lets us key it by device name.

The environment variables are set here on import so they apply before any conv
or compiled kernel runs, including in child processes (the backend is a
separate python process spawned by the frontend).
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache directory resolution: <appdata>/cache/tuning/
# ---------------------------------------------------------------------------


def _app_data_dir() -> Path:
    """Mirror of src/constants.py CWD logic, but without PySide6 imports so it
    works in the standalone backend process and benchmark scripts."""
    home = os.path.expanduser("~")
    platform_name = sys.platform

    if "FLATPAK_ID" in os.environ:
        return Path(home) / ".var/app/io.github.tntwise.REAL-Video-Enhancer/cache/tuning"

    if "APPIMAGE" in os.environ and not ("SteamAppId" in os.environ):
        # AppImage squashfs is read-only; .local/share/REAL-Video-Enhancer exists.
        return Path(home) / ".local/share/REAL-Video-Enhancer/cache/tuning"

    if platform_name == "win32":
        base = Path(os.path.join(Path.home(), "AppData", "Local"))
        return base / "REAL-Video-Enhancer/cache/tuning"
    elif platform_name == "darwin":
        return Path(home) / "Library/REAL-Video-Enhancer/cache/tuning"
    else:
        # Linux and other unix-like systems.
        return Path(home) / ".local/share/REAL-Video-Enhancer/cache/tuning"


def tuning_cache_dir() -> Path:
    """Persistent cache root for compiled kernels and MIOpen results."""
    path = _app_data_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only FS (e.g. running as non-root without a home dir): fall back
        # to the system temp cache so tuning still works, just not persistently.
        import tempfile

        fallback = Path(tempfile.gettempdir()) / "REAL-Video-Enhancer-tuning-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return path


_DEVICE_NAME_SUFFIX: str | None = None


def _device_name_suffix(device_type: str) -> str:
    """Shorten the GPU name so it is safe as a filename component."""
    global _DEVICE_NAME_SUFFIX
    if _DEVICE_NAME_SUFFIX is not None:
        return _DEVICE_NAME_SUFFIX

    suffix = "cpu"
    try:
        import torch

        if device_type == "cuda":
            available = getattr(torch.cuda, "is_available", lambda: False)()
            if available and torch.cuda.device_count():
                name = torch.cuda.get_device_name(0).strip().replace(" ", "_")
                suffix = f"gpu-{name}"[:64]
        elif device_type == "xpu":
            available = getattr(torch.xpu, "is_available", lambda: False)()
            if available and torch.xpu.device_count():
                name = torch.xpu.get_device_name(0).strip().replace(" ", "_")
                suffix = f"gpu-{name}"[:64]
        elif device_type == "mps":
            suffix = "apple-silicon"
    except Exception:
        pass

    _DEVICE_NAME_SUFFIX = suffix
    return suffix


# ---------------------------------------------------------------------------
# Environment setup (idempotent; safe to call multiple times)
# ---------------------------------------------------------------------------


def configure_tuning_env(device_type: str | None = None, cache_dir: Path | None = None):
    """Point Inductor/Triton and MIOpen at the persistent cache.

    Call once before any convolutions run (module import is fine). No-op for
    CPU/MPS where there is nothing to tune.
    """
    if device_type in ("cpu", "mps"):
        return None

    base = Path(cache_dir) if cache_dir else tuning_cache_dir()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(base / "inductor"))

    # MIOpen (ROCm) autotune database. Keyed by device name so a multi-GPU box
    # does not mix results from different chip generations.
    if "MIOPEN_USER_DB_PATH" in os.environ:
        return base  # user chose to manage the cache themselves

    try:
        import torch

        suffix = _device_name_suffix(device_type or ("cuda" if torch.cuda.is_available() else ""))
    except Exception:
        suffix = "gpu-unknown"

    db_path = str(base / f"miopen-{suffix}.fdb.txt")
    os.environ["MIOPEN_USER_DB_PATH"] = db_path

    return base


_LOGGED_MESSAGES = set()


def tune_device(device_type: str):
    """Enable per-shape autotuning at startup for supported GPU device types.

    The render pipeline uses a fixed set of input shapes (video resolution and
    tile sizes), so letting the backend benchmark convolutions once up front
    costs seconds on first use but removes that cost from every frame, on both
    CUDA (cuDNN) and ROCm (MIOpen). Intel XPU builds expose the same cudnn-style
    flags through torch.xpu.

    Safe to call multiple times; log messages are only shown once per process.
    """
    import torch as _torch  # local scope keeps module-level imports light

    def _log(msg):
        if msg in _LOGGED_MESSAGES:
            return
        _LOGGED_MESSAGES.add(msg)
        try:
            from ..utils.Util import log

            log(f"[tuning] {msg}")
        except Exception:
            pass  # standalone/CLI usage where the backend logger is unavailable

    if device_type in ("cuda", "xpu"):
        try:
            # torch.backends.cudnn works for CUDA, ROCm (MIOpen) and Intel XPU.
            _torch.backends.cudnn.benchmark = True
            _log(f"{device_type} autotune enabled")
        except Exception as e:  # pragma: no cover - defensive, device may be unavailable
            _log(f"Could not enable {device_type} autotune: {e}")

