"""Hardware-aware onnxruntime provider upgrade manager.

The bundled build ships CPU onnxruntime (+ DirectML on Windows, which
covers every GPU vendor via DX12). On Linux the manager recommends and
installs a faster build for the detected GPU vendor:

- NVIDIA -> ``onnxruntime-gpu`` (CUDA EP, from PyPI)
- Intel  -> ``onnxruntime-openvino`` (OpenVINO EP, from PyPI)
- AMD    -> ``onnxruntime-rocm`` (MIOpen/ROCm build from AMD's wheel index
  at ``repo.radeon.com/rocm/manylinux/<release>/`` — newest release that
  supports the detected GPU)

Everything installs into a writable data dir (never the read-only AppImage
mount). Activation happens on the next start: the backend prepends the
providers dir to ``sys.path`` before importing onnxruntime. The bundled CPU
build remains the fallback whenever anything fails.
"""

import json
import os
import subprocess
import sys
import urllib.request

from apps.gui.util import log

AMD_ROCM_INDEX = "https://repo.radeon.com/rocm/manylinux/"

# appdata location for downloaded providers + the active-provider marker
PROVIDERS_DIR_NAME = os.path.join("REAL-Video-Enhancer", "providers")
MARKER_FILE = "active_provider.json"


def providers_dir() -> str:
    """Writable per-user dir for downloaded provider builds."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, PROVIDERS_DIR_NAME)
    base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, PROVIDERS_DIR_NAME)


def _marker_path() -> str:
    return os.path.join(providers_dir(), MARKER_FILE)


def provider_upgrade_status() -> dict:
    """Which provider build is currently active (bundled or downloaded)."""
    try:
        with open(_marker_path()) as f:
            return json.load(f)
    except Exception:
        return {"provider": "bundled", "package": None}


def get_recommended_provider() -> dict | None:
    """The faster onnxruntime build for this hardware, or None if the
    bundled build is already the best option.

    Uses the existing torch-free GPU detection; never defaults to a vendor
    that isn't present.
    """
    if sys.platform == "win32":
        return None  # bundled DirectML covers every vendor
    if sys.platform == "darwin":
        return None  # bundled CPU/CoreML

    from apps.gui.utils.backends.BackendDetect import detect_gpu_hardware

    vendors = detect_gpu_hardware()
    if "nvidia" in vendors:
        return {"package": "onnxruntime-gpu", "index_url": None}
    if "intel" in vendors:
        return {"package": "onnxruntime-openvino", "index_url": None}
    if "amd" in vendors:
        release = _newest_supported_rocm_release()
        if release:
            return {
                "package": "onnxruntime-rocm",
                "index_url": f"{AMD_ROCM_INDEX}{release}/",
            }
    return None


def _newest_supported_rocm_release() -> str | None:
    """Newest ROCm release on AMD's index publishing a wheel for our python.

    Walking the plain HTML directory listing is enough — AMD keeps it
    sorted and stable. Architecture filtering (gfx capability) is left to
    the runtime: an unsupported card simply gets no EP and the bundled CPU
    build remains active.
    """
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    try:
        with urllib.request.urlopen(AMD_ROCM_INDEX, timeout=30) as r:
            listing = r.read().decode("utf-8", "replace")
    except Exception as e:
        log(f"Could not list AMD ROCm index: {e}")
        return None

    import re

    releases = re.findall(r'href="(rocm-rel-[^"/]+)/"', listing)
    for version in sorted(
        releases,
        key=lambda r: [int(x) for x in r.replace("rocm-rel-", "").split(".") if x.isdigit()],
        reverse=True,
    ):
        try:
            with urllib.request.urlopen(f"{AMD_ROCM_INDEX}{version}/", timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        if f"onnxruntime_rocm-1.{py_tag}" in page or f"-{py_tag}-" in page:
            return version
    return None


def install_recommended_provider(progress=None, cancel_check=None) -> bool:
    """Download and install the recommended provider build.

    Uses the running interpreter's pip with ``--target`` so the wheels (and
    their dependency closure, e.g. the nvidia CUDA libs) land in the
    providers dir instead of the frozen application.

    Returns True when a new provider was installed successfully.
    """
    recommendation = get_recommended_provider()
    if recommendation is None:
        log("No faster onnxruntime build available for this hardware.")
        return False

    package = recommendation["package"]
    target = providers_dir()
    os.makedirs(target, exist_ok=True)

    cmd = [sys.executable, "-m", "pip", "install", "--target", target, package]
    if recommendation.get("index_url"):
        cmd[3:3] = ["--extra-index-url", recommendation["index_url"], "--trusted-host", "repo.radeon.com"]

    log("Installing provider: " + " ".join(cmd))
    from apps.gui.util import subprocess_popen_without_terminal

    process = subprocess_popen_without_terminal(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    for line in process.stdout:
        if cancel_check is not None and cancel_check():
            process.kill()
            log("Provider download cancelled.")
            return False
        if progress is not None:
            progress(line.strip())
        log(line.strip())
    process.wait()

    if process.returncode != 0:
        log(f"Provider install failed with code {process.returncode}")
        return False

    with open(_marker_path(), "w") as f:
        json.dump(
            {"provider": package, "package": package, "index_url": recommendation.get("index_url")},
            f,
            indent=2,
        )
    log(f"Provider installed to {target}. Restart the app to use it.")
    return True


def activate_providers_dir() -> bool:
    """Backend-side: put the downloaded providers dir ahead on sys.path so
    its onnxruntime shadows the bundled CPU build. Call before importing
    onnxruntime. Returns True when an upgraded provider was activated."""
    marker = os.path.join(
        os.environ.get("RVE_ONNX_PROVIDERS_DIR")
        or os.path.join(
            (
                os.environ.get("APPDATA")
                if sys.platform == "win32"
                else os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
            ),
            PROVIDERS_DIR_NAME,
        ),
        MARKER_FILE,
    )
    try:
        with open(marker) as f:
            info = json.load(f)
        if not info.get("package"):
            return False
        providers = os.path.dirname(marker)
        if providers not in sys.path:
            sys.path.insert(0, providers)
        return True
    except Exception:
        return False
