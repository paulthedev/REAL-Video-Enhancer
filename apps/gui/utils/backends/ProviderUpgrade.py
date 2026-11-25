"""Hardware-aware onnxruntime provider upgrade manager.

The bundled build ships CPU onnxruntime (+ DirectML on Windows, which
covers every GPU vendor via DX12). The manager recommends and installs a
faster build for the detected GPU vendor in the background (Windows: NVIDIA
only, DirectML covers AMD/Intel/no-GPU):

- NVIDIA -> ``onnxruntime-gpu`` (CUDA EP, from PyPI) and, when the GPU has
  compute capability >= 7.5, the matching ``tensorrt`` libs so the TensorRT
  EP can be used (CUDA remains the automatic fallback).
- Intel  -> ``onnxruntime-openvino`` (OpenVINO EP, from PyPI)
- AMD    -> ``onnxruntime-rocm`` (MIOpen/ROCm build from AMD's wheel index
  at ``repo.radeon.com/rocm/manylinux/<release>/`` — newest release that
  supports the detected GPU architecture)

Everything installs into a writable data dir (never the read-only AppImage
mount). Activation happens on the next start: the backend prepends the
providers dir to ``sys.path`` before importing onnxruntime. The bundled CPU
build remains the fallback whenever anything fails.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

from apps.gui.util import log

AMD_ROCM_INDEX = "https://repo.radeon.com/rocm/manylinux/"

# appdata location for downloaded providers + the active-provider marker
PROVIDERS_DIR_NAME = os.path.join("REAL-Video-Enhancer", "providers")
MARKER_FILE = "active_provider.json"
TRT_CACHE_DIR_NAME = "trt_cache"

# onnxruntime-gpu 1.20+ is built against TensorRT 10.x. Keep the pin in one
# place so an ORT upgrade only needs a single edit here.
TENSORRT_REQUIREMENT = "tensorrt>=10,<11"
MIN_TENSORRT_COMPUTE_CAPABILITY = (7, 5)

# pip flags shared by every install: wheels only (the app has no compiler),
# into the writable providers dir, without trying to write launcher scripts.
_PIP_FLAGS = [
    "install",
    "--upgrade",
    "--only-binary=:all:",
    "--no-warn-script-location",
]

# Packages that already ship with the frozen app. pip installs a fresh copy
# into --target; leaving them on the shadowing sys.path entry can swap numpy
# (ABI-sensitive) or protobuf (onnx) underneath the app. Keep the bundled
# ones and delete the copies pip pulled in for onnxruntime.
_SHADOWED_PACKAGES = {
    "numpy",
    "numpy.libs",
    "google",
    "protobuf",
    "packaging",
    "flatbuffers",
    "sympy",
    "mpmath",
}

# gfx architecture -> minimum ROCm release that supports it. Cards older
# than every entry have no accelerated wheel and stay on the bundled CPU
# build. Unknown architectures are skipped (an APU or unsupported discrete
# card must not trigger a useless ~1 GB download).
_GFX_MIN_ROCM = {
    "gfx906": (2, 10),
    "gfx908": (3, 5),
    "gfx90a": (4, 5),
    "gfx1030": (4, 0),
    "gfx1031": (4, 5),
    "gfx1032": (4, 5),
    "gfx1100": (5, 7),
    "gfx1101": (5, 7),
    "gfx1102": (5, 7),
    "gfx1200": (6, 4),
    "gfx1201": (6, 4),
}


def providers_dir() -> str:
    """Writable per-user dir for downloaded provider builds."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, PROVIDERS_DIR_NAME)
    base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, PROVIDERS_DIR_NAME)


def trt_cache_dir() -> str:
    """Directory where the TensorRT EP caches engines and timing tables."""
    return os.path.join(providers_dir(), TRT_CACHE_DIR_NAME)


def _marker_path() -> str:
    return os.path.join(providers_dir(), MARKER_FILE)


def provider_upgrade_status() -> dict:
    """Which provider build is currently active (bundled or downloaded)."""
    try:
        with open(_marker_path()) as f:
            return json.load(f)
    except Exception:
        return {"provider": "bundled", "package": None}


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_recommended_provider() -> dict | None:
    """The faster onnxruntime build for this hardware, or None if the
    bundled build is already the best option.

    Uses the existing torch-free GPU detection; never defaults to a vendor
    that isn't present.
    """
    if sys.platform == "darwin":
        return None  # bundled CPU/CoreML

    from apps.gui.utils.backends.BackendDetect import detect_gpu_hardware

    vendors = detect_gpu_hardware()
    if "nvidia" in vendors:
        capability = _nvidia_compute_capability()
        return {
            "package": "onnxruntime-gpu",
            "index_url": None,
            "compute_capability": capability,
            "tensorrt": bool(
                capability and capability >= MIN_TENSORRT_COMPUTE_CAPABILITY
            ),
        }
    if sys.platform == "win32":
        return None  # bundled DirectML covers AMD/Intel/no-GPU on Windows
    if "intel" in vendors:
        return {"package": "onnxruntime-openvino", "index_url": None}
    if "amd" in vendors:
        gfx = _detect_amd_gfx()
        release = _newest_supported_rocm_release(gfx)
        if release:
            return {
                "package": "onnxruntime-rocm",
                "index_url": f"{AMD_ROCM_INDEX}{release}/",
            }
    return None


def _nvidia_compute_capability() -> tuple[int, int] | None:
    """NVIDIA compute capability via nvidia-smi (no torch dependency)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        log(f"Could not query NVIDIA compute capability: {e}")
        return None
    match = re.search(r"(\d+)\.(\d+)", out.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _detect_amd_gfx() -> str | None:
    """AMD GPU architecture (gfxXXXX) via rocminfo, or None."""
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=20
        )
    except Exception:
        return None
    match = re.search(r"\bgfx[0-9a-f]{3,4}\b", out.stdout)
    return match.group(0) if match else None


def _parse_release(version: str) -> tuple[int, ...]:
    return tuple(
        int(part)
        for part in version.replace("rocm-rel-", "").split(".")
        if part.isdigit()
    )


def _release_supports_gfx(release: str, gfx: str | None) -> bool:
    if not gfx or gfx not in _GFX_MIN_ROCM:
        return False
    minimum = _GFX_MIN_ROCM[gfx]
    return _parse_release(release) >= minimum


def _newest_supported_rocm_release(gfx: str | None = None) -> str | None:
    """Newest ROCm release on AMD's index publishing a wheel for our python
    and supporting the detected GPU architecture.

    Without a known gfx (rocminfo missing) we cannot tell an APU/unsupported
    card from a supported one, so the bundled CPU build stays active."""
    if not gfx:
        log(
            "Could not identify the AMD GPU architecture (rocminfo); "
            "staying on the bundled CPU build."
        )
        return None
    py_tag = _python_tag()
    try:
        with urllib.request.urlopen(AMD_ROCM_INDEX, timeout=30) as r:
            listing = r.read().decode("utf-8", "replace")
    except Exception as e:
        log(f"Could not list AMD ROCm index: {e}")
        return None

    releases = re.findall(r'href="(rocm-rel-[^"/]+)/"', listing)
    for version in sorted(releases, key=_parse_release, reverse=True):
        if not _release_supports_gfx(version, gfx):
            continue
        try:
            with urllib.request.urlopen(f"{AMD_ROCM_INDEX}{version}/", timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        if f"-{py_tag}-" in page:
            return version
    return None


def _run_pip(packages: list[str], index_url: str | None, progress, cancel_check) -> bool:
    """Install wheels into the providers dir with the frozen-safe pip mode."""
    target = providers_dir()
    os.makedirs(target, exist_ok=True)

    cmd = [sys.executable, "-m", "pip", *_PIP_FLAGS]
    if index_url:
        cmd += ["--extra-index-url", index_url, "--trusted-host", "repo.radeon.com"]
    cmd += ["--target", target, *packages]

    log("Installing providers: " + " ".join(cmd))
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
    return True


def _prune_shadowed_packages(target: str) -> None:
    """Drop copies of packages the frozen app already provides."""
    try:
        entries = os.listdir(target)
    except OSError:
        return
    for entry in entries:
        stem = entry.split("-")[0].lower() if entry.endswith(".dist-info") else entry
        if stem in _SHADOWED_PACKAGES:
            path = os.path.join(target, entry)
            try:
                if os.path.isdir(path):
                    import shutil

                    shutil.rmtree(path)
                else:
                    os.remove(path)
                log(f"Kept bundled copy of {stem}; removed {entry} from providers dir.")
            except OSError as e:
                log(f"Could not prune {entry}: {e}")


def install_recommended_provider(progress=None, cancel_check=None) -> bool:
    """Download and install the recommended provider build.

    Uses the running interpreter's pip with ``--target`` so the wheels (and
    their dependency closure, e.g. the nvidia CUDA libs) land in the
    providers dir instead of the frozen application. In a frozen build the
    executable's ``-m pip`` entry point runs bundled pip in a child process
    (see the dispatch at the top of ``apps/gui/REAL-Video-Enhancer.py``).

    Returns True when a new provider was installed successfully.
    """
    recommendation = get_recommended_provider()
    if recommendation is None:
        log("No faster onnxruntime build available for this hardware.")
        return False

    package = recommendation["package"]
    os.makedirs(providers_dir(), exist_ok=True)

    if not _run_pip(
        [package], recommendation.get("index_url"), progress, cancel_check
    ):
        return False

    tensorrt = bool(recommendation.get("tensorrt"))
    if tensorrt:
        os.makedirs(trt_cache_dir(), exist_ok=True)
        if not _run_pip(
            [TENSORRT_REQUIREMENT],
            recommendation.get("index_url"),
            progress,
            cancel_check,
        ):
            # CUDA still works without TensorRT — keep the installed build.
            log("TensorRT install failed; continuing with the CUDA EP.")
            tensorrt = False

    _prune_shadowed_packages(providers_dir())

    with open(_marker_path(), "w") as f:
        json.dump(
            {
                "provider": package,
                "package": package,
                "index_url": recommendation.get("index_url"),
                "tensorrt": tensorrt,
                "compute_capability": recommendation.get("compute_capability"),
                "python": _python_tag(),
            },
            f,
            indent=2,
        )
    log(f"Provider installed to {providers_dir()}. Restart the app to use it.")
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
        os.environ["RVE_ONNX_PROVIDERS_DIR"] = providers
        if info.get("tensorrt"):
            cache = os.path.join(providers, TRT_CACHE_DIR_NAME)
            os.makedirs(cache, exist_ok=True)
            os.environ["RVE_ONNX_TRT_CACHE_DIR"] = cache
        return True
    except Exception:
        return False
