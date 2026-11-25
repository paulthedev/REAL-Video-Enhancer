"""Detect the GPU vendors physically present on this system.

The detection is independent of installed drivers or GPU framework builds,
so it works on machines where the hardware changed since the initial install.
ONNX Runtime is the only inference backend; this module reports the hardware
and the ONNX execution providers it maps to.
"""

import os
import platform
import re
import subprocess
from typing import Optional

from .BackendsTables import ONNX_PROVIDER_HARDWARE
from apps.gui.constants import PLATFORM
from apps.gui.util import log


def detect_gpu_hardware() -> list:
    """Detect which GPU vendors are physically present on this system,
    independently of any installed drivers or torch builds.

    Returns a list of hardware vendor names, e.g. ["nvidia", "amd"].
    Only discrete GPUs are reported; integrated graphics (which share
    the "intel" vendor name with discrete Intel GPUs) are skipped."""
    hardware = set()
    if platform == "darwin":
        # Apple Silicon is the only Apple GPU torch can run on; older
        # Macs fall back to the CPU build.
        if "arm" in platform.machine().lower() or "arm" in platform.processor().lower():
            hardware.add("apple")
        return sorted(hardware)
    if PLATFORM == "win32":
        try:
            output = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.lower()
        except Exception as e:
            log(f"Could not query Windows GPU info: {e}")
            return sorted(hardware)
        for line in output.splitlines():
            if "nvidia" in line:
                hardware.add("nvidia")
            if "radeon" in line or "amd" in line:
                hardware.add("amd")
            # Integrated UHD/Iris graphics can't run the xpu backend;
            # only discrete Arc GPUs count.
            if "intel" in line and "arc" in line:
                hardware.add("intel")
        return sorted(hardware)
    # PCI vendor IDs from sysfs, present on every Linux machine with a
    # discrete GPU regardless of which drivers are installed.
    pci_vendor_names = {
        "1002": "amd",
        "10de": "nvidia",
        "8086": "intel",
    }
    try:
        for card in os.listdir("/sys/class/drm"):
            if not card.startswith("card"):
                continue
            try:
                with open(f"/sys/class/drm/{card}/device/uevent") as f:
                    content = f.read()
            except OSError:
                continue
            # Integrated Intel graphics (iGPU) use the i915 driver;
            # discrete Intel GPUs (Arc/Xe) use xe, so this skips iGPUs.
            if re.search(r"^DRIVER=i915$", content, re.MULTILINE):
                continue
            match = re.search(r"PCI_ID=([0-9a-f]{4}):", content)
            if match:
                hardware.add(pci_vendor_names.get(match.group(1).lower(), "unknown"))
    except Exception as e:
        log(f"Could not query PCI devices: {e}")
    return sorted(hardware)


def get_compatible_backends(hardware: Optional[list] = None) -> list:
    """ONNX Runtime is the only backend; kept for API compatibility."""
    return ["onnx"]


def get_preferred_backend(hardware: Optional[list] = None) -> str:
    """ONNX Runtime is the only backend."""
    return "onnx"


def get_onnx_providers(hardware: Optional[list] = None) -> list:
    """ONNX Runtime providers in priority order for the detected hardware."""
    if hardware is None:
        hardware = detect_gpu_hardware()

    providers = []
    for provider, vendor in ONNX_PROVIDER_HARDWARE.items():
        if vendor in hardware:
            providers.append(provider)
    if PLATFORM == "win32" and "DmlExecutionProvider" not in providers:
        providers.append("DmlExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def check_backend_health(backend: str = "onnx") -> dict:
    """Probe the backend interpreter for ONNX Runtime health.

    The backend runs on our own interpreter, but the probe stays in a
    subprocess so an upgraded provider directory (activated by the backend's
    ``constants.py``) is reflected here too.
    """
    import json

    from apps.gui.constants import PYTHON_EXECUTABLE_PATH

    result = {"healthy": False, "error": None, "details": {}}
    if backend != "onnx":
        result["error"] = f"Unknown backend: {backend} (only 'onnx' exists)"
        return result

    probe = (
        "import json, onnxruntime as ort; providers = ort.get_available_providers();"
        "print(json.dumps({"
        "'version': ort.__version__,"
        "'providers': providers,"
        "'gpu_providers_available': any(p not in "
        "('CPUExecutionProvider', 'AzureExecutionProvider') for p in providers)}))"
    )

    try:
        proc = subprocess.run(
            [PYTHON_EXECUTABLE_PATH, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            result["error"] = f"onnx probe failed: {proc.stderr.strip()[:500]}"
            return result
        output = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        result["details"] = json.loads(output)
        result["healthy"] = True
    except FileNotFoundError:
        result["error"] = "Backend interpreter not found"
    except subprocess.TimeoutExpired:
        result["error"] = "onnx health check timed out"
    except json.JSONDecodeError as e:
        result["error"] = f"Could not parse health check output: {e}"
    except Exception as e:
        result["error"] = f"onnx health check failed: {e}"

    return result


def get_backend_reinstall_info(backend: str, hardware: Optional[list] = None) -> dict:
    """Information about the provider build that matches this hardware."""
    if hardware is None:
        hardware = detect_gpu_hardware()

    info = {
        "backend": backend,
        "hardware": hardware,
        "recommended_provider": None,
        "pip_packages": [],
    }
    if backend != "onnx":
        info["error"] = "only 'onnx' is supported"
        return info

    info["recommended_provider"] = get_onnx_providers(hardware)
    return info
