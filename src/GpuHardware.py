"""Detect the GPU vendors physically present on this system.

The detection is independent of installed drivers or torch builds, so it
works on machines where the hardware changed since the initial install.
"""

import os
import platform
import re
import subprocess

from .constants import PLATFORM
from .Util import log

# The hardware vendor name each torch backend family runs on.
FAMILY_HARDWARE = {
    "cuda": "nvidia",
    "rocm": "amd",
    "xpu": "intel",
    "mps": "apple",
}
# The torch backend family for each hardware vendor.
HARDWARE_FAMILIES = {v: k for k, v in FAMILY_HARDWARE.items()}


def detect_gpu_hardware() -> list:
    """Detect which GPU vendors are physically present on this system,
    independently of any installed drivers or torch builds.

    Returns a list of hardware vendor names, e.g. ["nvidia", "amd"].
    Only discrete GPUs are reported; integrated graphics (which share
    the "intel" vendor name with discrete Intel GPUs) are skipped."""
    hardware = set()
    if PLATFORM == "darwin":
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
