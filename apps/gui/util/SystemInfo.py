"""
System information queries (disk, OS, RAM, CPU).

Part of the `apps.gui.util` package.
"""

import platform
import shutil
import subprocess

import cpuinfo
import distro
import psutil

from apps.gui.constants import CWD, CPU_ARCH, PLATFORM
from apps.gui.util.Logging import log


def getAvailableDiskSpace() -> float:
    """
    Returns the available disk space in GB.
    """
    try:
        total, used, free = shutil.disk_usage(CWD)
        available_space = free / (1024**3)
        return available_space
    except Exception as e:
        log(f"An error occurred while getting available disk space: {e}")
        return "Unknown"


def getOSInfo() -> str:
    try:
        """
        Returns the exact name of the operating system along with additional information like 64-bit.
        """
        system = platform.system()
        release = platform.release()
        architecture = platform.machine()
        if system == "Linux":
            distro_name = distro.name()
            distro_version = distro.version()
            return f"{distro_name} {distro_version} {architecture}"
        return f"{system} {release} {architecture}"
    except Exception as e:
        log(f"An error occurred while getting OS information: {e}")
        return "Unknown"


def getRAMAmount() -> str:
    """
    Returns the amount of RAM in the system.
    """
    try:
        ram = psutil.virtual_memory().total
        ram_gb = ram / (1024**3)
        return f"{ram_gb:.2f} GB"
    except Exception as e:
        log(f"An error occurred while getting RAM amount: {e}")
        return "Unknown"


def getCPUInfo() -> str:
    """
    Returns the CPU information of the system.
    """
    try:
        if PLATFORM == "win32":
            try:
                # Run the 'wmic' command to get CPU information
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Split the result by lines and return the second line which contains the CPU name
                return result.stdout.split("\n")[2].strip()
            except Exception as e:
                print(f"An error occurred while getting CPU brand: {e}")
                return "X86_64 CPU" if CPU_ARCH == "x86_64" else "ARM64 CPU"
        else:
            return cpuinfo.get_cpu_info()["brand_raw"]
    except Exception as e:
        log(f"An error occurred while getting CPU information: {e}")
        return "Unknown"
