"""
GUI utility subpackages.

This package re-exports all symbols in one place so call sites can import
from ``apps.gui.util`` directly, or target the specific module for a symbol
(e.g. ``from apps.gui.util.FileHandler import FileHandler``).

Modules:
    - ``Logging``      -> log, warnAndLog, errorAndLog
    - ``FileHandler``  -> FileHandler + standalone file helpers
    - ``SystemInfo``   -> getAvailableDiskSpace / getOSInfo / getRAMAmount / getCPUInfo
    - ``Networking``   -> networkCheck
    - ``Permissions``  -> checkForWritePermissions
    - ``Subprocess``   -> subprocess_popen_without_terminal, create_independent_process
    - ``Misc``         -> openLink / open_folder / print_execution_time
"""

from .FileHandler import (
    FileHandler,
    copy,
    copyFile,
    createDirectory,
    currentDirectory,
    downloadFile,
    extractTarGZ,
    makeExecutable,
    move,
    removeFolder,
    removeFile,
)
from .Logging import errorAndLog, log, warnAndLog
from .Permissions import checkForWritePermissions
from .SystemInfo import (
    getAvailableDiskSpace,
    getCPUInfo,
    getOSInfo,
    getRAMAmount,
)
from .Misc import openLink, open_folder, print_execution_time
from .Networking import networkCheck
from .Subprocess import create_independent_process, subprocess_popen_without_terminal

__all__ = [
    # FileHandler
    "FileHandler",
    "copy",
    "copyFile",
    "createDirectory",
    "currentDirectory",
    "downloadFile",
    "extractTarGZ",
    "makeExecutable",
    "move",
    "removeFolder",
    "removeFile",
    # Log
    "errorAndLog",
    "log",
    "warnAndLog",
    # Permissions
    "checkForWritePermissions",
    # SystemInfo
    "getAvailableDiskSpace",
    "getCPUInfo",
    "getOSInfo",
    "getRAMAmount",
    # misc
    "openLink",
    "open_folder",
    "print_execution_time",
    # network
    "networkCheck",
    # subprocess
    "create_independent_process",
    "subprocess_popen_without_terminal",
]

