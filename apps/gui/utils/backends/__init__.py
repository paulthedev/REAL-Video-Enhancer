"""Decomposed backend/dependency install services for the GUI.

Each concern that lived in the legacy monolithic ``DownloadDeps.py``,
``GpuHardware.py``, ``BuiltInTorchVersions.py`` and ``DownloadModels.py``
now lives in its own module in this package:

- Dependency             -> shared base class + run_executable helper
- Backend                -> backend bundle installer
- Python                 -> standalone CPython installer
- FFmpeg                 -> ffmpeg binary installer
- VcRedlist              -> MSVC redist installer (Windows)
- DownloadDependencies   -> orchestrator (torch pairing, ROCm targets, pip)
- BackendDetect          -> runtime GPU detection and backend health checks
- BackendsTables         -> static vendor/backend/provider tables
- TorchVersions          -> bundled-in PyTorch version matrix
- BackendsModels         -> model download service

Importers can pull any of these straight from the package::

    from apps.gui.utils.backends import Dependency, detect_gpu_hardware
"""

from .BackendsModels import DownloadModel
from .Backend import Backend
from .BackendDetect import (
    _check_ncnn_health,
    _check_onnx_health,
    _check_pytorch_health,
    check_backend_health,
    detect_gpu_hardware,
    get_backend_reinstall_info,
    get_compatible_backends,
    get_ncnn_backend,
    get_onnx_providers,
    get_preferred_backend,
    get_torch_backend_family,
)
from .BackendsTables import (
    BACKEND_HARDWARE_COMPATIBILITY,
    FAMILY_HARDWARE,
    HARDWARE_FAMILIES,
    NCNN_BACKEND_HARDWARE,
    ONNX_PROVIDER_HARDWARE,
)
from .Dependency import Dependency, run_executable
from .DownloadDependencies import DownloadDependencies
from .FFmpeg import FFMpeg
from .Python import Python
from .TorchVersions import Torch2_14, Torch2_15, TorchVersion
from .VcRedlist import VCRedList

__all__ = [
    "BACKEND_HARDWARE_COMPATIBILITY",
    "Backend",
    "DownloadDependencies",
    "DownloadModel",
    "FFMpeg",
    "FAMILY_HARDWARE",
    "HARDWARE_FAMILIES",
    "NCNN_BACKEND_HARDWARE",
    "ONNX_PROVIDER_HARDWARE",
    "Python",
    "Torch2_14",
    "Torch2_15",
    "TorchVersion",
    "VCRedList",
    "_check_ncnn_health",
    "_check_onnx_health",
    "_check_pytorch_health",
    "check_backend_health",
    "detect_gpu_hardware",
    "get_backend_reinstall_info",
    "get_compatible_backends",
    "get_ncnn_backend",
    "get_onnx_providers",
    "get_preferred_backend",
    "get_torch_backend_family",
    "run_executable",
]

