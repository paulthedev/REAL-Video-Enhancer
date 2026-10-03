"""App-level shared services package.

Historically the small util helpers (log, FileHandler, SystemInfo, network,
Permissions, subprocess, misc) lived in the singular ``apps/gui/util``
package. This package holds the decomposed backend/dependency install
services under :mod:`apps.gui.utils.backends` and re-exports their public
symbols here so callers can import from either place::

    from apps.gui.utils import detect_gpu_hardware, Dependency
"""

from .backends import (  # noqa: F401
    BACKEND_HARDWARE_COMPATIBILITY,
    Backend,
    Dependency,
    DownloadDependencies,
    DownloadModel,
    FFMpeg,
    FAMILY_HARDWARE,
    HARDWARE_FAMILIES,
    NCNN_BACKEND_HARDWARE,
    ONNX_PROVIDER_HARDWARE,
    Python,
    Torch2_14,
    Torch2_15,
    TorchVersion,
    VCRedList,
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
    run_executable,
)
