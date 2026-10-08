"""Backend services for the GUI.

ONNX Runtime ships bundled with the app, so this package is now limited to
model downloads, GPU/provider detection and the runtime provider upgrade.

- BackendDetect         -> GPU detection + backend health checks
- BackendsTables        -> static vendor/backend/provider tables
- BackendsModels        -> model download service
- DownloadDependencies  -> onnxruntime provider upgrade + model downloads
- ProviderUpgrade       -> hardware-aware onnxruntime upgrade manager
"""

from .BackendsModels import DownloadModel
from .BackendDetect import (
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
from .DownloadDependencies import DownloadDependencies
from .ProviderUpgrade import (
    get_recommended_provider,
    install_recommended_provider,
    provider_upgrade_status,
)

__all__ = [
    "BACKEND_HARDWARE_COMPATIBILITY",
    "BackendsTables",
    "DownloadDependencies",
    "DownloadModel",
    "FAMILY_HARDWARE",
    "HARDWARE_FAMILIES",
    "NCNN_BACKEND_HARDWARE",
    "ONNX_PROVIDER_HARDWARE",
    "ProviderUpgrade",
    "check_backend_health",
    "detect_gpu_hardware",
    "get_backend_reinstall_info",
    "get_compatible_backends",
    "get_ncnn_backend",
    "get_onnx_providers",
    "get_preferred_backend",
    "get_torch_backend_family",
    "get_recommended_provider",
    "install_recommended_provider",
    "provider_upgrade_status",
]
