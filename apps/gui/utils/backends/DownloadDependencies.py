"""Model downloads and onnxruntime provider upgrade for the GUI.

The dependency-install machinery (torch/ncnn/onnxruntime pip installs into a
standalone python env) is gone — ONNX Runtime and ffmpeg ship bundled. What
remains is model downloads (see BackendsModels.DownloadModel) and the
hardware-aware onnxruntime provider upgrade (see ProviderUpgrade).
"""

import re

from apps.gui.constants import (
    PLATFORM,
    TEMP_DOWNLOAD_PATH,
)
from apps.gui.lib.QTcustom import DisplayCommandOutputPopup
from apps.gui.util import log

from .BackendDetect import (
    check_backend_health,
    detect_gpu_hardware,
    get_backend_reinstall_info,
)
from .ProviderUpgrade import (
    get_recommended_provider,
    install_recommended_provider,
    provider_upgrade_status,
)


class DownloadDependencies:
    """Thin facade over the provider upgrade + health checks.

    Kept as a class because MainWindow/process code holds an instance.
    """

    def __init__(self, use_torch_nightly: bool = False):
        # torch nightly support died with the torch backend
        pass

    def check_backend_health(self, backend: str) -> dict:
        return check_backend_health(backend)

    def recommend_provider(self) -> dict:
        """Recommended onnxruntime build for this hardware (may be None)."""
        return get_recommended_provider()

    def install_recommended(self, progress=None, cancel_check=None) -> bool:
        """Download + activate the recommended provider. Returns success."""
        return install_recommended_provider(progress=progress, cancel_check=cancel_check)

    def provider_status(self) -> dict:
        return provider_upgrade_status()
