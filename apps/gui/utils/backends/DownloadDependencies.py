"""Model downloads and onnxruntime provider upgrade for the GUI.

The dependency-install machinery (torch/ncnn/onnxruntime pip installs into a
standalone python env) is gone — ONNX Runtime and ffmpeg ship bundled. What
remains is model downloads (see BackendsModels.DownloadModel) and the
hardware-aware onnxruntime provider upgrade (see ProviderUpgrade).
"""

from .BackendDetect import check_backend_health
from .ProviderUpgrade import (
    get_recommended_provider,
    install_recommended_provider,
    provider_upgrade_status,
)


class DownloadDependencies:
    """Thin facade over the provider upgrade + health checks.

    Kept as a class because MainWindow/process code held an instance.
    """

    def check_backend_health(self, backend: str = "onnx") -> dict:
        return check_backend_health(backend)

    def recommend_provider(self) -> dict:
        """Recommended onnxruntime build for this hardware (may be None)."""
        return get_recommended_provider()

    def install_recommended(self, progress=None, cancel_check=None) -> bool:
        """Download + activate the recommended provider. Returns success."""
        return install_recommended_provider(progress=progress, cancel_check=cancel_check)

    def provider_status(self) -> dict:
        return provider_upgrade_status()
