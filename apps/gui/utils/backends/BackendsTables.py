"""Static ONNX Runtime provider → hardware vendor mapping.

Pure data, no runtime logic. ONNX Runtime is the only inference backend;
GPU builds are installed at runtime by ``ProviderUpgrade.py``.
"""

# Preferred provider order per hardware vendor. DirectML is handled by
# platform (Windows) rather than vendor, so it is not listed here.
ONNX_PROVIDER_HARDWARE = {
    "TensorrtExecutionProvider": "nvidia",
    "CUDAExecutionProvider": "nvidia",
    "MIGraphXExecutionProvider": "amd",
    "ROCMExecutionProvider": "amd",
    "OpenVINOExecutionProvider": "intel",
    "QnnExecutionProvider": "qualcomm",
    "CoreMLExecutionProvider": "apple",
}

__all__ = ["ONNX_PROVIDER_HARDWARE"]
