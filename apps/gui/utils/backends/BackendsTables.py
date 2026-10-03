"""Static vendor/backend/provider compatibility tables.

Extracted from ``GpuHardware.py``. These module-level mappings describe
which hardware vendor a torch backend family runs on, which backends are
compatible with which vendors, and which ONNX/NCNN providers map to which
hardware. They are pure data and carry no runtime logic.
"""


# The hardware vendor name each torch backend family runs on.
FAMILY_HARDWARE = {
    "cuda": "nvidia",
    "rocm": "amd",
    "xpu": "intel",
    "mps": "apple",
}
# The torch backend family for each hardware vendor.
HARDWARE_FAMILIES = {v: k for k, v in FAMILY_HARDWARE.items()}

# Backend compatibility with hardware vendors
# Each backend lists which hardware vendors it supports
BACKEND_HARDWARE_COMPATIBILITY = {
    "pytorch": ["nvidia", "amd", "intel", "apple"],
    "onnx": ["nvidia", "amd", "intel", "apple"],
    "ncnn": ["nvidia", "amd", "intel", "apple"],
}

# ONNX Runtime provider to hardware vendor mapping
ONNX_PROVIDER_HARDWARE = {
    "TensorrtExecutionProvider": "nvidia",
    "CUDAExecutionProvider": "nvidia",
    "MIGraphXExecutionProvider": "amd",
    "OpenVINOExecutionProvider": "intel",
    "QNNExecutionProvider": "qualcomm",
    "DirectMLExecutionProvider": "amd",
    "CoreMLExecutionProvider": "apple",
    "WebGPUExecutionProvider": "webgpu",
}

# NCNN backend to hardware vendor mapping
NCNN_BACKEND_HARDWARE = {
    "vulkan": ["nvidia", "amd", "intel"],  # Vulkan works on all discrete GPUs
    "metal": ["apple"],  # Metal only on Apple Silicon
    "cuda": ["nvidia"],
    "opencl": ["nvidia", "amd", "intel"],
}


__all__ = [
    "FAMILY_HARDWARE",
    "HARDWARE_FAMILIES",
    "BACKEND_HARDWARE_COMPATIBILITY",
    "ONNX_PROVIDER_HARDWARE",
    "NCNN_BACKEND_HARDWARE",
]
