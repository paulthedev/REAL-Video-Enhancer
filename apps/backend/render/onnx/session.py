"""Shared ONNX session construction for the render engines.

One place for fp16 conversion, vendor-aware provider selection and session
options — UpscaleONNX and InterpolateONNX both build through this.
"""

import numpy as np
import onnx
import onnxruntime as ort

try:
    from onnxconverter_common import float16 as _fp16
except ImportError:  # fp16 conversion unavailable — fp32 keeps rendering
    _fp16 = None

# Preferred GPU provider order (performance-first: TRT before CUDA).
PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
]


def resolve_precision(precision: str) -> np.dtype:
    """auto -> fp16 only when a GPU EP is available and the converter exists."""
    if precision == "float16" and _fp16 is not None:
        return np.float16
    if precision == "float32":
        return np.float32
    gpu_present = any(
        not p.startswith(("CPU", "Azure")) for p in ort.get_available_providers()
    )
    if gpu_present and _fp16 is not None:
        return np.float16
    return np.float32


def prepare_model(model_path: str, np_dtype: np.dtype) -> onnx.ModelProto:
    model = onnx.load(model_path)
    if np_dtype == np.float16:
        model = _fp16.convert_float_to_float16(model, check_fp16_ready=False)
    return model


def create_session(model: onnx.ModelProto, device_id: int = 0):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    available = ort.get_available_providers()
    providers = []
    for name in PREFERRED_PROVIDERS:
        if name in available:
            options = {"device_id": device_id}
            if name != "DmlExecutionProvider":
                options["do_copy_in_default_stream"] = True
            providers.append((name, options))
    if not providers:
        print(
            f"[onnx] No GPU provider available ({available}); using CPU."
        )
        providers = [("CPUExecutionProvider", {})]
    return ort.InferenceSession(
        model.SerializeToString(), session_options, providers=providers
    )
