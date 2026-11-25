"""Shared ONNX session construction for the render engines.

One place for fp16 conversion, vendor-aware provider selection and session
options — UpscaleONNX and InterpolateONNX both build through this.
"""

import os

import numpy as np
import onnx
import onnxruntime as ort

try:
    from onnxconverter_common import float16 as _fp16
except ImportError:  # fp16 conversion unavailable — fp32 keeps rendering
    _fp16 = None

# Preferred GPU provider order (performance-first: TRT before CUDA). ORT
# falls through to the next provider automatically when an EP rejects a
# node, so the full chain is registered rather than just the first match.
PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
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


def _provider_options(name: str, device_id: int, fp16: bool) -> dict:
    cache_dir = os.environ.get("RVE_ONNX_TRT_CACHE_DIR")
    if name == "TensorrtExecutionProvider":
        options = {
            "device_id": device_id,
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": bool(cache_dir),
            "trt_timing_cache_enable": bool(cache_dir),
        }
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            options["trt_engine_cache_path"] = cache_dir
            options["trt_timing_cache_path"] = cache_dir
        return options
    if name == "OpenVINOExecutionProvider":
        return {"device_type": "GPU"}
    if name == "DmlExecutionProvider":
        return {"device_id": device_id}
    options = {"device_id": device_id}
    if name == "CUDAExecutionProvider":
        options["do_copy_in_default_stream"] = True
    return options


def create_session(
    model: onnx.ModelProto,
    device_id: int = 0,
    fp16: bool = False,
    use_gpu: bool = True,
):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    available = ort.get_available_providers()
    providers = []
    if use_gpu:
        for name in PREFERRED_PROVIDERS:
            if name in available:
                providers.append(
                    (name, _provider_options(name, device_id, fp16))
                )
    if "CPUExecutionProvider" in available:
        providers.append(("CPUExecutionProvider", {}))
    if not providers:
        providers = [("CPUExecutionProvider", {})]
    return ort.InferenceSession(
        model.SerializeToString(), session_options, providers=providers
    )
