"""Detect the GPU vendors physically present on this system.

The detection is independent of installed drivers or torch builds, so it
works on machines where the hardware changed since the initial install.

This module is backend-agnostic: it detects hardware and determines which
backends (PyTorch, ONNX, NCNN) are compatible with the detected hardware.
"""

import os
import platform
import re
import subprocess
from typing import Optional

from .constants import PLATFORM
from .Util import log

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


def detect_gpu_hardware() -> list:
    """Detect which GPU vendors are physically present on this system,
    independently of any installed drivers or torch builds.

    Returns a list of hardware vendor names, e.g. ["nvidia", "amd"].
    Only discrete GPUs are reported; integrated graphics (which share
    the "intel" vendor name with discrete Intel GPUs) are skipped."""
    hardware = set()
    if PLATFORM == "darwin":
        # Apple Silicon is the only Apple GPU torch can run on; older
        # Macs fall back to the CPU build.
        if "arm" in platform.machine().lower() or "arm" in platform.processor().lower():
            hardware.add("apple")
        return sorted(hardware)
    if PLATFORM == "win32":
        try:
            output = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.lower()
        except Exception as e:
            log(f"Could not query Windows GPU info: {e}")
            return sorted(hardware)
        for line in output.splitlines():
            if "nvidia" in line:
                hardware.add("nvidia")
            if "radeon" in line or "amd" in line:
                hardware.add("amd")
            # Integrated UHD/Iris graphics can't run the xpu backend;
            # only discrete Arc GPUs count.
            if "intel" in line and "arc" in line:
                hardware.add("intel")
        return sorted(hardware)
    # PCI vendor IDs from sysfs, present on every Linux machine with a
    # discrete GPU regardless of which drivers are installed.
    pci_vendor_names = {
        "1002": "amd",
        "10de": "nvidia",
        "8086": "intel",
    }
    try:
        for card in os.listdir("/sys/class/drm"):
            if not card.startswith("card"):
                continue
            try:
                with open(f"/sys/class/drm/{card}/device/uevent") as f:
                    content = f.read()
            except OSError:
                continue
            # Integrated Intel graphics (iGPU) use the i915 driver;
            # discrete Intel GPUs (Arc/Xe) use xe, so this skips iGPUs.
            if re.search(r"^DRIVER=i915$", content, re.MULTILINE):
                continue
            match = re.search(r"PCI_ID=([0-9a-f]{4}):", content)
            if match:
                hardware.add(pci_vendor_names.get(match.group(1).lower(), "unknown"))
    except Exception as e:
        log(f"Could not query PCI devices: {e}")
    return sorted(hardware)


def get_compatible_backends(hardware: Optional[list] = None) -> list:
    """Return the list of backends compatible with the detected hardware.
    
    Args:
        hardware: Optional list of hardware vendors. If None, detected from system.
        
    Returns:
        List of backend names that are compatible with the hardware.
    """
    if hardware is None:
        hardware = detect_gpu_hardware()
    
    compatible = []
    for backend in BACKEND_HARDWARE_COMPATIBILITY:
        backend_hardware = BACKEND_HARDWARE_COMPATIBILITY[backend]
        # Backend is compatible if any detected hardware is in its supported list
        if any(h in backend_hardware for h in hardware):
            compatible.append(backend)
    
    return compatible


def get_preferred_backend(hardware: Optional[list] = None) -> str:
    """Return the preferred backend for the detected hardware.
    
    Priority: ONNX > PyTorch > NCNN (ONNX is primary, PyTorch is fallback)
    
    Args:
        hardware: Optional list of hardware vendors. If None, detected from system.
        
    Returns:
        Preferred backend name.
    """
    if hardware is None:
        hardware = detect_gpu_hardware()
    
    # ONNX is the primary backend
    if "onnx" in get_compatible_backends(hardware):
        return "onnx"
    
    # PyTorch is the fallback
    if "pytorch" in get_compatible_backends(hardware):
        return "pytorch"
    
    # NCNN is tertiary
    if "ncnn" in get_compatible_backends(hardware):
        return "ncnn"
    
    return "onnx"  # default fallback


def get_torch_backend_family(hardware: Optional[list] = None) -> str:
    """Get the recommended PyTorch backend family for the detected hardware.
    
    Args:
        hardware: Optional list of hardware vendors. If None, detected from system.
        
    Returns:
        Backend family name (cuda, rocm, xpu, mps, or cpu).
    """
    if hardware is None:
        hardware = detect_gpu_hardware()
    
    if not hardware:
        return "cpu"
    
    # Use the first detected hardware vendor
    vendor = hardware[0]
    return HARDWARE_FAMILIES.get(vendor, "cpu")


def get_onnx_providers(hardware: Optional[list] = None) -> list:
    """Get the list of ONNX Runtime providers for the detected hardware.
    
    Args:
        hardware: Optional list of hardware vendors. If None, detected from system.
        
    Returns:
        List of provider names in priority order.
    """
    if hardware is None:
        hardware = detect_gpu_hardware()
    
    providers = []
    
    # Add providers based on hardware
    for provider, vendor in ONNX_PROVIDER_HARDWARE.items():
        if vendor in hardware:
            providers.append(provider)
    
    # Always add CPU as fallback
    providers.append("CPUExecutionProvider")
    
    return providers


def get_ncnn_backend(hardware: Optional[list] = None) -> str:
    """Get the recommended NCNN backend for the detected hardware.
    
    Args:
        hardware: Optional list of hardware vendors. If None, detected from system.
        
    Returns:
        Backend name (vulkan, metal, cuda, opencl, or cpu).
    """
    if hardware is None:
        hardware = detect_gpu_hardware()
    
    if not hardware:
        return "cpu"
    
    vendor = hardware[0]
    
    # Check each backend's supported hardware
    for backend, supported in NCNN_BACKEND_HARDWARE.items():
        if vendor in supported:
            return backend
    
    return "cpu"


def check_backend_health(backend: str) -> dict:
    """Check if a backend is healthy and working.
    
    Args:
        backend: Backend name (pytorch, onnx, ncnn).
        
    Returns:
        Dictionary with health status:
        - healthy: bool
        - error: str or None
        - details: dict with additional info
    """
    result = {
        "healthy": False,
        "error": None,
        "details": {}
    }
    
    try:
        if backend == "pytorch":
            return _check_pytorch_health(result)
        elif backend == "onnx":
            return _check_onnx_health(result)
        elif backend == "ncnn":
            return _check_ncnn_health(result)
        else:
            result["error"] = f"Unknown backend: {backend}"
            return result
    except Exception as e:
        result["error"] = str(e)
        return result


def _check_pytorch_health(result: dict) -> dict:
    """Check PyTorch backend health."""
    try:
        import torch
        
        result["details"]["version"] = torch.__version__
        result["details"]["cuda_available"] = torch.cuda.is_available()
        result["details"]["mps_available"] = torch.backends.mps.is_available()
        
        if torch.cuda.is_available():
            result["details"]["cuda_device_count"] = torch.cuda.device_count()
            result["details"]["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        result["healthy"] = True
    except ImportError as e:
        result["error"] = f"PyTorch not installed: {e}"
    except Exception as e:
        result["error"] = f"PyTorch health check failed: {e}"
    
    return result


def _check_onnx_health(result: dict) -> dict:
    """Check ONNX Runtime health."""
    try:
        import onnxruntime as ort
        
        result["details"]["version"] = ort.__version__
        result["details"]["providers"] = ort.get_available_providers()
        
        # Check if any GPU provider is available
        gpu_providers = [p for p in ort.get_available_providers() 
                        if p not in ["CPUExecutionProvider"]]
        result["details"]["gpu_providers_available"] = len(gpu_providers) > 0
        
        result["healthy"] = True
    except ImportError as e:
        result["error"] = f"ONNX Runtime not installed: {e}"
    except Exception as e:
        result["error"] = f"ONNX Runtime health check failed: {e}"
    
    return result


def _check_ncnn_health(result: dict) -> dict:
    """Check NCNN backend health."""
    try:
        # Try importing ncnn
        import ncnn
        
        result["details"]["version"] = getattr(ncnn, "__version__", "unknown")
        result["healthy"] = True
    except ImportError:
        # Try alternative imports
        try:
            import rife_ncnn_vulkan_python_tntwise
            result["details"]["rife_ncnn_version"] = rife_ncnn_vulkan_python_tntwise.__version__
            result["healthy"] = True
        except ImportError:
            try:
                import upscale_ncnn_py
                result["details"]["upscale_ncnn_version"] = upscale_ncnn_py.__version__
                result["healthy"] = True
            except ImportError as e:
                result["error"] = f"NCNN not installed: {e}"
    except Exception as e:
        result["error"] = f"NCNN health check failed: {e}"
    
    return result


def get_backend_reinstall_info(backend: str, hardware: Optional[list] = None) -> dict:
    """Get information needed to reinstall a backend.
    
    Args:
        backend: Backend name (pytorch, onnx, ncnn).
        hardware: Optional list of hardware vendors. If None, detected from system.
        
    Returns:
        Dictionary with reinstall information:
        - backend: str
        - hardware: list
        - recommended_family: str (for pytorch)
        - recommended_provider: list (for onnx)
        - recommended_backend: str (for ncnn)
        - pip_packages: list
    """
    if hardware is None:
        hardware = detect_gpu_hardware()
    
    info = {
        "backend": backend,
        "hardware": hardware,
        "recommended_family": None,
        "recommended_provider": None,
        "recommended_backend": None,
        "pip_packages": []
    }
    
    if backend == "pytorch":
        info["recommended_family"] = get_torch_backend_family(hardware)
        info["pip_packages"] = ["torch", "torchvision"]
    elif backend == "onnx":
        info["recommended_provider"] = get_onnx_providers(hardware)
        info["pip_packages"] = ["onnxruntime"]
    elif backend == "ncnn":
        info["recommended_backend"] = get_ncnn_backend(hardware)
        info["pip_packages"] = ["ncnn", "rife-ncnn-vulkan-python-tntwise", "upscale_ncnn_py"]
    
    return info
