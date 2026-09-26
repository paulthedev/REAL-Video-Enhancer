import requests
try:
    from .constants import HAS_NETWORK_ON_STARTUP
except ImportError:
    from constants import HAS_NETWORK_ON_STARTUP
from dataclasses import dataclass
from .ui.SettingsTab import Settings

@dataclass
class TorchVersion:
    torch_version: str
    torchvision_version: str
    cuda_version: str
    rocm_version: str
    xpu_version: str
    mps_version: str
    is_nightly: bool = False

class Torch2_15(TorchVersion):
    torch_version = "2.15.0.dev"
    torchvision_version = "0.30.0.dev"
    cuda_version = "+cu134"
    rocm_version = "+rocm10.0"
    xpu_version = "+xpu"
    mps_version = ""
    is_nightly = True

class Torch2_14(TorchVersion):
    torch_version = "2.14.0"
    torchvision_version = "0.29.0"
    cuda_version = "+cu132"
    rocm_version = "+rocm7.14"
    xpu_version = "+xpu"
    mps_version = ""

class Torch2_13(TorchVersion):
    torch_version = "2.13.0"
    torchvision_version = "0.28.0"
    cuda_version = "+cu132"
    rocm_version = ""
    xpu_version = "+xpu"
    mps_version = ""

class Torch2_12(TorchVersion):
    torch_version = "2.12.1"
    torchvision_version = "0.27.1"
    cuda_version = "+cu132"
    rocm_version = ""
    xpu_version = "+xpu"
    mps_version = ""


