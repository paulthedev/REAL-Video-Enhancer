
from .Util import log_error, suppress_stdout_stderr

# Compute-capability floors for backend/feature availability, queried strictly
# through torch.cuda.get_device_capability (works for CUDA and ROCm, since
# ROCm exposes a CUDA-compatible interface; XPU via torch.xpu). A GPU must meet
# or exceed (major, minor) to offer the feature.
#   - PyTorch inference: cc >= 6.0 (Pascal+), ROCm, or XPU.
#   - torch.compile (Inductor): cc >= 8.0 (Ampere+ on CUDA); ROCm/XPU archs
#     all qualify. Capability is the sole gate — an arch is never disabled
#     because a trace happened to crash (that is an OOM/runtime error caught by
#     the eager fallback in UpscaleTorch._load, not a capability floor).
MINIMUM_PYTORCH_CAP = (6, 0)
MINIMUM_COMPILE_CAP = (8, 0)


class BackendDetect:
    def __init__(self):
        self.__torch = None
        self.__tensorrt = None
        self.__ncnn = None
        self.pytorch_device = None
        self.pytorch_version = None
        try:
            import torch
            import torchvision
            self.__torch = torch
            self.pytorch_device = self.__get_pytorch_device()
            self.pytorch_version = self.__torch.__version__
            try:
                with suppress_stdout_stderr():
                    import tensorrt
                    import torch_tensorrt
                self.__tensorrt = tensorrt
            except ImportError as e:
                pass
            except Exception as e:
                log_error("FATAL: " + str(e))
        except ImportError as e:
            pass
        except Exception as e:
            log_error("FATAL: " + str(e))
        try:
            from rife_ncnn_vulkan_python import Rife
            import ncnn

            try:
                from upscale_ncnn_py import UPSCALE
            except ImportError:
                log_error(
                    "Warning: Cannot import upscale_ncnn, falling back to default ncnn processing. (Please install vcredlist on your computer to fix this!)"
                )
            self.__ncnn = ncnn
        except ImportError as e:
            pass
        except Exception as e:
            log_error("FATAL: " + str(e))



    def __get_pytorch_device(self):
        if "cu" in self.__torch.__version__: return "cuda" 
        if "rocm" in self.__torch.__version__: return "rocm"
        if self.__torch.xpu.is_available(): return "xpu"
        if self.__torch.backends.mps.is_available(): return "mps"
        return "CPU"

    def get_tensorrt(self):
        if self.__tensorrt: return self.__tensorrt.__version__
    
    def get_ncnn(self):
        if self.__ncnn: return self.__ncnn.__version__

    def get_half_precision(self):
        """
        Function that checks if the torch backend supports bfloat16
        """

        try:
            x = self.__torch.tensor([1.0], dtype=self.__torch.float16).to(device="cuda" if self.pytorch_device == "rocm" else self.pytorch_device)
            return True
        except Exception as e:
            log_error(str(e))
            return False    
    
    def get_gpus_torch(self):
        """
        Function that returns a list of available GPU names using PyTorch.
        """
        
        devices = []
        
        if self.__torch:
            if self.pytorch_device == "CPU": return self.pytorch_device
            if self.pytorch_device.lower() == "mps": return [{"index": 0, "name": "Apple MPS"}]
            torch_cmd_dict = {
            "cuda": self.__torch.cuda,
            "xpu": self.__torch.xpu,
            "rocm": self.__torch.cuda,  
            }

            torch_cmd = torch_cmd_dict[self.pytorch_device]
            if torch_cmd.is_available():
                for dev_index in range(torch_cmd.device_count()):
                    props = torch_cmd.get_device_properties(dev_index)
                    devices.append(props.name)
            if not devices:
                devices.append("CPU")
       
        return devices

    def _torch_api(self):
        """Return the torch device API bound to this backend (cuda/xpu)."""
        if not self.__torch:
            return None
        if self.pytorch_device == "xpu":
            return self.__torch.xpu
        return self.__torch.cuda

    def get_device_capability(self, gpu_id: int = 0):
        """Return (major, minor) compute capability of the requested GPU via
        torch.cuda.get_device_capability. Returns (0, 0) when the backend has
        no CUDA-style device, or when the requested index is out of range."""
        api = self._torch_api()
        if api is None or self.pytorch_device == "cpu" or self.pytorch_device == "mps":
            return (0, 0)
        try:
            if not api.is_available() or gpu_id >= api.device_count():
                return (0, 0)
            return tuple(api.get_device_capability(gpu_id))
        except Exception:
            return (0, 0)

    def meets_capability(self, gpu_id: int = 0, minimum=MINIMUM_PYTORCH_CAP):
        """True if the GPU's compute capability meets the given (major, minor)
        floor. (0, 0) from get_device_capability is always treated as below any
        positive floor."""
        major, minor = self.get_device_capability(gpu_id)
        return (major, minor) >= minimum

    def pytorch_available(self, gpu_id: int = 0):
        """Whether the PyTorch upscale backend is offered on this GPU (cc >=
        MINIMUM_PYTORCH_CAP)."""
        return self.meets_capability(gpu_id, MINIMUM_PYTORCH_CAP)

    def compile_available(self, gpu_id: int = 0):
        """Whether torch.compile is offered on this GPU (cc >= MINIMUM_COMPILE_CAP)."""
        return self.meets_capability(gpu_id, MINIMUM_COMPILE_CAP)

    def get_gpus_ncnn(self):
        if self.__ncnn:
            from ..constants import PLATFORM
            if PLATFORM == "win32":
                # this is to prevent ncnn from creating a crashdump file on windows, despite working.
                # Dont know the side effects of this, but if there are thats for a later me to figure out.
                try:
                    import ctypes
                    SEM_NOGPFAULTERRORBOX = 0x0002
                    SEM_FAILCRITICALERRORS = 0x0001

                    ctypes.windll.kernel32.SetErrorMode(
                        SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
                    )
                except Exception as e:
                    log_error(str(e))
            devices = []
            try:
                with suppress_stdout_stderr():

                    gpu_count = self.__ncnn.get_gpu_count()
                    if gpu_count < 1:
                        return ["CPU"]
                    for i in range(gpu_count):
                        device = self.__ncnn.get_gpu_device(i)
                        gpu_info = device.info()
                        devices.append(gpu_info.device_name())
                return devices
            except Exception:
                return ["CPU"]
            except Exception as e:
                log_error(str(e))
                return "Unable to get NCNN GPU"