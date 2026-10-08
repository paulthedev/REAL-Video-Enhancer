"""
DnCNN (Deep CNN) model definition.

This module provides a backend-agnostic DnCNN model that can be used with
PyTorch, ONNX, or NCNN backends.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from abc import abstractmethod
import os

from apps.backend.models.denoise.base import BaseDenoiseModel
from apps.backend.models.base import ModelTask, ModelFormat
from apps.backend.models.registry import register_model


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CBR"):
    """Create convolution layer with optional activation."""
    L = []
    for t in mode:
        if t == "C":
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
    return nn.Sequential(*L) if L else nn.Identity()


def sequential(*args):
    """Create sequential module."""
    if len(args) == 1:
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class DnCNN(nn.Module):
    """DnCNN model - simplified for testing."""
    
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=17, act_mode="BR", mode="DnCNN"):
        super().__init__()
        self.mode = mode
        
        assert "R" in act_mode or "L" in act_mode, "Examples of activation function: R, L, BR, BL, IR, IL"
        bias = True
        
        m_head = conv(in_nc, nc, mode="C" + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode="C" + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode="C", bias=bias)
        
        self.model = sequential(m_head, *m_body, m_tail)
    
    def forward(self, x):
        """Forward pass for DnCNN."""
        if self.mode == "DnCNN":
            n = self.model(x)
            return x - n
        else:
            return self.model(x)


@dataclass
class DnCNNConfig:
    """Configuration for DnCNN model."""
    in_nc: int = 3
    out_nc: int = 3
    nc: int = 64
    nb: int = 17
    act_mode: str = "BR"
    mode: str = "DnCNN"
    width: int = 1920
    height: int = 1080


class DnCNNModel(BaseDenoiseModel):
    """DnCNN model implementation."""
    
    def __init__(self, config: DnCNNConfig, model_path: Optional[str] = None):
        super().__init__(name="DnCNN", architecture="DnCNN")
        self.config = config
        self.model_path = model_path
        self.model = None
        self.backend = None
        self.onnx_loader = None
        self.ncnn_net = None
    
    def load(self, backend_name: str) -> None:
        """
        Load model with specified backend.
        
        Args:
            backend_name: Backend to use ("pytorch", "onnx", "ncnn")
        """
        self.backend = backend_name
        
        if backend_name == "pytorch":
            self._load_pytorch()
        elif backend_name == "onnx":
            self._load_onnx()
        elif backend_name == "ncnn":
            self._load_ncnn()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
    
    def _load_pytorch(self) -> None:
        """Load DnCNN model using PyTorch backend."""
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.model = DnCNN(
            in_nc=self.config.in_nc,
            out_nc=self.config.out_nc,
            nc=self.config.nc,
            nb=self.config.nb,
            act_mode=self.config.act_mode,
            mode=self.config.mode,
        ).to(device)
        
        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"Loaded DnCNN PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load DnCNN model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded DnCNN ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load DnCNN model using NCNN backend."""
        import os
        
        param_path = self.model_path.replace('.bin', '.param') if self.model_path.endswith('.bin') else self.model_path
        bin_path = self.model_path.replace('.param', '.bin') if self.model_path.endswith('.param') else self.model_path
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"NCNN model files not found: {param_path} or {bin_path}")
        
        import ncnn
        self.ncnn_net = ncnn.Net()
        self.ncnn_net.load_param(param_path)
        self.ncnn_net.load_model(bin_path)
        
        self.ncnn_param_path = param_path
        self.ncnn_bin_path = bin_path
        
        print(f"Loaded DnCNN NCNN model: {param_path}")
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to denoise method."""
        return self.denoise(*args, **kwargs)
    
    def denoise(self, frame: torch.Tensor) -> torch.Tensor:
        """Denoise a frame."""
        if self.backend == "onnx":
            return self._denoise_onnx(frame)
        elif self.backend == "ncnn":
            return self._denoise_ncnn(frame)
        elif self.backend == "pytorch":
            return self._denoise_pytorch(frame)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    def _denoise_pytorch(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform PyTorch denoise inference."""
        with torch.no_grad():
            return self.model(frame.to(self._device) if hasattr(self, "_device") else frame)
    
    def _denoise_onnx(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform ONNX denoise inference."""
        import numpy as np
        
        # Convert tensor to numpy (NCHW to NHWC)
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = frame_np.astype(np.float32)
        
        # Run inference
        input_name = self.onnx_inputs[0].name
        output_name = self.onnx_outputs[0].name
        
        outputs = self.onnx_loader.run(
            {input_name: frame_np},
            [output_name]
        )
        
        # Convert back to tensor (NHWC to NCHW)
        result = torch.from_numpy(outputs[0]).permute(2, 0, 1).unsqueeze(0)
        
        return result
    
    def _denoise_ncnn(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform NCNN denoise inference."""
        import ncnn  # optional dependency; imported here so the module imports without it
        import numpy as np
        
        # Convert tensor to numpy (NCHW to NHWC)
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255.0).astype(np.uint8)
        
        # Create NCNN Mat
        h, w, c = frame_np.shape
        mat = ncnn.Mat(h, w, c, frame_np.flatten())
        
        # Create extractor
        extractor = self.ncnn_net.create_extractor()
        extractor.set_num_thread(4)
        extractor.input("input.1", mat)
        
        # Run inference
        _, out_mat = extractor.extract("output.1")
        
        # Convert output to tensor
        output_data = out_mat.to_pixels()
        result = torch.from_numpy(output_data).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        return result
    
    def unload(self) -> None:
        """Unload model and free memory."""
        import gc
        
        if self.backend == "onnx" and hasattr(self, 'onnx_loader'):
            self.onnx_loader.unload()
        elif self.backend == "ncnn" and hasattr(self, 'ncnn_net'):
            self.ncnn_net = None
        elif self.backend == "pytorch":
            self.model = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "task": ModelTask.DENOISE,
            "format": ModelFormat.PT,
            "in_nc": self.config.in_nc,
            "out_nc": self.config.out_nc,
            "nc": self.config.nc,
            "nb": self.config.nb,
            "act_mode": self.config.act_mode,
            "mode": self.config.mode,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, self.config.in_nc, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, self.config.out_nc, self.config.height, self.config.width]


# Register the model (create instance for registration)
_dncnn_instance = DnCNNModel(config=DnCNNConfig())
register_model(
    name="dncnn",
    model=_dncnn_instance,
    formats={
        ModelFormat.PT: "models/denoise/dncnn.pth",
        ModelFormat.ONNX: "models/denoise/dncnn.onnx",
        ModelFormat.NCNN: "models/denoise/dncnn.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
