"""
TSPAN (Temporal SPAN) model definition.

This module provides a backend-agnostic TSPAN model that can be used with
PyTorch, ONNX, or NCNN backends.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import os

from apps.backend.models.base import BaseUpscaleModel, ModelTask, ModelFormat
from apps.backend.models.registry import register_model


def conv_layer(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Create convolution layer with adaptive padding."""
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def pixelshuffle_block(in_channels: int, out_channels: int, upscale_factor: int = 2, kernel_size: int = 3) -> nn.Module:
    """Create pixel shuffle block for upsampling."""
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    """3x3 convolution with optional 1x1 skip connection."""
    
    def __init__(self, c_in: int, c_out: int, gain1: int = 1, s: int = 1, bias: bool = True):
        super().__init__()
        self.stride = s
        gain = gain1
        
        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, padding=0 if s == 1 else None, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, 1, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, 3, stride=s, padding=1, bias=bias),
            nn.Conv2d(c_out * gain, c_out, 1, bias=bias),
        )
    
    def forward(self, x):
        return self.sk(x) + self.conv(x)


class SPAB(nn.Module):
    """Spatially Adaptive Block for TSPAN."""
    
    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv1 = conv_layer(c, c, 3)
        self.conv2 = conv_layer(c, c, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dilation = dilation
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        # Apply dilation to second conv
        if self.dilation > 1:
            out = F.pad(out, (self.dilation - 1, self.dilation - 1, self.dilation - 1, self.dilation - 1))
            out = self.conv2(out)
        else:
            out = self.conv2(out)
        return x + out


class TemporalSPAN(nn.Module):
    """TSPAN model - simplified for testing."""
    
    def __init__(self, upscale: int = 2):
        super().__init__()
        self.upscale = upscale
        
        # Simplified architecture
        self.conv_first = conv_layer(3, 64, 3)
        self.body = nn.Sequential(
            *[SPAB(64) for _ in range(16)]
        )
        self.conv_hr = conv_layer(64, 64, 3)
        self.conv_last = conv_layer(64, 64, 3)
        self.pixel_shuffle = pixelshuffle_block(64, 64, upscale_factor=upscale)
        self.conv_final = conv_layer(64, 3, 3)
    
    def forward(self, x):
        """Forward pass for TSPAN."""
        residual = x
        out = self.conv_first(x)
        out = self.body(out)
        out = self.conv_hr(out)
        out = self.conv_last(out)
        out = self.pixel_shuffle(out)
        out = self.conv_final(out)
        return out + residual


@dataclass
class TSPANConfig:
    """Configuration for TSPAN model."""
    scale: int = 2
    width: int = 1920
    height: int = 1080
    num_frames: int = 5  # TSPAN uses multiple frames


class TSPANModel(BaseUpscaleModel):
    """TSPAN model implementation."""
    
    def __init__(self, config: TSPANConfig, model_path: Optional[str] = None):
        super().__init__(name="TSPAN", architecture="TSPAN", scale=config.scale)
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
        """Load TSPAN model using PyTorch backend."""
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.model = TemporalSPAN(upscale=self.config.scale).to(device)
        
        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"Loaded TSPAN PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load TSPAN model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded TSPAN ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load TSPAN model using NCNN backend."""
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
        
        print(f"Loaded TSPAN NCNN model: {param_path}")
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to upscale method."""
        return self.upscale(*args, **kwargs)
    
    def upscale(self, frame: torch.Tensor) -> torch.Tensor:
        """Upscale a frame."""
        if self.backend == "onnx":
            return self._upscale_onnx(frame)
        elif self.backend == "ncnn":
            return self._upscale_ncnn(frame)
        elif self.backend == "pytorch":
            return self._upscale_pytorch(frame)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    def _upscale_pytorch(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform PyTorch upscale inference."""
        with torch.no_grad():
            return self.model(frame.to(self._device) if hasattr(self, "_device") else frame)
    
    def _upscale_onnx(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform ONNX upscale inference."""
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
    
    def _upscale_ncnn(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform NCNN upscale inference."""
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
            "task": ModelTask.UPSCALE,
            "format": ModelFormat.PT,
            "scale": self.config.scale,
            "width": self.config.width,
            "height": self.config.height,
            "num_frames": self.config.num_frames,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, 3, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, 3, self.config.height * self.config.scale, self.config.width * self.config.scale]


# Register the model (create instance for registration)
_tspan_instance = TSPANModel(config=TSPANConfig())
register_model(
    name="tspan",
    model=_tspan_instance,
    formats={
        ModelFormat.PT: "models/upscale/tspan.pth",
        ModelFormat.ONNX: "models/upscale/tspan.onnx",
        ModelFormat.NCNN: "models/upscale/tspan.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
