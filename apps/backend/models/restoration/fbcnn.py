"""
FBCNN (Fully Convolutional Basic Network) model definition.

This module provides a backend-agnostic FBCNN model that can be used with
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

from apps.backend.models.base import BaseRestorationModel, ModelTask, ModelFormat
from apps.backend.models.registry import register_model


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CBR"):
    """Create convolution layer with optional activation."""
    L = []
    for t in mode:
        if t == "C":
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
    return nn.Sequential(*L) if L else nn.Identity()


class ResBlock(nn.Module):
    """Residual block for FBCNN."""
    
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size, padding=padding, mode="C")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, in_channels, kernel_size, padding=padding, mode="C")
    
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class QFAttention(nn.Module):
    """Quantization-aware Feature attention for FBCNN."""
    
    def __init__(self, channels=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x * self.attention(x)


class FBCNN(nn.Module):
    """FBCNN model - simplified for testing."""
    
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale
        
        # Simplified architecture
        self.conv_first = conv(3, 64, 3, padding=1, mode="C")
        self.body = nn.Sequential(
            *[ResBlock(64) for _ in range(16)]
        )
        self.conv_hr = conv(64, 64, 3, padding=1, mode="C")
        self.conv_last = conv(64, 3, 3, padding=1, mode="C")
        self.attention = QFAttention(64)
    
    def forward(self, x):
        """Forward pass for FBCNN."""
        residual = x
        out = self.conv_first(x)
        out = self.body(out)
        out = self.conv_hr(out)
        out = self.attention(out)
        out = self.conv_last(out)
        return out + residual


@dataclass
class FBCNNConfig:
    """Configuration for FBCNN model."""
    scale: int = 1
    width: int = 1920
    height: int = 1080


class FBCNNModel(BaseRestorationModel):
    """FBCNN model implementation."""
    
    def __init__(self, config: FBCNNConfig, model_path: Optional[str] = None):
        super().__init__(name="FBCNN", architecture="FBCNN")
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
        """Load FBCNN model using PyTorch backend."""
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.model = FBCNN(scale=self.config.scale).to(device)
        
        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"Loaded FBCNN PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load FBCNN model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded FBCNN ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load FBCNN model using NCNN backend."""
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
        
        print(f"Loaded FBCNN NCNN model: {param_path}")
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to restore method."""
        return self.restore(*args, **kwargs)
    
    def restore(self, frame: torch.Tensor) -> torch.Tensor:
        """Restore a frame."""
        if self.backend == "onnx":
            return self._restore_onnx(frame)
        elif self.backend == "ncnn":
            return self._restore_ncnn(frame)
        elif self.backend == "pytorch":
            return self._restore_pytorch(frame)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    def _restore_pytorch(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform PyTorch restoration inference."""
        with torch.no_grad():
            return self.model(frame.to(self._device) if hasattr(self, "_device") else frame)
    
    def _restore_onnx(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform ONNX restoration inference."""
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
    
    def _restore_ncnn(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform NCNN restoration inference."""
        import ncnn  # optional dependency; imported here so the module imports without it
        import numpy as np
        
        # Convert tensor to numpy (NCHW to NHWC)
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255.0).astype(np.uint8)
        
        # Create NCNN Mat
        h, w, c = frame_np.shape
        mat = ncnn.Mat.from_pixels(frame_np, ncnn.Mat.PixelType.PIXEL_RGB, w, h)
        
        # Create extractor
        extractor = self.ncnn_net.create_extractor()
        extractor.set_num_thread(4)
        extractor.input("input.1", mat)
        
        # Run inference
        _, out_mat = extractor.extract("output.1")
        
        # Convert output to tensor
        output_data = out_mat.numpy()  # CHW float
        result = torch.from_numpy(output_data).unsqueeze(0).clamp(0.0, 1.0)
        
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
            "task": ModelTask.RESTORATION,
            "format": ModelFormat.PT,
            "scale": self.config.scale,
            "width": self.config.width,
            "height": self.config.height,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, 3, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, 3, self.config.height, self.config.width]


# Register the model (create instance for registration)
_fbcnn_instance = FBCNNModel(config=FBCNNConfig())
register_model(
    name="fbcnn",
    model=_fbcnn_instance,
    formats={
        ModelFormat.PT: "models/restoration/fbcnn.pth",
        ModelFormat.ONNX: "models/restoration/fbcnn.onnx",
        ModelFormat.NCNN: "models/restoration/fbcnn.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
