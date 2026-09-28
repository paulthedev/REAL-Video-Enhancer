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

from backend.models.base import BaseRestorationModel, ModelTask, ModelFormat
from backend.models.registry import register_model


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
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to restore method."""
        return self.restore(*args, **kwargs)
    
    def restore(self, frame: torch.Tensor) -> torch.Tensor:
        """Restore a frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model(frame)
    
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
    formats={ModelFormat.PT: "models/restoration/fbcnn.pth"},
    backends=["pytorch"],
)
