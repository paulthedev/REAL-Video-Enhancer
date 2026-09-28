"""
SPAN (Spatially-Adaptive Pyramidal Network) model definition.

This module provides a backend-agnostic SPAN model that can be used with
PyTorch, ONNX, or NCNN backends.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from apps.backend.models.base import BaseUpscaleModel, ModelTask, ModelFormat
from apps.backend.models.registry import register_model


class ResidualBlock(nn.Module):
    """Residual block for SPAN."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class SPAN(nn.Module):
    """SPAN model - simplified for testing."""
    
    def __init__(self, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        
        # Simplified architecture
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 64, 3, 1, 1)
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
        )
        self.conv_final = nn.Conv2d(64, 3, 3, 1, 1)
    
    def forward(self, x):
        """Forward pass for SPAN."""
        identity = x
        out = self.conv_first(x)
        out = self.body(out)
        out = self.conv_hr(out)
        out = self.conv_last(out)
        out = self.pixel_shuffle(out)
        out = self.conv_final(out)
        # Resize identity to match output size
        identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + identity


@dataclass
class SPANConfig:
    """Configuration for SPAN model."""
    scale: int = 4
    width: int = 1920
    height: int = 1080


class SPANModel(BaseUpscaleModel):
    """SPAN model implementation."""
    
    def __init__(self, config: SPANConfig, model_path: Optional[str] = None):
        super().__init__(name="SPAN", architecture="SPAN", scale=config.scale)
        self.config = config
        self.model_path = model_path
        self.model = None
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to upscale method."""
        return self.upscale(*args, **kwargs)
    
    def upscale(self, frame: torch.Tensor) -> torch.Tensor:
        """Upscale a frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model(frame)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "task": ModelTask.UPSCALE,
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
        return [1, 3, self.config.height * self.config.scale, self.config.width * self.config.scale]


# Register the model (create instance for registration)
_span_instance = SPANModel(config=SPANConfig())
register_model(
    name="span",
    model=_span_instance,
    formats={ModelFormat.PT: "models/upscale/span.pth"},
    backends=["pytorch"],
)
