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
    formats={ModelFormat.PT: "models/upscale/tspan.pth"},
    backends=["pytorch"],
)
