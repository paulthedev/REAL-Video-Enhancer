"""
AnimeSR model definition.

This module provides a backend-agnostic AnimeSR model that can be used with
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


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN."""
    
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)
    
    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class MyPixelShuffle(nn.Module):
    """Pixel shuffle for upsampling."""
    
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        b, c, hh, hw = x.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = x.view(b, out_channel, self.upscale_factor, self.upscale_factor, hh, hw)
        return x_view.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channel, h, w)


class AnimeSR(nn.Module):
    """AnimeSR model - simplified for testing."""
    
    def __init__(self, upscale=4):
        super().__init__()
        self.upscale = upscale
        
        # Simplified architecture
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlockNoBN(64) for _ in range(16)]
        )
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 64, 3, 1, 1)
        self.pixel_shuffle = MyPixelShuffle(upscale)
        self.conv_final = nn.Conv2d(64, 3, 3, 1, 1)
    
    def forward(self, x):
        """Forward pass for AnimeSR."""
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
class AnimeSRConfig:
    """Configuration for AnimeSR model."""
    scale: int = 4
    width: int = 1920
    height: int = 1080


class AnimeSRModel(BaseUpscaleModel):
    """AnimeSR model implementation."""
    
    def __init__(self, config: AnimeSRConfig, model_path: Optional[str] = None):
        super().__init__(name="AnimeSR", architecture="AnimeSR", scale=config.scale)
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
_animesr_instance = AnimeSRModel(config=AnimeSRConfig())
register_model(
    name="animesr",
    model=_animesr_instance,
    formats={ModelFormat.PT: "models/upscale/animesr.pth"},
    backends=["pytorch"],
)
