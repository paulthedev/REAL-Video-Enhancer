"""
GIMM (Generalized Image Motion Model) model definition.

This module provides a backend-agnostic GIMM model that can be used with
PyTorch, ONNX, or NCNN backends.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

from apps.backend.models.base import BaseInterpolateModel, ModelTask, ModelFormat
from apps.backend.models.registry import register_model


def warp(img, flow):
    """Warp image using flow field."""
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    pd = 'border'
    if img.device.type == "mps":
        pd = 'zeros'
        grid_ = grid_.clamp(-1, 1)
    return F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode=pd, align_corners=True)


def resize(x, scale_factor):
    """Resize tensor."""
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


@dataclass
class GimmConfig:
    """Configuration for GIMM model."""
    width: int = 1920
    height: int = 1080
    scale: float = 0.5  # GIMM uses lower resolution for VRAM efficiency
    ceil_interpolate_factor: int = 2
    ensemble: bool = False
    hdr_mode: bool = False


class GIMMVFI_R(nn.Module):
    """GIMM VFI R model - simplified for testing."""
    
    def __init__(self, model_path: str = None, width: int = 1920, height: int = 1080):
        super().__init__()
        self.width = width
        self.height = height
        self.scale = 0.5
        
        # Simplified flow estimator
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),
        )
        
        # AMT-style decoder
        self.amt_last_cproj = nn.Conv2d(32, 64, 1)
        self.amt_second_last_cproj = nn.Conv2d(32, 64, 1)
        self.amt_fproj = nn.Conv2d(64, 64, 1)
        
        # Flow refinement
        self.amt_update4_low = nn.Conv2d(32, 64, 3, 1, 1)
        self.amt_update4_high = nn.Conv2d(64, 2, 3, 1, 1)
        
        # Combining block
        self.amt_comb_block = nn.Sequential(
            nn.Conv2d(9, 18, 7, 1, 3),
            nn.PReLU(18),
            nn.Conv2d(18, 3, 7, 1, 3),
        )
        
        # Coord sampler
        self.coord_sampler = nn.Identity()
        
        # CNN encoder for pixel features
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
        )
        
        # Residual convolution
        self.res_conv = nn.Sequential(
            nn.Conv2d(96, 48, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(96, 64, 3, 1, 1),
        )
        
        # HypoNet for output
        self.hyponet = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        
        # Parameters
        self.alpha_v = nn.Parameter(torch.Tensor([1.0]))
        self.alpha_fe = nn.Parameter(torch.Tensor([1.0]))
        
        # Smoothing filter
        self.g_filter = nn.Parameter(
            torch.Tensor([
                [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
                [1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0],
                [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
            ]).reshape(1, 1, 3, 3),
            requires_grad=False
        )
    
    def cal_bidirection_flow(self, im0, im1, iters=20):
        """Calculate bidirectional flow."""
        # Simplified flow estimation
        combined = torch.cat([im0, im1], dim=1)
        flow01 = self.flow_estimator(combined)
        flow10 = self.flow_estimator(torch.cat([im1, im0], dim=1))
        
        return flow01, flow10
    
    def forward(self, img0, img1, timestep=0.5):
        """Forward pass for GIMM."""
        # Calculate flows
        flow01, flow10 = self.cal_bidirection_flow(img0, img1)
        
        # Warp features
        feat0 = self.cnn_encoder(img0)
        feat1 = self.cnn_encoder(img1)
        
        # Softsplat-like warping (simplified)
        warped0 = warp(feat0, flow01 * timestep)
        warped1 = warp(feat1, -flow10 * (1 - timestep))
        
        # Combine
        combined = torch.cat([warped0, warped1, feat0, feat1], dim=1)
        combined = self.res_conv(combined)
        
        # Output
        output = self.hyponet(combined)
        
        return output


class GimmModel(BaseInterpolateModel):
    """GIMM model implementation."""
    
    def __init__(self, config: GimmConfig, model_path: Optional[str] = None):
        super().__init__(name="GIMM", architecture="GIMM")
        self.config = config
        self.model_path = model_path
        self.model = None  # Will be created when load() is called
        
        # Padding for flow alignment
        _pad = 64
        tmp = max(_pad, int(_pad / config.scale))
        self.pw = math.ceil(config.width / tmp) * tmp
        self.ph = math.ceil(config.height / tmp) * tmp
        self.padding = (0, self.pw - config.width, 0, self.ph - config.height)
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to interpolate method."""
        return self.interpolate(*args, **kwargs)
    
    def interpolate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """Interpolate between two frames."""
        # Apply padding
        frame1 = F.pad(frame1, self.padding)
        frame2 = F.pad(frame2, self.padding)
        
        # Forward pass
        output = self.model(frame1, frame2, timestep)
        
        # Remove padding
        output = output[:, :, :self.config.height, :self.config.width]
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "task": ModelTask.INTERPOLATE,
            "format": ModelFormat.PT,
            "width": self.config.width,
            "height": self.config.height,
            "scale": self.config.scale,
            "ceil_interpolate_factor": self.config.ceil_interpolate_factor,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, 3, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, 3, self.config.height, self.config.width]


# Register the model (create instance for registration)
_gimm_instance = GimmModel(config=GimmConfig())
register_model(
    name="gimm",
    model=_gimm_instance,
    formats={ModelFormat.PT: "models/interpolate/gimm.pkl"},
    backends=["pytorch"],
)
