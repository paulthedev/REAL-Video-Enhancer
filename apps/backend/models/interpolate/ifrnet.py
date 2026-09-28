"""
IFRNet (Implicit Flow-free Residual Network) model definition.

This module provides a backend-agnostic IFRNet model that can be used with
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


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    """Convolution + PReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )


class ResBlock(nn.Module):
    """Residual block for IFRNet."""
    
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    """Encoder for IFRNet."""
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1),
            convrelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1),
            convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1),
            convrelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1),
            convrelu(96, 96, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    """Decoder for pyramid level 4."""
    
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(192+1, 192),
            ResBlock(192, 32),
            nn.ConvTranspose2d(192, 76, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        x = torch.cat((f0, f1, embt), 1)
        x = self.convblock(x)
        return x


class Decoder3(nn.Module):
    """Decoder for pyramid level 3."""
    
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(144+1, 144),
            ResBlock(144, 48),
            nn.ConvTranspose2d(144, 40, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        x = torch.cat((f0, f1, embt), 1)
        x = self.convblock(x)
        return x


class Decoder2(nn.Module):
    """Decoder for pyramid level 2."""
    
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(96+1, 96),
            ResBlock(96, 48),
            nn.ConvTranspose2d(96, 24, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        x = torch.cat((f0, f1, embt), 1)
        x = self.convblock(x)
        return x


class Decoder1(nn.Module):
    """Decoder for pyramid level 1."""
    
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(64+1, 64),
            ResBlock(64, 32),
            nn.ConvTranspose2d(64, 16, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        x = torch.cat((f0, f1, embt), 1)
        x = self.convblock(x)
        return x


class FlowHead(nn.Module):
    """Flow head for IFRNet."""
    
    def __init__(self, in_channels):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 2, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class ChulFlowHead(nn.Module):
    """Channel-wise flow head for IFRNet."""
    
    def __init__(self, in_channels):
        super(ChulFlowHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 2, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class IFRNet(nn.Module):
    """IFRNet (Implicit Flow-free Residual Network) for video interpolation."""
    
    def __init__(self, scale_factor: float = 1.0):
        super(IFRNet, self).__init__()
        self.scale_factor = scale_factor
        
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.flow_head = FlowHead(192)
        self.chul_flow_head = ChulFlowHead(192)
        
        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.PReLU(16),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # Refine head
        self.refine_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.PReLU(16),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for IFRNet.
        
        Args:
            img0: First frame (B, C, H, W)
            img1: Second frame (B, C, H, W)
            timestep: Timestep tensor (B, 1, 1, 1)
            
        Returns:
            Interpolated frame (B, C, H, W)
        """
        # Encode
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        
        # Embed timestep
        embt = timestep.expand(-1, -1, img0.size(2), img0.size(3))
        
        # Decoder 4
        x4 = self.decoder4(f0_4, f1_4, embt)
        flow4 = self.flow_head(x4)
        
        # Warp and decode
        f0_3_warped = warp(f0_3, resize(flow4, 0.5))
        f1_3_warped = warp(f1_3, resize(flow4, 0.5))
        x3 = self.decoder3(torch.cat([f0_3_warped, f1_3_warped], dim=1), embt)
        flow3 = self.flow_head(x3) + resize(flow4, 0.5)
        
        # Warp and decode
        f0_2_warped = warp(f0_2, resize(flow3, 0.5))
        f1_2_warped = warp(f1_2, resize(flow3, 0.5))
        x2 = self.decoder2(torch.cat([f0_2_warped, f1_2_warped], dim=1), embt)
        flow2 = self.flow_head(x2) + resize(flow3, 0.5)
        
        # Warp and decode
        f0_1_warped = warp(f0_1, resize(flow2, 0.5))
        f1_1_warped = warp(f1_1, resize(flow2, 0.5))
        x1 = self.decoder1(torch.cat([f0_1_warped, f1_1_warped], dim=1), embt)
        flow1 = self.flow_head(x1) + resize(flow2, 0.5)
        
        # Generate mask and refine
        mask = self.mask_head(x1)
        img0_warped = warp(img0, flow1)
        img1_warped = warp(img1, resize(flow1, 2.0))
        
        # Apply mask
        result = mask * img0_warped + (1 - mask) * img1_warped
        
        # Refine
        refine = self.refine_head(x1)
        result = result + refine
        
        return result


@dataclass
class IFRNetConfig:
    """Configuration for IFRNet model."""
    scale: float = 1.0
    ensemble: bool = False
    width: int = 1920
    height: int = 1080
    ceil_interpolate_factor: int = 2


class IFRNetModel(BaseInterpolateModel):
    """
    IFRNet model for video interpolation.
    
    This model uses implicit flow estimation to interpolate between frames.
    """
    
    def __init__(
        self,
        config: IFRNetConfig,
        model_path: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "fp32",
    ):
        super().__init__("IFRNet", "ifrnet")
        self.config = config
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.stream = None
        self.prepare_stream = None
        
    def load(self, backend: str = "pytorch") -> None:
        """Load the model weights."""
        if backend != "pytorch":
            raise ValueError(f"IFRNet only supports PyTorch backend, got {backend}")
        
        if self.model is None:
            self.model = IFRNet(scale_factor=self.config.scale)
        
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
        
        # Move to device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(self.device)
        dtype_obj = torch.float16 if self.dtype == "fp16" else torch.float32
        self.model.eval().to(device=device_obj, dtype=dtype_obj)
        
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to interpolate method."""
        return self.interpolate(*args, **kwargs)
    
    def interpolate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between two frames.
        
        Args:
            frame1: First frame (C, H, W)
            frame2: Second frame (C, H, W)
            timestep: Interpolation timestep (0.0 to 1.0)
            
        Returns:
            Interpolated frame (C, H, W)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Add batch dimension
        frame1_batch = frame1.unsqueeze(0)
        frame2_batch = frame2.unsqueeze(0)
        
        # Create timestep tensor
        timestep_tensor = torch.tensor([[timestep]], dtype=self.model.dtype, device=self.model.device)
        timestep_tensor = timestep_tensor.view(1, 1, 1, 1)
        
        # Run inference
        with torch.no_grad():
            result = self.model(frame1_batch, frame2_batch, timestep_tensor)
        
        # Remove batch dimension
        return result.squeeze(0)
    
    def get_input_shape(self) -> tuple:
        """Return the expected input shape."""
        return (3, self.config.height, self.config.width)
    
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Return the output shape."""
        return input_shape
    
    def get_info(self) -> dict:
        """Return model metadata."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "task": "interpolate",
            "scale": self.config.scale,
            "supported_formats": ["pt", "pkl"],
            "supported_backends": ["pytorch"],
        }


# Register the model (create instance for registration)
_ifrnet_instance = IFRNetModel(config=IFRNetConfig())
register_model(
    name="ifrnet",
    model=_ifrnet_instance,
    formats={ModelFormat.PT: "models/interpolate/ifrnet.pkl"},
    backends=["pytorch"],
)
