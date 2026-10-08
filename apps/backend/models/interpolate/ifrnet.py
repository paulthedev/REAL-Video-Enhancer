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
        self.backend = None
        self.onnx_loader = None
        self.ncnn_net = None
        self.stream = None
        self.prepare_stream = None
        
    def load(self, backend: str = "pytorch") -> None:
        """
        Load the model weights.
        
        Args:
            backend: Backend to use ("pytorch", "onnx", "ncnn")
        """
        self.backend = backend
        
        if backend == "pytorch":
            self._load_pytorch()
        elif backend == "onnx":
            self._load_onnx()
        elif backend == "ncnn":
            self._load_ncnn()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _load_pytorch(self) -> None:
        """Load IFRNet model using PyTorch backend."""
        import torch
        
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
        self._device_obj = device_obj
        self._dtype_obj = dtype_obj
        self.model.eval().to(device=device_obj, dtype=dtype_obj)
        
        print(f"Loaded IFRNet PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load IFRNet model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded IFRNet ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load IFRNet model using NCNN backend."""
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
        
        print(f"Loaded IFRNet NCNN model: {param_path}")
    
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
        if self.backend == "onnx":
            return self._interpolate_onnx(frame1, frame2, timestep)
        elif self.backend == "ncnn":
            return self._interpolate_ncnn(frame1, frame2, timestep)
        elif self.backend == "pytorch":
            return self._interpolate_pytorch(frame1, frame2, timestep)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    # backends/pytorch runner dispatches through model.infer() (RIFE-style name)
    infer = interpolate

    def _interpolate_pytorch(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """Perform PyTorch interpolation inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Add batch dimension
        frame1_batch = frame1.unsqueeze(0)
        frame2_batch = frame2.unsqueeze(0)
        
        # Create timestep tensor
        timestep_tensor = torch.tensor(
            [[timestep]], dtype=self._dtype_obj, device=self._device_obj
        )
        timestep_tensor = timestep_tensor.view(1, 1, 1, 1)
        
        # Run inference
        with torch.no_grad():
            result = self.model(frame1_batch, frame2_batch, timestep_tensor)
        
        # Remove batch dimension
        return result.squeeze(0)
    
    def _interpolate_onnx(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """Perform ONNX interpolation inference."""
        import numpy as np
        
        # Convert tensors to numpy (NCHW to NHWC)
        frame1_np = frame1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        frame2_np = frame2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        
        # Create timestep array
        timestep_np = np.array([[[[timestep]]]], dtype=np.float32)
        
        # Run inference
        input_name0 = self.onnx_inputs[0].name
        input_name1 = self.onnx_inputs[1].name
        input_name2 = self.onnx_inputs[2].name
        output_name = self.onnx_outputs[0].name
        
        outputs = self.onnx_loader.run(
            {input_name0: frame1_np, input_name1: frame2_np, input_name2: timestep_np},
            [output_name]
        )
        
        # Convert back to tensor (NHWC to NCHW)
        result = torch.from_numpy(outputs[0]).permute(2, 0, 1).unsqueeze(0)
        
        return result.squeeze(0)
    
    def _interpolate_ncnn(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """Perform NCNN interpolation inference."""
        import numpy as np
        
        # Convert tensors to numpy (NCHW to NHWC)
        frame1_np = frame1.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame2_np = frame2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame1_np = (frame1_np * 255.0).astype(np.uint8)
        frame2_np = (frame2_np * 255.0).astype(np.uint8)
        
        # Create NCNN Mats
        h, w, c = frame1_np.shape
        mat1 = ncnn.Mat(h, w, c, frame1_np.flatten())
        mat2 = ncnn.Mat(h, w, c, frame2_np.flatten())
        
        # Create extractor
        extractor = self.ncnn_net.create_extractor()
        extractor.set_num_thread(4)
        extractor.input("input.1", mat1)
        extractor.input("input.2", mat2)
        
        # Run inference
        _, out_mat = extractor.extract("output.1")
        
        # Convert output to tensor
        output_data = out_mat.to_pixels()
        result = torch.from_numpy(output_data).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        return result.squeeze(0)
    
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
            "supported_formats": ["pt", "pkl", "onnx", "ncnn"],
            "supported_backends": ["pytorch", "onnx", "ncnn"],
        }


# Register the model (create instance for registration)
_ifrnet_instance = IFRNetModel(config=IFRNetConfig())
register_model(
    name="ifrnet",
    model=_ifrnet_instance,
    formats={
        ModelFormat.PT: "models/interpolate/ifrnet.pkl",
        ModelFormat.ONNX: "models/interpolate/ifrnet.onnx",
        ModelFormat.NCNN: "models/interpolate/ifrnet.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
