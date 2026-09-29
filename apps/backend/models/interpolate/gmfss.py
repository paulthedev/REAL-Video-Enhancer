"""
GMFSS (GMFlow-based Multi-scale Feature Selection and Synthesis) model definition.

This module provides a backend-agnostic GMFSS model that can be used with
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


def warp(img, flow, metric=None, strMode="soft"):
    """Warp image using flow field (simplified softsplat)."""
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
class GmfssConfig:
    """Configuration for GMFSS model."""
    width: int = 1920
    height: int = 1080
    scale: float = 1.0
    ceil_interpolate_factor: int = 2
    ensemble: bool = False
    max_timestep: float = 1.0
    hdr_mode: bool = False


class GMFSS(nn.Module):
    """GMFSS model - simplified for testing."""
    
    def __init__(self, model_path: str = None, scale: float = 1.0, width: int = 1920, height: int = 1080):
        super().__init__()
        self.scale = scale
        self.width = width
        self.height = height
        
        # Feature extraction
        self.feat_ext = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
        )
        
        # Flow estimation (simplified GMFlow)
        self.flownet = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1, 1),
        )
        
        # Metric network
        self.metricnet = nn.Sequential(
            nn.Conv2d(10, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, 1, 1),
        )
        
        # RIFE-style IFNet (simplified)
        self.ifnet = nn.Sequential(
            nn.Conv2d(12, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        
        # Fusion network
        self.fusionnet = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        
        # State variables
        self.flow01 = None
        self.flow10 = None
        self.feat11 = None
        self.feat12 = None
        self.feat13 = None
        self.metric0 = None
        self.metric1 = None
    
    def forward(self, img0, img1, timestep, scale=None):
        """Forward pass for GMFSS."""
        if scale is not None:
            self.scale = scale
        
        # Extract features
        if self.feat11 is None:
            feats = self.feat_ext(img0)
            self.feat11 = feats[:, :32, :, :]
            self.feat12 = feats[:, 32:64, :, :]
            self.feat13 = feats[:, 64:, :, :]
        
        feat2 = self.feat_ext(img1)
        feat21 = feat2[:, :32, :, :]
        feat22 = feat2[:, 32:64, :, :]
        feat23 = feat2[:, 64:, :, :]
        
        # Downsample images
        img0_small = F.interpolate(img0, scale_factor=0.5, mode="bilinear")
        img1_small = F.interpolate(img1, scale_factor=0.5, mode="bilinear")
        
        # Flow estimation
        if self.flow01 is None:
            combined = torch.cat([img0_small, img1_small], dim=1)
            self.flow01 = self.flownet(combined)
            self.flow10 = self.flownet(torch.cat([img1_small, img0_small], dim=1))
        
        # Scale flows if needed
        if self.scale != 1.0:
            self.flow01 = F.interpolate(self.flow01, scale_factor=1.0 / self.scale, mode="bilinear") / self.scale
            self.flow10 = F.interpolate(self.flow10, scale_factor=1.0 / self.scale, mode="bilinear") / self.scale
        
        # Metric estimation
        if self.metric0 is None:
            metric_input = torch.cat([img0_small, img1_small, self.flow01, self.flow10], dim=1)
            self.metric0, self.metric1 = torch.chunk(self.metricnet(metric_input), 2, dim=1)
        
        # Compute warped flows and metrics
        F1t = timestep * self.flow01
        F2t = (1 - timestep) * self.flow10
        Z1t = timestep * self.metric0
        Z2t = (1 - timestep) * self.metric1
        
        # Warp images
        I1t = warp(img0_small, F1t, Z1t)
        I2t = warp(img1_small, F2t, Z2t)
        
        # Get RIFE output
        rife = self.ifnet(torch.cat([img0_small, I1t, I2t, img1_small], dim=1))
        
        # Warp features at different scales
        feat1t1 = warp(self.feat11, F1t, Z1t)
        feat2t1 = warp(feat21, F2t, Z2t)
        
        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear") * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear")
        feat1t2 = warp(self.feat12, F1td, Z1d)
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear") * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear")
        feat2t2 = warp(feat22, F2td, Z2d)
        
        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear") * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear")
        feat1t3 = warp(self.feat13, F1tdd, Z1dd)
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear") * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear")
        feat2t3 = warp(feat23, F2tdd, Z2dd)
        
        # Fusion
        in1 = torch.cat([I1t, rife, I2t], dim=1)
        in2 = torch.cat([feat1t1, feat2t1], dim=1)
        in3 = torch.cat([feat1t2, feat2t2], dim=1)
        in4 = torch.cat([feat1t3, feat2t3], dim=1)
        
        out = self.fusionnet(torch.cat([in1, in2, in3, in4], dim=1))
        
        # Crop to original size
        out = out[:, :, :self.height, :self.width]
        
        return out


class GmfssModel(BaseInterpolateModel):
    """GMFSS model implementation."""
    
    def __init__(self, config: GmfssConfig, model_path: Optional[str] = None):
        super().__init__(name="GMFSS", architecture="GMFSS")
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
            "max_timestep": self.config.max_timestep,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, 3, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, 3, self.config.height, self.config.width]
    
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
        """Load GMFSS model using PyTorch backend."""
        import torch
        
        # Create model
        self.model = GMFSS(
            model_path=self.model_path,
            scale=self.config.scale,
            width=self.config.width,
            height=self.config.height,
        )
        
        # Load weights if provided
        if self.model_path and self.model_path.endswith('.pkl'):
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        
        self.model.eval()
        print(f"Loaded GMFSS PyTorch model: {self.model_path}")
    
    def _interpolate_pytorch(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """Perform PyTorch interpolation inference."""
        # Apply padding
        frame1 = F.pad(frame1, self.padding)
        frame2 = F.pad(frame2, self.padding)
        
        # Forward pass
        output = self.model(frame1, frame2, timestep)
        
        # Remove padding
        output = output[:, :, :self.config.height, :self.config.width]
        
        return output
    
    def _load_onnx(self) -> None:
        """Load GMFSS model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded GMFSS ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load GMFSS model using NCNN backend."""
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
        
        print(f"Loaded GMFSS NCNN model: {param_path}")
    
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
_gmfss_instance = GmfssModel(config=GmfssConfig())
register_model(
    name="gmfss",
    model=_gmfss_instance,
    formats={
        ModelFormat.PT: "models/interpolate/gmfss.pkl",
        ModelFormat.ONNX: "models/interpolate/gmfss.onnx",
        ModelFormat.NCNN: "models/interpolate/gmfss.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
