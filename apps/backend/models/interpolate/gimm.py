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
        self.backend = None
        self.onnx_loader = None
        self.ncnn_net = None
        
        # Padding for flow alignment
        _pad = 64
        tmp = max(_pad, int(_pad / config.scale))
        self.pw = math.ceil(config.width / tmp) * tmp
        self.ph = math.ceil(config.height / tmp) * tmp
        self.padding = (0, self.pw - config.width, 0, self.ph - config.height)
    
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
        """Load GIMM model using PyTorch backend."""
        import torch
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(self.device)
        self.model = GIMMVFI_R(
            model_path=self.model_path,
            width=self.config.width,
            height=self.config.height,
        ).to(device)
        
        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"Loaded GIMM PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load GIMM model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded GIMM ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load GIMM model using NCNN backend."""
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
        
        print(f"Loaded GIMM NCNN model: {param_path}")
    
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
        # Apply padding
        frame1 = F.pad(frame1, self.padding)
        frame2 = F.pad(frame2, self.padding)
        
        # Forward pass
        output = self.model(frame1, frame2, timestep)
        
        # Remove padding
        output = output[:, :, :self.config.height, :self.config.width]
        
        return output
    
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
    
    def get_output_shape(self, input_shape: tuple = None) -> List[int]:
        """Get expected output shape."""
        return [1, 3, self.config.height, self.config.width]


# Register the model (create instance for registration)
_gimm_instance = GimmModel(config=GimmConfig())
register_model(
    name="gimm",
    model=_gimm_instance,
    formats={
        ModelFormat.PT: "models/interpolate/gimm.pkl",
        ModelFormat.ONNX: "models/interpolate/gimm.onnx",
        ModelFormat.NCNN: "models/interpolate/gimm.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
