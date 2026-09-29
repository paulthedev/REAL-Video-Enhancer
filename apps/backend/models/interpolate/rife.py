"""
RIFE (Real-Time Intermediate Flow Estimation) model definition.

This module provides a backend-agnostic RIFE model that can be used with
PyTorch, ONNX, or NCNN backends.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import os
import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod

from apps.backend.models.base import BaseInterpolateModel, ModelTask, ModelFormat
from apps.backend.models.registry import register_model


class RifeVersion(Enum):
    """RIFE model versions."""
    RIFE413 = "rife413"
    RIFE420 = "rife420"
    RIFE421 = "rife421"
    RIFE422_LITE = "rife422_lite"
    RIFE425 = "rife425"
    RIFE425_HEAVY = "rife425_heavy"
    RIFE46 = "rife46"
    RIFE47 = "rife47"


@dataclass
class RifeConfig:
    """Configuration for RIFE model."""
    version: RifeVersion
    scale: float = 1.0
    ensemble: bool = False
    dynamic_scaled_optical_flow: bool = False
    drba: bool = False
    drba_model_path: Optional[str] = None
    drba_times: int = 2
    drba_dst_fps: int = 48
    drba_fps: int = 24
    hdr_mode: bool = False
    width: int = 1920
    height: int = 1080
    ceil_interpolate_factor: int = 2


class RifeHead(nn.Module):
    """Feature extraction head for RIFE."""
    
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x = x.clamp(0.0, 1.0)
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class RifeResConv(nn.Module):
    """Residual convolution block for RIFE."""
    
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class RifeIFBlock(nn.Module):
    """Intermediate Flow Block for RIFE."""
    
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, c // 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(c // 2, c, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.convblock = nn.Sequential(*[RifeResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        from torch.nn.functional import interpolate
        x = interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class RifeIFNet(nn.Module):
    """RIFE Interpolation Network."""
    
    def __init__(
        self,
        scale: float = 1.0,
        ensemble: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.scale = scale
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        
        # Block configurations vary by version
        self.block0 = RifeIFBlock(7 + 8, c=192)
        self.block1 = RifeIFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = RifeIFBlock(8 + 4 + 8 + 8, c=64)
        self.block3 = RifeIFBlock(8 + 4 + 8 + 8, c=32)
        
        self.encode = None
        self.blocks = [self.block0, self.block1, self.block2, self.block3]
        
    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid, f0, f1, scale=None):
        """Forward pass for RIFE interpolation."""
        from apps.backend.pytorch.InterpolateArchs.RIFE.warplayer import warp
        from torch.nn.functional import interpolate
        
        img0 = img0.clamp(0., 1.)
        img1 = img1.clamp(0., 1.)
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        
        if scale is not None:
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
            
        for i in range(4):
            if flow is None:
                flow, mask, feat = self.blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=self.scaleList[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
                wf1 = warp(f1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
                flow, mask, feat = self.blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], wf0, wf1, timestep), 1),
                    flow,
                    scale=self.scaleList[i],
                )
                
            f0 = warp(f0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            f1 = warp(f1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
            flow = flow + torch.cat(
                [
                    (tenHorizontal := torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=self.device)
                     .view(1, 1, 1, self.pw).expand(-1, -1, self.ph, -1)),
                    (tenVertical := torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=self.device)
                     .view(1, 1, self.ph, 1).expand(-1, -1, -1, self.pw)),
                ], 1
            ) * flow[:, :2]
            
        return flow, mask, feat


class RifeModel(BaseInterpolateModel):
    """
    Backend-agnostic RIFE model implementation.
    
    This model can be used with PyTorch, ONNX, or NCNN backends.
    The actual inference is delegated to the backend.
    """
    
    def __init__(
        self,
        config: RifeConfig,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        gpu_id: int = 0,
    ):
        """
        Initialize RIFE model.
        
        Args:
            config: RIFE configuration
            model_path: Path to model weights
            device: Device to run on ("auto", "cuda", "cpu", etc.)
            dtype: Data type ("auto", "fp16", "fp32")
            gpu_id: GPU ID for multi-GPU systems
        """
        super().__init__(
            name="RIFE",
            architecture="RIFE",
        )
        
        self.model_path = model_path
        self.device_type = device
        self.dtype_str = dtype
        self.gpu_id = gpu_id
        self.backend = None  # Set by backend wrapper
        self.model = None  # Actual model instance
        
        # RIFE-specific attributes
        self.width = config.width
        self.height = config.height
        self.ceil_interpolate_factor = config.ceil_interpolate_factor
        self.scale = config.scale
        self.ensemble = config.ensemble
        self.dynamic_scaled_optical_flow = config.dynamic_scaled_optical_flow
        self.hdr_mode = config.hdr_mode
        
        # Padding calculation
        self._pad = 32 if config.version not in [RifeVersion.RIFE425, RifeVersion.RIFE425_HEAVY] else 64
        tmp = max(self._pad, int(self._pad / self.scale))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        
        # Pre-computed tensors
        self.timestep_dict: Dict[float, torch.Tensor] = {}
        self.tenFlow_div: Optional[torch.Tensor] = None
        self.backwarp_tenGrid: Optional[torch.Tensor] = None
        
    def load(self, backend_name: str, use_tensorrt: bool = False) -> None:
        """
        Load model with specified backend.
        
        Args:
            backend_name: Backend to use ("pytorch", "onnx", "ncnn")
            use_tensorrt: Whether to use TensorRT optimization (PyTorch backend only)
        """
        self.backend = backend_name
        self.use_tensorrt = use_tensorrt
        
        if backend_name == "pytorch":
            self._load_pytorch()
        elif backend_name == "onnx":
            self._load_onnx()
        elif backend_name == "ncnn":
            self._load_ncnn()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
    
    def _load_pytorch(self) -> None:
        """Load RIFE model using PyTorch backend."""
        from apps.backend.pytorch.InterpolateArchs.RIFE.warplayer import warp
        from apps.backend.pytorch.TorchUtils import TorchUtils
        
        device = TorchUtils.handle_device(self.device_type, gpu_id=self.gpu_id)
        dtype = TorchUtils.handle_precision(self.dtype_str)
        
        # Detect architecture version
        from apps.backend.pytorch.InterpolateArchs.DetectInterpolateArch import ArchDetect
        ad = ArchDetect(self.model_path)
        arch_name = ad.getArchName().lower()
        
        # Import appropriate architecture
        arch_map = {
            "rife46": "apps.backend.pytorch.InterpolateArchs.RIFE.rife46IFNET",
            "rife47": "apps.backend.pytorch.InterpolateArchs.RIFE.rife47IFNET",
            "rife413": "apps.backend.pytorch.InterpolateArchs.RIFE.rife413IFNET",
            "rife420": "apps.backend.pytorch.InterpolateArchs.RIFE.rife420IFNET",
            "rife421": "apps.backend.pytorch.InterpolateArchs.RIFE.rife421IFNET",
            "rife422lite": "apps.backend.pytorch.InterpolateArchs.RIFE.rife422_liteIFNET",
            "rife425": "apps.backend.pytorch.InterpolateArchs.RIFE.rife425IFNET",
            "rife425_heavy": "apps.backend.pytorch.InterpolateArchs.RIFE.rife425_heavyIFNET",
        }
        
        if arch_name not in arch_map:
            raise ValueError(f"Unknown RIFE architecture: {arch_name}")
        
        # Dynamic import
        import importlib
        module = importlib.import_module(arch_map[arch_name])
        IFNet = module.IFNet
        Head = getattr(module, 'Head', None)
        
        # Create model
        self.flownet = IFNet(
            scale=self.scale,
            ensemble=self.ensemble,
            dtype=dtype,
            device=device,
        )
        
        # Load encode head if present
        self.encode = None
        if Head and arch_name in ["rife413", "rife420", "rife421", "rife422lite", "rife425", "rife425_heavy"]:
            self.encode = Head()
        
        # Load state dict
        state_dict = torch.load(
            self.model_path,
            map_location=device,
            weights_only=True,
            mmap=True,
        )
        
        # Remove module. prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load encode head
        if self.encode:
            head_state_dict = {k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k}
            self.encode.load_state_dict(head_state_dict, strict=True)
            self.encode.eval().to(device=device, dtype=dtype)
        
        # Load flownet
        self.flownet.load_state_dict(state_dict=state_dict, strict=False)
        self.flownet.eval().to(device=device, dtype=dtype)
        
        # Initialize utilities
        self.torchUtils = TorchUtils(
            width=self.width,
            height=self.height,
            hdr_mode=self.hdr_mode,
            device_type=self.device_type,
        )
        
        # Create streams
        self.stream = self.torchUtils.init_stream()
        self.prepareStream = self.torchUtils.init_stream()
        self.copyStream = self.torchUtils.init_stream()
        self.f2tStream = self.torchUtils.init_stream()
        
        # Pre-compute timestep tensors
        for n in range(self.ceil_interpolate_factor):
            timestep = n / self.ceil_interpolate_factor
            timestep_tens = torch.full(
                (1, 1, self.ph, self.pw),
                timestep,
                dtype=dtype,
                device=device,
            )
            self.timestep_dict[timestep] = timestep_tens
        
        # Pre-compute flow division and backward warp grid
        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=torch.float32,
            device=device,
        )
        
        tenHorizontal = torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, self.pw).expand(-1, -1, self.ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=device)
        tenVertical = tenVertical.view(1, 1, self.ph, 1).expand(-1, -1, -1, self.pw)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)
        
        # DRBA setup
        if self.config.drba:
            from apps.backend.pytorch.DRBA.infer import DRBA_RVE
            self.drba = DRBA_RVE(
                model_type="rife",
                model_path=self.config.drba_model_path or "./flownet.pkl",
                times=self.config.drba_times,
                dst_fps=self.config.drba_dst_fps,
                fps=self.config.drba_fps,
                scale=1,
            )
    
    def _load_onnx(self) -> None:
        """Load RIFE model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        # Load ONNX model using reusable loader
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        # Store model info
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        self.onnx_width = self.width
        self.onnx_height = self.height
        
        print(f"Loaded RIFE ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load RIFE model using NCNN backend."""
        import numpy as np
        
        # Check if ncnn is available
        try:
            import ncnn
        except ImportError:
            raise ImportError("NCNN package not installed. Install with: pip install ncnn")
        
        # Check if model files exist
        param_path = self.model_path.replace('.bin', '.param') if self.model_path.endswith('.bin') else self.model_path
        bin_path = self.model_path.replace('.param', '.bin') if self.model_path.endswith('.param') else self.model_path
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"NCNN model files not found: {param_path} or {bin_path}")
        
        # Load NCNN model
        self.ncnn_net = ncnn.Net()
        self.ncnn_net.load_param(param_path)
        self.ncnn_net.load_model(bin_path)
        
        # Store model info
        self.ncnn_param_path = param_path
        self.ncnn_bin_path = bin_path
        self.ncnn_width = self.width
        self.ncnn_height = self.height
        
        print(f"Loaded RIFE NCNN model: {param_path}")
    
    def infer(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """
        Perform interpolation inference.
        
        Args:
            img0: First frame tensor [B, C, H, W]
            img1: Second frame tensor [B, C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)
            
        Returns:
            Interpolated frame tensor [B, C, H, W]
        """
        if self.backend == "onnx":
            return self._infer_onnx(img0, img1, timestep)
        elif self.backend == "ncnn":
            return self._infer_ncnn(img0, img1, timestep)
        elif self.backend == "pytorch":
            return self._infer_pytorch(img0, img1, timestep)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    def _infer_pytorch(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Perform PyTorch interpolation inference."""
        from apps.backend.pytorch.InterpolateArchs.RIFE.warplayer import warp
        
        # Get pre-computed timestep tensor
        timestep_tensor = self.timestep_dict.get(timestep)
        if timestep_tensor is None:
            raise ValueError(f"Invalid timestep: {timestep}")
        
        # Pad frames if needed
        if any(p > 0 for p in self.padding):
            img0 = torch.nn.functional.pad(img0, self.padding)
            img1 = torch.nn.functional.pad(img1, self.padding)
        
        # Initialize feature maps
        f0 = self.encode(img0) if self.encode is not None else None
        f1 = self.encode(img1) if self.encode is not None else None
        
        # Run interpolation
        with torch.cuda.stream(self.stream):
            flow, mask, feat = self.flownet(
                img0, img1, timestep_tensor,
                self.tenFlow_div, self.backwarp_tenGrid,
                f0, f1,
            )
            
            # Warp and combine
            warped_img0 = warp(img0, flow[:, :2], self.tenFlow_div, self.backwarp_tenGrid)
            warped_img1 = warp(img1, flow[:, 2:4], self.tenFlow_div, self.backwarp_tenGrid)
            img = warped_img0 * mask + warped_img1 * (1.0 - mask)
            
            # Apply features
            img = img + feat
            
        return img
    
    def _infer_onnx(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Perform ONNX interpolation inference."""
        import numpy as np
        
        # Convert tensors to numpy
        img0_np = img0.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img1_np = img1.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        
        # Normalize to [0, 1]
        img0_tensor = img0_np.astype(np.float32)
        img1_tensor = img1_np.astype(np.float32)
        
        # Prepare timestep
        timestep_array = np.array([[timestep]], dtype=np.float32)
        
        # Run inference using loader
        input_names = self.onnx_loader.get_input_names()
        output_names = self.onnx_loader.get_output_names()
        
        # Handle different input signatures
        if len(self.onnx_inputs) == 3:
            # Standard RIFE ONNX: img0, img1, timestep
            outputs = self.onnx_loader.run(
                {
                    input_names[0]: img0_tensor,
                    input_names[1]: img1_tensor,
                    input_names[2]: timestep_array,
                }
            )
        else:
            # Fallback: try with just img0 and img1
            outputs = self.onnx_loader.run(
                {
                    input_names[0]: img0_tensor,
                    input_names[1]: img1_tensor,
                }
            )
        
        # Get result
        result = outputs[0]
        
        # Convert back to tensor
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)
        else:
            result = torch.from_numpy(result.cpu().numpy()).permute(2, 0, 1).unsqueeze(0)
        
        return result
    
    def _infer_ncnn(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Perform NCNN interpncnn" and hasattr(self, 'ncnn_net'):
            self.ncnn_net = None
        elif self.backend == "olation inference."""
        import numpy as np
        
        # Convert tensors to numpy (NHWC format for NCNN)
        img0_np = img0.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img1_np = img1.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        
        # Create NCNN mats
        mat0 = self.ncnn_net.create_input("input.1")
        mat0.from_pixels(img0_np)
        
        mat1 = self.ncnn_net.create_input("input.2")
        mat1.from_pixels(img1_np)
        
        # Create timestep mat
        timestep_array = np.array([[timestep]], dtype=np.float32)
        mat_t = self.ncnn_net.create_input("input.3")
        mat_t.from_pixels_blocking(timestep_array.flatten().astype(np.float32))
        
        # Run inference
        extractor = self.ncnn_net.create_extractor()
        extractor.set_input(mat0, "input.1")
        extractor.set_input(mat1, "input.2")
        extractor.set_input(mat_t, "input.3")
        
        # Extract output
        _, out_mat = extractor.extract("output.1")
        
        # Convert output to tensor
        output_data = out_mat.to_pixels()
        result = torch.from_numpy(output_data).permute(2, 0, 1).unsqueeze(0)
        
        return result
    
    def unload(self) -> None:
        """Unload model and free memory."""
        import gc
        
        if self.backend == "onnx" and hasattr(self, 'onnx_loader'):
            self.onnx_loader.unload()
        elif self.backend == "ncnn" and hasattr(self, 'ncnn_net'):
            self.ncnn_net = None
        elif self.backend == "pytorch":
            self.flownet = None
            self.encode = None
            self.tenFlow_div = None
            self.backwarp_tenGrid = None
            self.timestep_dict.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()
    
    def forward(self, *args, **kwargs):
        """Forward pass (delegates to backend)."""
        raise NotImplementedError("Use backend-specific inference methods")
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return expected input shape."""
        return (1, 3, self.height, self.width)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return output shape for given input shape."""
        return input_shape
    
    def interpolate(self, frame1, frame2, timestep=0.5):
        """Interpolate between two frames."""
        return self.infer(frame1, frame2, timestep)


# Register RIFE model
register_model(
    name="RIFE",
    model=RifeModel(
        config=RifeConfig(version=RifeVersion.RIFE422_LITE),
        model_path="",
        device="cpu",
        dtype="fp32",
    ),
)
