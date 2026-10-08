"""
NAFNet (Nonlinear Activation Free Network) model definition.

This module provides a backend-agnostic NAFNet model that can be used with
PyTorch, ONNX, or NCNN backends.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import os

from apps.backend.models.base import BaseRestorationModel, ModelTask, ModelFormat
from apps.backend.models.registry import register_model


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D features."""
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SimpleGate(nn.Module):
    """Simple gate for NAFNet."""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFBlock for NAFNet."""
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
    
    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet model - simplified for testing."""
    
    def __init__(self, width=16, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        
        self.padder_size = 2 ** len(self.encoders)
    
    def forward(self, inp):
        """Forward pass for NAFNet."""
        B, C, H, W = inp.shape
        # Pad input if needed
        self.mod = H % self.padder_size
        if self.mod != 0:
            inp = F.pad(inp, (0, self.mod, 0, self.mod - self.mod if self.mod > 0 else 0))
        
        x = self.intro(inp)
        
        enc_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_outputs.append(x)
            x = self.downs[len(enc_outputs) - 1](x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(enc_outputs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        
        if self.mod != 0:
            x = x[..., :H, :W]
        
        return x


@dataclass
class NAFNetConfig:
    """Configuration for NAFNet model."""
    width: int = 16
    enc_blk_nums: list = None
    middle_blk_num: int = 1
    dec_blk_nums: list = None
    width: int = 1920
    height: int = 1080
    
    def __post_init__(self):
        if self.enc_blk_nums is None:
            self.enc_blk_nums = [1, 1, 1, 28]
        if self.dec_blk_nums is None:
            self.dec_blk_nums = [1, 1, 1, 1]


class NAFNetModel(BaseRestorationModel):
    """NAFNet model implementation."""
    
    def __init__(self, config: NAFNetConfig, model_path: Optional[str] = None):
        super().__init__(name="NAFNet", architecture="NAFNet")
        self.config = config
        self.model_path = model_path
        self.model = None
        self.backend = None
        self.onnx_loader = None
        self.ncnn_net = None
    
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
        """Load NAFNet model using PyTorch backend."""
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.model = NAFNet(
            width=self.config.width,
            enc_blk_nums=self.config.enc_blk_nums,
            middle_blk_num=self.config.middle_blk_num,
            dec_blk_nums=self.config.dec_blk_nums,
        ).to(device)
        
        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"Loaded NAFNet PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load NAFNet model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded NAFNet ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load NAFNet model using NCNN backend."""
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
        
        print(f"Loaded NAFNet NCNN model: {param_path}")
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to restore method."""
        return self.restore(*args, **kwargs)
    
    def restore(self, frame: torch.Tensor) -> torch.Tensor:
        """Restore a frame."""
        if self.backend == "onnx":
            return self._restore_onnx(frame)
        elif self.backend == "ncnn":
            return self._restore_ncnn(frame)
        elif self.backend == "pytorch":
            return self._restore_pytorch(frame)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    def _restore_pytorch(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform PyTorch restoration inference."""
        with torch.no_grad():
            return self.model(frame.to(self._device) if hasattr(self, "_device") else frame)
    
    def _restore_onnx(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform ONNX restoration inference."""
        import numpy as np
        
        # Convert tensor to numpy (NCHW to NHWC)
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = frame_np.astype(np.float32)
        
        # Run inference
        input_name = self.onnx_inputs[0].name
        output_name = self.onnx_outputs[0].name
        
        outputs = self.onnx_loader.run(
            {input_name: frame_np},
            [output_name]
        )
        
        # Convert back to tensor (NHWC to NCHW)
        result = torch.from_numpy(outputs[0]).permute(2, 0, 1).unsqueeze(0)
        
        return result
    
    def _restore_ncnn(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform NCNN restoration inference."""
        import ncnn  # optional dependency; imported here so the module imports without it
        import numpy as np
        
        # Convert tensor to numpy (NCHW to NHWC)
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255.0).astype(np.uint8)
        
        # Create NCNN Mat
        h, w, c = frame_np.shape
        mat = ncnn.Mat.from_pixels(frame_np, ncnn.Mat.PixelType.PIXEL_RGB, w, h)
        
        # Create extractor
        extractor = self.ncnn_net.create_extractor()
        extractor.set_num_thread(4)
        extractor.input("input.1", mat)
        
        # Run inference
        _, out_mat = extractor.extract("output.1")
        
        # Convert output to tensor
        output_data = out_mat.numpy()  # CHW float
        result = torch.from_numpy(output_data).unsqueeze(0).clamp(0.0, 1.0)
        
        return result
    
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
            "task": ModelTask.RESTORATION,
            "format": ModelFormat.PT,
            "width": self.config.width,
            "height": self.config.height,
            "enc_blk_nums": self.config.enc_blk_nums,
            "middle_blk_num": self.config.middle_blk_num,
            "dec_blk_nums": self.config.dec_blk_nums,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, 3, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, 3, self.config.height, self.config.width]


# Register the model (create instance for registration)
_nafnet_instance = NAFNetModel(config=NAFNetConfig())
register_model(
    name="nafnet",
    model=_nafnet_instance,
    formats={
        ModelFormat.PT: "models/restoration/nafnet.pth",
        ModelFormat.ONNX: "models/restoration/nafnet.onnx",
        ModelFormat.NCNN: "models/restoration/nafnet.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
