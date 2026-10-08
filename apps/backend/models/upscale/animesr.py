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
import os

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
        """Load AnimeSR model using PyTorch backend."""
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.model = AnimeSR(upscale=self.config.scale).to(device)
        
        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"Loaded AnimeSR PyTorch model: {self.model_path}")
    
    def _load_onnx(self) -> None:
        """Load AnimeSR model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        self.onnx_loader = OnnxModelLoader(provider="auto")
        self.onnx_loader.load(self.model_path)
        
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        
        print(f"Loaded AnimeSR ONNX model: {self.model_path}")
        print(f"  Provider: {self.onnx_provider}")
    
    def _load_ncnn(self) -> None:
        """Load AnimeSR model using NCNN backend."""
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
        
        print(f"Loaded AnimeSR NCNN model: {param_path}")
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to upscale method."""
        return self.upscale(*args, **kwargs)
    
    def upscale(self, frame: torch.Tensor) -> torch.Tensor:
        """Upscale a frame."""
        if self.backend == "onnx":
            return self._upscale_onnx(frame)
        elif self.backend == "ncnn":
            return self._upscale_ncnn(frame)
        elif self.backend == "pytorch":
            return self._upscale_pytorch(frame)
        else:
            raise RuntimeError(f"Inference not supported for backend: {self.backend}")
    
    def _upscale_pytorch(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform PyTorch upscale inference."""
        with torch.no_grad():
            return self.model(frame.to(self._device) if hasattr(self, "_device") else frame)
    
    def _upscale_onnx(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform ONNX upscale inference."""
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
    
    def _upscale_ncnn(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform NCNN upscale inference."""
        import ncnn  # optional dependency; imported here so the module imports without it
        import numpy as np
        
        # Convert tensor to numpy (NCHW to NHWC)
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255.0).astype(np.uint8)
        
        # Create NCNN Mat
        h, w, c = frame_np.shape
        mat = ncnn.Mat(h, w, c, frame_np.flatten())
        
        # Create extractor
        extractor = self.ncnn_net.create_extractor()
        extractor.set_num_thread(4)
        extractor.input("input.1", mat)
        
        # Run inference
        _, out_mat = extractor.extract("output.1")
        
        # Convert output to tensor
        output_data = out_mat.to_pixels()
        result = torch.from_numpy(output_data).permute(2, 0, 1).unsqueeze(0) / 255.0
        
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
    formats={
        ModelFormat.PT: "models/upscale/animesr.pth",
        ModelFormat.ONNX: "models/upscale/animesr.onnx",
        ModelFormat.NCNN: "models/upscale/animesr.ncnn",
    },
    backends=["pytorch", "onnx", "ncnn"],
)
