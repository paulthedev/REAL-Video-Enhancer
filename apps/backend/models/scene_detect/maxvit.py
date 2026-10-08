"""
MaxViT scene detection model definition.

This module provides a backend-agnostic scene detection model using MaxViT.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from abc import abstractmethod
import os
from apps.backend.models.scene_detect.base import BaseSceneDetectModel
from apps.backend.models.base import ModelTask, ModelFormat
from apps.backend.models.registry import register_model

class MaxViTSceneDetect(nn.Module):
    """MaxViT model for scene detection - simplified for testing."""

    def __init__(self, num_classes=2, in_chans=6):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.features = nn.Sequential(nn.Conv2d(in_chans, 32, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        """Forward pass for scene detection."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

@dataclass
class MaxViTConfig:
    """Configuration for MaxViT scene detection model."""
    num_classes: int = 2
    in_chans: int = 6
    width: int = 256
    height: int = 256
    threshold: float = 0.3

class MaxViTSceneDetectModel(BaseSceneDetectModel):
    """MaxViT scene detection model implementation."""

    def __init__(self, config: MaxViTConfig, model_path: Optional[str]=None):
        super().__init__(name='scene_detect', architecture='maxvit')
        self.config = config
        self.model_path = model_path
        self.model = None

    def forward(self, *args, **kwargs):
        """Forward pass - delegates to detect method."""
        return self.detect(*args, **kwargs)

    def detect(self, frame: torch.Tensor, prev_frame: Optional[torch.Tensor]=None) -> bool:
        """
        Detect scene change.
        
        Args:
            frame: Current frame [C, H, W]
            prev_frame: Previous frame [C, H, W]
        
        Returns:
            True if scene change detected
        """
        if prev_frame is None:
            return False
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load() first.')
        if prev_frame is None:
            return False
        input_tensor = torch.cat((prev_frame, frame), dim=0)
        with torch.inference_mode():
            output = self.model(input_tensor.unsqueeze(0))
        return output[0][0] > self.config.threshold

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {'task': ModelTask.SCENE_DETECT, 'format': ModelFormat.PT, 'num_classes': self.config.num_classes, 'in_chans': self.config.in_chans, 'threshold': self.config.threshold}

    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, self.config.in_chans, self.config.height, self.config.width]

    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, self.config.num_classes]

    def load(self, backend_name: str) -> None:
        """
        Load model with specified backend.

        Args:
            backend_name: Backend to use (only "onnx" is supported)
        """
        if backend_name != 'onnx':
            raise ValueError(f'Unsupported backend: {backend_name}')
        self.backend = backend_name
        self._load_onnx()

    def _load_onnx(self) -> None:
        """Load MaxViT model using ONNX backend."""
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        self.onnx_loader = OnnxModelLoader(provider='auto')
        self.onnx_loader.load(self.model_path)
        self.onnx_inputs = self.onnx_loader.inputs
        self.onnx_outputs = self.onnx_loader.outputs
        self.onnx_provider = self.onnx_loader.provider
        print(f'Loaded MaxViT ONNX model: {self.model_path}')
        print(f'  Provider: {self.onnx_provider}')

    def _detect_onnx(self, frame: torch.Tensor, prev_frame: Optional[torch.Tensor]=None) -> bool:
        """Perform ONNX scene detection inference."""
        import numpy as np
        if prev_frame is None:
            return False
        input_tensor = torch.cat((prev_frame, frame), dim=0)
        input_np = input_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)
        input_name = self.onnx_inputs[0].name
        output_name = self.onnx_outputs[0].name
        outputs = self.onnx_loader.run({input_name: input_np}, [output_name])
        return outputs[0][0][0] > self.config.threshold

    def unload(self) -> None:
        """Unload model and free memory."""
        import gc
        if self.backend == 'onnx' and hasattr(self, 'onnx_loader'):
            self.onnx_loader.unload()
        elif self.backend == 'ncnn' and hasattr(self, 'ncnn_net'):
            self.ncnn_net = None
        elif self.backend == 'pytorch':
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_info(self) -> dict:
        """Return model metadata."""
        return {'name': self.name, 'architecture': self.architecture, 'task': 'scene_detect', 'num_classes': self.config.num_classes, 'supported_formats': ['pt', 'onnx', 'ncnn'], 'supported_backends': ['pytorch', 'onnx', 'ncnn']}
_scene_detect_instance = MaxViTSceneDetectModel(config=MaxViTConfig())
register_model(name='scene_detect', model=_scene_detect_instance, formats={ModelFormat.PT: 'models/scene_detect/sudo_maxxvit_scenedetect.pt', ModelFormat.ONNX: 'models/scene_detect/sudo_maxxvit_scenedetect.onnx', ModelFormat.NCNN: 'models/scene_detect/sudo_maxxvit_scenedetect.ncnn'}, backends=['pytorch', 'onnx', 'ncnn'])