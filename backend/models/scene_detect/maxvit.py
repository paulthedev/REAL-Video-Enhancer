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

from backend.models.scene_detect.base import BaseSceneDetectModel
from backend.models.base import ModelTask, ModelFormat
from backend.models.registry import register_model


class MaxViTSceneDetect(nn.Module):
    """MaxViT model for scene detection - simplified for testing."""
    
    def __init__(self, num_classes=2, in_chans=6):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans
        
        # Simplified architecture for testing
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
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
    
    def __init__(self, config: MaxViTConfig, model_path: Optional[str] = None):
        super().__init__(name="scene_detect", architecture="maxvit")
        self.config = config
        self.model_path = model_path
        self.model = None
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to detect method."""
        return self.detect(*args, **kwargs)
    
    def detect(self, frame: torch.Tensor, prev_frame: Optional[torch.Tensor] = None) -> bool:
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
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if prev_frame is None:
            return False
        
        # Concatenate frames
        input_tensor = torch.cat((prev_frame, frame), dim=0)
        
        # Run inference
        with torch.inference_mode():
            output = self.model(input_tensor.unsqueeze(0))
        
        # Return True if scene change detected
        return output[0][0] > self.config.threshold
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "task": ModelTask.SCENE_DETECT,
            "format": ModelFormat.PT,
            "num_classes": self.config.num_classes,
            "in_chans": self.config.in_chans,
            "threshold": self.config.threshold,
        }
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, self.config.in_chans, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, self.config.num_classes]


# Register the model (create instance for registration)
_scene_detect_instance = MaxViTSceneDetectModel(config=MaxViTConfig())
register_model(
    name="scene_detect",
    model=_scene_detect_instance,
    formats={ModelFormat.PT: "models/scene_detect/sudo_maxxvit_scenedetect.pt"},
    backends=["pytorch"],
)
