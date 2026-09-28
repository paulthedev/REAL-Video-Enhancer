"""
ONNX scene detection model definition.

This module provides an ONNX-based scene detection model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os
import numpy as np

from apps.backend.models.scene_detect.base import BaseSceneDetectModel
from apps.backend.models.base import ModelTask, ModelFormat


@dataclass
class ONNXSceneDetectConfig:
    """Configuration for ONNX scene detection model."""
    model_path: str
    threshold: float = 0.3
    width: int = 256
    height: int = 256


class ONNXSceneDetectModel(BaseSceneDetectModel):
    """ONNX scene detection model implementation."""
    
    def __init__(self, config: ONNXSceneDetectConfig, model_path: Optional[str] = None):
        super().__init__(name="scene_detect", architecture="onnx_efficientnet")
        self.config = config
        self.model_path = model_path or config.model_path
        self.session = None
        self.inputs = None
        self.outputs = None
    
    def load(self, backend_name: str = "onnx") -> None:
        """
        Load ONNX scene detection model.
        
        Args:
            backend_name: Backend name (should be "onnx")
        """
        from apps.backend.utils.OnnxLoader import OnnxModelLoader
        
        # Load ONNX model using reusable loader
        self.loader = OnnxModelLoader(provider="CPUExecutionProvider")
        self.loader.load(self.model_path)
        
        # Store model info
        self.session = self.loader.session
        self.inputs = self.loader.inputs
        self.outputs = self.loader.outputs
        
        print(f"Loaded ONNX scene detection model: {self.model_path}")
    
    def detect(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> bool:
        """
        Detect scene change.
        
        Args:
            frame: Current frame [C, H, W] or [H, W, C]
            prev_frame: Previous frame [C, H, W] or [H, W, C]
        
        Returns:
            True if scene change detected
        """
        if prev_frame is None:
            return False
        
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Handle different input formats
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            # NCHW format (C, H, W)
            frame_input = frame.astype(np.float32) / 255.0 if frame.max() > 1.0 else frame.astype(np.float32)
            prev_frame_input = prev_frame.astype(np.float32) / 255.0 if prev_frame.max() > 1.0 else prev_frame.astype(np.float32)
        else:
            # NHWC format (H, W, C)
            frame_input = frame.astype(np.float32) / 255.0 if frame.max() > 1.0 else frame.astype(np.float32)
            prev_frame_input = prev_frame.astype(np.float32) / 255.0 if prev_frame.max() > 1.0 else prev_frame.astype(np.float32)
            frame_input = np.transpose(frame_input, (2, 0, 1))
            prev_frame_input = np.transpose(prev_frame_input, (2, 0, 1))
        
        # Concatenate frames (expected format: [6, H, W])
        input_tensor = np.concatenate([prev_frame_input, frame_input], axis=0)
        
        # Run inference
        input_name = self.inputs[0].name
        output_name = self.outputs[0].name
        
        output = self.session.run(
            [output_name],
            {input_name: input_tensor.reshape(1, *input_tensor.shape)}
        )[0]
        
        # Return True if scene change detected
        return output[0][0] > self.config.threshold
    
    def get_input_shape(self) -> List[int]:
        """Get expected input shape."""
        return [1, 6, self.config.height, self.config.width]
    
    def get_output_shape(self) -> List[int]:
        """Get expected output shape."""
        return [1, 2]
    
    def unload(self) -> None:
        """Unload model."""
        if hasattr(self, 'loader'):
            self.loader.unload()
        self.session = None
