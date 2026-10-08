"""
ONNX backend runner for backend-agnostic models.

This module handles running inference with ONNX models.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from queue import Queue

from apps.backend.backends.base import InterpolateBackend, UpscaleBackend
from apps.backend.models.base import BaseModel


class ONNXInterpolateRunner(InterpolateBackend):
    """
    ONNX backend runner for interpolation models.
    """
    
    def __init__(self, loader: 'ONNXBackendLoader'):
        super().__init__(name="onnx_interpolate", loader=loader)
        self.loader = loader
    
    def run(
        self,
        model: Dict[str, Any],
        img0: np.ndarray,
        img1: np.ndarray,
        timestep: float,
        writeQueue: Optional[Queue] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Run interpolation inference.
        
        Args:
            model: Model info dict from ONNX loader
            img0: First frame [H, W, C]
            img1: Second frame [H, W, C]
            timestep: Interpolation timestep (0.0 to 1.0)
            writeQueue: Optional queue to write results
            **kwargs: Additional arguments
            
        Returns:
            Interpolated frame [H, W, C]
        """
        session = model['session']
        inputs = model['inputs']

        # ONNX models expect NCHW float input: img0, img1, [timestep]
        img0_tensor = (img0.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        img1_tensor = (img1.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        feed = {
            inputs[0].name: img0_tensor,
            inputs[1].name: img1_tensor,
        }
        if len(inputs) > 2:
            feed[inputs[2].name] = np.array([[timestep]], dtype=np.float32)

        # Run inference
        output_names = [out.name for out in model['outputs']]
        result = session.run(output_names, feed)[0]

        # NCHW float result -> HWC uint8 (session.run always returns ndarrays)
        if result.ndim == 4:
            result = result[0]
        result = (np.clip(result, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        
        # Write to queue if provided
        if writeQueue is not None:
            writeQueue.put(result)
        
        return result
    
    def run_batch(
        self,
        model: Dict[str, Any],
        frames: List[Tuple[np.ndarray, np.ndarray]],
        timestep: float,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Run batch interpolation inference.
        
        Args:
            model: Model info dict
            frames: List of (img0, img1) tuples
            timestep: Interpolation timestep
            **kwargs: Additional arguments
            
        Returns:
            List of interpolated frames
        """
        results = []
        for img0, img1 in frames:
            result = self.run(model, img0, img1, timestep, **kwargs)
            results.append(result)
        return results


class ONNXUpscaleRunner(UpscaleBackend):
    """
    ONNX backend runner for upscaling models.
    """
    
    def __init__(self, loader: 'ONNXBackendLoader'):
        super().__init__(name="onnx_upscale", loader=loader)
        self.loader = loader
    
    def run(
        self,
        model: Dict[str, Any],
        img: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Run upscaling inference.
        
        Args:
            model: Model info dict
            img: Input frame [H, W, C]
            **kwargs: Additional arguments
            
        Returns:
            Upscaled frame
        """
        session = model['session']
        inputs = model['inputs']
        
        # Prepare input data (NCHW)
        img_tensor = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        # Run inference
        output_names = [out.name for out in model['outputs']]
        outputs = session.run(
            output_names,
            {inputs[0].name: img_tensor}
        )
        
        # Get result
        result = outputs[0]
        
        # float result -> HWC uint8 (session.run always returns ndarrays)
        result = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        return result
