"""
PyTorch backend runner for backend-agnostic models.

This module handles running inference with PyTorch models.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from queue import Queue

from apps.backend.backends.base import InterpolateBackend, UpscaleBackend
from apps.backend.models.base import BaseModel, ModelTask


class PyTorchInterpolateRunner(InterpolateBackend):
    """
    PyTorch backend runner for interpolation models.
    """
    
    def __init__(self, loader: 'PyTorchBackendLoader'):
        super().__init__(name="pytorch_interpolate", loader=loader)
        self.loader = loader
    
    def run(
        self,
        model: BaseModel,
        img0: np.ndarray,
        img1: np.ndarray,
        timestep: float,
        writeQueue: Optional[Queue] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Run interpolation inference.
        
        Args:
            model: Model instance
            img0: First frame [H, W, C]
            img1: Second frame [H, W, C]
            timestep: Interpolation timestep (0.0 to 1.0)
            writeQueue: Optional queue to write results
            **kwargs: Additional arguments
            
        Returns:
            Interpolated frame [H, W, C]
        """
        # Convert numpy to tensor
        img0_tensor = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        if self.loader.device:
            img0_tensor = img0_tensor.to(self.loader.device)
            img1_tensor = img1_tensor.to(self.loader.device)
        
        # Run inference
        with torch.inference_mode():
            result = model.infer(img0_tensor, img1_tensor, timestep)
        
        # Convert back to numpy — normalize rank: some models return
        # [B,C,H,W], others [C,H,W]
        if result.ndim == 3:
            result = result.unsqueeze(0)
        result = result.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result = (result * 255.0).astype(np.uint8)
        
        # Write to queue if provided
        if writeQueue is not None:
            writeQueue.put(result)
        
        return result
    
    def run_batch(
        self,
        model: BaseModel,
        frames: List[Tuple[np.ndarray, np.ndarray]],
        timestep: float,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Run batch interpolation inference.
        
        Args:
            model: Model instance
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


class PyTorchUpscaleRunner(UpscaleBackend):
    """
    PyTorch backend runner for upscaling models.
    """
    
    def __init__(self, loader: 'PyTorchBackendLoader'):
        super().__init__(name="pytorch_upscale", loader=loader)
        self.loader = loader
    
    def run(
        self,
        model: BaseModel,
        img: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Run upscaling inference.
        
        Args:
            model: Model instance
            img: Input frame [H, W, C]
            **kwargs: Additional arguments
            
        Returns:
            Upscaled frame
        """
        # Convert numpy to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        if self.loader.device:
            img_tensor = img_tensor.to(self.loader.device)
        
        # Run inference
        with torch.inference_mode():
            result = model.infer(img_tensor, **kwargs)
        
        # Convert back to numpy — normalize rank: some models return
        # [B,C,H,W], others [C,H,W]
        if result.ndim == 3:
            result = result.unsqueeze(0)
        result = result.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result = (result * 255.0).astype(np.uint8)
        
        return result
