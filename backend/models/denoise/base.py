"""
Denoise model base class.
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from backend.models.base import BaseModel, ModelTask, ModelFormat


class BaseDenoiseModel(BaseModel):
    """Base class for denoise models."""
    
    def __init__(self, name: str, architecture: str):
        super().__init__(name, architecture, ModelTask.DENOISE)
    
    @abstractmethod
    def denoise(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Denoise a frame.
        
        Args:
            frame: Input noisy frame
        
        Returns:
            Denoised frame
        """
        pass
