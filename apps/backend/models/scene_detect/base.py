"""
Scene detection model base class.
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from apps.backend.models.base import BaseModel, ModelTask, ModelFormat


class BaseSceneDetectModel(BaseModel):
    """Base class for scene detection models."""
    
    def __init__(self, name: str, architecture: str):
        super().__init__(name, architecture, ModelTask.SCENE_DETECT)
    
    @abstractmethod
    def detect(self, frame: torch.Tensor, prev_frame: Optional[torch.Tensor] = None) -> bool:
        """
        Detect scene change.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (optional)
        
        Returns:
            True if scene change detected
        """
        pass
