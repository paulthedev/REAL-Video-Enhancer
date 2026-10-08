"""
Base classes for backend implementations.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from ..models.base import BaseModel, ModelTask


class BackendType(Enum):
    """Backend execution type."""
    TORCH = "pytorch"
    ONNX = "onnx"
    NCNN = "ncnn"


class BaseBackend(ABC):
    """
    Abstract base class for all backends.
    
    Backends are thin wrappers that handle:
    - Loading models from disk
    - Running inference
    - Memory management
    """
    
    def __init__(self, name: str, backend_type: Optional[BackendType] = None, supported_tasks: Optional[List[ModelTask]] = None, device: str = "auto", loader: Any = None):
        self.name = name
        self.backend_type = backend_type
        self.supported_tasks = supported_tasks or []
        self.device = device
        self.loader = loader
        self._model: Optional[BaseModel] = None
    
    def load(self, model: BaseModel, model_path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            model: The model to load
            model_path: Path to the model file
        
        Returns:
            True if loaded successfully
        """
        raise NotImplementedError
    
    def run(self, model: BaseModel, input_data: Any, **kwargs) -> Any:
        """
        Run inference on the model.
        
        Args:
            model: The model to run
            input_data: Input data
            **kwargs: Additional arguments
        
        Returns:
            Model output
        """
        raise NotImplementedError
    
    def unload(self, model: Optional[BaseModel] = None) -> bool:
        """
        Unload a model.
        
        Args:
            model: The model to unload (optional, defaults to current model)
        
        Returns:
            True if unloaded successfully
        """
        raise NotImplementedError
    
    def is_loaded(self, model: BaseModel) -> bool:
        """Check if a model is loaded."""
        return self._model is not None and self._model.name == model.name
    
    def get_info(self) -> Dict[str, Any]:
        """Return backend information."""
        return {
            "name": self.name,
            "device": self.device,
            "model": self._model.name if self._model else None,
        }


class InterpolateBackend(BaseBackend):
    """Base class for interpolation backends."""
    
    def interpolate(
        self,
        model: BaseModel,
        frame1: Any,
        frame2: Any,
        timestep: float = 0.5,
        **kwargs
    ) -> Any:
        """Interpolate between two frames."""
        raise NotImplementedError
    
    def run(
        self,
        model: BaseModel,
        img0: Any,
        img1: Any,
        timestep: float,
        **kwargs
    ) -> Any:
        """Run interpolation inference."""
        return self.interpolate(model, img0, img1, timestep, **kwargs)


class UpscaleBackend(BaseBackend):
    """Base class for upscaling backends."""
    
    def upscale(
        self,
        model: BaseModel,
        frame: Any,
        **kwargs
    ) -> Any:
        """Upscale a frame."""
        raise NotImplementedError
    
    def run(
        self,
        model: BaseModel,
        frame: Any,
        **kwargs
    ) -> Any:
        """Run upscaling inference."""
        return self.upscale(model, frame, **kwargs)


class RestorationBackend(BaseBackend):
    """Base class for restoration backends."""
    
    @abstractmethod
    def restore(
        self,
        model: BaseModel,
        frame: Any,
        **kwargs
    ) -> Any:
        """Restore a frame."""
        pass

    def run(
        self,
        model: BaseModel,
        frame: Any,
        **kwargs
    ) -> Any:
        """Run restoration inference."""
        return self.restore(model, frame, **kwargs)


class SceneDetectBackend(BaseBackend):
    """Base class for scene detection backends."""
    
    @abstractmethod
    def detect(
        self,
        model: BaseModel,
        frame: Any,
        prev_frame: Optional[Any] = None,
        **kwargs
    ) -> bool:
        """Detect scene change."""
        pass

    def run(
        self,
        model: BaseModel,
        frame: Any,
        prev_frame: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Run scene detection."""
        return self.detect(model, frame, prev_frame, **kwargs)
