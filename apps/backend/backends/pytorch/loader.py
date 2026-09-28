"""
PyTorch backend loader for backend-agnostic models.

This module handles loading models with the PyTorch backend.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import torch
import gc

from apps.backend.backends.base import BaseBackend, BackendType
from apps.backend.models.base import BaseModel, ModelTask
from apps.backend.models.registry import get_model_info


class PyTorchBackendLoader(BaseBackend):
    """
    PyTorch backend loader.
    
    Handles loading and managing PyTorch models.
    """
    
    def __init__(self):
        super().__init__(
            name="pytorch",
            backend_type=BackendType.TORCH,
            supported_tasks=[ModelTask.INTERPOLATE, ModelTask.UPSCALE, ModelTask.RESTORATION, ModelTask.DENOISE, ModelTask.SCENE_DETECT],
        )
        self.models: Dict[str, BaseModel] = {}
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        
    def initialize(
        self,
        device: str = "auto",
        dtype: str = "auto",
        gpu_id: int = 0,
    ) -> None:
        """
        Initialize PyTorch backend.
        
        Args:
            device: Device to run on ("auto", "cuda", "cpu")
            dtype: Data type ("auto", "fp16", "fp32")
            gpu_id: GPU ID
        """
        from apps.backend.pytorch.TorchUtils import TorchUtils
        
        self.device = TorchUtils.handle_device(device, gpu_id=gpu_id)
        self.dtype = TorchUtils.handle_precision(dtype)
        
        print(f"PyTorch backend initialized on {self.device} with dtype {self.dtype}")
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseModel:
        """
        Load a model with PyTorch backend.
        
        Args:
            model_name: Name of the model
            model_path: Path to model weights
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        from apps.backend.models.registry import get_model
        
        # Get model class from registry
        model_class = get_model(model_name)
        if model_class is None:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Create model instance
        model = model_class(
            model_path=model_path,
            device=self.device.type if self.device else "auto",
            dtype=self.dtype if self.dtype else "auto",
            gpu_id=kwargs.get("gpu_id", 0),
            **(config or {}),
        )
        
        # Load with PyTorch backend
        model.load("pytorch")
        
        # Store in registry
        self.models[model_name] = model
        
        return model
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'unload'):
                model.unload()
            del self.models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all(self) -> None:
        """Unload all models."""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """
        Get a loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None
        """
        return self.models.get(model_name)
    
    def is_available(self) -> bool:
        """Check if PyTorch backend is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        import torch
        
        return {
            "name": "pytorch",
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": self.device.type if self.device else None,
            "models_loaded": len(self.models),
        }
