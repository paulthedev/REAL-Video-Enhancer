"""
NCNN backend loader for backend-agnostic models.

This module handles loading models with the NCNN backend.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import numpy as np

from apps.backend.backends.base import BaseBackend, BackendType
from apps.backend.models.base import BaseModel, ModelTask


class NCNNBackendLoader(BaseBackend):
    """
    NCNN backend loader.
    
    Uses NCNN (Neural Network Inferencer) for fast inference.
    """
    
    def __init__(self):
        super().__init__(
            name="ncnn",
            backend_type=BackendType.NCNN,
            supported_tasks=[ModelTask.INTERPOLATE, ModelTask.UPSCALE, ModelTask.RESTORATION, ModelTask.DENOISE],
        )
        self.models: Dict[str, Dict[str, Any]] = {}
        self.ncnn = None
        self._try_import_ncnn()
    
    def _try_import_ncnn(self):
        """Try to import ncnn, set to None if not available."""
        try:
            import ncnn
            self.ncnn = ncnn
        except ImportError:
            self.ncnn = None
    
    def initialize(
        self,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize NCNN backend.
        
        Args:
            provider_options: Provider-specific options
        """
        if self.ncnn is None:
            raise RuntimeError("NCNN package not installed. Install with: pip install ncnn")
        
        print("NCNN backend initialized")
    
    def load_model(
        self,
        model_path: str,
        model_name: str = "ncnn_model",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Load an NCNN model.
        
        Args:
            model_path: Path to NCNN model file (.param and .bin)
            model_name: Name for the loaded model
            input_names: Input tensor names
            output_names: Output tensor names
        
        Returns:
            Model info dictionary
        """
        if self.ncnn is None:
            raise RuntimeError("NCNN package not installed")
        
        # NCNN expects .param and .bin files
        param_path = model_path.replace('.bin', '.param') if model_path.endswith('.bin') else model_path
        
        # Load model
        net = self.ncnn.Net()
        net.load_param(param_path)
        net.load_model(model_path)
        
        model_info = {
            "net": net,
            "path": model_path,
            "input_names": input_names or ["input"],
            "output_names": output_names or ["output"],
        }
        
        self.models[model_name] = model_info
        print(f"Loaded NCNN model: {model_name}")
        
        return model_info
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            print(f"Unloaded NCNN model: {model_name}")
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get a loaded model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model info dictionary
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not loaded")
        return self.models[model_name]
    
    def is_loaded(self, model_name: str) -> bool:
        """
        Check if a model is loaded.
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if model is loaded
        """
        return model_name in self.models
    
    def get_available_models(self) -> List[str]:
        """
        Get list of loaded model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get backend information.
        
        Returns:
            Backend info dictionary
        """
        return {
            "name": self.name,
            "type": self.backend_type.value,
            "supported_tasks": [t.value for t in self.supported_tasks],
            "models_loaded": len(self.models),
            "ncnn_available": self.ncnn is not None,
        }
