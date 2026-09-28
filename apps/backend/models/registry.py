"""
Model registry for tracking available models and their formats.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from pathlib import Path

from .base import BaseModel, ModelFormat, ModelTask


class ModelRegistry:
    """
    Registry for tracking available models and their formats.
    
    Usage:
        registry = ModelRegistry()
        registry.register("rife422", RIFEModel(), ["pytorch", "onnx", "ncnn"])
        model = registry.get("rife422")
    """
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._formats: Dict[str, Dict[ModelFormat, str]] = {}
        self._backends: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        model: BaseModel,
        formats: Optional[Dict[ModelFormat, str]] = None,
        backends: Optional[List[str]] = None,
    ):
        """
        Register a model in the registry.
        
        Args:
            name: Unique model name
            model: Model instance
            formats: Mapping of format to file path
            backends: List of supported backends
        """
        self._models[name] = model
        if formats:
            self._formats[name] = formats
        if backends:
            self._backends[name] = backends
    
    def get(self, name: str) -> Optional[BaseModel]:
        """Get a model by name."""
        return self._models.get(name)
    
    def get_formats(self, name: str) -> Optional[Dict[ModelFormat, str]]:
        """Get available formats for a model."""
        return self._formats.get(name)
    
    def get_backends(self, name: str) -> Optional[List[str]]:
        """Get supported backends for a model."""
        return self._backends.get(name)
    
    def list_models(self, task: Optional[ModelTask] = None) -> List[str]:
        """List all registered model names, optionally filtered by task."""
        if task is None:
            return list(self._models.keys())
        return [
            name for name, model in self._models.items()
            if model.task == task
        ]
    
    def has_model(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._models
    
    def remove(self, name: str) -> bool:
        """Remove a model from the registry."""
        if name in self._models:
            del self._models[name]
            self._formats.pop(name, None)
            self._backends.pop(name, None)
            return True
        return False
    
    def get_model_info(self, name: str) -> Optional[dict]:
        """Get complete info about a model."""
        model = self._models.get(name)
        if model is None:
            return None
        return {
            "name": name,
            "architecture": model.architecture,
            "task": model.task.value,
            "formats": self._formats.get(name, {}),
            "backends": self._backends.get(name, []),
        }


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry


def register_model(
    name: str,
    model: BaseModel,
    formats: Optional[Dict[ModelFormat, str]] = None,
    backends: Optional[List[str]] = None,
):
    """Register a model in the global registry."""
    _registry.register(name, model, formats, backends)


def get_model(name: str) -> Optional[BaseModel]:
    """Get a model from the global registry."""
    return _registry.get(name)


def list_models(task: Optional[ModelTask] = None) -> List[str]:
    """List all registered models."""
    return _registry.list_models(task)


def get_model_info(name: str) -> Optional[dict]:
    """Get info about a registered model."""
    return _registry.get_model_info(name)
