"""
Backend-agnostic model definitions.
"""
from .base import (
    BaseModel,
    BaseInterpolateModel,
    BaseUpscaleModel,
    BaseRestorationModel,
    BaseSceneDetectModel,
    ModelTask,
    ModelFormat,
    ModelInfo,
)
from .registry import (
    ModelRegistry,
    get_registry,
    register_model,
    get_model,
    list_models,
)

__all__ = [
    "BaseModel",
    "BaseInterpolateModel",
    "BaseUpscaleModel",
    "BaseRestorationModel",
    "BaseSceneDetectModel",
    "ModelTask",
    "ModelFormat",
    "ModelInfo",
    "ModelRegistry",
    "get_registry",
    "register_model",
    "get_model",
    "list_models",
]
