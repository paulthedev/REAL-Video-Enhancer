"""
Model registry loader.

Loads model catalog from config/models.yaml and provides
functions to get model info for both test and production use.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path

# Global registry loaded from YAML
_model_registry: Optional[Dict[str, Any]] = None
_test_mode: bool = False


def set_test_mode(enabled: bool = True):
    """
    Enable test mode. In test mode, models are loaded from local paths
    instead of being downloaded.
    
    Args:
        enabled: True to enable test mode
    """
    global _test_mode
    _test_mode = enabled


def is_test_mode() -> bool:
    """Check if test mode is enabled."""
    return _test_mode


def load_model_registry(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model registry from YAML config.
    
    Args:
        config_path: Path to models.yaml config file.
                    Defaults to config/models.yaml in repo root.
    
    Returns:
        Model registry dictionary
    """
    global _model_registry
    
    if _model_registry is not None:
        return _model_registry
    
    if config_path is None:
        # Try to find config/models.yaml relative to this file
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        _model_registry = yaml.safe_load(f)
    
    return _model_registry


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get model info by ID.
    
    Args:
        model_id: Model ID (e.g., "rife426", "span_animation")
    
    Returns:
        Model info dictionary or None if not found
    """
    registry = load_model_registry()
    return registry.get("models", {}).get(model_id)


def get_models_by_task(task: str) -> List[Dict[str, Any]]:
    """
    Get all models for a specific task.
    
    Args:
        task: Task type (interpolate, upscale, restoration, denoise, scene_detect)
    
    Returns:
        List of model info dictionaries
    """
    registry = load_model_registry()
    models = registry.get("models", {})
    
    return [
        {"id": k, **v}
        for k, v in models.items()
        if v.get("task") == task
    ]


def get_models_by_backend(backend: str) -> List[Dict[str, Any]]:
    """
    Get all models that support a specific backend.
    
    Args:
        backend: Backend name (pytorch, onnx, ncnn)
    
    Returns:
        List of model info dictionaries
    """
    registry = load_model_registry()
    models = registry.get("models", {})
    
    return [
        {"id": k, **v}
        for k, v in models.items()
        if backend in v.get("backends", [])
    ]


def get_model_path(model_id: str, format: str = "pytorch") -> Optional[str]:
    """
    Get the path to a model file.
    
    In test mode, returns local path from repo.
    In production, returns path where model would be downloaded.
    
    Args:
        model_id: Model ID
        format: Model format (pytorch, onnx, ncnn)
    
    Returns:
        Path to model file or None
    """
    model = get_model(model_id)
    if model is None:
        return None
    
    formats = model.get("formats", {})
    format_info = formats.get(format, {})
    
    if not format_info:
        return None
    
    # In test mode, use local paths
    if _test_mode:
        return format_info.get("path")
    
    # In production, use CWD-relative paths
    from src.constants import MODELS_PATH
    return os.path.join(MODELS_PATH, format_info.get("file", ""))


def get_model_download_url(model_id: str, format: str = "pytorch") -> Optional[str]:
    """
    Get the download URL for a model.
    
    Args:
        model_id: Model ID
        format: Model format (pytorch, onnx, ncnn)
    
    Returns:
        Download URL or None
    """
    model = get_model(model_id)
    if model is None:
        return None
    
    formats = model.get("formats", {})
    format_info = formats.get(format, {})
    
    return format_info.get("download")


def get_defaults() -> Dict[str, str]:
    """
    Get default model selections for each task.
    
    Returns:
        Dictionary mapping task to default model ID
    """
    registry = load_model_registry()
    return registry.get("defaults", {})


def list_models() -> List[str]:
    """
    List all available model IDs.
    
    Returns:
        List of model IDs
    """
    registry = load_model_registry()
    return list(registry.get("models", {}).keys())


def get_recommended_models(task: Optional[str] = None) -> List[str]:
    """
    Get recommended model IDs.
    
    Args:
        task: Optional task filter
    
    Returns:
        List of recommended model IDs
    """
    registry = load_model_registry()
    models = registry.get("models", {})
    
    recommended = []
    for model_id, model_info in models.items():
        if model_info.get("recommended", False):
            if task is None or model_info.get("task") == task:
                recommended.append(model_id)
    
    return recommended


if __name__ == "__main__":
    # Test the loader
    print("Loading model registry...")
    registry = load_model_registry()
    
    print(f"\nTotal models: {len(registry.get('models', {}))}")
    print(f"\nAvailable tasks: {set(m.get('task') for m in registry.get('models', {}).values())}")
    
    print("\nRecommended models:")
    for model_id in get_recommended_models():
        model = get_model(model_id)
        print(f"  - {model_id}: {model.get('display_name')}")
    
    print("\nPyTorch models:")
    for model in get_models_by_backend("pytorch")[:5]:
        print(f"  - {model['id']}: {model['display_name']}")
