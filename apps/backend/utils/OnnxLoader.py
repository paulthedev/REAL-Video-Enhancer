"""
ONNX model loading utilities.

This module provides reusable utilities for loading and running ONNX models.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import onnxruntime as ort
import numpy as np


class OnnxModelLoader:
    """
    Reusable ONNX model loader with automatic provider selection.
    
    Supports multiple execution providers:
    - CUDA (onnxruntime-gpu)
    - MIGraphX (onnxruntime-migraphx)
    - OpenVINO (onnxruntime-openvino)
    - DirectML (onnxruntime-directml)
    - CPU (onnxruntime)
    """
    
    # Provider priority order
    PROVIDER_PRIORITY = [
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    
    def __init__(
        self,
        provider: str = "auto",
        provider_options: Optional[Dict[str, Any]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ):
        """
        Initialize ONNX model loader.
        
        Args:
            provider: Execution provider ("auto", or specific provider name)
            provider_options: Provider-specific options
            session_options: ONNX Runtime session options
        """
        self.available_providers: List[str] = ort.get_available_providers()
        self.provider: str = self._select_provider(provider)
        self.provider_options: Dict[str, Any] = provider_options or {}
        self.session_options: ort.SessionOptions = session_options or ort.SessionOptions()
        self.session: Optional[ort.InferenceSession] = None
        self.inputs: List[ort.ValueInfo] = []
        self.outputs: List[ort.ValueInfo] = []
    
    def _select_provider(self, provider: str) -> str:
        """
        Select the best available execution provider.
        
        Args:
            provider: Requested provider ("auto" or specific name)
            
        Returns:
            Selected provider name
        """
        if provider == "auto":
            for preferred_provider in self.PROVIDER_PRIORITY:
                if preferred_provider in self.available_providers:
                    return preferred_provider
            return "CPUExecutionProvider"
        return provider
    
    def load(self, model_path: str) -> None:
        """
        Load an ONNX model from file.
        
        Args:
            model_path: Path to ONNX model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=self.session_options,
            providers=[self.provider],
            provider_options=[self.provider_options] if self.provider_options else None,
        )
        
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        print(f"Loaded ONNX model: {model_path}")
        print(f"  Provider: {self.provider}")
        print(f"  Inputs: {[inp.name for inp in self.inputs]}")
        print(f"  Outputs: {[out.name for out in self.outputs]}")
    
    def run(
        self,
        input_data: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Run inference on the loaded model.
        
        Args:
            input_data: Dictionary of input names to numpy arrays
            output_names: Optional list of output names to fetch
            
        Returns:
            List of output arrays
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        return self.session.run(output_names, input_data)
    
    def get_input_names(self) -> List[str]:
        """Get input tensor names."""
        return [inp.name for inp in self.inputs]
    
    def get_output_names(self) -> List[str]:
        """Get output tensor names."""
        return [out.name for out in self.outputs]
    
    def unload(self) -> None:
        """Unload the model."""
        self.session = None
        self.inputs = []
        self.outputs = []


def load_onnx_model(
    model_path: str,
    provider: str = "auto",
    provider_options: Optional[Dict[str, Any]] = None,
) -> OnnxModelLoader:
    """
    Convenience function to load an ONNX model.
    
    Args:
        model_path: Path to ONNX model file
        provider: Execution provider ("auto" or specific name)
        provider_options: Provider-specific options
        
    Returns:
        Loaded OnnxModelLoader instance
    """
    loader = OnnxModelLoader(provider=provider, provider_options=provider_options)
    loader.load(model_path)
    return loader
