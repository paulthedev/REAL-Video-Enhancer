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
    
    Supports multiple execution providers (priority order):
    - TensorRT (onnxruntime-tensorrt): NVIDIA GPU max throughput
    - CUDA (onnxruntime-gpu): NVIDIA GPU standard path
    - MIGraphX (onnxruntime-migraphx): AMD GPU (ROCm)
    - OpenVINO (onnxruntime-openvino): Intel CPU / iGPU / NPU (Arc & Core Ultra)
    - QNN (onnxruntime-qualcomm): Snapdragon HTP/NPU + Adreno GPU (ARM64 & Android)
    - DirectML (onnxruntime-directml): Any DirectX 12 GPU (Windows)
    - CoreML (onnxruntime-coreml): Apple Silicon (ANE/GPU/CPU)
    - WebGPU (onnxruntime-webgpu): Browser/native WebGPU
    - CPU (onnxruntime): MLAS + Eigen (x86), XNNPACK (Arm + x86)
    """
    
    # Provider priority order
    # Based on ONNX Runtime execution providers:
    # - TensorRT: NVIDIA GPU max throughput (compiles subgraphs to TRT engines)
    # - CUDA: NVIDIA GPU standard path
    # - MIGraphX: AMD GPU (ROCm)
    # - OpenVINO: Intel CPU / iGPU / NPU (best for Intel Arc & Core Ultra NPUs)
    # - QNN: Qualcomm Snapdragon HTP/NPU + Adreno GPU (Windows ARM64 & Android)
    # - DirectML: Any DirectX 12 GPU (Windows cross-vendor)
    # - CoreML: Apple Silicon (ANE/GPU/CPU)
    # - WebGPU: Browser/native WebGPU
    # - CPU: MLAS + Eigen (x86), XNNPACK (Arm + x86), always available fallback
    PROVIDER_PRIORITY = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "OpenVINOExecutionProvider",
        "QnnExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "WebGPUExecutionProvider",
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
        # Configure session options with optimization enabled
        self.session_options: ort.SessionOptions = session_options or ort.SessionOptions()
        self.session_options.enable_cpu_mem_arena = True
        self.session_options.enable_mem_pattern = True
        self.session_options.enable_mem_reuse = True
        self.session_options.enable_profiling = False
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
