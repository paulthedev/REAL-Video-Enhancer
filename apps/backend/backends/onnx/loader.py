"""
ONNX backend loader for backend-agnostic models.

This module handles loading models with the ONNX Runtime backend.
Supports multiple execution providers: CUDA, DirectML, CPU.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import onnxruntime as ort
import numpy as np

from apps.backend.backends.base import BaseBackend, BackendType
from apps.backend.models.base import BaseModel, ModelTask


class ONNXBackendLoader(BaseBackend):
    """
    ONNX Runtime backend loader.
    
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
    
    def __init__(self):
        super().__init__(
            name="onnx",
            backend_type=BackendType.ONNX,
            supported_tasks=[ModelTask.INTERPOLATE, ModelTask.UPSCALE, ModelTask.RESTORATION],
        )
        self.models: Dict[str, Dict[str, Any]] = {}
        self.providers: List[str] = []
        self.available_providers: List[str] = ort.get_available_providers()
        self.provider: str = "CPUExecutionProvider"
        self.provider_options: Dict[str, Any] = {}
        
    def initialize(
        self,
        provider: str = "auto",
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ONNX backend.
        
        Args:
            provider: Execution provider ("auto", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
            provider_options: Provider-specific options
        """
        # Select provider
        if provider == "auto":
            self.provider = self._select_best_provider()
        else:
            self.provider = provider
        
        # Default provider options
        self.provider_options = provider_options or {}
        
        print(f"ONNX backend initialized with provider: {self.provider}")
        print(f"Available providers: {self.available_providers}")
    
    def _select_best_provider(self) -> str:
        """
        Select the best available execution provider.
        
        Priority order based on ONNX Runtime execution providers:
        - TensorRT: NVIDIA GPU max throughput (compiles subgraphs to TRT engines)
        - CUDA: NVIDIA GPU standard path
        - MIGraphX: AMD GPU (ROCm)
        - OpenVINO: Intel CPU / iGPU / NPU (best for Intel Arc & Core Ultra NPUs)
        - QNN: Qualcomm Snapdragon HTP/NPU + Adreno GPU (Windows ARM64 & Android)
        - DirectML: Any DirectX 12 GPU (Windows cross-vendor)
        - CoreML: Apple Silicon (ANE/GPU/CPU)
        - WebGPU: Browser/native WebGPU
        - CPU: MLAS + Eigen (x86), XNNPACK (Arm + x86), always available fallback
        
        Returns:
            Best provider name
        """
        # Priority order based on hardware capabilities
        if "TensorrtExecutionProvider" in self.available_providers:
            return "TensorrtExecutionProvider"
        elif "CUDAExecutionProvider" in self.available_providers:
            return "CUDAExecutionProvider"
        elif "MIGraphXExecutionProvider" in self.available_providers:
            return "MIGraphXExecutionProvider"
        elif "OpenVINOExecutionProvider" in self.available_providers:
            return "OpenVINOExecutionProvider"
        elif "QnnExecutionProvider" in self.available_providers:
            return "QnnExecutionProvider"
        elif "DmlExecutionProvider" in self.available_providers:
            return "DmlExecutionProvider"
        elif "CoreMLExecutionProvider" in self.available_providers:
            return "CoreMLExecutionProvider"
        elif "WebGPUExecutionProvider" in self.available_providers:
            return "WebGPUExecutionProvider"
        else:
            return "CPUExecutionProvider"
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseModel:
        """
        Load an ONNX model.
        
        Args:
            model_name: Name of the model
            model_path: Path to ONNX model file
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model info dict (ONNX models are loaded differently)
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Create session options with optimization enabled
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.enable_profiling = False
        
        # Load model
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[self.provider],
            provider_options=[self.provider_options] if self.provider_options else None,
        )
        
        # Get model info
        model_info = {
            "name": model_name,
            "path": model_path,
            "session": session,
            "provider": self.provider,
            "inputs": session.get_inputs(),
            "outputs": session.get_outputs(),
            "config": config or {},
        }
        
        # Store in registry
        self.models[model_name] = model_info
        
        print(f"Loaded ONNX model: {model_name} ({model_path})")
        print(f"  Provider: {self.provider}")
        print(f"  Inputs: {[inp.name for inp in model_info['inputs']]}")
        print(f"  Outputs: {[out.name for out in model_info['outputs']]}")
        
        return model_info
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            model_info = self.models[model_name]
            if 'session' in model_info:
                model_info['session'] = None
            del self.models[model_name]
            print(f"Unloaded ONNX model: {model_name}")
    
    def unload_all(self) -> None:
        """Unload all models."""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model info dict or None
        """
        return self.models.get(model_name)
    
    def run_inference(
        self,
        model_name: str,
        input_data: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a loaded model.
        
        Args:
            model_name: Name of the model
            input_data: Dictionary of input names to numpy arrays
            output_names: Optional list of output names to fetch
            
        Returns:
            Dictionary of output names to numpy arrays
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model_info = self.models[model_name]
        session = model_info['session']
        
        # Run inference
        outputs = session.run(
            output_names,
            input_data
        )
        
        # Get output names
        if output_names is None:
            output_names = [out.name for out in model_info['outputs']]
        
        # Return as dictionary
        return dict(zip(output_names, outputs))
    
    def is_available(self) -> bool:
        """Check if ONNX backend is available."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "name": "onnx",
            "version": ort.__version__,
            "provider": self.provider,
            "available_providers": self.available_providers,
            "models_loaded": len(self.models),
        }
