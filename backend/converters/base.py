"""
Base classes for model format converters.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models.base import BaseModel, ModelFormat


class BaseConverter(ABC):
    """
    Abstract base class for model converters.
    
    Converters handle translation between model formats.
    """
    
    def __init__(self, name: str = "converter"):
        self.name = name
        self.source_format = None
        self.target_format = None
    
    @abstractmethod
    def convert(
        self,
        model: BaseModel,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> bool:
        """
        Convert a model from one format to another.
        
        Args:
            model: The model to convert
            input_path: Path to input model file
            output_path: Path to output model file
            **kwargs: Additional conversion arguments
        
        Returns:
            True if conversion successful
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> tuple[ModelFormat, ModelFormat]:
        """Return (source_format, target_format)."""
        pass


class TorchToOnnxConverter(BaseConverter):
    """Converter from PyTorch to ONNX."""
    
    def __init__(self, name: str = "torch_to_onnx"):
        super().__init__(name)
    
    def get_supported_formats(self) -> tuple[ModelFormat, ModelFormat]:
        return (ModelFormat.PT, ModelFormat.ONNX)
    
    @abstractmethod
    def convert(
        self,
        model: BaseModel,
        input_path: str,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None,
        use_pnnx: bool = False,
        **kwargs
    ) -> bool:
        """
        Convert PyTorch model to ONNX.
        
        Args:
            model: The model to convert
            input_path: Path to PyTorch model
            output_path: Path to output ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axis mapping
            use_pnnx: Use pnnx for conversion
        """
        pass


class OnnxToNcnnConverter(BaseConverter):
    """Converter from ONNX to NCNN."""
    
    def __init__(self, name: str = "onnx_to_ncnn"):
        super().__init__(name)
    
    def get_supported_formats(self) -> tuple[ModelFormat, ModelFormat]:
        return (ModelFormat.ONNX, ModelFormat.NCNN)
    
    @abstractmethod
    def convert(
        self,
        model: BaseModel,
        input_path: str,
        output_dir: str,
        use_pnnx: bool = True,
        fp16_mode: bool = False,
        **kwargs
    ) -> bool:
        """
        Convert ONNX model to NCNN.
        
        Args:
            model: The model to convert
            input_path: Path to ONNX model
            output_dir: Directory for output NCNN files
            use_pnnx: Use pnnx for conversion
            fp16_mode: Convert to FP16
        """
        pass


class TorchToNcnnConverter(BaseConverter):
    """Converter from PyTorch to NCNN (via pnnx)."""
    
    def __init__(self):
        super().__init__("torch_to_ncnn")
    
    def get_supported_formats(self) -> tuple[ModelFormat, ModelFormat]:
        return (ModelFormat.PT, ModelFormat.NCNN)
    
    @abstractmethod
    def convert(
        self,
        model: BaseModel,
        input_path: str,
        output_dir: str,
        **kwargs
    ) -> bool:
        """
        Convert PyTorch model to NCNN using pnnx.
        
        Args:
            model: The model to convert
            input_path: Path to PyTorch model
            output_dir: Directory for output NCNN files
        """
        pass
