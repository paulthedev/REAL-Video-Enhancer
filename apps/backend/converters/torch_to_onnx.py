"""
Torch to ONNX converter.

This module handles converting PyTorch models to ONNX format.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import onnx
import onnxsim
import os
from pathlib import Path

from apps.backend.converters.base import BaseConverter, TorchToOnnxConverter as BaseTorchToOnnxConverter
from apps.backend.models.base import BaseModel, ModelFormat


class TorchToOnnxConverter(BaseTorchToOnnxConverter):
    """
    Converter from PyTorch to ONNX format.
    
    Uses torch.onnx.export() to convert models.
    """
    
    def __init__(
        self,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        simplify: bool = True,
        fuse_conv_bn: bool = True,
        enable_fusion: bool = True,
    ):
        """
        Initialize converter.
        
        Args:
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
            simplify: Whether to simplify the model after export
            fuse_conv_bn: Whether to fuse Conv+BatchNorm layers
            enable_fusion: Whether to enable operator fusion optimizations
        """
        super().__init__("torch_to_onnx")
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes or {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
        self.simplify = simplify
        self.fuse_conv_bn = fuse_conv_bn
        self.enable_fusion = enable_fusion
    
    def convert(
        self,
        model: BaseModel,
        model_path: str,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 1080, 1920),
        **kwargs,
    ) -> str:
        """
        Convert PyTorch model to ONNX.
        
        Args:
            model: PyTorch model instance
            model_path: Path to PyTorch model weights
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            **kwargs: Additional arguments
            
        Returns:
            Path to converted ONNX model
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load model — the wrapper is a plain object; eval() the inner nn.Module
        model.load("pytorch")
        nn_module = getattr(model, "model", None)
        if isinstance(nn_module, torch.nn.Module):
            nn_module.eval()

        # Interpolation models take (img0, img1, timestep); upscale/restoration
        # take a single frame. Export the signature the loaders expect.
        is_interpolate = getattr(model, "infer", None) is not None or hasattr(model, "ceil_interpolate_factor")
        if is_interpolate:
            c, h, w = input_shape[1], input_shape[2], input_shape[3]
            dummy_inputs = (
                torch.randn(1, c, h, w),
                torch.randn(1, c, h, w),
                torch.zeros(1, 1, 1, 1),
            )
            input_names = ["img0", "img1", "timestep"]
            dynamic_axes = {
                "img0": {2: "height", 3: "width"},
                "img1": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            }
        else:
            dummy_inputs = torch.randn(*input_shape)
            input_names = ["input"]
            dynamic_axes = self.dynamic_axes

        # Export to ONNX
        torch.onnx.export(
            nn_module if isinstance(nn_module, torch.nn.Module) else model,
            dummy_inputs,
            output_path,
            opset_version=self.opset_version,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        
        print(f"Exported PyTorch model to ONNX: {output_path}")
        
        # Simplify model if requested
        if self.simplify:
            try:
                model_onnx = onnx.load(output_path)
                model_onnx_simplified, check = onnxsim.simplify(
                    model_onnx,
                    check_n=3,
                )
                if check:
                    onnx.save(model_onnx_simplified, output_path)
                    print(f"Simplified ONNX model: {output_path}")
                else:
                    print(f"Model simplification failed, keeping original")
            except Exception as e:
                print(f"Model simplification failed: {e}")
        
        return output_path
    
    def convert_rife(
        self,
        model_path: str,
        output_path: str,
        version: str = "rife422_lite",
        input_shape: Tuple[int, ...] = (1, 3, 1080, 1920),
        **kwargs,
    ) -> str:
        """
        Convert RIFE model to ONNX.
        
        Args:
            model_path: Path to PyTorch RIFE model
            output_path: Path to save ONNX model
            version: RIFE version
            input_shape: Input tensor shape
            **kwargs: Additional arguments
            
        Returns:
            Path to converted ONNX model
        """
        # Import RIFE model
        from apps.backend.models.interpolate.rife import RifeModel, RifeConfig, RifeVersion
        
        # Create config
        config = RifeConfig(
            version=getattr(RifeVersion, version.upper()),
            width=input_shape[3],
            height=input_shape[2],
        )
        
        # Create model
        model = RifeModel(
            config=config,
            model_path=model_path,
            device="cpu",
            dtype="fp32",
        )
        
        # Convert
        return self.convert(model, model_path, output_path, input_shape, **kwargs)
    
    def convert_span(
        self,
        model_path: str,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 1080, 1920),
        **kwargs,
    ) -> str:
        """
        Convert SPAN model to ONNX.
        
        Args:
            model_path: Path to PyTorch SPAN model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            **kwargs: Additional arguments
            
        Returns:
            Path to converted ONNX model
        """
        from apps.backend.models.upscale.span import SPANModel, SPANConfig
        
        # Create config
        config = SPANConfig(
            width=input_shape[3],
            height=input_shape[2],
        )
        
        # Create model
        model = SPANModel(config=config, model_path=model_path)
        
        # Convert
        return self.convert(model, model_path, output_path, input_shape, **kwargs)


def convert_model(
    model_path: str,
    output_path: str,
    model_type: str = "rife",
    version: Optional[str] = None,
    input_shape: Tuple[int, ...] = (1, 3, 1080, 1920),
    opset_version: int = 17,
    simplify: bool = True,
) -> str:
    """
    Convert a model from PyTorch to ONNX.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Path to save ONNX model
        model_type: Model type ("rife", "span", etc.)
        version: Model version
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        simplify: Whether to simplify the model
        
    Returns:
        Path to converted ONNX model
    """
    converter = TorchToOnnxConverter(
        opset_version=opset_version,
        simplify=simplify,
    )
    
    if model_type == "rife":
        return converter.convert_rife(
            model_path=model_path,
            output_path=output_path,
            version=version or "rife422_lite",
            input_shape=input_shape,
        )
    elif model_type == "span":
        return converter.convert_span(
            model_path=model_path,
            output_path=output_path,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to PyTorch model")
    parser.add_argument("--output-path", required=True, help="Path to save ONNX model")
    parser.add_argument("--model-type", default="rife", help="Model type (rife, span)")
    parser.add_argument("--version", help="Model version")
    parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 1080, 1920], help="Input shape")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable model simplification")
    
    args = parser.parse_args()
    
    convert_model(
        model_path=args.model_path,
        output_path=args.output_path,
        model_type=args.model_type,
        version=args.version,
        input_shape=tuple(args.input_shape),
        opset_version=args.opset_version,
        simplify=not args.no_simplify,
    )
