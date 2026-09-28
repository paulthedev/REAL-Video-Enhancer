"""
ONNX to NCNN converter.

This module handles converting ONNX models to NCNN format.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import os
import subprocess
import shutil
from pathlib import Path

from backend.converters.base import BaseConverter, OnnxToNcnnConverter as BaseOnnxToNcnnConverter
from backend.models.base import ModelFormat


class OnnxToNcnnConverter(BaseOnnxToNcnnConverter):
    """
    Converter from ONNX to NCNN format.
    
    Uses pnnx and ncnnoptimize to convert models.
    """
    
    def __init__(
        self,
        pnnx_path: Optional[str] = None,
        ncnnoptimize_path: Optional[str] = None,
        fp16: bool = False,
    ):
        """
        Initialize converter.
        
        Args:
            pnnx_path: Path to pnnx binary
            ncnnoptimize_path: Path to ncnnoptimize binary
            fp16: Whether to convert to FP16
        """
        super().__init__("onnx_to_ncnn")
        self.pnnx_path = pnnx_path or self._find_pnnx()
        self.fp16 = fp16
    
    def _find_pnnx(self) -> str:
        """Find pnnx binary."""
        # Check common locations
        candidates = [
            "pnnx",
            os.path.expanduser("~/.ncnn/bin/pnnx"),
            "/usr/local/bin/pnnx",
            "./pnnx",
        ]
        
        for candidate in candidates:
            if shutil.which(candidate):
                return candidate
        
        # Try to find in venv
        venv_pnnx = os.path.join(os.path.dirname(sys.executable), "pnnx")
        if os.path.exists(venv_pnnx):
            return venv_pnnx
        
        # Try to find in site-packages
        import site
        for path in site.getsitepackages():
            pnnx_path = os.path.join(path, "pnnx", "pnnx")
            if os.path.exists(pnnx_path):
                return pnnx_path
        
        raise FileNotFoundError("pnnx not found. Please install pnnx: pip install pnnx")
    
    def _find_ncnnoptimize(self) -> str:
        """Find ncnnoptimize binary."""
        # Check common locations
        candidates = [
            "ncnnoptimize",
            os.path.expanduser("~/.ncnn/bin/ncnnoptimize"),
            "/usr/local/bin/ncnnoptimize",
            "./ncnnoptimize",
        ]
        
        for candidate in candidates:
            if shutil.which(candidate):
                return candidate
        
        # Try to find in common installation paths
        import glob
        patterns = [
            "**/ncnnoptimize",
            "**/bin/ncnnoptimize",
        ]
        
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
        
        raise FileNotFoundError("ncnnoptimize not found. Please install NCNN tools.")
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> str:
        """
        Convert ONNX model to NCNN using pnnx.
        
        Args:
            model_path: Path to ONNX model
            output_path: Path to save NCNN model (without extension)
            input_shape: Input tensor shape (optional)
            **kwargs: Additional arguments
            
        Returns:
            Path to converted NCNN model
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Use pnnx to convert ONNX to NCNN directly
        pnnx_path = kwargs.get("pnnx_path", self.pnnx_path)
        onnx_file = model_path
        ncnn_param = output_path + ".param"
        ncnn_bin = output_path + ".bin"
        
        print(f"Converting ONNX to NCNN: {onnx_file} -> {ncnn_param}")
        
        cmd = [
            pnnx_path,
            onnx_file,
            f"ncnnparam={ncnn_param}",
            f"ncnnbin={ncnn_bin}",
        ]
        
        if input_shape:
            cmd.extend([f"inputshape=[1,{input_shape[1]},{input_shape[2]},{input_shape[3]}]"])
        
        if self.fp16:
            cmd.append("fp16=1")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"PNNX conversion failed: {result.stderr}")
            raise RuntimeError(f"PNNX conversion failed: {result.stderr}")
        
        print(f"NCNN conversion successful: {ncnn_param}, {ncnn_bin}")
        
        return ncnn_bin
    
    def convert_rife(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 1080, 1920),
        **kwargs,
    ) -> str:
        """
        Convert RIFE ONNX model to NCNN.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save NCNN model
            input_shape: Input tensor shape
            **kwargs: Additional arguments
            
        Returns:
            Path to converted NCNN model
        """
        return self.convert(
            model_path=onnx_path,
            output_path=output_path,
            input_shape=input_shape,
            **kwargs,
        )


def convert_model(
    model_path: str,
    output_path: str,
    model_type: str = "rife",
    input_shape: Optional[Tuple[int, ...]] = None,
    fp16: bool = False,
) -> str:
    """
    Convert a model from ONNX to NCNN.
    
    Args:
        model_path: Path to ONNX model
        output_path: Path to save NCNN model
        model_type: Model type ("rife", etc.)
        input_shape: Input tensor shape
        fp16: Whether to convert to FP16
        
    Returns:
        Path to converted NCNN model
    """
    converter = OnnxToNcnnConverter(fp16=fp16)
    
    if model_type == "rife":
        return converter.convert_rife(
            onnx_path=model_path,
            output_path=output_path,
            input_shape=input_shape or (1, 3, 1080, 1920),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ONNX model to NCNN")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument("--output-path", required=True, help="Path to save NCNN model")
    parser.add_argument("--model-type", default="rife", help="Model type (rife, etc.)")
    parser.add_argument("--input-shape", nargs=4, type=int, help="Input shape")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16")
    
    args = parser.parse_args()
    
    convert_model(
        model_path=args.model_path,
        output_path=args.output_path,
        model_type=args.model_type,
        input_shape=tuple(args.input_shape) if args.input_shape else None,
        fp16=args.fp16,
    )
