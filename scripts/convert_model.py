#!/usr/bin/env python3
"""
CLI tool for converting models between formats.

Usage:
    python scripts/convert_model.py --model-path model.pt --output-path model.onnx --model-type rife --version rife422_lite
    python scripts/convert_model.py --model-path model.onnx --output-path model.ncnn --model-type rife --to-ncnn
"""

import argparse
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.converters.torch_to_onnx import convert_model as convert_torch_to_onnx
from backend.converters.onnx_to_ncnn import convert_model as convert_onnx_to_ncnn


def main():
    parser = argparse.ArgumentParser(
        description="Convert models between PyTorch, ONNX, and NCNN formats"
    )
    parser.add_argument("--model-path", required=True, help="Path to input model")
    parser.add_argument("--output-path", required=True, help="Path to save output model")
    parser.add_argument("--model-type", default="rife", help="Model type (rife, span, etc.)")
    parser.add_argument("--version", help="Model version")
    parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 1080, 1920], help="Input shape [batch, channels, height, width]")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX model simplification")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16 (for NCNN)")
    parser.add_argument("--to-ncnn", action="store_true", help="Convert from ONNX to NCNN")
    
    args = parser.parse_args()
    
    # Determine conversion type
    if args.to_ncnn:
        # ONNX to NCNN
        print(f"Converting ONNX to NCNN: {args.model_path} -> {args.output_path}")
        convert_onnx_to_ncnn(
            model_path=args.model_path,
            output_path=args.output_path,
            model_type=args.model_type,
            input_shape=tuple(args.input_shape),
            fp16=args.fp16,
        )
    else:
        # PyTorch to ONNX
        print(f"Converting PyTorch to ONNX: {args.model_path} -> {args.output_path}")
        convert_torch_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            model_type=args.model_type,
            version=args.version,
            input_shape=tuple(args.input_shape),
            opset_version=args.opset_version,
            simplify=not args.no_simplify,
        )
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()
