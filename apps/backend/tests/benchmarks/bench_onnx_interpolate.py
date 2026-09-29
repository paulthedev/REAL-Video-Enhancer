"""
Frame interpolation (RIFE family) performance benchmark for ONNX backend.

Loads a RIFE model through the ONNX backend and times one interpolated
frame at 2x, i.e. the workload RenderVideo does per pair of input frames.
Also benchmarks different execution providers (CUDA, TensorRT, CPU).

Usage (run from the backend/ directory with the app python):
    <app-python> tests/benchmarks/bench_onnx_interpolate.py \
        --model /path/to/rife.onnx \
        [--width 1920] [--height 1080] \
        [--provider CUDAExecutionProvider|TensorrtExecutionProvider|CPUExecutionProvider] \
        [--iterations 10]

Notes:
- ONNX Runtime provider selection: TensorRT > CUDA > CPU
- First run includes model loading and provider initialization time
- Use --provider to force a specific provider for comparison
"""
import argparse
import os
import sys
import time

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX interpolation benchmark")
    parser.add_argument("--model", required=True, help="Path to the RIFE ONNX model")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution provider",
    )
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    from apps.backend.backends.onnx.loader import ONNXBackendLoader
    from apps.backend.backends.onnx.runner import ONNXInterpolateRunner

    # Initialize ONNX backend
    loader = ONNXBackendLoader()
    loader.initialize(provider=args.provider)

    # Load model
    print(f"Loading model: {args.model}")
    model_info = loader.load_model(args.model)
    print(f"Model loaded: {model_info['model_name']}")
    print(f"Provider: {loader.provider}")

    # Create runner
    runner = ONNXInterpolateRunner(loader)

    # Create random input frames
    img0 = np.random.randint(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    img1 = np.random.randint(0, 256, (args.height, args.width, 3), dtype=np.uint8)

    # Warmup run
    print("Running warmup...")
    _ = runner.run(model_info, img0, img1, timestep=0.5)

    # Benchmark
    start = time.perf_counter()
    for _ in range(args.iterations):
        _ = runner.run(model_info, img0, img1, timestep=0.5)
    elapsed = time.perf_counter() - start

    ms_per_pair = elapsed / args.iterations * 1000.0
    print(f"\n--- ONNX Interpolation Benchmark ---")
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Provider: {loader.provider}")
    print(f"Iterations: {args.iterations}")
    print(f"Time per pair: {ms_per_pair:.1f} ms")
    print(f"FPS: {1000.0 / ms_per_pair:.1f}")


if __name__ == "__main__":
    main()
