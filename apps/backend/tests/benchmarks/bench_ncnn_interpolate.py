"""
Frame interpolation (RIFE family) performance benchmark for NCNN backend.

Loads a RIFE model through the NCNN backend and times one interpolated
frame at 2x, i.e. the workload RenderVideo does per pair of input frames.
Also benchmarks different NCNN backends (Vulkan, OpenGL, CPU).

Usage (run from the backend/ directory with the app python):
    <app-python> tests/benchmarks/bench_ncnn_interpolate.py \
        --model /path/to/rife.param \
        [--width 1920] [--height 1080] \
        [--backend vulkan|opengl|cpu] \
        [--iterations 10]

Notes:
- NCNN Vulkan backend is fastest on GPUs with Vulkan support
- NCNN CPU backend uses different optimizations (ARM vs x86)
- First run includes model loading time
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
    parser = argparse.ArgumentParser(description="NCNN interpolation benchmark")
    parser.add_argument("--model", required=True, help="Path to the NCNN model (.param file)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--backend",
        default="vulkan",
        choices=["vulkan", "opengl", "cpu"],
        help="NCNN backend to use",
    )
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import ncnn
    except ImportError:
        print("ERROR: NCNN package not installed. Install with: pip install ncnn")
        return

    # Initialize NCNN
    net = ncnn.Net()
    
    # Set backend
    if args.backend == "vulkan":
        net.set_vulkan_device(0)
        print(f"Using NCNN Vulkan backend")
    elif args.backend == "opengl":
        net.set_gl_device(0)
        print(f"Using NCNN OpenGL backend")
    else:
        print(f"Using NCNN CPU backend")

    # Load model
    print(f"Loading model: {args.model}")
    net.load_param(args.model)
    model_bin = args.model.replace('.param', '.bin')
    net.load_model(model_bin)
    print(f"Model loaded: {os.path.basename(args.model)}")

    # Create extractor
    extractor = net.create_extractor()

    # Create input
    input_name = net.input_names()[0] if net.input_names() else "input"
    mat = ncnn.Mat(args.width, args.height, 3)
    
    # Fill with random data
    mat.from_pixels_resize(
        np.random.randint(0, 256, (args.height, args.width, 3), dtype=np.uint8).tobytes(),
        ncnn.Mat.PIXEL_RGB,
        args.width,
        args.height
    )

    # Warmup run
    print("Running warmup...")
    extractor.set_input(input_name, mat)
    _ = extractor.extract("output")

    # Benchmark
    start = time.perf_counter()
    for _ in range(args.iterations):
        extractor.set_input(input_name, mat)
        _ = extractor.extract("output")
    elapsed = time.perf_counter() - start

    ms_per_frame = elapsed / args.iterations * 1000.0
    print(f"\n--- NCNN Interpolation Benchmark ---")
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Backend: {args.backend}")
    print(f"Iterations: {args.iterations}")
    print(f"Time per frame: {ms_per_frame:.1f} ms")
    print(f"FPS: {1000.0 / ms_per_frame:.1f}")


if __name__ == "__main__":
    main()
