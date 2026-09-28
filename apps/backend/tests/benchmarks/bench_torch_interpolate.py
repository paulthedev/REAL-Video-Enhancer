"""
Frame interpolation (RIFE family) performance benchmark for PyTorch on ROCm/CUDA/XPU.

Loads a RIFE model through the real InterpolateFactory path and times one
interpolated frame at 2x, i.e. the workload RenderVideo does per pair of
input frames. Also benchmarks torch.compile around the flownet to measure
the Triton/inductor kernel path vs eager MIOpen kernels.

Usage (run from the backend/ directory with the app python):
    <app-python> tests/benchmarks/bench_torch_interpolate.py \
        --model /path/to/flownet.pkl \
        [--width 1920] [--height 1080] \
        [--compile-mode default|reduce-overhead|max-autotune] \
        [--iterations 3]

Notes:
- RIFE is much more memory-bound than upscale models; compile gains are
  usually smaller here. Measure before enabling anything in the pipeline.
- The first compiled call includes compilation time and is reported separately.
"""
import argparse
import os
import sys
import time

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="RIFE interpolation benchmark")
    parser.add_argument("--model", required=True, help="Path to the RIFE model (.pkl)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--compile-mode",
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="also benchmark torch.compile around the flownet with this mode",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    from apps.backend.pytorch.InterpolateTorch import InterpolateFactory
    from apps.backend.utils.Frame import Frame

    device_name = torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else "cpu"
    print(f"device: cuda:{args.gpu_id} | name: {device_name}")

    # Build the interpolation method exactly like RenderVideo.setupInterpolate.
    interpolate = InterpolateFactory.build_interpolation_method(
        args.model, "pytorch", drba=False
    )(
        modelPath=args.model,
        ceilInterpolateFactor=2,
        width=args.width,
        height=args.height,
        device="auto",
        dtype="auto",
        backend="pytorch",
        gpu_id=args.gpu_id,
        UHDMode=False,
        drba=False,
    )

    # Two random frames (random uint8 video data), fed as tensors.
    frame_a = Frame(
        backend="pytorch", width=args.width, height=args.height,
        device="auto", gpu_id=0, hdr_mode=False, dtype="auto",
    ).set_frame_tensor(torch.randn(1, 3, args.height, args.width))
    frame_b = Frame(
        backend="pytorch", width=args.width, height=args.height,
        device="auto", gpu_id=0, hdr_mode=False, dtype="auto",
    ).set_frame_tensor(torch.randn(1, 3, args.height, args.width))

    def one_pair():
        # First call primes frame0; the second yields interpolated frames.
        list(interpolate(frame_a))
        return list(interpolate(frame_b))

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start = time.perf_counter()
    one_pair()  # warmup (MIOpen autotune, grid allocation)
    sync()
    print(f"warmup+first-pair: {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    for _ in range(args.iterations):
        one_pair()
    sync()
    ms_per_pair = (time.perf_counter() - start) / args.iterations * 1000.0
    print(f"eager 2x interpolation:     {ms_per_pair:.1f} ms/pair")

    if args.compile_mode:
        # Swap the reference used by __call__ for a compiled version.
        original_flownet = interpolate.flownet
        interpolate.flownet = torch.compile(
            original_flownet, mode=args.compile_mode, dynamic=False
        )

        start = time.perf_counter()
        one_pair()  # compile + first run
        sync()
        print(f"compile+first-run: {time.perf_counter() - start:.1f}s")

        start = time.perf_counter()
        for _ in range(args.iterations):
            one_pair()
        sync()
        compiled_ms = (time.perf_counter() - start) / args.iterations * 1000.0
        print(f"compiled({args.compile_mode}): {compiled_ms:.1f} ms/pair")

        interpolate.flownet = original_flownet


if __name__ == "__main__":
    main()
