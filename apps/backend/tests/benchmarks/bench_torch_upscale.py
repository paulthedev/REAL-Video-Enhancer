"""
Upscale model performance benchmark for PyTorch on ROCm/CUDA/XPU.

Compares eager vs channels_last memory format vs torch.compile modes (which
generate Triton kernels via inductor) on a given upscale model (.pth /
.safetensors). The wrapper's own precision test determines the runtime dtype
(fp16 models that fail fp16 inference fall back to float32 automatically,
exactly like in RenderVideo).

Usage (run from the backend/ directory with the app python):
    <app-python> tests/benchmarks/bench_torch_upscale.py \
        --model /path/to/model.pth \
        [--width 512] [--height 512] \
        [--modes default reduce-overhead max-autotune] \
        [--iterations 8]

Notes:
- torch.compile on ROCm generates Triton kernels via the inductor backend.
- The first compiled call includes compilation time; steady-state numbers
  are printed separately from "compile+first-run".
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
    parser = argparse.ArgumentParser(description="Upscale model benchmark")
    parser.add_argument("--model", required=True, help="Path to the upscale model file")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--modes", nargs="+", default=["default"],
        choices=["none"] + ["default", "reduce-overhead", "max-autotune"],
        help="torch.compile modes to benchmark (use 'none' to skip compile entirely)",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def bench(fn, x, n: int, label: str):
    with torch.inference_mode():
        fn(x)  # warmup (MIOpen kernel autotune / Triton compile cache)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n):
            fn(x)
    torch.cuda.synchronize()
    ms_per_frame = (time.perf_counter() - start) / n * 1000.0
    print(f"{label:32s} {ms_per_frame:9.1f} ms/frame")
    return ms_per_frame


def main():
    args = parse_args()

    import torch

    from apps.backend.pytorch.UpscaleModelWrapper import UpscaleModelWrapper
    from apps.backend.pytorch.TorchUtils import TorchUtils

    device = torch.device("cuda", args.gpu_id)
    print(f"device: {device} | name: {torch.cuda.get_device_name(args.gpu_id)}")

    # Same tuning the render pipeline applies at startup:
    # persistent cache dirs + device autotune via src.pytorch.tuning.
    utils = TorchUtils(args.width, args.height, "auto", gpu_id=args.gpu_id)
    _ = utils  # configure_tuning_env() / tune_device() run in the constructor

    wrapper = UpscaleModelWrapper(args.model, device, torch.half)
    model = wrapper.get_model().eval()
    precision = next(model.parameters()).dtype
    print(f"model: {os.path.basename(args.model)} | scale {wrapper.get_scale()} "
          f"| runtime precision: {precision}")

    x = torch.randn(1, 3, args.height, args.width, device=device, dtype=precision)
    results = {}
    results["eager"] = bench(model, x, args.iterations, f"eager ({precision})")

    model_cl = model.to(memory_format=torch.channels_last).eval()
    x_cl = x.contiguous(memory_format=torch.channels_last)
    results["channels_last"] = bench(
        model_cl, x_cl, args.iterations, "channels_last eager"
    )

    for mode in args.modes:
        if mode == "none":
            continue
        print(f"\n--- torch.compile mode={mode} (inductor -> Triton kernels) ---")
        start = time.perf_counter()
        try:
            compiled = torch.compile(model_cl, mode=mode, dynamic=False)
            with torch.inference_mode():
                compiled(x_cl)  # compilation + first run
            torch.cuda.synchronize()
            print(f"compile+first-run: {time.perf_counter() - start:.1f}s")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"COMPILE FAILED ({mode}): {type(e).__name__}: {str(e)[:300]}")
            continue
        results[f"compiled_{mode}"] = bench(
            lambda t, c=compiled: c(t), x_cl, args.iterations, f"compiled({mode})"
        )

    baseline = min(results.values()) if results else float("inf")
    print("\n--- summary ---")
    for label, ms in results.items():
        speedup = baseline / ms
        print(f"{label:32s} {ms:9.1f} ms/frame   x{speedup:.2f}")


if __name__ == "__main__":
    main()
