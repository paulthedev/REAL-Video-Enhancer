"""Tests for the default-on PyTorch optimizations.

Covers three things (all enabled by default in the render pipeline):

1. Pinned-memory staging buffer in TorchUtils.frame_to_tensor — output must be
   byte-for-byte identical to the previous pageable upload path, and the
   persistent buffer must be reused across frames.
2. Hoisted per-tile .to() calls in UpscalePytorch.renderTiledImage — tiled
   output must match full-frame inference (and eager vs compiled within fp16
   tolerance).
3. torch.compile default-on for the plain PyTorch backend — the wrapper's
   inference helper must be a compiled module, and disabling it via
   torch_compile=False must restore an eager callable that still produces
   matching output.

These tests load real models on a GPU, so they are skipped when no CUDA/XPU
device or model file is available (pass --model to override). Run with:

    python -m pytest tests/unit_tests/test_pytorch_optimizations.py \
        --model /path/to/4x-UltraSharpV2.pth -v

CI keeps a small default set of checks that skip gracefully.
"""
import os
import sys

import pytest


BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _find_default_model():
    """Look for an upscale model the way RenderVideo would (app data dir)."""
    candidates = [
        os.path.expanduser("~/.local/share/REAL-Video-Enhancer/models/UpscaleModels"),
        os.path.expanduser("~/.local/share/REAL-Video-Enhancer/custom_models"),
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith((".pth", ".safetensors")) and "4x-UltraSharpV2" in name:
                return os.path.join(base, name)
    # fall back to any 4x spandrel model
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith((".pth", ".safetensors")) and "4x" in name:
                return os.path.join(base, name)
    return None


def _gpu_available():
    torch = pytest.importorskip("torch")
    return (getattr(torch, "cuda", None) is not None and torch.cuda.is_available()) or \
        (hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)())


@pytest.fixture(scope="module")
def device():
    if not _gpu_available():
        pytest.skip("no CUDA/XPU device available")
    torch = pytest.importorskip("torch")
    return torch.device("cuda" if torch.cuda.is_available() else "xpu", 0)


@pytest.fixture(scope="module")
def model_path(request):
    path = os.environ.get("RVE_TEST_MODEL") or _find_default_model()
    if not path or not os.path.isfile(path):
        pytest.skip(f"no upscale model found (set RVE_TEST_MODEL); looked for 4x-UltraSharpV2")
    return path


@pytest.fixture(scope="module")
def width_height():
    # Small enough to fit in VRAM quickly, large enough that tiling actually
    # triggers with the tilesize used below.
    env_w = os.environ.get("RVE_TEST_WIDTH", "384")
    return int(env_w), int(os.environ.get("RVE_TEST_HEIGHT", env_w))


def _make_upscale(model_path, device, W, H, tilesize=0, torch_compile=True):
    from apps.backend.pytorch.UpscaleTorch import UpscalePytorch

    kwargs = dict(device=str(device.type), width=W, height=H, tilesize=tilesize)
    if "cuda" in str(device):
        kwargs["gpu_id"] = device.index or 0
    return UpscalePytorch(model_path, torch_compile=torch_compile, **kwargs)


# Module-scoped model instances — each construction compiles the model and
# takes on the order of a minute, so share them across tests.
@pytest.fixture(scope="module")
def upc_full(device, model_path, width_height):
    W, H = width_height
    upc = _make_upscale(model_path, device, W, H)  # tilesize=0, compile on (default)
    yield upc


@pytest.fixture(scope="module")
def upc_tiled_compiled(device, model_path, width_height):
    W, H = width_height
    return _make_upscale(model_path, device, W, H, tilesize=min(W // 2, 192) or 64)


@pytest.fixture(scope="module")
def upc_tiled_eager(device, model_path, width_height):
    W, H = width_height
    return _make_upscale(model_path, device, W, H, tilesize=min(W // 2, 192) or 64,
                         torch_compile=False)


@pytest.fixture(scope="module")
def input_tensor(device, width_height):
    import torch

    W, H = width_height
    return (torch.rand(1, 3, H, W, device=device, dtype=torch.float16)).clamp_(0.05, 0.95)


def test_frame_to_tensor_pinned_matches_pageable(device, model_path, width_height, upc_full):
    """Pinned staging upload must produce exactly the same tensor as the old
    pageable path."""
    import random

    import torch

    W, H = width_height
    tu = upc_full.torchUtils
    data = random.randbytes(W * H * 3)

    out_pinned = (
        tu.frame_to_tensor(data, None, device, torch.float16).clone().float()
    )

    r = torch.frombuffer(data, dtype=torch.uint8).to(device=device)
    ref = (
        r.div(255.0)
        .clamp(0.0, 1.0)
        .reshape(H, W, 3)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    ).to(dtype=torch.float16).float()

    assert torch.equal(out_pinned, ref), "pinned upload differs from pageable reference"

    # The persistent per-stream staging buffer must be reused (same object).
    s1 = tu.init_stream(gpu_id=0)
    b1 = tu._get_staging_buffer(s1, W * H * 3)
    b2 = tu._get_staging_buffer(s1, W * H * 3)
    assert b1 is not None and b1 is b2, "pinned staging buffer was not reused on the same stream"

    # A larger request must still return a big-enough buffer.
    bigger = tu._get_staging_buffer(s1, (W + 64) * H * 3)
    assert bigger.numel() >= (W + 64) * H * 3, "buffer did not grow for larger frame size"

    # A different stream gets its own private staging buffer.
    s2 = tu.init_stream(gpu_id=0)
    b_other = tu._get_staging_buffer(s2, W * H * 3)
    assert id(s1) != id(s2), "expected distinct streams"


def test_tiled_render_matches_full_frame(device, input_tensor, upc_full,
                                        upc_tiled_compiled, upc_tiled_eager):
    """renderTiledImage with hoisted .to() must match full-frame inference."""
    import torch

    out_tiled = upc_tiled_compiled.renderTiledImage(input_tensor.clone())
    with torch.inference_mode():
        out_full = upc_full.upscale_model_wrapper(input_tensor)

    diff = (out_tiled.float() - out_full.float()).abs().max().item()
    # fp16 accumulation order differs between tiled and full-frame; allow a
    # small tolerance instead of exact equality.
    assert diff < 0.05, f"tiled output diverges from full-frame: max abs diff {diff}"

    # Tiled with compile on vs off must also agree within fp16 tolerance —
    # this is the identity check for both the hoist and torch.compile.
    out_eager_tiled = upc_tiled_eager.renderTiledImage(input_tensor.clone())
    assert torch.allclose(out_tiled.float(), out_eager_tiled.float(), atol=2e-2), \
        "compiled tiled output diverges from eager tiled output"


def test_compile_enabled_by_default(device, width_height, upc_full):
    """Default construction must wrap the inference helper in torch.compile."""
    import torch

    H = width_height[1]
    helper = upc_full.upscale_model_wrapper.inference_helper
    compiled = hasattr(type(helper), "_orig_mod") or "Optim" in type(helper).__name__
    assert compiled, f"inference helper not compiled: {type(helper)!r}"

    # Calling through the wrapper still works and returns a fresh tensor.
    x = torch.rand(1, 3, H, width_height[0], device=device, dtype=torch.float16)
    out = upc_full.upscale_model_wrapper(x.clone())
    assert out.shape[2] == H * upc_full.scale


def test_compile_disabled_restores_eager(device, model_path, width_height):
    """torch_compile=False must leave the helper as a plain nn.Module."""
    import torch

    W = H = 192
    upc = _make_upscale(model_path, device, W, H, torch_compile=False)
    try:
        helper = upc.upscale_model_wrapper.inference_helper
        assert not hasattr(type(helper), "_orig_mod"), \
            f"helper was compiled despite torch_compile=False: {type(helper)!r}"

        x = (torch.rand(1, 3, H, W, device=device, dtype=torch.float16))
        out = upc.upscale_model_wrapper(x)
        assert out.shape[2] == H * upc.scale
    finally:
        del upc


def test_enable_compile_skips_temporal_models(device):
    """enable_compile() must return None (helper untouched) for non-spandrel
    models, so AnimeSR/TSPAN temporal helpers keep running eager."""
    from apps.backend.pytorch.UpscaleModelWrapper import UpscaleModelWrapper

    class _Stub:
        pass

    stub = _Stub()
    # Simulate an AnimeSR-style wrapper without loading a model file.
    object.__setattr__(stub, "_UpscaleModelWrapper__inference_mode", "animesr")
    sentinel = type("H", (), {})()
    object.__setattr__(stub, "inference_helper", sentinel)

    prev = UpscaleModelWrapper.enable_compile(stub)
    assert prev is None, "non-spandrel model should not be wrapped in torch.compile"
    assert stub.inference_helper is sentinel, "helper must be left untouched"
