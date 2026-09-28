# PyTorch Optimization Tracker

Status of GPU performance work on the `update_dependencies` branch.
Benchmarks below were measured on an RX 9070 XT (gfx1201) with ROCm, using
4x-UltraSharpV2.pth at 512×512 in fp16 unless noted otherwise.

Legend: ✅ done & verified · 🚧 in progress · ⬜ planned · ❌ evaluated & skipped

---

## Completed

### ✅ Correctness fixes that unblocked GPU inference (commit `fe54b4f0`)
These were prerequisites — models crashed before any speed-up was possible.

| Item | Detail |
|------|--------|
| Spandrel attention-mask dtype casts | 10 archs built fp32 masks added to fp16 activations → mixed-dtype crash on GPU. Fixed by casting the mask to `attn.dtype`: DAT, SwinIR, DRCT, Swin2SR, HAT, RGT, ATD, CRAFT, Uformer (×2), GRL (`mixed_attn_block_efficient`). |
| `UpscaleModelWrapper.load_model` under `inference_mode` | Freshly built model params became inference tensors → state_dict loads failed ("Inference tensors do not track version counter"). Moved loading outside; inference itself stays in `inference_mode`. |
| fp16→fp32 fallback test input | On precision fallback the dummy test tensor was left in fp16 while the model went fp32. Now cast to match. |

Verified: **8/8 custom models run fp16 at render resolution** (SwinIR-L, 4x-BSRGAN,
ClearRealityV1, UltraSharpV2 + Lite, FaceUpSharpLDAT, Nomos8kDAT, BSRGANx2).

### ✅ Persistent tuning cache (`backend/src/pytorch/tuning.py`)
| Item | Detail |
|------|--------|
| MIOpen kernel DB persisted | `MIOPEN_USER_DB_PATH` → `<appdata>/cache/tuning/miopen-<gpu-name>.fdb.txt`, keyed per GPU so autotuned kernels survive across runs. Respects a user-set value (does not override). |
| Inductor cache dir pinned | `TORCHINDUCTOR_CACHE_DIR` → `<appdata>/cache/tuning/inductor` — compiled artifacts persist, so any future `torch.compile` only pays the compile cost once per model+shape combo. |
| Startup autotune | `tune_device()` sets `torch.backends.cudnn.benchmark = True`, which applies to CUDA, ROCm (MIOpen) and Intel XPU alike; verified on this build (`torch.xpu.backends` does not exist here — top-level flag is the correct API). Log-once per process. |
| Integration | Wired into `TorchUtils.__init__`; old inline device-tuning method removed. No-op for cpu/mps. App-data dir mirrors frontend logic without PySide6 deps (works in standalone backend process), with tempdir fallback. |

### ✅ Benchmark tooling
`backend/tests/benchmarks/`:
- `bench_torch_upscale.py` — upscale model perf, per-model timing at render resolution
- `bench_torch_interpolate.py` — frame interpolation model perf

Both apply the same startup tuning as the real pipeline (via `TorchUtils`).

### ✅ TensorRT pin (`src/DownloadDeps.py`)
Pinned to 11.1.0.106 because torch-tensorrt 2.14.0 requires tensorrt>=11.1.0,<11.2.0.

---

## Baseline numbers (RX 9070 XT, fp16)

| Configuration | ms/frame | vs eager | One-time cost |
|---------------|---------:|----------|---------------|
| eager fp16 (baseline today) | ~2785 | — | — |
| + `channels_last` | ~2665 | 4% faster | none |
| + `torch.compile` default mode | ~1085 | **~2.6×** | ~125 s compile |
| + `max-autotune` | ~667 | **~4.2×** | ~270 s compile |

---

## Optimizations implemented (default-on, no opt-in setting)

### ✅ 1. `torch.compile` on by default for PyTorch backend
- `UpscalePytorch.__init__` takes `torch_compile: bool = True`; `_load()` wraps the spandrel inference helper via `UpscaleModelWrapper.enable_compile()` (`torch.compile(model, dynamic=False)`) and warms up with a dummy input at the exact render shape (padded tile size when tiled, video dimensions otherwise) so no recompile mid-render.
- **Root cause fixed along the way:** `@torch.inference_mode()` on `__init__`/`_load` made model parameters inference tensors, which broke compile tracing (`Inference tensors cannot be saved for backward`). Decorators removed from those methods; inference is still wrapped at call sites. Warmup itself runs under `no_grad`.
- Compile cost lands in the persistent Inductor cache → paid **once per model+shape** across runs (~125 s on 9070 XT, cached thereafter).
- **Capability gate:** compile availability is decided strictly through
  `torch.cuda.get_device_capability()` (works for CUDA and ROCm; XPU via
  `torch.xpu`), not by arch name. `BackendDetect` exposes the check as
  `compile_available()` with a floor of `cc >= 8.0` (Ampere+ on CUDA; every
  ROCm/XPU arch qualifies), and `pytorch_available()` with `cc >= 6.0`. The
  pytorch backend is dropped from `rve-backend.py --list_backends` when any
  PyTorch GPU falls below the floor. Capability is the sole gate — an arch is
  never disabled because a trace happened to crash; that is a runtime error
  caught by the eager fallback below, not a capability floor.
- Fallback: any tracing error logs "torch.compile failed, falling back to eager
  execution" and restores the eager helper — a bad model never blocks
  rendering (verified on the RX 6800 XT, where the warmup trace fails at first
  run but eager inference runs fine at 512²).
- Temporal helpers (AnimeSR/TSPAN) are not traceable → `enable_compile()`
  returns None for them; they stay eager.

### ✅ 2. Hoist per-tile `.to()` out of `renderTiledImage` loop
In `UpscaleTorch.renderTiledImage`: one device/dtype normalization of the input before the tile loop, and `output = torch.zeros(output_shape, ...)` allocated directly on-device with the right dtype — no more per-tile `.to()` dispatch (hundreds of tiles/frame × many frames).

### ✅ 3. Pinned memory for CPU→GPU frame upload
`TorchUtils.frame_to_tensor` now copies raw bytes into a persistent **pinned staging buffer** and uploads `non_blocking=True`, so the DMA is actually async on AMD instead of driver-staged pageable copy:
- Buffers keyed by `id(stream)` — concurrent streams never share one (async transfer may still be reading it); sequential calls on the same stream reuse it (`frame_to_tensor` syncs that stream before returning).
- Grow-only allocation with 25% headroom, min 16 MB; reuse threshold is `numel() >= nbytes`. Falls back to pageable `torch.frombuffer` if pinned alloc fails.
- D2H path (`tensor_to_frame`) already syncs via `.cpu()` — unchanged.

---

## Evaluated & skipped

### ❌ `channels_last` by default
Measured only ~4% on ROCm (MIOpen), and spandrel/VSF archs assume NCHW internally in places;
forcing the format through arbitrary user models risks silent breakage for little gain.
Revisit if it ever becomes part of a compile pipeline that benefits from NHWC kernels.

### ❌ Per-device tuning flag branching (`torch.xpu.backends.cudnn`)
Doesn't exist on this torch build (2.14.0+rocm7.14); top-level `torch.backends.cudnn.benchmark`
is the documented API and covers CUDA/ROCm/XPU — confirmed by reading installed source
(`PropModule` → `_C._set_cudnn_benchmark`).

---

## Verification checklist
- [x] Pytest suite (`backend/tests/unit_tests/test_pytorch_optimizations.py`): pinned-upload byte equality, tiled-vs-full-frame output parity, compile default-on / opt-out behavior — 5/5 pass on RX 9070 XT (fp16)
- [x] `bench_torch_upscale.py` numbers measured on 4x-UltraSharpV2.pth @ 512² fp16 (full run serial across both GPUs):
  - **RX 9070 XT (cc 12.0):** eager 2518 ms → compiled default 1074 ms (**2.34×**) → compiled max-autotune 609 ms (**4.14×**)
  - **RX 6800 XT (cc 10.3):** eager 8208 ms; compile path fails at warmup trace and falls back to eager (renders fine, never compiled)
- [ ] Real end-to-end render of a short clip with 4x-UltraSharpV2 — first frame slow (compile), rest fast; second run fast from cache

> **Note**: See [TRACKER.md](TRACKER.md) for tracking this item.

## Known limitations / follow-ups
- Pinned staging buffers are per-process (one TorchUtils instance per backend); no cross-process sharing needed since each render run owns its GPU context.
