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

## Remaining work (default-on, per decision — no opt-in setting)

### ✅ 2. Hoist per-tile `.to()` out of `renderTiledImage` loop
In `UpscaleTorch.renderTiledImage`:
- `output = img.new_zeros(output_shape).to(device=..., dtype=...)` → allocate directly on device with the right dtype (input is already there at render time; single no-op check before the loop instead of a per-tile `.to()` dispatch).
- Per-tile `img[...].to(device=self.device, dtype=self.dtype)` → hoist to one normalization of the input tensor up front.
Small but free win on long tiled renders (hundreds of tiles/frame × many frames).

### ✅ 3. Pinned memory for CPU→GPU frame upload
`TorchUtils.frame_to_tensor` builds tensors via `torch.frombuffer(...)` — a *pageable*
CPU buffer, so the `.to(device=..., non_blocking=True)` upload is not actually async on AMD:
- Replace with a persistent pinned staging tensor (allocated once per resolution), copy bytes in, then upload `non_blocking`. Expected single-digit % at 4K; also removes one allocation/frame.
- D2H path (`tensor_to_frame`) already syncs via `.cpu()` — fine as-is.

### 🚧 1. `torch.compile` on by default for PyTorch backend
- Wrap the model with `torch.compile(...)` inside `UpscaleModelWrapper` after test-inference passes (avoid tracing during load/test), guarded to cuda/xpu devices only.
- Uses plain mode first (`max-autotune` is another ~1.6× but also a much longer compile — revisit once stable).
- Compile cost lands in the persistent Inductor cache, so it pays **once per model+shape** across runs. First render of each new model will stall 1–5 min; log a clear one-time message ("compiling model, subsequent frames faster").
- Tiled rendering: tile shapes repeat → still fine with `dynamic=False`; keep pad sizes fixed as they already are in `_load`.
- **Risk: RDNA2 / early-RDNA3 AMD cards gfx1103 (RX 6800 XT) crash torch.compile's warmup trace with a device-side assertion (`hipErrorIllegalState`, CUDA error 401). On ROCm that assertion permanently corrupts the GPU context for the rest of the process/session — even eager ops fail afterward.** Mitigated in `UpscaleTorch._load`: the compile block reads `torch.cuda.get_device_capability(device)[0]` and, when it equals 10 (gfx1103), skips compile and runs eager with a log message. The warmup trace still lives inside a `try/except` that falls back to eager on any runtime error, so a bad model never blocks rendering. Verified working on RX 9070 XT (gfx1201, cc 12); RX 6800 XT (gfx1103, cc 10) path runs eager.
- Risk to watch: models that allocate inside forward (some temporal/SPAN archs) can fail tracing — fall back to eager on first compile/runtime error and log it, so a bad model never blocks rendering.

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

## Verification checklist after remaining work lands
- [ ] 8-model fp16 check still passes (`/tmp/final_check.py`)
- [ ] `bench_torch_upscale.py` before/after numbers on RX 9070 XT (expect ~2.5–4× with compile)
- [ ] Real end-to-end render of a short clip with 4x-UltraSharpV2 — first frame slow (compile), rest fast; second run fast from cache
- [ ] Tiled path: force `tilesize > 0`, confirm no per-frame regression and identical output vs eager
- [ ] Pinned upload: verify frames match byte-for-byte vs pageable path on a sample clip
