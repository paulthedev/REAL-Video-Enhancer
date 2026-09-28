# ONNX Backend — Implementation Plan

**Status**: Core implementation complete (2026-09-29)
**Goal**: Make ONNX the primary backend with full hardware support and UI integration

## Current Status

### ✅ Completed
- Backend architecture migrated to `apps/backend/backends/onnx/`
- Provider selection implemented: TensorRT > CUDA > MIGraphX > OpenVINO > QNN > DirectML > CoreML > WebGPU > CPU
- `OnnxModelLoader` utility created in `apps/backend/utils/OnnxLoader.py`
- ONNX backend registered as `"onnx"` (not "directml")
- Models implemented: RIFE interpolation, SPAN upscale, scene detection
- Converters: PyTorch → ONNX, ONNX → NCNN

### 🔄 In Progress
- UI integration (DownloadTab, backend selection)
- End-to-end testing with real models
- Performance benchmarking

### ⏳ Pending
- Full UI wiring (installer, model selection)
- Scene detection ONNX integration
- Interpolate ONNX refinement

## 2. Execution Provider Matrix

**Status**: ✅ Implemented in `OnnxModelLoader.PROVIDER_PRIORITY`

| Priority | Provider | Hardware | pip package | Runtime name |
|----------|----------|----------|-------------|--------------|
| 1 | TensorRT | NVIDIA GPU (max throughput) | `onnxruntime-tensorrt` | `TensorrtExecutionProvider` |
| 2 | CUDA | NVIDIA GPU (standard) | `onnxruntime-gpu` | `CUDAExecutionProvider` |
| 3 | MIGraphX | AMD GPU (ROCm) | `onnxruntime-migraphx` | `MIGraphXExecutionProvider` |
| 4 | OpenVINO | Intel CPU/iGPU/NPU | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| 5 | QNN | Qualcomm Snapdragon | `onnxruntime-qualcomm` | `QnnExecutionProvider` |
| 6 | DirectML | Windows DX12 GPUs | `onnxruntime-directml` | `DmlExecutionProvider` |
| 7 | CoreML | Apple Silicon | (built-in) | `CoreMLExecutionProvider` |
| 8 | WebGPU | Browser/native | (built-in) | `WebGPUExecutionProvider` |
| 9 | CPU | Universal fallback | `onnxruntime` | `CPUExecutionProvider` |

**Selection logic**: Automatic via `OnnxModelLoader._select_provider()` — checks availability and returns best match.

## 3. Install Path

**Status**: 🔄 In Progress

- [ ] Add ONNX package selection to `DownloadDeps.py`
- [ ] Implement per-hardware package selection (mirrors PyTorch strategy)
- [ ] Version pinning strategy (single ORT version vs per-variant)
- [ ] Log which EP was chosen at runtime

**Note**: Variant wheels have different release dates (`onnxruntime-migraphx` is newer than `onnxruntime-openvino`).

## 4. Backend availability reporting (`backend/rve-backend.py`)

Add an ONNX section next to tensorrt/pytorch/ncnn:
- `import onnxruntime` (guarded), report version + `ort.get_available_providers()`.
- Only advertise the backend if at least one GPU-capable EP is present when a GPU exists, else still advertise it as CPU fallback.
- Print each provider so UI/logs show e.g. `ONNX Runtime: 1.x [CUDAExecutionProvider]`.

## 5. Backend code fixes (required before wiring up)

### `backend/src/onnx/UpscaleONNX.py`
- Provider selection is already dynamic (CUDA → DML → CPU fallback); extend it to
  also try `MIGraphXExecutionProvider` / `OpenVINOExecutionProvider`. Note:
  MIGraphX EP requires a one-time JIT compile of the model and OpenVINO wants its
  own options — both should be validated with real models.
- The constructor takes `gpu_id`, but `RenderVideo.upscaleONNXObject()` passes
  `deviceID=` → `TypeError`. Fix the kwargs to match (prefer aligning with how
  pytorch objects receive `device`/`gpu_id`).
- `precision` is hardcoded `np.float16` in `__init__`; wire through a real
  precision argument and only convert the model to fp16 when the provider supports it.

### `backend/src/onnx/InterpolateONNX.py` — mostly broken, needs work before use
- Imports `checkForDirectMLHalfPrecisionSupport` from `utils.Util`, which does not exist anymore → import error. Replace with the same precision logic as UpscaleONNX (or a new shared helper).
- The class is named `UpscaleONNX` but holds interpolate code; rename to match its role, and export it properly in `src/onnx/__init__.py`.
- `render()` references undefined attributes (`self.ph`, `self.pw`, `self.dtype`,
  `self.device`) and a torch-style `np.full(...)` — rewrite against the actual RIFE ONNX model's input signature.
- `bytesToFrame` hardcodes 1080p; make width/height dynamic like UpscaleONNX does with preallocated buffers.

### `backend/src/RenderVideo.py`
- Wire `setupInterpolate()` for the onnx backend (currently only ncnn/pytorch/tensorrt are handled — ONNX RIFE would silently leave `interpolateOption = None`).
- Scene detection: `SceneDetect` maps unknown backends to the NCNN scene-detect model; decide whether onnx uses pytorch-scenedetect, an ONNX export of sudo scenedetect (there is already a `sudo_efficientnet_scenedetect.onnx` produced in `pytorch/scenechangedetect/sudo_scenechange.py`), or falls back to mean/pyscenedetect.
- Pass the correct gpu id/device for onnx (`self.device`, `gpu_id`) consistently.

## 6. UI changes

- `testRVEInterface.ui`: replace the "DirectML - NOT IMPLEMENTED YET" row with an ONNX installer (download/uninstall buttons + label), or rename in place and keep widget names to minimize churn.
- `src/REAL-Video-Enhancer.py` line ~127: remove/hide logic for `directMLBackendInstallerContainer`.
- `src/ui/DownloadTab.py`: currently disables the DirectML button unconditionally; wire it to `self.download("onnx", True)` and add the installed-state handling in `showUninstallButton()`.
- `getModels()` already returns ONNX model lists (RIFE 4.22 onnx, SPAN 2x) — verify those files are actually downloadable from the models repo before enabling interpolate; otherwise start with upscale-only for v1.

## 7. Verification plan

Per EP: load a small test image through `UpscaleONNX.__main__`-style benchmark
(mirror of `tests/benchmarks/`) and confirm output matches pytorch SPAN within tolerance, plus an fps number to decide default provider ordering (CUDA vs MIGraphX vs DirectML) on each vendor.

## Suggested order

1. Rename backend `"directml"` → `"onnx"` across the codebase (pure refactor).
2. Fix `UpscaleONNX` kwargs + dynamic providers; get upscale working end-to-end via CLI (`--backend onnx`).
3. Add install path in `DownloadDeps.py` with per-hardware package selection.
4. Backend availability reporting in `rve-backend.py`.
5. UI row + DownloadTab wiring.
6. Interpolate (RIFE ONNX) — rewrite `InterpolateONNX`, scene-detect fallback, then enable interpolate models in the UI.
