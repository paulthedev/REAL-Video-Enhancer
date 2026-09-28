# ONNX Backend — Implementation Plan (what needs to happen first)

Goal: make ONNX a first-class backend selectable in the UI, with the correct
`onnxruntime-*` wheel installed per platform/hardware at install time, mirroring
how PyTorch builds are chosen (`BuiltInTorchVersions.py` / `downloadPythonDeps`).

## 1. Naming decision (do this first)

The backend is currently registered as `"directml"`, but ONNX Runtime supports
far more than DirectML. Decide on a single backend name — recommend renaming to
`"onnx"` everywhere:

- `backend/rve-backend.py`: help text for `--backend` (`pytorch/ncnn/tensorrt/directml`)
  and the availability check that adds it to `availableBackends`.
- `backend/src/RenderVideo.py`: all `self.backend == "directml"` branches.
- `src/ModelHandler.py::getModels()`: `case "directml":` → `case "onnx":`.
- UI: the DirectML installer row in `testRVEInterface.ui` / labels, and any
  strings referencing "DirectML".

## 2. Execution-provider matrix (pip package per platform)

Verified against PyPI as of this writing; pin a single ORT version line for all
variants so they can be swapped without ABI surprises:

| OS | Hardware | pip package | Provider name at runtime |
|----|----------|-------------|--------------------------|
| Windows/Linux/macOS* | NVIDIA (CUDA 12.x) | `onnxruntime-gpu` | `CUDAExecutionProvider` |
| Windows, Linux, macOS | AMD GPU (RX 6000+) | `onnxruntime-migraphx` | `MIGraphXExecutionProvider` |
| Win/Linux/macOS | Intel CPU / iGPU / Arc / NPU | `onnxruntime-openvino` | `OpenVINOExecutionProvider` |
| Windows (any DX12 GPU) | fallback / AMD/Intel on Win | `onnxruntime-directml` | `DmlExecutionProvider` (sustained engineering — optional, maybe skip for new installs) |
| macOS | Apple Silicon | CoreML EP exists but is **not** a PyPI wheel; CPU (`onnxruntime`) or WebGPU only. Ship plain `onnxruntime`. |
| any | baseline / no GPU detected | `onnxruntime` (CPU, oneDNN/MLAS) |

\* `onnxruntime-gpu` requires system CUDA + cuDNN on Linux (or torch-tensorrt's
CUDA libs via LD_LIBRARY_PATH — needs testing since RVE bundles its own Python).

Selection logic mirrors `GpuHardware.detect_gpu_hardware()`:
- nvidia → `onnxruntime-gpu`
- amd → `onnxruntime-migraphx` (Linux/Win), else DirectML on Windows, else CPU
- intel discrete/iGPU/NPU → `onnxruntime-openvino`, else CPU
- apple / no GPU → `onnxruntime`

## 3. Install path (`src/DownloadDeps.py`)

- Add a case to `downloadPythonDeps()`: match on `"onnx"` (and legacy
  `"directml"`) and pip-install the chosen package, plus platform-independent deps.
- Decide version pinning strategy like torch: either hard-pin one ORT version for
  all variants in constants (simplest), or a small `OnnxBuiltInVersions` dataclass
  if per-variant versions diverge. Note variant wheels don't share the same latest
  release date (`onnxruntime-migraphx` is newer than `onnxruntime-openvino`).
- Reuse existing GPU vendor detection to pick the package; log which EP was chosen so users know what they got.

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
