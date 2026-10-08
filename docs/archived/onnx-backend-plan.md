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

## 2. Execution Provider Matrix (Automatic Selection)

**Status**: ✅ Implemented in `OnnxModelLoader.PROVIDER_PRIORITY`

**User-facing**: 3 backend options (ONNX, PyTorch, NCNN)
**Internal**: Execution providers auto-selected based on hardware

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

**Selection logic**: Automatic via `OnnxModelLoader._select_provider()` — checks availability and returns best match. Users never see these options.

## 3. Install Path

**Status**: 🔄 In Progress

- [ ] Add ONNX package selection to `DownloadDeps.py`
- [ ] Implement per-hardware package selection (mirrors PyTorch strategy)
- [ ] Version pinning strategy (single ORT version vs per-variant)
- [ ] Log which EP was chosen at runtime

**Note**: Variant wheels have different release dates (`onnxruntime-migraphx` is newer than `onnxruntime-openvino`).
Availability Reporting

**Status**: ✅ Implemented

- ONNX backend registered in `BackendDetect`
- Reports version and available providers
- Logs provider selection at initializations, else still advertise it as CPU fallback.
- Print each pCode Status

**Status**: ✅ Core implementation complete

### `apps/backend/backends/onnx/loader.py`
- ✅ Provider selection with full matrix (TensorRT > CUDA > MIGraphX > OpenVINO > QNN > DirectML > CoreML > WebGPU > CPU)
- ✅ Dynamic provider detection via `ort.get_available_providers()`
- ✅ Session options and provider options handling

### `apps/backend/backends/onnx/runner.py`
- ✅ `ONNXInterpolateRunner` for RIFE interpolation
- ✅ `ONNXUpscaleRunner` for SPAN upscale
- ✅ Dynamic input/output handling

### `apps/backend/utils/OnnxLoader.py`
- ✅ Reusable `OnnxModelLoader` class
- ✅ Automatic provider selection
- ✅ Session management

### `apps/backend/models/`
- ✅ `interpolate/rife.py` - RIFE with ONNX support
- ✅ `upscale/span.py` - SPAN with ONNX support
- ✅ `scene_detect/onnx.py` - ONNX scene detectionsorrt are handled — ONNX RIFE would silently leave `interpolateOption = None`).
- Scene detection: `SceneDetect` maps unknown backends to the NCNN scene-detect model; decide whether onnx uses pytorch-scenedetect, an ONNX export of sudo scenedetect (there is already a `sudo_efficientnet_scenedetect.onnx` produced in `pytorch/scenechangedetect/sudo_scenechange.py`), or falls back to mean/pyscenedetect.
- Pass the correct gpu id/device for onnx (`self.device`, `gpu_id`) consistently.

## 6. UI Integration

**Status**: 🔄 In Progress

- [ ] Add ONNX installer row to `testRVEInterface.ui`
- [ ] Wire `DownloadTab.py` to download ONNX packages
- [ ] Add backend selection in main UI
- [ ] Verify model & Testing

**Status**: 🔄 In Progress

- [ ] End-to-end upscale test (SPAN ONNX)
- [ ] End-to-end interpolate test (RIFE ONNX)
- [ ] Performance benchmarking per EP
- [ ] Cross-platform testing (Windows, Linux, macOS)
- [ ] GPU vendor testing (NVIDIA, AMD, Intel)

## 8. Remaining Work

### High Priority
1. **UI Integration** - Add ONNX to download tab and backend selector
2. **Install Path** - Add ONNX package selection to `DownloadDeps.py`
3. **End-to-End Testing** - Verify full pipeline with real models

### Medium Priority
4. **Scene Detection** - Complete ONNX scene detection integration
5. **Performance Tuning** - Optimize provider selection and memory usage
6. **Error Handling** - Improve fallback logic when providers fail

### Low Priority
7. **Documentation** - User-facing docs for ONNX backend
8. **Benchmarking** - Compare ONNX vs PyTorch performance
9. **Mobile Support** - Explore QNN for Android deployment

## Suggested Order

1. ✅ Backend architecture (complete)
2. ✅ Provider selection (complete)
3. ✅ Model implementations (complete)
4. 🔄 UI integration (in progress)
5. 🔄 Install path (in progress)
6. ⏳ End-to-end testing (pending)
7. ⏳ Performance optimization (pending)
4. Backend availability reporting in `rve-backend.py`.
5. UI row + DownloadTab wiring.
6. Interpolate (RIFE ONNX) — rewrite `InterpolateONNX`, scene-detect fallback, then enable interpolate models in the UI.
