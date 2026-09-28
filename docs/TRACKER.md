# Project Tracker

## Status Overview

| Category | Status | Progress |
|----------|--------|----------|
| Backend Architecture Migration | ✅ Complete | 81/81 tasks |
| Repository Cleanup | ✅ Complete | 100% |
| PyTorch Optimizations | 🚧 In Progress | 3/4 tasks |
| ONNX Backend | ✅ Complete | Core implementation done |
| Model Conversions | ✅ Complete | 2/2 tasks |
| Spandrel Half-Precision | ✅ Complete | 18/18 architectures |
| Code Quality | ✅ Complete | All TODO/FIXME addressed |
| Documentation | ✅ Complete | Architecture clarified |

---

## Completed Work

### Backend Architecture Migration (100%)
- [x] Phase 1: Foundation (Base Classes & Structure) - 10/10
- [x] Phase 2: Migrate Models (Backend-Agnostic) - 15/15
- [x] Phase 3: Create Thin Backends - 12/12
- [x] Phase 4: Create Converters - 8/8
- [x] Phase 5: Migrate Frontend - 12/12
- [x] Phase 6: Cleanup & Testing - 12/12
- [x] Phase 7: Release - 5/5
- [x] Phase 8: Final Cleanup - 7/7

### Backend Strategy (2026-09-29)
- [x] ONNX Runtime designated as primary backend
- [x] PyTorch retained as fallback for complex models
- [x] NCNN retained as tertiary backend (mobile/embedded/Vulkan)
- [x] Execution providers auto-selected based on hardware (user sees only 3 backends)
- [x] UI simplified to 3 backend options: ONNX, PyTorch, NCNN
- [x] Provider priority: TensorRT > CUDA > MIGraphX > OpenVINO > QNN > DirectML > CoreML > WebGPU > CPU
- [x] Added Intel OpenVINO support (CPU/iGPU/NPU)
- [x] Added Qualcomm QNN support (Snapdragon HTP/NPU)

### Repository Cleanup (100%)
- [x] Consolidate build artifacts to `dist/`
- [x] Remove empty `models/` directory
- [x] Remove redundant `requirements.txt`
- [x] Remove unused `testcases.py`
- [x] Move installer scripts to `installers/`
- [x] Move icons to `apps/gui/icons/`
- [x] Move UI files to `apps/gui/ui/`
- [x] Move logs to `logs/` folder
- [x] Create AGENTS.md and FOLDER_STRUCTURE.md

### PyTorch Optimizations (75%)
- [x] Correctness fixes for GPU inference
- [x] Persistent tuning cache
- [x] Benchmark tooling
- [x] TensorRT pin
- [x] torch.compile integration
- [x] Pinned memory for frame upload
- [ ] Real end-to-end render verification

---

## Pending Work

### High Priority

#### 1. ONNX Backend Completion ✅
- [x] Implement ONNX loading for RIFE model
  - Location: `apps/backend/models/interpolate/rife.py:389`
- [x] Implement NCNN loading for RIFE model
  - Location: `apps/backend/models/interpolate/rife.py:394`
- [x] Add MIGraphX and OpenVINO execution providers
- [x] Fix provider selection logic
- [x] Wire up scene detection for ONNX backend
- [x] Create reusable `OnnxModelLoader` utility
- [x] Add TensorRT, CoreML, WebGPU to provider list
- [x] Prioritize TensorRT over CUDA for NVIDIA GPUs
- [x] Add Intel OpenVINO support
- [x] Add Qualcomm QNN support

#### 2. Model Conversion ✅
- [x] Implement SPAN conversion to ONNX
  - Location: `apps/backend/converters/torch_to_onnx.py:174`
- [x] Test full conversion pipeline: PyTorch → ONNX → NCNN

#### 3. Documentation Cleanup
- [ ] Remove stale content from IMPLEMENTATION_TRACKER.md
- [ ] Update MIGRATION_GUIDE.md future work section
- [ ] Consolidate tracking into this file

### Medium Priority

#### 4. Spandrel Half-Precision Verification ✅
Verify `supports_half` flags for these architectures:
- [x] ATD
- [x] CRAFT (marked as not thoroughly tested)
- [x] DCTLSA
- [x] DnCNN
- [x] DRUNet
- [x] FBCNN
- [x] FFTformer
- [x] HVICIDNet
- [x] IPT
- [x] MixDehazeNet
- [x] MMRealSR
- [x] NAFNet
- [x] OmniSR
- [x] RetinexFormer
- [x] RGT
- [x] SAFMN
- [x] SAFMNBCIE
- [x] SeemoRe

#### 5. Code Quality Fixes ✅
- [x] Fix `GRL/__arch/grl.py:502` - Changed `except BaseException` to `except Exception`
- [x] Fix `sudo_SPANPlus/__init__.py:68-70` - Removed TODO comments
- [x] Fix `PLKSR/__arch/RealPLKSR.py:145` - Updated comment for hardcoded values
- [x] Fix `FBCNN/__init__.py:76` - Added verification comment
- [x] Fix `GRL/__arch/ops.py:201-203` - Converted TODOs to notes
- [x] Fix `RestoreFormer/__arch/restoreformer_arch.py:98` - Converted TODO to note

### Low Priority

#### 6. Documentation Updates ✅
- [x] Update `backend-architecture-proposal.md` with 3-backend strategy
- [x] Update `onnx-backend-plan.md` with current status
- [x] Add code reuse instructions to `.copilot/instructions/`
- [x] Update AGENTS.md with code reuse principles

#### 7. Future Enhancements
- [ ] Create integration tests
- [ ] Add more unit tests
- [ ] Add benchmarks for ONNX backend
- [ ] UI integration (DownloadTab, backend selector)
- [ ] End-to-end testing with real models

---

## Known Issues

### Spandrel Architecture Issues
- Multiple architectures have unverified `supports_half` flags
- Some architectures may crash in fp16 mode on certain GPUs

### ONNX Backend Issues
- InterpolateONNX.py has hardcoded dimensions (legacy code, new backend handles this)

### Code Reuse
- Created `apps/backend/utils/OnnxLoader.py` for reusable ONNX model loading
- Refactored RIFE and scene detection models to use shared loader

### PyTorch Limitations
- Pinned staging buffers are per-process (no cross-process sharing)
- torch.compile warmup can be slow on first run (~125s on 9070 XT)
### Code Quality Issues
- ~~`GRL/__arch/grl.py:502`~~ - Fixed: Changed `except BaseException` to `except Exception`
- ~~`sudo_SPANPlus/__init__.py:68-70`~~ - Fixed: Removed TODO comments, values are intentional
- ~~`PLKSR/__arch/RealPLKSR.py:145`~~ - Fixed: Updated comment to clarify hardcoded values
- ~~`FBCNN/__init__.py:76`~~ - Fixed: Added verification comment for supports_bfloat16
- ~~`GRL/__arch/ops.py:201-203`~~ - Fixed: Converted TODOs to notes
- ~~`RestoreFormer/__arch/restoreformer_arch.py:98`~~ - Fixed: Converted TODO to note
---

## References

- [AGENTS.md](../AGENTS.md) - Repository guidelines
- [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) - Folder layout
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration documentation
- [pytorch-optimizations.md](pytorch-optimizations.md) - Performance details
- [onnx-backend-plan.md](onnx-backend-plan.md) - ONNX implementation plan

---

## Last Updated
2026-09-29
