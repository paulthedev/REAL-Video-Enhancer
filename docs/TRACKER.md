# Project Tracker

## Status Overview

| Category | Status | Progress |
|----------|--------|----------|
| Backend Architecture Migration | ✅ Complete | 81/81 tasks |
| Repository Cleanup | ✅ Complete | 100% |
| PyTorch Optimizations | 🚧 In Progress | 3/4 tasks |
| ONNX Backend | 🚧 In Progress | Partial |
| Model Conversions | ⬜ Pending | 0/2 |
| Spandrel Half-Precision | ⬜ Pending | 0/20 |

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

#### 1. ONNX Backend Completion
- [x] Implement ONNX loading for RIFE model
  - Location: `apps/backend/models/interpolate/rife.py:389`
- [x] Implement NCNN loading for RIFE model
  - Location: `apps/backend/models/interpolate/rife.py:394`
- [x] Add MIGraphX and OpenVINO execution providers
- [x] Fix provider selection logic
- [x] Wire up scene detection for ONNX backend

#### 2. Model Conversion
- [x] Implement SPAN conversion to ONNX
  - Location: `apps/backend/converters/torch_to_onnx.py:174`
- [x] Test full conversion pipeline: PyTorch → ONNX → NCNN

#### 3. Documentation Cleanup
- [ ] Remove stale content from IMPLEMENTATION_TRACKER.md
- [ ] Update MIGRATION_GUIDE.md future work section
- [ ] Consolidate tracking into this file

### Medium Priority

#### 4. Spandrel Half-Precision Verification
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

### Low Priority

#### 5. Code Cleanup
- [x] Fix FIXME: MULTI-DIMENSION in `GIMM/raft.py:64`
- [x] Fix FIXME in `Swin2SR/__arch/Swin2SR.py:637`
- [x] Fix FIXME in `timm/__drop.py:163`

#### 6. TODO/FIXME Review
- [ ] Fix `GRL/__arch/grl.py:502` - `except BaseException` is suspicious
- [ ] Fix `sudo_SPANPlus/__init__.py:68-70` - Hardcoded scale/input_channels/output_channels
- [ ] Fix `PLKSR/__arch/RealPLKSR.py:145` - Hardcoded in_ch/out_ch
- [ ] Fix `FBCNN/__init__.py:76` - Verify supports_bfloat16
- [ ] Review `GRL/__arch/ops.py:201-203` - Pretrained window size TODOs
- [ ] Review `RestoreFormer/__arch/restoreformer_arch.py:98` - nn.Embedding handling

#### 7. Future Enhancements
- [ ] Add TensorRT backend
- [ ] Create integration tests
- [ ] Add more unit tests
- [ ] Add benchmarks for ONNX backend

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
