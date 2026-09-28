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
- [ ] Implement ONNX loading for RIFE model
  - Location: `apps/backend/models/interpolate/rife.py:389`
- [ ] Implement NCNN loading for RIFE model
  - Location: `apps/backend/models/interpolate/rife.py:394`
- [ ] Add MIGraphX and OpenVINO execution providers
- [ ] Fix provider selection logic
- [ ] Wire up scene detection for ONNX backend

#### 2. Model Conversion
- [ ] Implement SPAN conversion to ONNX
  - Location: `apps/backend/converters/torch_to_onnx.py:174`
- [ ] Test full conversion pipeline: PyTorch → ONNX → NCNN

#### 3. Documentation Cleanup
- [ ] Remove stale content from IMPLEMENTATION_TRACKER.md
- [ ] Update MIGRATION_GUIDE.md future work section
- [ ] Consolidate tracking into this file

### Medium Priority

#### 4. Spandrel Half-Precision Verification
Verify `supports_half` flags for these architectures:
- [ ] ATD
- [ ] CRAFT (marked as not thoroughly tested)
- [ ] DCTLSA
- [ ] DnCNN
- [ ] DRUNet
- [ ] FBCNN
- [ ] FFTformer
- [ ] HVICIDNet
- [ ] IPT
- [ ] MixDehazeNet
- [ ] MMRealSR
- [ ] NAFNet
- [ ] OmniSR
- [ ] RetinexFormer
- [ ] RGT
- [ ] SAFMN
- [ ] SAFMNBCIE
- [ ] SeemoRe

### Low Priority

#### 5. Code Cleanup
- [ ] Fix FIXME: MULTI-DIMENSION in `GIMM/raft.py:64`
- [ ] Fix FIXME in `Swin2SR/__arch/Swin2SR.py:637`
- [ ] Fix FIXME in `timm/__drop.py:163`

#### 6. Future Enhancements
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
- Provider selection needs refinement for AMD GPUs
- Scene detection not wired up for ONNX
- InterpolateONNX.py has hardcoded dimensions

### PyTorch Limitations
- Pinned staging buffers are per-process (no cross-process sharing)
- torch.compile warmup can be slow on first run (~125s on 9070 XT)

---

## References

- [AGENTS.md](../AGENTS.md) - Repository guidelines
- [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) - Folder layout
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration documentation
- [pytorch-optimizations.md](pytorch-optimizations.md) - Performance details
- [onnx-backend-plan.md](onnx-backend-plan.md) - ONNX implementation plan

---

## Last Updated
2026-09-28
