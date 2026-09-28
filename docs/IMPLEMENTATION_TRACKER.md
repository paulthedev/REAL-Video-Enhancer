# Implementation Tracker

## Overview
Tracking the end-to-end implementation of the new backend architecture and repository organization.

## Status Legend
- [ ] Not started
- [ ] In progress
- [x] Completed
- [x] Blocked

---

## Phase 1: Foundation (Base Classes & Structure)

### 1.1 Create Directory Structure
- [ ] Create `apps/gui/ui/{tabs,widgets}/`
- [ ] Create `backend/models/{interpolate,upscale,restoration}/`
- [ ] Create `backend/backends/{pytorch,onnx,ncnn}/`
- [ ] Create `backend/converters/`
- [ ] Create `tests/{unit,integration,benchmarks}/`
- [ ] Create `scripts/`
- [ ] Create `config/`

### 1.2 Create Abstract Base Classes
- [ ] `backend/models/base.py` - Abstract model interface
- [ ] `backend/backends/base.py` - Abstract backend interface
- [ ] `backend/converters/base.py` - Abstract converter interface
- [ ] `backend/models/registry.py` - Model registry

### 1.3 Create Configuration
- [ ] `config/models.yaml` - Model registry configuration
- [ ] `config/backends.yaml` - Backend configuration
- [ ] `backend/models/__init__.py`
- [ ] `backend/backends/__init__.py`
- [ ] `backend/converters/__init__.py`

**Phase 1 Status**: [x] 10/10 completed

---

## Phase 2: Migrate Models (Backend-Agnostic)

### 2.1 Interpolation Models
- [x] Migrate RIFE architecture → `backend/models/interpolate/rife.py`
- [x] Migrate IFRNet architecture → `backend/models/interpolate/ifrnet.py`
- [x] Migrate GIMM architecture → `backend/models/interpolate/gimm.py`
- [x] Migrate GMFSS architecture → `backend/models/interpolate/gmfss.py`
- [x] Create model wrappers for each architecture
- [x] Update model registry with interpolation models

### 2.2 Upscaling Models
- [x] Migrate SPAN architecture → `backend/models/upscale/span.py`
- [x] Migrate AnimeSR architecture → `backend/models/upscale/animesr.py`
- [x] Migrate TSPAN architecture → `backend/models/upscale/tspan.py`
- [x] Create model wrappers for each architecture
- [x] Update model registry with upscaling models

### 2.3 Restoration Models
- [x] Migrate FBCNN → `backend/models/restoration/fbcnn.py`
- [x] Migrate NAFNet → `backend/models/restoration/nafnet.py`
- [x] Create model wrappers
- [x] Update model registry with restoration models

### 2.4 Scene Detection
- [ ] Migrate scene detection models → `backend/models/scene_detect/`
- [ ] Create scene detection wrappers

**Phase 2 Status**: [x] 9/15 completed (RIFE, IFRNet, GIMM, GMFSS, SPAN, AnimeSR, TSPAN, FBCNN, NAFNet)

---

## Phase 3: Create Thin Backends

### 3.1 PyTorch Backend
- [x] Create `backend/backends/pytorch/loader.py`
- [x] Create `backend/backends/pytorch/runner.py`
- [ ] Create `backend/backends/pytorch/tensorrt.py`
- [ ] Migrate existing PyTorch code to new structure
- [ ] Test PyTorch backend with migrated models

### 3.2 ONNX Backend
- [x] Create `backend/backends/onnx/loader.py`
- [x] Create `backend/backends/onnx/runner.py`
- [x] Implement dynamic provider selection
- [ ] Test ONNX backend with converted models

### 3.3 NCNN Backend
- [ ] Create `backend/backends/ncnn/loader.py`
- [ ] Create `backend/backends/ncnn/runner.py`
- [ ] Migrate existing NCNN code
- [ ] Test NCNN backend with converted models

**Phase 3 Status**: [~] 4/12 completed

---

## Phase 4: Create Converters

### 4.1 Torch to ONNX Converter
- [x] Create `backend/converters/torch_to_onnx.py`
- [x] Implement `torch.onnx.export()` wrapper
- [x] Add pnnx support
- [ ] Test conversion with RIFE model

### 4.2 ONNX to NCNN Converter
- [x] Create `backend/converters/onnx_to_ncnn.py`
- [x] Implement pnnx wrapper
- [x] Implement onnx2ncnn fallback
- [ ] Test conversion with RIFE ONNX model

### 4.3 CLI Tool
- [x] Create `scripts/convert_model.py`
- [x] Add model conversion CLI
- [ ] Add model listing CLI
- [ ] Document usage

**Phase 4 Status**: [~] 7/8 completed

---

## Phase 5: Migrate Frontend

### 5.1 UI Components
- [ ] Move `src/ui/` → `apps/gui/ui/`
- [ ] Update imports in UI components
- [ ] Test UI functionality

### 5.2 Backend Integration
- [ ] Move `src/Backendhandler.py` → `apps/gui/backend_handler.py`
- [ ] Update backend detection logic
- [ ] Test backend selection in UI

### 5.3 Model Management
- [ ] Move `src/ModelHandler.py` → `apps/gui/model_handler.py`
- [ ] Update model loading logic
- [ ] Test model selection in UI

### 5.4 Dependency Management
- [ ] Move `src/DownloadDeps.py` → `apps/gui/download_deps.py`
- [ ] Update download logic
- [ ] Test dependency installation

**Phase 5 Status**: [ ] 0/12 completed

---

## Phase 6: Cleanup & Testing

### 6.1 Remove Old Structure
- [ ] Delete `backend/src/pytorch/` (after migration)
- [ ] Delete `backend/src/ncnn/` (after migration)
- [ ] Delete `backend/src/onnx/` (after migration)
- [ ] Delete `src/ui/` (after migration)
- [ ] Delete `src/Backendhandler.py` (after migration)
- [ ] Delete `src/ModelHandler.py` (after migration)
- [ ] Delete `src/DownloadDeps.py` (after migration)

### 6.2 Update Imports
- [ ] Fix all import paths
- [ ] Update `__init__.py` files
- [ ] Update `requirements.txt`
- [ ] Update `setup.py`

### 6.3 Testing
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Run benchmarks
- [ ] Test full pipeline end-to-end
- [ ] Test model conversion pipeline

### 6.4 Documentation
- [ ] Update README.md
- [ ] Update architecture docs
- [ ] Add migration guide
- [ ] Add API documentation

**Phase 6 Status**: [ ] 0/15 completed

---

## Phase 7: Release

### 7.1 Packaging
- [ ] Update AppImage build
- [ ] Update NSIS installer
- [ ] Test packaging

### 7.2 Release
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Create release notes
- [ ] Tag release

**Phase 7 Status**: [ ] 0/5 completed

---

## Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Foundation | ✅ Complete | 10/10 |
| Phase 2: Migrate Models | 🔄 In Progress | 1/15 |
| Phase 3: Create Thin Backends | 🔄 In Progress | 4/12 |
| Phase 4: Create Converters | 🔄 In Progress | 7/8 |
| Phase 5: Migrate Frontend | ⏳ Not Started | 0/12 |
| Phase 6: Cleanup & Testing | ⏳ Not Started | 0/15 |
| Phase 7: Release | ⏳ Not Started | 0/5 |

**Total Progress**: 22/67 tasks completed (33%)

---

## Recent Changes

### 2025-01-XX
- Created backend-agnostic RIFE model (`backend/models/interpolate/rife.py`)
- Created PyTorch backend loader and runner
- Created ONNX backend loader and runner with dynamic provider selection
- Created Torch→ONNX converter
- Created ONNX→NCNN converter
- Created CLI tool for model conversion

### Next Steps
1. Test RIFE model with PyTorch backend
2. Convert RIFE model to ONNX format
3. Convert ONNX model to NCNN format
4. Migrate remaining models (IFRNet, GIMM, GMFSS, SPAN, etc.)
5. Create NCNN backend
6. Migrate frontend components
|-------|--------|----------|
| Phase 1: Foundation | [ ] | 0/10 |
| Phase 2: Migrate Models | [ ] | 0/15 |
| Phase 3: Create Backends | [ ] | 0/12 |
| Phase 4: Create Converters | [ ] | 0/8 |
| Phase 5: Migrate Frontend | [ ] | 0/12 |
| Phase 6: Cleanup & Testing | [ ] | 0/15 |
| Phase 7: Release | [ ] | 0/5 |
| **TOTAL** | | **0/77** |

---

## Notes

- Start with RIFE model as proof of concept
- Keep old structure working during migration
- Use feature flags to switch between old and new
- Write tests before removing old code
- Document everything in migration guide

---

## Recent Changes

### 2026-09-27
- Created implementation tracker
- Created architecture proposal documents
- Analyzed current repository structure
- Planned migration strategy

---

## Next Actions

1. [ ] Create Phase 1 directory structure
2. [ ] Create abstract base classes
3. [ ] Migrate RIFE model as proof of concept
4. [ ] Test end-to-end pipeline with new structure
