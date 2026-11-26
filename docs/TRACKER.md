# Project Tracker

> Historical project record. The current architecture is **ONNX Runtime only**;
> see [onnx-only-migration-plan.md](onnx-only-migration-plan.md) for the
> authoritative status. Sections below that mention PyTorch/NCNN describe work
> that has since been retired or moved to dev-only conversion tooling.

## Status Overview

| Category | Status | Progress |
|----------|--------|----------|
| Backend Architecture Migration | ✅ Complete | 81/81 tasks |
| Repository Cleanup | ✅ Complete | 100% |
| PyTorch Optimizations | 🗄️ Superseded | ONNX-only migration retired the torch backend |
| ONNX Backend | ✅ Complete | Core implementation done |
| Model Conversions | ✅ Complete | 2/2 tasks |
| Spandrel Half-Precision | ✅ Complete | 18/18 architectures |
| Code Quality | ✅ Complete | All TODO/FIXME addressed |
| Documentation | ✅ Complete | Architecture clarified |
| Backend-Agnostic Hardware Detection | ✅ Complete | `utils/backends/BackendDetect.py` |
| GUI Organization | ✅ Complete | REAL-Video-Enhancer.py moved to apps/gui/ |
| Unified Backend Selection UI | 🗄️ Superseded | ONNX-only UI (see ONNX-Only Migration) |
| Benchmark Tests | ✅ Complete | ONNX + torch-conversion benchmarks |
| VSR Model Support | ✅ Complete | AnimeSR, TSPAN (ONNX runtime) |
| Code Review Fixes (2026-10-08) | ✅ Done | 75/78 fixed; S5 open, C13/C14 by design — see [BUG_TRACKER.md](BUG_TRACKER.md) |
| ONNX-Only Migration | ✅ Phases 0-4 done | Phase 5 hardware/TRT verification pending — [onnx-only-migration-plan.md](onnx-only-migration-plan.md) |
| UI Redesign "Studio Polish" | ✅ Done | Workstreams shipped; checklist retained as history — [UI_REDESIGN_TRACKER.md](UI_REDESIGN_TRACKER.md) |

---

## Completed Work

### ONNX-Only Migration, Phases 0-4 (2026-10-08 → 2026-11-25)
- [x] Phase 0: cp314 wheels confirmed (onnxruntime/-directml/-gpu, imageio-ffmpeg); imageio-ffmpeg rawvideo piping verified
- [x] Phase 1: GUI ONNX-only — ModelHandler rebuilt (ONNX tables), DownloadTab slimmed to updates/uninstall, backend fixed to 'onnx'
- [x] Phase 2: bundled runtime — ffmpeg via imageio-ffmpeg, backend runs on the app interpreter, download-the-backend machinery deleted, vendor-aware ProviderUpgrade manager (NVIDIA/Intel/AMD)
- [x] Phase 2b: provider manager actually wired (opt-in startup prompt), frozen-safe `RVE -m pip` dispatch, TensorRT tier + engine cache, AMD gfx-aware release pick
- [x] Phase 3: rve-backend onnx-only args + listBackends; backends/pytorch, backends/ncnn, render/ncnn, onnx_to_ncnn converter removed; RenderVideo torch/ncnn branches removed
- [x] Phase 4: CI split into dev (convert extra) + torch-free runtime-smoke; docs reconciled
- [~] Phase 5 (in progress): frozen cx_Freeze bundle + pip dispatch + provider activation + synthetic render verified; real-model/provider-hardware/TRT/perf checks pending

### Backend Architecture Migration (100%)
- [x] Phase 1: Foundation (Base Classes & Structure) - 10/10
- [x] Phase 2: Migrate Models (Backend-Agnostic) - 15/15
- [x] Phase 3: Create Thin Backends - 12/12
- [x] Phase 4: Create Converters - 8/8
- [x] Phase 5: Migrate Frontend - 12/12
- [x] Phase 6: Cleanup & Testing - 12/12
- [x] Phase 7: Release - 5/5
- [x] Phase 8: Final Cleanup - 7/7

### Backend Strategy (2026-09-29) — superseded 2026-10-08
Historical decision; the ONNX-only migration replaced it.
- ONNX Runtime was designated the primary backend, with PyTorch and NCNN as
  fallbacks.
- PyTorch (`apps/backend/pytorch/`, model loaders) and NCNN support were later
  removed from the shipping path; they survive only as dev-only conversion
  tooling.
- Current provider priority lives in
  `apps/backend/render/onnx/session.py`: TensorRT > CUDA > MIGraphX/ROCm >
  OpenVINO > DirectML > CPU.

### Backend-Agnostic Hardware Detection (2026-09-29) — historical
- [x] Extended GPU detection (`apps/gui/lib/GPUDetect.py` at the time, now
  `apps/gui/utils/backends/BackendDetect.py`) with vendor detection
- [x] Added ONNX_PROVIDER_HARDWARE mapping (PyTorch/NCNN tables later removed)
- [x] Added get_compatible_backends() / get_preferred_backend() (now ONNX-only)
- [x] Added get_onnx_providers() function
- [x] Added check_backend_health() function
- [x] Added get_backend_reinstall_info() function
- [x] Integrated with DownloadDependencies (`apps/gui/utils/backends/`)
- [x] All 293 tests passing at the time (current suite: 324)

### GUI Refactor Wiring Fix (2026-10-08)
- [x] `assemble_pages` re-points every page widget onto MainWindow (flat namespace preserved) — ~140 broken references fixed
- [x] Restored `DownloadTab.hideUninstallButtons` / `showUninstallButton` (lost in the ui/ → pages/ move)
- [x] Fixed `resources.qrc` paths (rcc resolves relative to the .qrc dir) — icon bundle compiled again (0-byte before)
- [x] `getModels`: merged `onnx | tensorrt | directml` case; unknown backends raise instead of returning `{}`
- [x] Removed dead refactor leftovers: empty `apps/gui/tests/`, `ModelRegistry.py` (no importers), stale local `mainwindow.py`
- [x] Verified by offscreen smoke test: MainWindow constructs, models populate, home page renders (details in BUG_TRACKER.md Wiring section)

### Unified Backend Selection UI (2026-09-29)
- [x] Replaced separate download buttons with unified backend selector
- [x] Added backend_type_combo dropdown (ONNX, PyTorch, NCNN)
- [x] Added backend_variant_combo dropdown (CUDA, ROCm, XPU, MPS for PyTorch)
- [x] Added "Load Backend" button to trigger backend loading
- [x] Dynamic variant filtering based on selected backend type
- [x] Platform-aware variant options (Linux: CUDA/ROCm/XPU, Windows: CUDA, macOS: MPS)
- [x] Integrated with ProcessTab for unified backend selection during processing
- [x] Updated REAL-Video-Enhancer.py to use unified backend selector
- [x] All 293 tests passing at the time (current suite: 324)
- Note (2026-10-08): the selector widgets were unreachable from MainWindow until the
  GUI Refactor Wiring Fix above — the `backend_type_combo` / `backend_variant_combo`
  accesses were `hasattr`-guarded and always False.

### Benchmark Tests
- [x] ONNX interpolation benchmark (`bench_onnx_interpolate.py`)
- [x] ONNX upscale benchmark (`bench_onnx_upscale.py`)
- [x] PyTorch conversion benchmarks (`bench_torch_*`, dev tooling only)
- [x] Provider comparison support (TensorRT, CUDA, CPU) for ONNX
- [x] NCNN benchmarks retired with the ONNX-only migration

### VSR Model Support (2026-09-29) — historical (now ONNX runtime)
- [x] AnimeSR ONNX model (PyTorch/NCNN paths retired)
- [x] TSPAN ONNX model (PyTorch/NCNN paths retired)
- [x] Added TSPAN to the GUI model selection
- [x] Added AnimeSR to the GUI model selection
- [x] VSR models registered in the model registry (converter tooling)
- [x] Temporal frame caching for video consistency

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

### PyTorch Optimizations (historical — superseded)
- [x] Correctness fixes for GPU inference
- [x] Persistent tuning cache
- [x] Benchmark tooling
- [x] TensorRT pin
- [x] torch.compile integration
- [x] Pinned memory for frame upload
- [~] Real end-to-end render verification — moot after the torch backend was
  removed; ONNX render verification is tracked in Phase 5 of the migration plan

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
- [x] Enable operator fusion (Conv+BatchNorm+ReLU)
- [x] Enable ONNX Runtime memory optimization

#### 2. Model ONNX Conversion (complete — converter tooling)
- [x] Add ONNX support for SPAN upscale model
  - Location: `apps/backend/models/upscale/span.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for FBCNN restoration model
  - Location: `apps/backend/models/restoration/fbcnn.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for DnCNN denoise model
  - Location: `apps/backend/models/denoise/dncnn.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for IFRNet interpolation model
  - Location: `apps/backend/models/interpolate/ifrnet.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for GIMM interpolation model
  - Location: `apps/backend/models/interpolate/gimm.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for NAFNet restoration model
  - Location: `apps/backend/models/restoration/nafnet.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for ANIMEsr upscale model
  - Location: `apps/backend/models/upscale/animesr.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only
- [x] Add ONNX support for TSPAN upscale model
  - Location: `apps/backend/models/upscale/tspan.py`
  - Status: Complete — ONNX model; PyTorch source is conversion tooling only

#### 2. Model Conversion ✅
- [x] Implement SPAN conversion to ONNX
  - Location: `apps/backend/converters/torch_to_onnx.py:174`
- [x] Test full conversion pipeline: PyTorch → ONNX (the NCNN step is retired)

#### 3. Documentation Cleanup
- [x] Remove stale content from IMPLEMENTATION_TRACKER.md (archived)
- [x] Update MIGRATION_GUIDE.md future work section (archived)
- [x] Consolidate tracking into this file + ONNX migration plan

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
- [x] Create integration tests
- [x] Add more unit tests
- [x] Add tests for GMFSS and MaxViT models
- [x] Add benchmarks for ONNX backend
- [x] UI integration (Models page: custom ONNX import + conversion hub)
- [x] End-to-end testing with synthetic ONNX models (upscale + interpolation)
- [ ] End-to-end testing with real downloaded models (Phase 5 spot-check)

### GUI Organization (2026-09-29)
- [x] Move REAL-Video-Enhancer.py to apps/gui/ folder
- [x] Update build.py references to new location

#### 8. Model Backend Expansion
- [x] Add ONNX and NCNN support to GMFSS (interpolation)
- [x] Add ONNX and NCNN support to MaxViT Scene Detect

---

## Known Issues

### Spandrel Architecture Issues
- Multiple architectures have unverified `supports_half` flags
- Some architectures may crash in fp16 mode on certain GPUs
- **Mitigation**: `UpscaleModelWrapper.__test_model_precision()` automatically falls back to float32 if fp16 inference fails
- **Mitigation**: Models with `supports_half=False` will raise `UnsupportedDtypeError` when forced to fp16, preventing crashes

### ONNX Backend Issues
- ~~InterpolateONNX.py has hardcoded dimensions~~ - **Fixed**: Legacy file removed during backend migration
- New ONNX backend handles dynamic dimensions correctly

### Code Reuse
- Created `apps/backend/utils/OnnxLoader.py` for reusable ONNX model loading
- Refactored RIFE and scene detection models to use shared loader

### PyTorch Limitations (historical — backend removed)
- Pinned staging buffers are per-process (no cross-process sharing)
- torch.compile warmup can be slow on first run — these paths exist only in
  the dev-only conversion tooling now.
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
- [onnx-only-migration-plan.md](onnx-only-migration-plan.md) - Current architecture and migration status
- [UI_REDESIGN_TRACKER.md](UI_REDESIGN_TRACKER.md) - UI redesign history
- [pytorch-optimizations.md](pytorch-optimizations.md) - Historical PyTorch performance notes (superseded)
- [BUG_TRACKER.md](BUG_TRACKER.md) - Open issues from code review (2026-10-08)

Archived (completed work):
- [backend-architecture-proposal.md](archived/backend-architecture-proposal.md) - Backend architecture proposal
- [MIGRATION_GUIDE.md](archived/MIGRATION_GUIDE.md) - Migration documentation
- [onnx-backend-plan.md](archived/onnx-backend-plan.md) - ONNX implementation plan

---

## Recent Changes (2026-11-25)

### ONNX-only gap closure
- Provider upgrade wired (opt-in prompt), TensorRT tier + engine cache, AMD
  gfx gating, frozen-safe `RVE -m pip` dispatch
- Removed torch/ncnn branches from the render path and GUI
- CPU baseline bundled on Linux; GPU builds runtime-installed
- CI split into dev + torch-free runtime-smoke; frozen build verified

### Backend-Agnostic Hardware Detection (historical)
- Detection now lives in `apps/gui/utils/backends/BackendDetect.py` and is
  ONNX-only (PyTorch/NCNN helpers removed)

### GUI Organization
- Moved `REAL-Video-Enhancer.py` to `apps/gui/REAL-Video-Enhancer.py`
- Updated build.py references

### Test Results
- 324 tests passing, 5 warnings

## Last Updated
2026-11-25
