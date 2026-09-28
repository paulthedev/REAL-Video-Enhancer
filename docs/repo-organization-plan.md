# Repository Organization Plan

## Current Structure Analysis

### Root Level (Mixed Concerns)
```
REAL-Video-Enhancer.py      # Main UI application
build.py                    # Build script
testcases.py                # Test cases
requirements.txt            # Dependencies
```

### Backend (`backend/`)
```
backend/
├── rve-backend.py          # Backend entry point
├── src/
│   ├── pytorch/            # 15+ files, multiple architectures
│   │   ├── InterpolateArchs/  # Model architectures (RIFE, GIMM, GMFSS, IFRNET)
│   │   ├── DRBA/            # Deep Blending
│   │   ├── spandrel/        # Restoration models
│   │   ├── scenechangedetect/ # Scene detection
│   │   ├── VSRArchs/        # Video super-resolution
│   │   ├── Interpolate*.py  # Interpolation implementations
│   │   ├── UpscaleTorch.py  # Upscaling
│   │   └── ...
│   ├── ncnn/               # 3 files (thin wrapper)
│   ├── onnx/               # 2 files (thin wrapper)
│   └── utils/              # Shared utilities
└── tests/
    ├── benchmarks/         # Performance benchmarks
    └── unit_tests/         # Unit tests
```

### Frontend (`src/`)
```
src/
├── Backendhandler.py       # Backend detection
├── ModelHandler.py         # Model management
├── DownloadDeps.py         # Dependency downloading
├── GpuHardware.py          # GPU detection
├── ui/                     # UI components
│   ├── HomeTab.py
│   ├── ProcessTab.py
│   ├── SettingsTab.py
│   ├── DownloadTab.py
│   └── ...
└── ...
```

### Issues with Current Structure

1. **Mixed concerns**: Models and backends are intertwined
2. **Inconsistent organization**: pytorch has 15+ files, ncnn/onnx have 2-3
3. **No clear conversion path**: Can't easily convert between backends
4. **Duplicate code**: Similar logic in multiple places
5. **Hard to add new backends**: Need to understand all existing backends

---

## Proposed New Structure

```
REAL-Video-Enhancer/
│
├── apps/                     # Application entry points
│   ├── gui/                  # Main GUI application
│   │   ├── __init__.py
│   │   ├── main.py          # Entry point
│   │   └── ui/              # UI components
│   │       ├── tabs/
│   │       │   ├── home.py
│   │       │   ├── process.py
│   │       │   ├── settings.py
│   │       │   └── download.py
│   │       └── widgets/
│   └── cli/                  # Command-line interface
│       ├── __init__.py
│       └── main.py
│
├── backend/                  # Core processing backend
│   ├── __init__.py
│   ├── engine.py            # Main rendering engine
│   ├── pipeline.py          # Processing pipeline
│   │
│   ├── models/              # Backend-agnostic model definitions
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base class
│   │   ├── registry.py      # Model registry
│   │   │
│   │   ├── interpolate/     # Interpolation models
│   │   │   ├── __init__.py
│   │   │   ├── rife.py      # RIFE architecture
│   │   │   ├── ifrnet.py    # IFRNet
│   │   │   ├── gimm.py      # GIMM
│   │   │   └── gmfss.py     # GMFSS
│   │   │
│   │   ├── upscale/         # Upscaling models
│   │   │   ├── __init__.py
│   │   │   ├── span.py      # SPAN
│   │   │   └── esrgan.py    # ESRGAN variants
│   │   │
│   │   └── restoration/     # Restoration models
│   │       ├── __init__.py
│   │       ├── fbcnn.py
│   │       └── nafnet.py
│   │
│   ├── backends/            # Backend implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract backend interface
│   │   │
│   │   ├── pytorch/         # PyTorch backend
│   │   │   ├── __init__.py
│   │   │   ├── loader.py    # Load PyTorch models
│   │   │   ├── runner.py    # Run inference
│   │   │   └── tensorrt.py  # TensorRT optimization
│   │   │
│   │   ├── onnx/            # ONNX Runtime backend
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   └── runner.py
│   │   │
│   │   └── ncnn/            # NCNN backend
│   │       ├── __init__.py
│   │       ├── loader.py
│   │       └── runner.py
│   │
│   ├── converters/          # Model format converters
│   │   ├── __init__.py
│   │   ├── torch_to_onnx.py
│   │   ├── onnx_to_ncnn.py
│   │   └── base.py          # Converter base class
│   │
│   └── utils/               # Shared utilities
│       ├── __init__.py
│       ├── frame.py         # Frame handling
│       ├── util.py          # General utilities
│       ├── ffmpeg.py        # FFmpeg integration
│       └── scene_detect.py  # Scene detection
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── unit/                # Unit tests
│   │   ├── test_models.py
│   │   ├── test_backends.py
│   │   └── test_converters.py
│   ├── integration/         # Integration tests
│   │   └── test_pipeline.py
│   └── benchmarks/          # Performance benchmarks
│       ├── bench_torch_interpolate.py
│       └── bench_torch_upscale.py
│
├── docs/                    # Documentation
│   ├── backend-architecture-proposal.md
│   ├── pytorch-to-ncnn.md
│   ├── onnx-backend-plan.md
│   └── repo-organization-plan.md
│
├── scripts/                 # Utility scripts
│   ├── convert_model.py     # Model conversion script
│   ├── benchmark.py         # Benchmarking tool
│   └── download_models.py   # Model downloader
│
├── config/                  # Configuration
│   ├── __init__.py
│   ├── models.yaml          # Model registry config
│   └── backends.yaml        # Backend configuration
│
├── models/                  # Model weights (gitignored)
│   ├── interpolate/
│   ├── upscale/
│   └── restoration/
│
├── dist/                    # Built distributions
│   └── backend/
│
├── appimage/                # AppImage packaging
│
├── requirements.txt         # Python dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.py                 # Package setup
└── README.md
```

---

## Migration Strategy

### Phase 1: Create New Structure (No Breaking Changes)

1. **Create directory structure**
   ```bash
   mkdir -p apps/gui/ui/{tabs,widgets}
   mkdir -p backend/{models/{interpolate,upscale,restoration},backends/{pytorch,onnx,ncnn},converters}
   mkdir -p tests/{unit,integration,benchmarks}
   mkdir -p scripts config
   ```

2. **Create abstract base classes**
   - `backend/models/base.py` - Abstract model interface
   - `backend/backends/base.py` - Abstract backend interface
   - `backend/converters/base.py` - Abstract converter interface

3. **Create model registry**
   - `backend/models/registry.py` - Track all models and their formats

### Phase 2: Migrate Models (Backend-Agnostic)

1. **Extract model architectures**
   - Move `InterpolateArchs/RIFE/` → `backend/models/interpolate/rife.py`
   - Move `InterpolateArchs/GIMM/` → `backend/models/interpolate/gimm.py`
   - Move `InterpolateArchs/GMFSS/` → `backend/models/interpolate/gmfss.py`
   - Move `InterpolateArchs/IFRNET/` → `backend/models/interpolate/ifrnet.py`

2. **Create model wrappers**
   - Each model gets a thin wrapper that implements the base interface
   - No backend-specific code in model definitions

### Phase 3: Create Thin Backends

1. **PyTorch backend** (refactor existing)
   - `backend/backends/pytorch/loader.py` - Load models from disk
   - `backend/backends/pytorch/runner.py` - Run inference
   - Keep existing architecture code, just reorganize

2. **ONNX backend** (rewrite)
   - `backend/backends/onnx/loader.py` - Load ONNX models
   - `backend/backends/onnx/runner.py` - Run inference
   - Use model registry to find models

3. **NCNN backend** (refactor existing)
   - `backend/backends/ncnn/loader.py` - Load NCNN models
   - `backend/backends/ncnn/runner.py` - Run inference

### Phase 4: Create Converters

1. **Torch to ONNX**
   - `backend/converters/torch_to_onnx.py`
   - Use `torch.onnx.export()` or `pnnx`

2. **ONNX to NCNN**
   - `backend/converters/onnx_to_ncnn.py`
   - Use `pnnx` or `onnx2ncnn`

3. **CLI tool**
   - `scripts/convert_model.py` - Convert models between formats

### Phase 5: Migrate Frontend

1. **Reorganize UI components**
   - Move `src/ui/` → `apps/gui/ui/`
   - Update imports

2. **Update backend integration**
   - `src/Backendhandler.py` → `apps/gui/backend_handler.py`
   - Use new backend interfaces

### Phase 6: Cleanup

1. **Remove old structure**
   - Delete `backend/src/pytorch/` (after migration)
   - Delete `backend/src/ncnn/` (after migration)
   - Delete `backend/src/onnx/` (after migration)

2. **Update imports**
   - Fix all import paths
   - Update `__init__.py` files

3. **Run tests**
   - Ensure all tests pass
   - Update test paths

---

## Benefits of New Structure

### 1. Clear Separation of Concerns
- Models are backend-agnostic
- Backends are thin wrappers
- Converters handle format translation

### 2. Easy to Add New Backends
```python
# Just implement the base interface
class MyBackend(BaseBackend):
    def load(self, model: BaseModel) -> Any:
        ...
    def run(self, model: BaseModel, input: Frame) -> Frame:
        ...
```

### 3. Model Portability
```bash
# Convert any model to any format
python scripts/convert_model.py --model rife422 --from pytorch --to onnx
python scripts/convert_model.py --model rife422 --from onnx --to ncnn
```

### 4. Better Testing
- Test models independently of backends
- Test backends with mock models
- Integration tests for full pipeline

### 5. Easier Maintenance
- Smaller, focused files
- Clear responsibilities
- Less duplicate code

---

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| 1 | 1 day | Create new structure, base classes |
| 2 | 2-3 days | Migrate models to backend-agnostic |
| 3 | 2 days | Create thin backend wrappers |
| 4 | 1 day | Create converters |
| 5 | 2 days | Migrate frontend |
| 6 | 1 day | Cleanup and testing |

**Total: ~1 week**

---

## Risk Mitigation

1. **Keep old structure working** during migration
2. **Use feature flags** to switch between old and new
3. **Write tests** before removing old code
4. **Migrate one model at a time** (start with RIFE)
5. **Document everything** in migration guide

---

## Next Steps

1. Review and approve this plan
2. Create Phase 1 directory structure
3. Start with RIFE model migration as proof of concept
4. Migrate remaining models
