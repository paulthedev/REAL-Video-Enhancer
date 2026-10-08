# Migration Guide: Backend Architecture

This guide documents the migration from the old monolithic structure to the new backend-agnostic architecture.

## Overview

The migration restructures the repository to separate concerns and make the backend truly backend-agnostic.

## Directory Structure Changes

### Old Structure
```
REAL-Video-Enhancer/
├── backend/
│   ├── src/
│   │   ├── pytorch/      # 15+ files, multiple architectures
│   │   ├── ncnn/         # 3 files
│   │   ├── onnx/         # 2 files
│   │   └── utils/        # Shared utilities
│   ├── tests/
│   └── rve-backend.py
├── src/
│   ├── Backendhandler.py
│   ├── ModelHandler.py
│   ├── DownloadDeps.py
│   ├── ui/
│   └── ...
└── tests/
    └── unit_tests/
```

### New Structure
```
REAL-Video-Enhancer/
├── apps/
│   ├── gui/              # Main GUI application
│   │   ├── ui/           # UI components
│   │   ├── Backendhandler.py
│   │   ├── ModelHandler.py
│   │   ├── DownloadDeps.py
│   │   └── ...
│   └── backend/          # Backend processing library
│       ├── models/       # Backend-agnostic model definitions
│       │   ├── interpolate/
│       │   ├── upscale/
│       │   ├── restoration/
│       │   ├── denoise/
│       │   └── scene_detect/
│       ├── backends/     # PyTorch, ONNX, NCNN backends
│       │   ├── pytorch/
│       │   ├── onnx/
│       │   └── ncnn/
│       ├── converters/   # Model format converters
│       ├── pytorch/      # PyTorch utilities and architectures
│       ├── utils/        # Shared utilities
│       └── tests/
├── config/               # Configuration files
├── scripts/              # CLI tools
└── docs/                 # Documentation
```

## Key Changes

### 1. Backend-Agnostic Models

Models are now defined independently of their backend implementation:

```python
# Old: Model tied to specific backend
from backend.src.pytorch.InterpolateRIFE import RIFE

# New: Backend-agnostic model
from apps.backend.models.interpolate.rife import RifeModel
```

### 2. Thin Backends

Backends are now thin wrappers that load and run models:

```python
# Old: Backend had model-specific code
class PyTorchBackend:
    def load_rife(self, path):
        # RIFE-specific code
        
    def load_span(self, path):
        # SPAN-specific code

# New: Backend loads any model
class PyTorchBackendLoader(BaseBackend):
    def load_model(self, model_name, model_path, config=None):
        model = get_model(model_name)
        model.load_pytorch(model_path, config)
```

### 3. Model Registry

Models are registered in a central YAML configuration:

```yaml
# config/models.yaml
models:
  rife422_lite:
    type: interpolate
    backend: pytorch
    path: ./models/rife/rife422_lite.pt
    config:
      scale: 1.0
      ensemble: false
```

### 4. Import Path Changes

| Old Import | New Import |
|------------|------------|
| `from backend.src.pytorch.TorchUtils import TorchUtils` | `from apps.backend.pytorch.TorchUtils import TorchUtils` |
| `from backend.src.pytorch.InterpolateArchs.RIFE.warplayer import warp` | `from apps.backend.pytorch.InterpolateArchs.RIFE.warplayer import warp` |
| `from src.Backendhandler import BackendHandler` | `from apps.gui.Backendhandler import BackendHandler` |
| `from src.ModelHandler import getModels` | `from apps.gui.ModelHandler import getModels` |

### 5. Test Location Changes

Tests are now co-located with their respective apps:

| Old Location | New Location |
|--------------|--------------|
| `tests/unit_tests/test_rife_model.py` | `apps/backend/tests/unit_tests/test_rife_model.py` |
| `tests/unit_tests/test_pytorch_backend.py` | `apps/backend/tests/unit_tests/test_pytorch_backend.py` |

## Running Tests

```bash
# Run all backend unit tests
uv run python -m unittest discover -s apps/backend/tests/unit_tests -p "test_*.py" -v

# Run specific test
uv run python -m unittest apps.backend.tests.unit_tests.test_rife_model -v
```

## Model Conversion

Use the CLI tool to convert models between formats:

```bash
# PyTorch to ONNX
python scripts/convert_model.py --model-path model.pt --output-path model.onnx --model-type rife --version rife422_lite

# ONNX to NCNN
python scripts/convert_model.py --model-path model.onnx --output-path model.ncnn --model-type rife --to-ncnn
```

## Migration Checklist

- [x] Create `apps/` directory structure
- [x] Migrate models to backend-agnostic definitions
- [x] Create thin backend wrappers
- [x] Create model converters
- [x] Migrate GUI components
- [x] Update all import paths
- [x] Move tests to co-located locations
- [x] Update configuration files
- [x] Update documentation
- [x] Run all tests and verify

## Backward Compatibility

The old `backend/src/` directory has been removed. All code should use the new import paths.

## Future Work

See [TRACKER.md](TRACKER.md) for current pending work and priorities.
