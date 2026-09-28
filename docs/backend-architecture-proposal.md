# Backend Architecture Proposal

## Current Problem

The current structure mixes model logic with backend logic:
- `pytorch/` has 15+ files with multiple architectures (RIFE, IFRNET, GIMM, GMFSS)
- `ncnn/` and `onnx/` are thin wrappers but still have their own model-specific code
- Models can't be easily converted between backends
- Each backend reimplements model loading/running logic

## Backend Strategy

### Primary Backend: ONNX Runtime

ONNX Runtime is now the **primary backend** for all model inference. It provides:

- **Universal hardware support**: TensorRT (NVIDIA), CUDA, MIGraphX (AMD), OpenVINO (Intel), QNN (Qualcomm), DirectML (Windows), CoreML (Apple), WebGPU, CPU
- **Best performance**: TensorRT compiles subgraphs to optimized engines
- **Cross-platform**: Windows, macOS, Linux, ARM64
- **Mature ecosystem**: Well-maintained, extensive documentation

### Fallback Backend: PyTorch

PyTorch remains as a **fallback backend** for:
- Models that don't export well to ONNX
- Training workflows
- Complex control flow or dynamic shapes
- Development and debugging

### Deprecated: NCNN

NCNN is **deprecated for desktop use**. It was designed for mobile/embedded deployment. For desktop applications, ONNX Runtime with TensorRT provides superior performance.

### Backend Priority Order

1. **ONNX Runtime** (primary) - Automatic provider selection: TensorRT > CUDA > MIGraphX > OpenVINO > QNN > DirectML > CoreML > WebGPU > CPU
2. **PyTorch** (fallback) - For models that can't use ONNX
3. **NCNN** (deprecated) - Mobile/embedded only

## Proposed Structure

```
backend/src/
├── models/                    # Backend-agnostic model definitions
│   ├── __init__.py
│   ├── base.py               # Abstract base class for all models
│   ├── interpolate/
│   │   ├── __init__.py
│   │   ├── rife.py           # RIFE architecture definition
│   │   ├── ifrnet.py         # IFRNet architecture
│   │   ├── gimm.py           # GIMM architecture
│   │   └── gmfss.py          # GMFSS architecture
│   └── upscale/
│       ├── __init__.py
│       ├── span.py           # SPAN architecture
│       └── ...
│
├── backends/
│   ├── __init__.py
│   ├── base.py               # Abstract backend interface
│   ├── pytorch/
│   │   ├── __init__.py
│   │   ├── loader.py         # PyTorch model loader
│   │   ├── runner.py         # PyTorch inference runner
│   │   └── ...
│   ├── onnx/
│   │   ├── __init__.py
│   │   ├── loader.py         # ONNX model loader
│   │   ├── runner.py         # ONNX Runtime runner
│   │   └── ...
│   └── ncnn/
│       ├── __init__.py
│       ├── loader.py         # NCNN model loader
│       ├── runner.py         # NCNN inference runner
│       └── ...
│
├── converters/
│   ├── __init__.py
│   ├── torch_to_onnx.py      # PyTorch → ONNX converter
│   ├── onnx_to_ncnn.py       # ONNX → NCNN converter
│   └── ...
│
└── utils/
    ├── Frame.py              # Frame handling (already exists)
    ├── Util.py               # Utilities (already exists)
    └── ...
```

## Key Principles

### 1. Model Definitions are Backend-Agnostic
- Define model architecture once in `models/`
- No backend-specific code (no torch, no onnxruntime, no ncnn imports)
- Just the model structure and weights

### 2. Backends are Thin Wrappers
- Each backend only handles:
  - Loading models from disk
  - Running inference
  - Memory management
- No model-specific logic

### 3. Converters Handle Format Translation
- `torch_to_onnx.py`: Export PyTorch models to ONNX
- `onnx_to_ncnn.py`: Convert ONNX to NCNN format
- Reusable, testable, independent

## Benefits

1. **Model Portability**: Convert a model from PyTorch to ONNX to NCNN with one command
2. **Backend Flexibility**: Swap backends without changing model code
3. **Easier Testing**: Test models independently of backends
4. **Smaller Backends**: Each backend is 2-3 files (loader + runner)
5. **Future-Proof**: Easy to add new backends (TensorRT, CoreML, etc.)

## Migration Path

### Phase 1: Create Abstract Interfaces
- Define `BaseModel` in `models/base.py`
- Define `BaseBackend` in `backends/base.py`
- No breaking changes yet

### Phase 2: Extract Model Definitions
- Move model architectures from `pytorch/` to `models/`
- Keep PyTorch implementations as-is for now
- Add converters

### Phase 3: Create Thin Backends
- Rewrite `onnx/` and `ncnn/` as thin wrappers
- Use model definitions from `models/`
- Deprecate old structure

### Phase 4: Full Migration
- Move PyTorch models to `models/`
- Remove duplicate code
- Clean up old structure

## Example: RIFE Interpolation

### Current (mixed)
```python
# pytorch/InterpolateRIFE.py
import torch
class InterpolateRifeTorch:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
    def __call__(self, frame1, frame2, timestep):
        # torch-specific code
        ...

# ncnn/InterpolateNCNN.py
import ncnn
class InterpolateRIFENCNN:
    def __init__(self, model_path):
        self.net = ncnn.Net()
        self.net.load(model_path)
    def __call__(self, frame1, frame2, timestep):
        # ncnn-specific code
        ...
```

### Proposed (separated)
```python
# models/interpolate/rife.py
class RIFE:
    def __init__(self, model_path):
        self.weights = load_weights(model_path)
    def forward(self, frame1, frame2, timestep):
        # pure Python/NumPy implementation
        ...

# backends/pytorch/runner.py
class PyTorchRunner:
    def __init__(self, model: RIFE):
        self.model = torch_model(model)
    def run(self, frame1, frame2, timestep):
        return self.model(frame1, frame2, timestep)

# backends/onnx/runner.py
class ONNXRunner:
    def __init__(self, model: RIFE):
        self.session = onnx.load(model)
    def run(self, frame1, frame2, timestep):
        return self.session.run(...)

# backends/ncnn/runner.py
class NCNNRunner:
    def __init__(self, model: RIFE):
        self.net = ncnn.Net()
        self.net.load(model)
    def run(self, frame1, frame2, timestep):
        return self.net.run(...)
```

## Next Steps

1. Review and approve this architecture
2. Create `models/base.py` and `backends/base.py`
3. Start with RIFE as a proof of concept
4. Migrate one model at a time

---

# Model Conversion Pipeline

## PyTorch (Safetensors) → ONNX → NCNN

See `docs/pytorch-to-ncnn.md` for the complete conversion guide.

### Quick Summary

```bash
# Install conversion tools
pip install torch safetensors onnx onnxruntime onnxsim
pip install pnnx          # recommended for ONNX→NCNN

# Convert PyTorch → ONNX → NCNN
pnnx model.pt              # produces model.pnnx.onnx + model.ncnn.param + model.ncnn.bin
python -m onnxsim model.onnx model_sim.onnx  # optional simplification
ncnnoptimize model.param model.bin model_opt.param model_opt.bin 1  # fp16 optimization
```

### Integration with Backend Architecture

The converter scripts should live in `backend/src/converters/`:

```python
# converters/torch_to_onnx.py
import torch
import onnx

def export_torch_to_onnx(model, input_shape, output_path, opset=14):
    """Export PyTorch model to ONNX format."""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )

# converters/onnx_to_ncnn.py
import subprocess

def export_onnx_to_ncnn(onnx_path, param_path, bin_path):
    """Convert ONNX to NCNN using pnnx."""
    # pnnx handles ONNX → NCNN conversion
    result = subprocess.run(
        ["pnnx", onnx_path],
        capture_output=True,
        text=True
    )
    # pnnx produces .param and .bin files automatically
    return result.returncode == 0
```

### Model Registry

Create a model registry to track available models and their formats:

```python
# models/registry.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ModelFormat(Enum):
    SAFETENSORS = "safetensors"
    PT = "pt"
    ONNX = "onnx"
    NCNN = "ncnn"

@dataclass
class ModelInfo:
    name: str
    architecture: str  # "rife", "span", etc.
    task: str  # "interpolate", "upscale", etc.
    formats: dict[ModelFormat, str]  # format → file path
    scale: Optional[int] = None  # for upscale models
    supported_backends: list[str] = None  # ["pytorch", "onnx", "ncnn"]

# Example registry entry
MODEL_REGISTRY = {
    "rife422": ModelInfo(
        name="rife422",
        architecture="rife",
        task="interpolate",
        formats={
            ModelFormat.SAFETENSORS: "models/rife422.safetensors",
            ModelFormat.ONNX: "models/rife422.onnx",
            ModelFormat.NCNN: "models/rife422.ncnn",
        },
        supported_backends=["pytorch", "onnx", "ncnn"]
    ),
    "span2x": ModelInfo(
        name="span2x",
        architecture="span",
        task="upscale",
        formats={
            ModelFormat.SAFETENSORS: "models/span2x.safetensors",
            ModelFormat.ONNX: "models/span2x.onnx",
            ModelFormat.NCNN: "models/span2x.ncnn",
        },
        scale=2,
        supported_backends=["pytorch", "onnx", "ncnn"]
    ),
}
```

### Conversion CLI

Create a CLI tool for model conversion:

```bash
# Convert a model to all supported formats
python -m backend.src.converters convert --model rife422 --all-formats

# Convert specific model to specific format
python -m backend.src.converters convert --model rife422 --format onnx

# List available models
python -m backend.src.converters list
```

### Benefits of This Approach

1. **One source of truth**: Models defined once, converted to all backends
2. **Easy to add new backends**: Just implement the backend interface
3. **Model versioning**: Track which format each model is in
4. **Automatic conversion**: Convert models on-demand or at build time
5. **Smaller downloads**: Users only download formats they need

### Migration Strategy

1. **Phase 1**: Create model registry and converter CLI
2. **Phase 2**: Convert RIFE model to ONNX and NCNN as proof of concept
3. **Phase 3**: Create thin backend wrappers that use the registry
4. **Phase 4**: Migrate existing backends to use the new architecture
5. **Phase 5**: Remove duplicate model code from backends
