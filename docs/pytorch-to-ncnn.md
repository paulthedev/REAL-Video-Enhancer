Here's the full pipeline as a single markdown doc:

# PyTorch (Safetensors) → ONNX → NCNN

## Prerequisites

```bash
pip install torch safetensors onnx onnxruntime onnxsim
pip install pnnx          # recommended for ONNX→NCNN
# OR build ncnn from source to get onnx2ncnn:
#   git clone https://github.com/Tencent/ncnn && cd ncnn
#   cmake -DNCNN_BUILD_TOOLS=ON .. && make -j

Step 1 — Load Safetensors into a PyTorch Model
import torch
from safetensors.torch import load_file

# Load weights into your model class
weights = load_file("model.safetensors")
model = MyModel()          # your nn.Module
model.load_state_dict(weights)
model.eval()

If the model comes from HuggingFace, use the library's loader instead:

from transformers import AutoModel
model = AutoModel.from_pretrained("org/name")  # reads .safetensors automatically

Step 2 — Export to ONNX
Option A: torch.onnx.export (classic)
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

Option B: pnnx (recommended — also gives you NCNN directly)
import pnnx

dummy_input = torch.randn(1, 3, 224, 224)
pnnx.export(model, "model.pt", (dummy_input,))
# Produces: model.pnnx.onnx  +  model.ncnn.param  +  model.ncnn.bin

Option C: optimum-cli (HuggingFace models)
optimum-cli export onnx --task image-classification --model org/name model_onnx/

Step 3 — Simplify the ONNX (optional but recommended)
python -m onnxsim model.onnx model_sim.onnx

Step 4 — Convert ONNX → NCNN
Option A: pnnx (recommended)
pnnx model_sim.onnx
# Produces: model_sim.ncnn.param  +  model_sim.ncnn.bin

Option B: onnx2ncnn (legacy, from ncnn build)
onnx2ncnn model_sim.onnx model.param model.bin

Option C: ncnnoptimize (post-process, reduce size)
ncnnoptimize model.param model.bin model_optimized.param model_optimized.bin 0
# 0 = float32, 1 = float16, 2 = int8

Step 5 — Inference with NCNN (C++)
#include <ncnn/net.h>

int main() {
    ncnn::Net net;
    net.load_param("model.param");
    net.load_model("model.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels(
        img_data, ncnn::Mat::PIXEL_BGR, width, height);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);
    // use out ...
}

Quick Reference
Step	Tool	Input → Output
Load weights	safetensors.torch.load_file	.safetensors → PyTorch model
Export ONNX	torch.onnx.export / pnnx / optimum-cli	PyTorch → .onnx
Simplify	onnxsim	.onnx → simplified .onnx
To NCNN	pnnx / onnx2ncnn	.onnx → .param + .bin
Optimize	ncnnoptimize	.param+.bin → smaller .param+.bin

Notes
pnnx is the modern, recommended path — it handles PyTorch→ONNX and ONNX→NCNN in one tool, with better operator coverage.
onnx2ncnn is the older C++ converter bundled in ncnn's build/tools/onnx/.
If your model uses operators not yet supported by NCNN, check the supported ONNX operator list and consider rewriting that subgraph in PyTorch with an equivalent that IS supported.
For dynamic shapes, export with dynamic_axes (classic) or set the shape in pnnx accordingly.
