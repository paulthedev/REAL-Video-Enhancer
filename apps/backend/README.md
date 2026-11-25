# RVE backend

ONNX Runtime is the only inference backend. The shipping render path lives in
`render/` (`RenderVideo.py` + `render/onnx/`); it is launched by the GUI with
the app's own interpreter:

```bash
python apps/backend/rve-backend.py -i input.mp4 -o output.mp4 -b onnx \
  --upscale_model models/2x_model.onnx
```

Dependencies are defined once in the repo-root `pyproject.toml` (run via
`uv`). The PyTorch code under `pytorch/`, the model definitions under
`models/`, and `converters/torch_to_onnx.py` are dev-only tooling for
producing new ONNX models; install them with `uv sync --extra convert`.
