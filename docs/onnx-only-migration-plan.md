# ONNX-Only Migration Plan

Status: **planned — not started**
Decided (user, 2026-10-08):

1. ONNX Runtime becomes the **only** inference backend. PyTorch and NCNN go away.
2. The runtime is **bundled from the get-go** — no "download backend later" flow.
   onnxruntime is a pip library, so the frozen build ships it.
3. Same for binary downloads: **ffmpeg ships via a pip package**
   (`imageio-ffmpeg` bundles a per-platform static ffmpeg build; RVE never uses
   ffprobe, so its missing ffprobe is not a problem). The BtbN download path
   and the standalone-CPython download both disappear.

Related: `docs/BUG_TRACKER.md` (S1/S2/S3 superseded by this plan for ffmpeg/
CPython provenance; D1/D2/D3/D4 download wiring mostly deleted), `docs/archived/
onnx-backend-plan.md`.

---

## Current architecture (what gets removed)

| Today | How it works |
|-------|--------------|
| Backend download | GUI runs `--list_backends` via a downloaded standalone CPython (`Python.py` → python-build-standalone); backend bundle `backend-v{ver}.tar.gz` (`Backend.py`) |
| Dependency install | `DownloadDependencies.pip()` installs torch/ncnn/onnxruntime into the standalone env at runtime |
| ffmpeg | downloaded from BtbN/evermeet at runtime (`FFMpeg.py`, `GetFFMpeg.py`) |
| Inference | GUI spawns `rve-backend.py` with `PYTHON_EXECUTABLE_PATH`; three backends (pytorch/ncnn/onnx), three model formats per model |

## Target architecture

- **onnxruntime ships inside the frozen app** (pip dependency picked up by
  cx_Freeze). No backend download, no dependency-install popup, no standalone
  CPython.
- **ffmpeg ships inside the frozen app** via `imageio-ffmpeg`
  (`imageio_ffmpeg.get_ffmpeg_exe()`), with `--ffmpeg_path` / `RVE_FFMPEG_PATH`
  kept as overrides. No runtime binary downloads at all.
- The GUI still spawns `rve-backend.py` as a subprocess, but with the **same
  frozen interpreter** (`sys.executable`) — crash/GIL isolation preserved
  without a second Python install. (Merging backend into the GUI process is a
  possible later step, not part of this migration.)
- **Models still download on demand** from the models repo — ONNX files only.
- Wheel variants: Windows → `onnxruntime-directml` (GPU via DX12, every
  vendor, no external CUDA); Linux → `onnxruntime-gpu` (CUDA via pip nvidia
  deps) with CPU fallback; macOS → `onnxruntime` (CPU/CoreML).
  **Decision point:** one combined bundle vs a small CPU + large GPU variant.

---

## Phase 0 — Verification gate (do before deleting anything)

The ONNX inference path is the least-exercised code in the repo (bugs G10/G11
etc. were fixed but never proven against a real model). Nothing below runs
until:

- [ ] Real ONNX render end-to-end: download one ONNX model from the models
  repo (e.g. `tspan.onnx`) and render a short clip through
  `rve-backend.py -b onnx` — verify output frames are sane.
- [ ] onnxruntime wheel availability for the **build venv's Python (3.14)**.
  If 3.14 wheels don't exist, move the build venv (and cx_Freeze target) to
  3.12/3.13 first. This is a hard gate for "bundled from the get-go".
- [ ] Confirm `imageio-ffmpeg`'s bundled build handles RVE's usage: rawvideo
  piped via stdin/stdout, `-filter_complex`, HDR pix fmts.

## Phase 1 — GUI: ONNX-only user flows

- [ ] `download/download.ui` + `DownloadTab`: remove the unified selector's
  type/variant combos and Load Backend button (ONNX is implicit); remove the
  PyTorch Version/Backend rows and the restack logic that manages them;
  remove `installRecommended`/`load_backend`/`download("torch"/"ncnn")` paths.
- [ ] `ModelHandler.getModels`: collapse to the ONNX model sets only.
- [ ] `lib/ModelHandler.py` model tables: keep `.onnx` entries only.
- [ ] Process page: fixed backend ("ONNX"); `get_current_backend` returns
  "onnx" unconditionally; `build_command` drops `-b` variants.
- [ ] Remove `TorchVersions`, torch-consistency/reinstall flows
  (`check_torch_hardware`, `check_torch_consistency`, `_reinstall_torch`,
  `detect_rocm_targets`, `install_rocm_meta`, `ensure_rocm_targets`),
  `GPUDetect` torch paths.
- [ ] `constants.py`: drop `BACKEND_RELEASE_URL_TEMPLATE`,
  `CPYTHON`/`PYTHON_EXECUTABLE_PATH` machinery (Phase 2), keep models URL.

## Phase 2 — Bundled runtime (kill "download backend later")

- [ ] `pyproject.toml`: add `imageio-ffmpeg`, and onnxruntime with PEP 508
  markers (`onnxruntime-directml; sys_platform == 'win32'`, `onnxruntime-gpu;
  sys_platform == 'linux'`, `onnxruntime; sys_platform == 'darwin'`).
- [ ] ffmpeg resolution: `FFMPEG_PATH` ← `imageio_ffmpeg.get_ffmpeg_exe()`;
  delete `utils/backends/FFmpeg.py`, `apps/backend/utils/GetFFMpeg.py`
  download logic, BtbN/evermeet URLs (keep the constants only if used as
  override docs — otherwise delete, superseding S1).
- [ ] Delete `utils/backends/Python.py` (standalone CPython download) and
  `utils/backends/Backend.py` (backend.tar.gz download); supersede S2.
- [ ] Backend launch: `BackendHandler`/`process.py`/`Updater.py` switch from
  `PYTHON_EXECUTABLE_PATH` to `sys.executable` (frozen) / dev venv python.
  `BACKEND_PATH` resolves to the bundled `backend/` folder.
- [ ] `DownloadDependencies` shrinks to **model downloads only**
  (`BackendsModels.py`); `downloadPythonDeps` and `pip()` are deleted.
- [ ] cx_Freeze build: include onnxruntime/cv2/numpy; verify the freeze picks
  the pip nvidia deps on Linux GPU builds; update `packages/` layout notes.
- [ ] Decide bundle variants (single GPU vs CPU+GPU) and update
  `build-appimage.sh`/CI accordingly.

## Phase 3 — Backend code removal

- [ ] `rve-backend.py`: drop `-b` choices except onnx, drop pytorch/ncnn/tensorrt
  args (`--precision` stays), `--pytorch_gpu_id`/`--ncnn_gpu_id` → `--gpu_id`,
  remove `--no_torch_compile`/TRT dynamic-shape flags.
- [ ] `listBackends` → report ONNX Runtime version + providers only
  (`BackendDetect` shrinks accordingly).
- [ ] Delete: `backends/pytorch/`, `backends/ncnn/`, `render/ncnn/`,
  `models/*/pytorch|ncnn` loader branches (registry models become
  ONNX-only), `models/scene_detect` NCNN path, `TorchVersions`,
  `converters/onnx_to_ncnn.py`.
- [ ] Keep `render/onnx/` wrappers and `backends/onnx/` (loader/runner) as the
  only inference path; keep `converters/torch_to_onnx.py` as dev tooling for
  producing new ONNX models (move to `scripts/` if desired).
- [ ] GUI spawns backend without `--pause_shared_memory_id`-adjacent backend
  flags that referenced other backends; settings page drops precision/backend
  options that were pytorch-specific.

## Phase 4 — Tests, docs, CI

- [ ] Tests: delete pytorch/ncnn model+backend tests, keep ONNX loader/runner,
  registry, utils tests; add the Phase 0 render as an integration smoke test.
- [ ] CI (`.github/workflows/prerelease.yml`): drop torch install steps;
  wheel variant per OS as decided above.
- [ ] Docs: AGENTS.md (key technologies), FOLDER_STRUCTURE.md, TRACKER.md —
  new "ONNX-only (2026-10)" completed section; archive superseded plans.
- [ ] BUG_TRACKER: mark S1/S2/D1/D2/D3 superseded-by-this-plan; G21 stays
  until the Phase 0 gate.

## Risks / open questions

1. **onnxruntime on Python 3.14** — wheel availability unverified (Phase 0 gate).
2. **GPU coverage loss** — NCNN was the only Vulkan path. Linux AMD without
   ROCm builds falls back to CPU (onnxruntime-rocm isn't on PyPI; would need a
   custom index — decision point).
3. **Bundle size** — Linux GPU bundle grows by the nvidia pip wheels
   (~1 GB+). Split variants mitigate.
4. **Custom .pth/.safetensors models** — dropped by default with the torch
   loader (decision point; a minimal torch-based custom loader could return
   later without restoring the full pytorch backend).
5. **TensorRT** — currently rides on torch_tensorrt; dies with PyTorch. ONNX
   Runtime's TensorRT EP is the replacement path if wanted later.
6. **Models repo** — ONNX files already hosted; keep on-demand model
   downloads (S5 version-pinning still applies).
