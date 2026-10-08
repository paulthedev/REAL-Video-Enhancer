# ONNX-Only Migration Plan

Status: **planned — not started**
Decided (user, 2026-10-08):

1. ONNX Runtime becomes the **only** inference backend. PyTorch and NCNN go away.
2. A **CPU baseline ships bundled** (pip dependency of the frozen build) — the
   app works offline out of the box. At runtime the app **auto-downloads the
   accelerated build matching the detected GPU vendor** into a writable data
   dir and uses it after a restart. The recommendation is **vendor-aware,
   never vendor-default**: NVIDIA → CUDA build (PyPI), Intel → OpenVINO build
   (PyPI), AMD → ROCm/MIOpen build (AMD's wheel index — not on PyPI),
   nothing → bundled CPU. Windows is covered by the bundled DirectML build
   regardless of vendor.
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

- **A CPU onnxruntime baseline ships inside the frozen app** (pip dependency
  picked up by cx_Freeze) — plus `onnxruntime-directml` on Windows (~50 MB,
  DX12 GPU on every vendor, no external CUDA). No dependency-install popup,
  no standalone CPython.
- **Runtime provider upgrade**: `detect_gpu_hardware()` (already torch-free)
  reports the vendor; the manager downloads the matching build in the
  background into the app's writable data dir. Bundled CPU stays as the
  guaranteed fallback if the vendor has no build or the download fails.
  Applies after a restart.
- **ffmpeg ships inside the frozen app** via `imageio-ffmpeg`
  (`imageio_ffmpeg.get_ffmpeg_exe()`), with `--ffmpeg_path` / `RVE_FFMPEG_PATH`
  kept as overrides. No runtime binary downloads at all.
- The GUI still spawns `rve-backend.py` as a subprocess, but with the **same
  frozen interpreter** (`sys.executable`) — crash/GIL isolation preserved
  without a second Python install. (Merging backend into the GUI process is a
  possible later step, not part of this migration.)
- **Models still download on demand** from the models repo — ONNX files only.
- Wheel strategy (decided): **bundled** = CPU everywhere + DirectML on
  Windows; **runtime-upgraded, vendor-aware on Linux** = NVIDIA →
  `onnxruntime-gpu` (PyPI), Intel → `onnxruntime-openvino` (PyPI), AMD →
  `onnxruntime-rocm` (AMD's MIOpen/ROCm build from
  `repo.radeon.com/rocm/manylinux/<release>/` — newest release that supports
  the detected GPU, see the provider manager below). macOS stays on
  CPU/CoreML (no download).

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

## Phase 2 — Bundled runtime + provider auto-upgrade

- [ ] `pyproject.toml`: add `imageio-ffmpeg`, `onnxruntime`, and
  `onnxruntime-directml; sys_platform == 'win32'` (bundled baseline;
  Linux/macOS bundle plain CPU — the GPU build is runtime-downloaded).
- [ ] ffmpeg resolution: `FFMPEG_PATH` ← `imageio_ffmpeg.get_ffmpeg_exe()`;
  delete `utils/backends/FFmpeg.py`, `apps/backend/utils/GetFFMpeg.py`
  download logic and BtbN/evermeet URLs (supersedes S1).
- [ ] Delete `utils/backends/Python.py` (standalone CPython download) and
  `utils/backends/Backend.py` (backend.tar.gz download); supersede S2.
- [ ] Backend launch: `BackendHandler`/`process.py`/`Updater.py` switch from
  `PYTHON_EXECUTABLE_PATH` to `sys.executable` (frozen) / dev venv python.
  `BACKEND_PATH` resolves to the bundled `backend/` folder.
- [ ] `DownloadDependencies` shrinks to **model downloads** +
  **provider upgrade** (`BackendsModels.py`); `downloadPythonDeps` and the
  generic `pip()` are deleted.
- [ ] **Provider upgrade manager** (new module, e.g. `utils/backends/
  ProviderUpgrade.py`):
  - vendor-aware recommendation built on the existing torch-free
    `detect_gpu_hardware()` — NVIDIA → `onnxruntime-gpu` (PyPI), Intel →
    `onnxruntime-openvino` (PyPI), AMD → `onnxruntime-rocm` from AMD's wheel
    index, nothing/unsupported → skip; Windows/macOS skip entirely (bundled
    builds cover them);
  - **AMD release selection: newest-supported-first** — detect the GPU's gfx
    architecture (`rocminfo`, or PCI-ID → gfx mapping from sysfs), walk
    `repo.radeon.com/rocm/manylinux/` releases newest-first, and pick the
    first whose ROCm version supports that architecture (newer ROCm releases
    add newer GPUs, e.g. RDNA4 gfx1200+ landed in 6.4.x); also gate on a
    wheel with the running python tag; fall back to older releases, then
    bundled CPU;
  - install into a **writable data dir** (`$XDG_DATA_HOME/REAL-Video-
    Enhancer/providers/`, `%APPDATA%` equivalent on Windows) — never into
    the AppImage mount (read-only squashfs);
  - mechanism: bootstrap pip via stdlib `ensurepip` and
    `pip install --target <providers dir> onnxruntime-gpu` (resolves the
    nvidia dep closure automatically) — alternative is a minimal wheel
    resolver over the PyPI JSON API (no pip, ~100 lines, but manual closure);
  - non-blocking background download with progress/cancel (no modal-kill —
    the G24/G25 lessons); bundled CPU remains the fallback on any failure;
  - activation by sys.path precedence (`providers/` before frozen libs) +
    one restart; a marker file records the active provider and version;
  - updater flow mirrors the app updater: check → recommend → apply on
    restart.
- [ ] cx_Freeze build: include onnxruntime/cv2/imageio_ffmpeg; verify the
  freeze on all three OSes; update `packages/` layout notes.

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
2. **AMD on Linux** — covered by `onnxruntime_rocm` from AMD's index with a
   newest-supported-first release pick. Caveats: ROCm supports specific GPU
   architectures (very old cards may match no release → bundled CPU), the
   wheel needs a matching python tag (cp310/cp312 published today), and the
   ROCm userspace stack (HIP/MIOpen) comes from the same index. Vulkan-only
   GPUs that ROCm doesn't support have no accelerator. Intel iGPUs are
   skipped by `detect_gpu_hardware()` (it reports discrete GPUs only), so
   OpenVINO upgrade for iGPU owners needs a detection extension first.
3. **Provider upgrade download size** — Linux GPU is ~1 GB+ of nvidia wheels;
   background, cancellable, and CPU keeps working without it. Bundled app
   stays small (CPU + DirectML).
4. **Custom .pth/.safetensors models** — dropped by default with the torch
   loader (decision point; a minimal torch-based custom loader could return
   later without restoring the full pytorch backend).
5. **TensorRT** — currently rides on torch_tensorrt; dies with PyTorch. ONNX
   Runtime's TensorRT EP is the replacement path if wanted later.
6. **Models repo** — ONNX files already hosted; keep on-demand model
   downloads (S5 version-pinning still applies).
