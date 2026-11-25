# ONNX-Only Migration Plan

Status: **Phases 0-4 complete (2026-11-25); Phase 5 verification in progress**.
Render verification happens **after** the refactor completes (Phase 5).
Decided (user, 2026-10-08):

1. ONNX Runtime becomes the **only** inference backend. PyTorch and NCNN go away.
2. A **CPU baseline ships bundled** (pip dependency of the frozen build) — the
   app works offline out of the box. At runtime the app **auto-downloads the
   accelerated build matching the detected GPU vendor** into a writable data
   dir and uses it after a restart. The recommendation is **vendor-aware,
   never vendor-default**: NVIDIA → CUDA build (PyPI), with an optional
   TensorRT tier when the GPU supports it; Intel → OpenVINO build (PyPI);
   AMD → ROCm/MIOpen build (AMD's wheel index — not on PyPI); nothing →
   bundled CPU. Windows is covered by the bundled DirectML build regardless
   of vendor.
3. **Performance first** — for every hardware configuration the app
   automatically selects the fastest runtime it can install (TensorRT > CUDA
   > DirectML > CPU on NVIDIA; ROCm on AMD; OpenVINO on Intel), downloading
   in the background and caching whatever it needs. Install cost is secondary
   to runtime performance; the bundled CPU build is only the offline
   fallback.
4. Same for binary downloads: **ffmpeg ships via a pip package**
   (`imageio-ffmpeg` bundles a per-platform static ffmpeg build; RVE never uses
   ffprobe, so its missing ffprobe is not a problem). The BtbN download path
   and the standalone-CPython download both disappear.

Related: `docs/BUG_TRACKER.md` (S1/S2/D1-D4 and G21 resolved/superseded by
this plan), `docs/archived/onnx-backend-plan.md`.

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
  Windows (universal baseline); **runtime-upgraded, vendor-aware** =
  NVIDIA → `onnxruntime-gpu` with the **TensorRT EP as the default
  provider** on SM ≥ 7.5 (CUDA EP registered as automatic fallback), Intel →
  `onnxruntime-openvino` (PyPI), AMD →
  `onnxruntime-rocm` (AMD's MIOpen/ROCm build from
  `repo.radeon.com/rocm/manylinux/<release>/` — newest release that supports
  the detected GPU, see the provider manager below). macOS stays on
  CPU/CoreML (no download).
- **NVIDIA TensorRT tier (default on SM ≥ 7.5)**: the `onnxruntime-gpu`
  wheel already ships the TensorRT EP; the manager additionally installs the
  matching `tensorrt` pip libs (NVIDIA publishes them on PyPI as `tensorrt`
  → `tensorrt_cu13`/`tensorrt-libs`). Gate on compute capability ≥ 7.5 (TRT
  10/11 dropped older SMs — those get the CUDA EP). Providers are registered
  in order `[TensorRT, CUDA, CPU]` so ORT falls back automatically if an
  engine fails to build. First-run engine building takes minutes per
  model/shape — mitigations: engines cached to the app data dir, a progress
  note on the first render, and CUDA EP as the fallback path. The
  `tensorrt` version must match what the chosen `onnxruntime-gpu` release
  was built against.
- **Windows upgrades too**: NVIDIA detected on Windows → download
  `onnxruntime-gpu` (win wheel) + TRT libs, same as Linux — DirectML remains
  the bundled baseline/fallback for AMD/Intel/no-GPU.

---

## Performance-first implementation principles (confirmed 2026-10-08)

Beyond provider selection, the performance-first rule applies to the
runtime itself:

- **fp16 everywhere it helps** — prefer fp16 model variants when the models
  repo offers them, default `--precision` to fp16 on GPU accelerators (CPU
  stays fp32), and run I/O conversions in fp16 where the model graph
  expects it.
- **ORT graph optimization at max** — set
  `session_options.graph_optimization_level = ORT_ENABLE_ALL` explicitly in
  `OnnxLoader` (plus the existing mem arena/pattern/reuse flags), so every
  session gets full kernel fusion regardless of defaults.
- **Engine/session caches treated as first-class** — TensorRT engine cache
  (Phase 2) and ORT's optimized-model caching live in the app data dir and
  survive upgrades keyed by model + provider + shape signature.
- **No unnecessary round-trips** — frames stay in their native dtype/range
  between pipeline stages (the G10/G11 fixes establish the single
  NCHW/0-1/uint8 convention at the boundaries only).

## Phase 0 — Build-prep checks

Quick engineering checks (not render gates) that inform the refactor:

- [x] onnxruntime wheel availability for the **build venv's Python (3.14)**.
  If 3.14 wheels don't exist, move the build venv (and cx_Freeze target) to
  3.12/3.13 as part of Phase 2.
- [x] Confirm `imageio-ffmpeg`'s bundled build handles RVE's usage: rawvideo
  piped via stdin/stdout, `-filter_complex`, HDR pix fmts.

The ONNX inference path is the least-exercised code in the repo — it gets
proven in **Phase 5** after the refactor, per decision (2026-10-08).

## Phase 1 — GUI: ONNX-only user flows

- [x] `download/download.ui` + `DownloadTab`: remove the unified selector's
  type/variant combos and Load Backend button (ONNX is implicit); remove the
  PyTorch Version/Backend rows and the restack logic that manages them;
  remove `installRecommended`/`load_backend`/`download("torch"/"ncnn")` paths.
- [x] `ModelHandler.getModels`: collapse to the ONNX model sets only.
- [x] `lib/ModelHandler.py` model tables: keep `.onnx` entries only.
- [x] Process page: fixed backend ("ONNX"); `get_current_backend` returns
  "onnx" unconditionally; `build_command` drops `-b` variants.
- [x] Remove `TorchVersions`, torch-consistency/reinstall flows
  (`check_torch_hardware`, `check_torch_consistency`, `_reinstall_torch`,
  `detect_rocm_targets`, `install_rocm_meta`, `ensure_rocm_targets`),
  `GPUDetect` torch paths.
- [x] `constants.py`: drop `BACKEND_RELEASE_URL_TEMPLATE`,
  `CPYTHON`/`PYTHON_EXECUTABLE_PATH` machinery (Phase 2), keep models URL.

## Phase 2 — Bundled runtime + provider auto-upgrade

- [x] `pyproject.toml`: add `imageio-ffmpeg`, `onnxruntime`, and
  `onnxruntime-directml; sys_platform == 'win32'` (bundled baseline;
  Linux/macOS bundle plain CPU — the GPU build is runtime-downloaded).
- [x] ffmpeg resolution: `FFMPEG_PATH` ← `imageio_ffmpeg.get_ffmpeg_exe()`;
  delete `utils/backends/FFmpeg.py`, `apps/backend/utils/GetFFMpeg.py`
  download logic and BtbN/evermeet URLs (supersedes S1).
- [x] Delete `utils/backends/Python.py` (standalone CPython download) and
  `utils/backends/Backend.py` (backend.tar.gz download); supersede S2.
- [x] Backend launch: `BackendHandler`/`process.py`/`Updater.py` switch from
  `PYTHON_EXECUTABLE_PATH` to `sys.executable` (frozen) / dev venv python.
  `BACKEND_PATH` resolves to the bundled `backend/` folder.
- [x] `DownloadDependencies` shrinks to **model downloads** +
  **provider upgrade** (`BackendsModels.py`); `downloadPythonDeps` and the
  generic `pip()` are deleted.
- [x] **Provider upgrade manager** (`apps/gui/utils/backends/ProviderUpgrade.py`):
  - vendor-aware recommendation built on the existing torch-free
    `detect_gpu_hardware()` — NVIDIA → `onnxruntime-gpu` (PyPI), Intel →
    `onnxruntime-openvino` (PyPI), AMD → `onnxruntime-rocm` from AMD's wheel
    index, nothing/unsupported → skip; Windows/macOS skip entirely (bundled
    builds cover them);
  - **NVIDIA TensorRT tier (default on SM ≥ 7.5)** — after the CUDA install,
    additionally install the `tensorrt` pip libs (`tensorrt>=10,<11`,
    version-matched to the onnxruntime-gpu generation); ORT engine cache
    lives in the app data dir (`providers/trt_cache`); providers registered
    as `[TensorRT, CUDA, ..., CPU]` so engine failures fall back
    automatically; SM < 7.5 → CUDA EP only;
  - **AMD release selection: newest-supported-first** — `rocminfo` supplies
    the gfx architecture, a small `gfx → minimum ROCm` table filters
    releases (unknown gfx falls back to the newest wheel with a matching
    python tag, logged);
  - install into a **writable data dir** (`$XDG_DATA_HOME/REAL-Video-
    Enhancer/providers/`, `%APPDATA%` equivalent on Windows) — never into
    the AppImage mount (read-only squashfs);
  - mechanism: pip is a bundled dependency; frozen builds dispatch
    `RVE -m pip …` to the bundled pip in a child process (no standalone
    Python required). Wheels only (`--only-binary=:all:`);
  - non-blocking background download with progress/cancel (opt-in prompt on
    startup, `ProviderUpgradeUI`; no modal-kill);
  - activation by sys.path precedence (`providers/` before frozen libs) +
    one restart; a marker file records the active package, TensorRT flag,
    compute capability and python tag;
  - updater flow mirrors the removed app updater: check → recommend → apply
    on restart.
- [x] cx_Freeze build: include onnxruntime/cv2/imageio_ffmpeg; verify the
  freeze on all three OSes; update `packages/` layout notes.

## Phase 3 — Backend code removal

- [x] `rve-backend.py`: drop `-b` choices except onnx, drop pytorch/ncnn/tensorrt
  args (`--precision` stays), `--pytorch_gpu_id`/`--ncnn_gpu_id` → `--gpu_id`,
  remove `--no_torch_compile`/TRT dynamic-shape flags.
- [x] `listBackends` → report ONNX Runtime version + providers only
  (`BackendDetect` shrinks accordingly).
- [x] Delete: `backends/pytorch/`, `backends/ncnn/`, `render/ncnn/`,
  `models/*/pytorch|ncnn` loader branches (registry models become
  ONNX-only), `models/scene_detect` NCNN path, `TorchVersions`,
  `converters/onnx_to_ncnn.py`.
- [x] Keep `render/onnx/` wrappers + `render/RenderVideo.py` as the only
  inference path; `backends/onnx/` loader/runner remains test/tooling-only.
  `models/` + `apps/backend/pytorch/` are converter tooling and are never
  imported by the shipping render path (guarded by
  `tests/unit_tests/test_runtime_torch_free.py`). `converters/torch_to_onnx.py`
  stays as dev tooling for producing new ONNX models.
- [x] GUI spawns backend without `--pause_shared_memory_id`-adjacent backend
  flags that referenced other backends; settings page drops precision/backend
  options that were pytorch-specific (`pytorch_version`/`pytorch_backend`
  removed, dead `pytorch (cuda|rocm|xpu|mps)` remap removed).

## Phase 4 — Tests, docs, CI

- [x] Tests: pytorch/ncnn benchmark entry points removed; ONNX loader/runner,
  registry, utils tests kept; synthetic-model ONNX render e2e
  (`test_onnx_render_e2e.py`) added; torch-free runtime smoke test added.
- [x] CI (`.github/workflows/tests.yml`): `ncnn` extra removed, torch comes
  from the dev-only `convert` extra, and a separate `runtime-smoke` job runs
  with **no torch installed** to guard the shipping path.
- [x] Docs: AGENTS.md (key technologies), FOLDER_STRUCTURE.md, TRACKER.md —
  ONNX-only notes added; superseded plans archived.
- [x] BUG_TRACKER: S1/S2/D1/D2/D3 marked superseded-by-this-plan; G21 resolved
  by deleting the NCNN path (ONNX 0-1 convention is the only one left).

## Phase 5 — Post-refactor verification (user decision 2026-10-08)

The render test runs here, once the refactor is complete:

- [~] Real ONNX render end-to-end: **synthetic-model e2e is automated**
  (`tests/integration_tests/test_onnx_render_e2e.py`: upscale + interpolation
  through the real CLI, output decoded and compared). A real downloaded model
  (e.g. `tspan.onnx`) is still a manual spot-check.
- [x] Frozen-bundle smoke (2026-11-25): cx_Freeze build completes;
  `RVE -m pip --version` works from the bundle; `--list_backends` reports the
  bundled ONNX Runtime; a marker-activated providers dir shadows it; a
  synthetic interpolation render succeeds through the frozen binary.
- [~] Provider upgrade flow: the install/activation mechanics are verified
  (frozen `pip --target`, marker activation, fallback), but real
  NVIDIA/AMD/Intel downloads still need hardware.
- [ ] TRT tier on capable hardware: engine build, cache reuse on second run.
- [x] imageio-ffmpeg path: render through the bundled ffmpeg binary (covered by
  the e2e render + ffmpeg round-trip tests).
- [ ] Performance spot-checks against the pre-refactor app (fp16 on/off).

## Risks / open questions

1. **onnxruntime on Python 3.14** — resolved: cp314 wheels confirmed in
   Phase 0; the build venv and cx_Freeze target stay on 3.14 while the
   project floor is 3.12.
2. **NVIDIA TensorRT coupling** — the manager pins `tensorrt>=10,<11` to the
   `onnxruntime-gpu` 1.20+ generation and re-pins here on ORT upgrades.
   First render after install spends minutes building engines (cached to
   disk afterwards, extra VRAM/disk) — mitigated by the
   `[TensorRT, CUDA, …, CPU]` fallback order.
2b. **AMD on Linux** — covered by `onnxruntime_rocm` from AMD's index with a
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
