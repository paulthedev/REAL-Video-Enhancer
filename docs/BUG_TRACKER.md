# Bug Tracker — Code Review 2026-10-08

Findings from the full code review of 2026-10-08 (backend review agent + manual verification).
All P0 items were verified directly (executed the entry point, grepped for symbols, read the code paths).
The 293 unit tests pass because they exercise dataclasses and mocked inference only — none of the
integration paths below are covered.

Status legend: `[ ]` open · `[x]` fixed · `[~]` won't fix / by design

---

## Summary

| Priority | Count | Fixed |
|----------|-------|-------|
| P0 — nothing runs end-to-end | 11 | 0 |
| P1 — GUI crashes / wrong output | 25 | 1 |
| Wiring — GUI refactor breakage (found + fixed 2026-10-08) | 4 | 4 |
| Download — backend download/install wiring (found 2026-10-08) | 4 | 0 |
| P2 — robustness | 15 | 1 |
| P3 — cleanup | 14 | 0 |

---

## P0 — Backend cannot run end-to-end

- [ ] **B1. Render pipeline module deleted.** `rve-backend.py:133` imports
  `apps.backend.RenderVideo.Render`, but `RenderVideo.py` does not exist anywhere. Every
  successful `renderVideo()` call crashes with `ModuleNotFoundError`. **Decision needed:**
  port the old pipeline from git history or write a new one against the `backends/` structure.
- [ ] **B2. Backend entry point can't run as a script.** `rve-backend.py:4` does
  `from apps.backend.version import __version__`; running
  `python apps/backend/rve-backend.py --version` → `ModuleNotFoundError: No module named 'apps'`
  (verified). The GUI launches it exactly this way (`process.py:537`, `BackendHandler.py:16`,
  `Updater.py:69`, `apps/gui/utils/VideoInfo.py:17`).
  Fix: `sys.path` bootstrap at top of `rve-backend.py`, or launch as `python -m apps.backend.rve-backend`.
- [ ] **B3. `--list_backends` imports nonexistent symbols.** `MINIMUM_PYTORCH_CAP` and
  `BackendDetect.meets_capability` (`rve-backend.py:77-80,108`) are defined nowhere.
  This is the first call the GUI makes at startup → `BackendHandler.py:26-34` shows
  "FATAL ERROR" popup and `exit(1)`. GUI cannot start against this backend.
  Fix: implement compute-capability check (e.g. `torch.cuda.get_device_capability(i) >= (6,0)`),
  define `MINIMUM_PYTORCH_CAP = (6, 0)`.
- [ ] **B4. `apps/backend/constants.py` missing.** `utils/GetFFMpeg.py:1` and
  `utils/BackendDetect.py:101` do `from ..constants import CPU_ARCH/PLATFORM` → ImportError
  on ffmpeg download and on NCNN GPU detection. `utils/Util.py:31-34` shows the correct
  guarded pattern — replicate or create the module.
- [ ] **B5. `RifeModel` never stores `self.config`.** `rife.py:377` reads `self.config.drba`
  but `__init__` (201-250) copies fields individually. `AttributeError` on every PyTorch RIFE load.
  Fix: `self.config = config` in `__init__` (every other model does this).
- [ ] **B6. PyTorch loader calls a registry instance as a class.**
  `backends/pytorch/loader.py:174-188`: `get_model(name)` returns a registered **instance**
  (all `register_model` calls register instances), then does `model_class(...)`.
  `TypeError: 'RifeModel' object is not callable`. Also `BaseModel.__init__` accepts none
  of the kwargs passed. Fix: use the instance and set attributes, or register factories.
- [ ] **B7. Missing `import os` in 7 models' `_load_pytorch`.** `os.path.exists` used but `os`
  only imported function-locally in `_load_ncnn` (or not at all — `span.py` never imports it).
  `NameError` on every real weight load. Files: `models/upscale/span.py:116`,
  `tspan.py:149`, `animesr.py:150`, `models/restoration/fbcnn.py:139`, `nafnet.py:208`,
  `models/denoise/dncnn.py:132`, `models/interpolate/gimm.py:206`.
- [ ] **B8. NCNN inference uses nonexistent API + missing imports.** `ncnn` imported only inside
  `_load_ncnn` but used in `_infer_ncnn` (NameError): `gimm.py:335`, `gmfss.py:395`,
  `ifrnet.py:518`, `dncnn.py:225`, `fbcnn.py:232`, `nafnet.py:301`, `animesr.py:243`,
  `span.py:209`, `tspan.py:242`, `scene_detect/maxvit.py:248`. Also `ncnn.Mat(h, w, c, data)`
  and `Mat.to_pixels()` are not valid ncnn-python API.
- [ ] **B9. `NCNNBackendRunner.run` written against a nonexistent ncnn API.**
  `backends/ncnn/runner.py:53-63`: `net.create_input()`, `predictor.set_input()`, `extract()`
  called twice (first result discarded), `output_mat.to_pixels()`, stray `net` variable (line 42).
  Rewrite with `ncnn.Mat.from_pixels(...)`, `extractor.input()`, `ret, out = extractor.extract()`,
  `out.numpy()`.
- [ ] **B10. IFRNet reads `.dtype`/`.device` off `nn.Module`.** `ifrnet.py:459`:
  `torch.tensor(..., dtype=self.model.dtype, device=self.model.device)` → `AttributeError`
  on first interpolation. Fix: store device/dtype objects at load time (as `_load_pytorch`
  at 374-376 already computes them).
- [ ] **B11. `getUnusedFileName` infinite loop.** `utils/FileHandler.py:114-122`: `while`
  checks the never-modified `base_file_name`, not the candidate `output_file`. If the output
  file exists the process hangs forever. Fix: `while os.path.isfile(output_file):`.

## P1 — GUI crashes / incorrect behavior

- [x] **G1. `getModels` crashes for real backend names.** `apps/gui/lib/ModelHandler.py:520-545`:
  `listBackends` emits `"tensorrt"` but there is no `tensorrt` case → `case _` returns `{}` →
  `ValueError` unpacking 6 values in `populateModels` (`process.py:93`) at GUI startup.
  The `directml` case binds only 3 of 6 variables → `UnboundLocalError`.
  Error path returns `{}` instead of raising.
  **Fixed 2026-10-08**: `onnx | tensorrt | directml` share one case assigning all six sets;
  unknown backends now raise `ValueError` instead of returning `{}`.
- [ ] **G2. Declining the overwrite dialog crashes and wipes the queue.** `process.py:244-251`
  calls `guiChangesOnRenderCompletion` directly; line 409 reads `self.currentRenderOptions.isPreview`
  — `None` on first run → `AttributeError`. Line 408 also `renderQueue.clear()`s the whole
  queue when the user declines.
- [ ] **G3. Discord RPC failure kills the render thread.** `process.py:281-287` swallows the
  `DiscordRPC()` constructor exception, so `self.discordRPC` may not exist;
  `onRenderCompletion` (`process.py:441-444`) calls `self.discordRPC.closeRPC()` unguarded →
  `AttributeError` inside `renderToPipeThread` → render thread dies, GUI stuck in "rendering".
- [ ] **G4. Preview shared-memory collision.** `QTcustom.py:253-260` `createNewSharedMemory`
  passes `create=True` with no `FileExistsError` handling (paused-shm code at `process.py:289-298`
  does it correctly). Stale segment from a crashed render kills the render thread silently.
- [ ] **G5. QImage lifetime bug.** `QTcustom.py:313-323`: preview `QImage` wraps the numpy
  buffer without `.copy()`; buffer is freed on the next poll iteration (0.2 s) while the main
  thread may still paint it — racy use-after-free.
- [ ] **G6. `--merge_subtitles` can never be disabled.** `rve-backend.py:494-499`:
  `action="store_true", default=True`. Fix: `default=False` or `BooleanOptionalAction`.
- [ ] **G7. Upscale/restore models run on the wrong device.** `_load_pytorch` moves the model
  to CUDA but inference never moves the input frame → "Expected all tensors to be on the same
  device" on any GPU machine. Files: `span.py:172-175`, `tspan.py:205-208`, `animesr.py:206-209`,
  `nafnet.py:264-267`, `fbcnn.py:195-198`, `dncnn.py:188-191`.
- [ ] **G8. Only RIFE implements `infer()`.** `backends/pytorch/runner.py:59,131` call
  `model.infer(...)`; GIMM/GMFSS/IFRNET only define `interpolate()` → `AttributeError`.
  Fix: aliases or task-based dispatch.
- [ ] **G9. GMFSS caches the first frame pair forever.** `gmfss.py:121-150`: `feat11`/`flow01`/
  `metric0` computed once and never shifted between pairs → every frame after the first is
  interpolated with the first pair's features → corrupted output on any real video.
- [ ] **G10. ONNX interpolate runner: `IndexError` + wrong input shape.**
  `backends/onnx/runner.py:61-65`: dict key `inputs[2].name` evaluated before the
  `if len(inputs) > 2` conditional value → crash on 2-input models. Frames fed as `[H,W,C]`
  with no NCHW transpose → shape error on real exported models.
- [ ] **G11. Unclamped uint8 conversion wraps to black pixels.** `(result * 255).astype(np.uint8)`
  without `np.clip(0,1)`: values >1.0 wrap modulo 256. `backends/onnx/runner.py:73,154`;
  NCNN paths divide by 255 unclamped too. PyTorch runner clamps — ONNX/NCNN don't.
- [ ] **G12. `bytesToImg` destroys HDR frames, ignores resize params.** `utils/Util.py:78-83`:
  uint16 → uint8 truncates the **high** byte (10-bit data becomes cyclic banding); resize
  hardcodes `(100, 100)` instead of the passed dimensions.
- [ ] **G13. Segmented scene detection collapses to one column.** `utils/SceneDetect.py:77-86`:
  `means[i] = np.mean(segment)` inside the `i`/`j` double loop keys by `i` only — each row's
  mean is overwritten per column; only the last column survives.
- [ ] **G14. MaxViT scene-detect weights never load.** `models/scene_detect/maxvit.py:143-144`
  checks `model_path.endswith('.pkl')` but the registry registers a `.pt` path (`maxvit.py:297`)
  → random-init model "detects" garbage. Same `.pkl`-only check in `gmfss.py:287`.
  Fix: `os.path.exists` instead of extension check.
- [ ] **G15. Torch→ONNX converter unusable.** `converters/torch_to_onnx.py:81-106`:
  `model.eval()` on plain model objects (not `nn.Module`); exports one input named `"input"`
  while every consumer expects 3 (`img0, img1, timestep`); dead `torch.utils.fusion.fuse_model` block.
- [ ] **G16. `Frame` singleton caches the first video's dims/device forever.**
  `utils/Frame.py:13-29`: module-level `_torch_utils` initialized once; later videos with new
  resolution/GPU silently reuse stale dims. `set_frame_tensor` (`Frame.py:75`) calls
  `_torch_utils.sync_all_streams()` unconditionally → `AttributeError` when backend is ncnn/onnx.
- [ ] **G17. `get_gpus_torch` returns a bare string, breaking `listBackends`.**
  `utils/BackendDetect.py:81-82` returns `"CPU"` (str) or a list of dicts, but
  `rve-backend.py:106-108` expects a list of names (`== ["CPU"]`, `range(len(...))`).
  Also duplicate unreachable `except` at 127-130.
- [ ] **G18. `sceneDetect` returns `None` (bare `return`) on the first frame.**
  `utils/SceneDetect.py:32-35, 89-94, 119-123` — callers expecting `False` misbehave.
- [ ] **G19. `--ffmpeg_path` default never triggers download; missing `-i` crashes.**
  `rve-backend.py:19-23`: default `"./bin/ffmpeg"` makes the `args.ffmpeg_path == None`
  download branch unreachable from CLI; `checkArguments` (`526-528`) evaluates
  `"http" not in self.args.input` before the None check → `TypeError` when `-i` omitted.
- [ ] **G20. Batch mode defects.** `rve-backend.py:60-74`: blank lines → `input=None` → G19's
  TypeError; `--ffmpeg_path`/`--cwd` not inherited into each re-parse. `--overlap`
  (233-238) parsed but never used. `fullModelPathandName` (515-516) references
  `self.args.modelPath/modelName` which don't exist.
- [ ] **G21. NCNN vs ONNX scaling convention inconsistent.** e.g. `fbcnn` NCNN path multiplies
  by 255 but ONNX path feeds 0-1 floats for the same model — one convention is wrong per model
  (washed-out or clipped output). Pick one convention per export and document in `models/base.py`.
- [ ] **G22. `RestorationBackend`/`SceneDetectBackend` don't map `run()`.**
  `backends/base.py:136-162`: only Upscale/Interpolate override `run()`; `backend.run(...)`
  on restore/scene-detect hits `BaseBackend.run` → `NotImplementedError`.
- [ ] **G23. Registry name drift.** `rife.py:637` registers `"RIFE"` (uppercase); all other
  models register lowercase. Lowercase lookups (matching the CLI help convention) miss RIFE.
- [ ] **G24. Startup update check silently kills or hangs the app.** Found during the
  2026-10-08 smoke test. `REAL-Video-Enhancer.py:169` (`if d.get_if_update_available()`)
  runs `rve-backend.py --version` (broken, see B2) → `CalledProcessError` path treats it as
  "backend missing" and calls `Backend.download()` (`Backend.py:43`) → **modal**
  `DownloadProgressPopup` downloading `BACKEND_RELEASE_URL_TEMPLATE`, which 404s. Then:
  `cancel_process` (`QTcustom.py:461-462`) calls `QApplication.quit()` + `sys.exit()` from
  a dialog slot → the whole app dies silently with exit 0 before the window shows.
  If instead the worker thread dies on the 404, nothing emits `finished` and the modal
  `exec()` never closes → hang. Fix: don't auto-download at startup (surface a prompt),
  handle HTTP errors in the worker, and never `sys.exit()` from a widget slot.
- [ ] **G25. `DownloadAndReportToQTThread.run` has no error handling.** `QTcustom.py:352`:
  `urlopen(...)` is outside any try — any HTTP/network error kills the worker thread without
  emitting `finished`/`canceled`, leaving `DownloadProgressPopup.exec()` open forever.
  Wrap the download, emit `canceled` (or a new `failed` signal) on error, and give the
  dialog a timeout path. Related: `setProgress`'s `sys.exit()` (`QTcustom.py:466`) raises
  SystemExit through the event loop — replace with dialog close.

## Wiring — GUI refactor breakage (found + fixed 2026-10-08)

The `.ui` split (monolithic `testRVEInterface.ui` → `mainwindow.ui` + per-page `.ui`)
was never wired through to the logic code: page widgets became attributes of throwaway
`Ui_*` instances inside `assemble_pages`, so every `self.<widget>` on MainWindow and
`self.parent.<widget>` in page modules (~140 references) pointed at nothing. All four
items below are fixed and verified by an offscreen `MainWindow()` smoke test
(constructs, populates models, renders — screenshot checked).

- [x] **W1. Page-widget wiring (~140 broken references).** `apps/gui/pages/__init__.py`
  `assemble_pages` now accepts `controller=` and re-points every widget each generated
  `Ui_*` class creates onto the MainWindow (same flat namespace as the pre-split
  monolithic `.ui`, same approach as `QTabNavigation.bind_controlled_attributes`).
  Call site: `REAL-Video-Enhancer.py:137` passes `controller=self`. Collision audit:
  only benign overlaps (page slot names, tab `title` attributes); `get_current_backend`
  now falls back when the unified selector is still empty at ProcessTab init time.
- [x] **W2. `DownloadTab.hideUninstallButtons` / `showUninstallButton` missing.**
  `REAL-Video-Enhancer.py:289-290` calls both; they were lost in the ui/ → pages/ move.
  Restored from history (`950abc80:src/ui/DownloadTab.py`); all referenced buttons
  verified present in `download.ui`.
- [x] **W3. `resources.qrc` paths made the whole icon bundle fail to compile.** Entries
  were repo-root-relative (`apps/gui/icons/...`) but rcc resolves them relative to the
  `.qrc` file's directory → `Cannot find file` → **0-byte `resources_rc.py`** → no icons
  at all. Fixed to qrc-relative (`icons/...`), matching the `:/icons/icons/...` paths the
  code uses; rebuilt bundle is ~211 KB with all 288 icons.
- [x] **W4. Dead refactor leftovers removed.** `apps/gui/tests/` (empty dir),
  `apps/gui/ModelRegistry.py` (zero importers), stale local `mainwindow.py` (see R14);
  duplicate names in `download.py:12` import deduped; unused `RegularQTPopup`/`errorAndLog`
  imports removed from `ModelHandler.py`.

## Download — backend download/install wiring (found 2026-10-08)

Trace of the download page's install paths against the restructured backend
(`download.py` → `DownloadDependencies.py` → `rve-backend.py`). PyTorch and
NCNN install paths are wired correctly (pip targets `PYTHON_EXECUTABLE_PATH`,
same env every launch path uses). ONNX is broken at both ends.

- [ ] **D1. ONNX install is a silent no-op.** `load_backend()` (`download.py:181`) and
  `installRecommended()` (`download.py:208`) call `self.download("onnx", install=True)` →
  `downloadPythonDeps` (`DownloadDependencies.py:593+`), whose `match` has **no
  `case "onnx"` and no `case _`** → installs only the platform-independent shared deps
  (opencv, scenedetect, …), returns 0 → UI pops "Download Complete! Please restart"
  while `onnxruntime` was never installed.
  Fix: add `case "onnx"` installing `onnxruntime` (+ provider/extra deps per platform).
- [ ] **D2. Backend never lists ONNX.** `rve-backend.py:listBackends` only appends
  `"tensorrt"`, `"pytorch (...)"`, `"ncnn"` — zero ONNX detection logic in the backend.
  Even with a working install (D1), `getAvailableBackends()` never includes `"onnx"` so
  the app treats it as not installed. Backend-side half of the "ONNX as primary backend"
  strategy (TRACKER 2026-09-29) was never implemented — same family as B1/B3.
  Fix: add an ONNX branch (probe `onnxruntime` + available providers in the backend env)
  mirroring the pytorch/ncnn blocks.
- [ ] **D3. Backend update/version check is dead until B2.** `Backend.get_if_update_available`
  (`Backend.py:50`) probes `PYTHON_EXECUTABLE_PATH rve-backend.py --version` →
  `ModuleNotFoundError` (B2) on every run → every startup takes the "backend not found"
  auto-download path (see G24). Also catches only `CalledProcessError` — a missing backend
  directory raises uncaught `FileNotFoundError`. Resolves via B2 + the G24/G25 error
  handling; the `FileNotFoundError` guard is a separate small fix.
- [ ] **D4. `check_backend_health` probes the wrong environment (dormant).**
  `BackendDetect.py:215-303` (`_check_pytorch/_onnx/_ncnn_health`) does
  `import torch` / `import onnxruntime` / `import ncnn` **in-process** — the GUI/frozen
  interpreter — while backend deps live in the standalone python env (`CWD/python/...`).
  Currently only reachable via `DownloadDependencies.check_backend_hardware_change` /
  `reinstall_backend`, which have no callers in the live flow. Fix before wiring them in:
  run the probes with `PYTHON_EXECUTABLE_PATH -c "import ..."` like the rest of the
  backend integration.


## P2 — Robustness / resource handling

- [ ] **R1. `FFMpegInfoWrapper` leaks subprocess, unguarded regexes.** `utils/VideoInfo.py:121`:
  `Popen` discarded without `wait()`/timeout (zombie). Lines 145/156/160 call
  `re.search(...).groups()` without None-checks → `AttributeError` on unusual files.
  `OpenCVInfo.get_duration_seconds` (`:260`) divides by `get_fps()` → `ZeroDivisionError`.
- [ ] **R2. `OpenCVInfo.__del__` can raise; `exit(1)` inside a library class.**
  `utils/VideoInfo.py:312-313`: if `__init__` raised, `self.cap` is unset → noisy `__del__`
  error. `_get_ffmpeg_info` calls `exit(1)` — should raise instead.
- [ ] **R3. ONNX provider never validated.** `backends/onnx/loader.py:59-62`,
  `utils/OnnxLoader.py:95`: unknown provider string passed straight to `InferenceSession` →
  cryptic error. Validate against `get_available_providers()` and fall back.
- [ ] **R4. NCNN loader maps `.param` paths wrong.** `backends/ncnn/loader.py:79-84`: a
  `.param` `model_path` is loaded as the weights bin. Fix: derive the `.bin` sibling
  (`rife.py:417-426` does this correctly).
- [ ] **R5. `unzipFile` chdir without try/finally.** `utils/FileHandler.py:38-45`: cwd never
  restored if `extractall` raises; chdir is process-global (thread-unsafe).
- [ ] **R6. `download_ffmpeg` never chmod +x, drops `.exe` on Windows, unknown platform saves
  an HTML page as the binary.** `utils/GetFFMpeg.py:21-29`.
- [ ] **R7. `onnx_to_ncnn` error path uses unimported `sys`; dead ncnnoptimize config.**
  `converters/onnx_to_ncnn.py:58` (`NameError` exactly when pnnx is missing);
  `ncnnoptimize_path` accepted but never used (28, 71-97).
- [ ] **R8. `get_frame_tensor` returns `None.clone()` on empty Frame.** `utils/Frame.py:86-103`.
  Raise a clear `RuntimeError` instead.
- [ ] **R9. HDR downconvert inconsistency.** `utils/Frame.py:164-172` divides uint16 by 65535
  while `Util.bytesToImg` treats the same data as 10-bit-in-16-bit — one mis-scales by ~64×.
  (`QTcustom.py` preview uses `>> 8`.) Settle one convention and share it.
- [ ] **R10. GUI `VideoLoader.getData` divides by fps without a guard.**
  `apps/gui/utils/VideoInfo.py:149`: `ZeroDivisionError` when OpenCV can't read fps; masked by
  `isValidVideo`'s blanket except. Note: GUI parses the backend's *printed* `print_video_info`
  output — keep formats in sync (coupling point: `rve-backend.py:315`).
- [ ] **R11. Render-output loop fragility.** `process.py:339`: bytes sentinel `b""` on a
  text-mode stream (loop exits only via the `poll()` check); `re.search(...).group(1)` at
  351-355 without None-guards, failures swallowed by bare `except: pass` (366); `self.fps`
  stored as `str` (354) but compared to `int` 0 (499) — works by accident.
- [ ] **R12. `eval()` on subprocess output.** `BackendHandler.py:52`: parses the backend's
  printed backends list with `eval` after string hacks — brittle; use `ast.literal_eval`
  or a structured output mode.
- [ ] **R13. Preview scrollbar connections accumulate.** `process.py:427`: a new lambda +
  `QMediaPlayer` connected per preview render; stale handlers fire on deleted players.
- [x] **R14. Stale `apps/gui/mainwindow.py`.** ~~Generated file imports **PySide2**~~
  Correction 2026-10-08: the file was **never tracked** — `.gitignore` already covers
  `mainwindow.py`; it was only a local stale artifact (PySide2-era build output) that could
  shadow the intended `dist/mainwindow.py` before a build. Deleted from disk 2026-10-08.
- [ ] **R15. Pause race.** `process.py:200-202`: `pauseRender` reads `self.pausedSharedMemory`
  created inside the render thread (`renderToPipeThread` → `createPausedSharedMemory`) —
  clicking pause in the first moments of a render → `AttributeError` in the GUI thread.

## P3 — Cleanup / dead code

- [ ] **C1.** `rve-backend.py:120` — hardcoded `"NCNN Version: 20220729"` instead of detected `ncnn_ver`.
- [ ] **C2.** `rve-backend.py:121` — `rife_ncnn_vulkan_python` imported, never used.
- [ ] **C3.** `rife.py:563-565` — `_infer_ncnn` docstring corrupted by a bad merge (dead text).
- [ ] **C4.** `gimm.py` `_load_pytorch` hardcodes cuda/cpu, ignores `self.device`/`self.gpu_id` (no MPS/XPU), unlike IFRNet.
- [ ] **C5.** `get_output_shape()` signature drift: `gimm.py:383`, `gmfss.py:252` drop the
  `input_shape` param required by `BaseModel`; `ifrnet.py:553` returns a 3-tuple vs 4-tuples elsewhere.
- [ ] **C6.** Runner rank conventions inconsistent: `RifeModel` returns `[B,C,H,W]`,
  GIMM/GMFSS/IFRNET return `[C,H,W]`; runners squeeze/permute assuming one convention.
- [ ] **C7.** `backends/onnx/runner.py:72-76` — `isinstance(result, np.ndarray)` else-branch is dead (session.run always returns ndarrays).
- [ ] **C8.** `converters/onnx_to_ncnn.py:139` — `input_shape[1:4]` unconditionally indexes a 4-tuple; batch dim hardcoded 1.
- [ ] **C9.** `SceneDetect.py:172` — stray `self.model.model` no-op; `RVESceneDetect` (192-197) unreachable from the CLI method dict.
- [ ] **C10.** `utils/Util.py:37-41`, `FileHandler.moveFolder:25-31` — misleading log messages; `removeFile` concatenates `str` + possibly-`Path`.
- [ ] **C11.** `utils/PySceneDetectUtils.py:1099` — debug `print(kernel_size)` left in.
- [ ] **C12.** `rve-backend.py:285-288` — `--scene_detect_method` help lists `mean, mean_segmented, none` but default is `pyscenedetect` and `mean_diff` also exists.
- [ ] **C13.** `SceneDetect.py:144` — PySceneDetect resizes every frame to fixed `(640, 360)`, distorting non-16:9 aspect ratios (thresholds calibrated to the distortion).
- [ ] **C14.** `rve-backend.py` — `--device`, `--precision` interaction: `renderVideo` forces
  `precision="float32"` when device is cpu but `--precision` is still passed; verify intended.

---

## Suggested fix order

1. **B1** (restore render pipeline — decision: port vs rewrite) — nothing works without it.
2. **B2, B3, B4** (entry point bootstrap, `MINIMUM_PYTORCH_CAP`/`meets_capability`, `constants.py`) — GUI starts again.
3. **B5, B6, B7, B10** (model load crashes) — PyTorch renders work.
4. **B8, B9 + G7, G8, G10, G11** (inference correctness) — frames come out right.
5. **G1, G2, G3, G4** (GUI startup + mid-render crashes).
6. **B11, G6, G12, G13, G14** and remaining P1s.
7. **D1 + D2 together** (ONNX install + backend listing — one is useless without the other);
   **D3** resolves via B2 + G24/G25; **D4** when wiring in the health checks.
8. P2/P3 in a cleanup pass.

## Verification notes

- `uv run pytest apps/backend/tests -x -q` → 293 passed, 5 skipped (pre-fix baseline).
- `python apps/backend/rve-backend.py --version` → `ModuleNotFoundError: No module named 'apps'` (B2 verified).
- Static import check (AST walk over all relative imports) → 0 unresolved — the breakage is
  runtime/behavioral, not import resolution.
- `apps/gui/dist/mainwindow.py` absent on this machine; `backend_type_combo` does not exist in
  `apps/gui/mainwindow.ui` (0 matches) — the unified-selector code paths are dormant; the GUI
  runs on the `backendComboBox` fallback.
