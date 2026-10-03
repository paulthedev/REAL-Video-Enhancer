# GUI Refactoring Plan

## Goal

Refactor the entire `apps/gui` folder so that each page lives in its own
folder containing its own `.ui` file (Qt Designer source) and `.py` file
(logic), reusable widget components move into `apps/gui/widgets/`, app-level
utils move into `apps/gui/utils/`, and the **main** `.ui` file becomes the
entry-point page. The current single monolithic `testRVEInterface.ui` (already
compiled to `mainwindow.py`) is split into per-page `.ui` files.

## Current State

- Single source UI file: `apps/gui/ui/testRVEInterface.ui`
  → compiled/generated output already exists as `apps/gui/mainwindow.py`
  (`Ui_MainWindow`).
- Main window = one `QMainWindow` holding a `QStackedWidget` (`stackedWidget`)
  with the following page widgets currently defined inline inside the single ui:
  - `homePage` — Home (changelog, logo, "Support Me", GitHub buttons)
  - `morePage` — currently an empty placeholder grid
  - `procPage` — Processing (preview label, render controls, ETA/FPS/STATUS,
    `processSettingsContainer` QTabWidget: General / Encoder Options / Render
    Queue / Advanced / Logs tabs)
  - `settingsPage` — has a `scrollArea` > `tabWidget` with 3 sub-tabs:
    Output Settings, Render Settings, RVE Settings (plus Reset Settings button)
  - `downloadPage` — Application update + Backends section (unified backend
    selector, torch/ncnn/tensorrt/directML install-uninstall buttons)
  - `leftMenuContainer` — side nav buttons: `homeBtn`, `processBtn`,
    `settingsBtn`, `downloadBtn`
- Python logic exists as "Page" classes in `apps/gui/ui/`:
  `HomeTab.py`, `ProcessTab.py`, `SettingsTab.py`, `DownloadTab.py`. These are
  pure-Python controller classes bound to the widgets via
  `self.parent.<widgetName>` — they contain no UI definition (UI comes from the
  compiled `mainwindow.py`).
- Shared/auxiliary Python modules in `apps/gui/ui/`:
  `AnimationHandler.py`, `QTcustom.py`, `QTstyle.py`, `RenderQueue.py`,
  `GPUDetect.py`, `Updater.py`; plus resources: `resources.qrc` and `colors.txt`.
- Empty folders already present and ready to populate:
  `apps/gui/ui/widgets/` and `apps/gui/utils/` all currently empty.
- Top-level entry app: `apps/gui/REAL-Video-Enhancer.py` imports
  `from mainwindow import Ui_MainWindow` (built via `build.py` →
  `dist/mainwindow.py`) and wires pages by name in `MainWindow.__init__` and
  `QConnect`.

## Target Structure

```
apps/gui/
├── REAL-Video-Enhancer.py              # entry point (unchanged logic, updated imports)
├── mainwindow.ui                        # NEW main/entry page .ui (the split-out big ui)
├── mainwindow.py                        # generated from mainwindow.ui -> stays as dist build
├── constants.py                         # unchanged
├── version.py                           # unchanged
├── Util.py                              # REMOVED: refactored into utils/ below
├── Backendhandler.py                    # unchanged business logic
├── ModelHandler.py                      # unchanged business logic
├── VideoInfo.py                         # unchanged business logic
├── DownloadModels.py                    # download-model service (moved to utils/backends/)
├── BuiltInTorchVersions.py              # torch version tables/dataclass -> backends/torch_versions.py 
├── DownloadDeps.py                      # backend/dependency installers -> backends/*.py (see Task 7)
├── GpuHardware.py                       # detect/health service -> backends/BackendDetect.py 
├── DiscordRPC.py                        # shared Discord RPC service (stays as top-level service)
│
├── utils/                              # app-level Python utilities (split from Util.py + decomposed service helpers)
│   ├── __init__.py                     # re-exports for backward-compat imports
│   ├── FileHandler.py                  # file/folder copy, move, remove, chmod, unzip/tar
│   ├── Log.py                          # log(), directory setup ("logs/log.txt")
│   ├── SystemInfo.py                   # getAvailableDiskSpace, getOSInfo, getRAMAmount, getCPUInfo, currentDirectory
│   ├── network.py                      # networkCheck(), downloadFile(), openLink()
│   ├── Permissions.py                  # checkForWritePermissions() (flatpak-aware)
│   ├── subprocess.py                   # subprocess_popen_without_terminal, create_independent_process
│   ├── misc.py                         # warnAndLog, errorAndLog, createDirectory, open_folder, print_execution_time
│   │
│   ├── backends/                       # decomposed service logic pulled from DownloadDeps.py / GpuHardware.py
│   ├── backends/__init__.py
│   ├── BackendsTables.py               # static vendor/backend/provider mappings (FAMILY_HARDWARE, HARDWARE_FAMILIES, compatibility tables) from GpuHardware.py / BuiltInTorchVersions.py static tables
│   ├── torch_versions.py               # TorchVersion dataclass + Torch2_15/Torch2_14 table (from BuiltInTorchVersions.py)
│   ├── BackendDetect.py                # runtime detect/health functions (detect_gpu_hardware, get_preferred_backend, check_backend_health …) from GpuHardware.py
│   ├── Dependency.py                   # shared base service class for the download/dependency model classes (from DownloadDeps.py)
│   ├── Backend.py                      # backend install/uninstall/download logic (from DownloadDeps.py)
│   ├── Python.py                       # pytorch Python package install/uninstall (from DownloadDeps.py)
│   ├── FFMpeg.py                       # ffmpeg installer/updater logic (from DownloadDeps.py)
│   ├── VCRedList.py                    # VC Redistributable installer logic (from DownloadDeps.py)
│   ├── DownloadDependencies.py         # orchestrates the per-backend dependency installs (from DownloadDeps.py)
│   ├── BackendsModels.py               # DownloadModel: download a model from url, install for pytorch/ncnn (from DownloadModels.py)
│
├── /pages/                               # ONE FOLDER PER PAGE (each has XPage.ui + XPage.py)
│   ├── home/home.ui                       # defines homePage subtree
│   ├── home/home.py                        # HomeTab logic -> HomeTab class lives here
│   ├── process/process.ui                 # defines procPage subtree
│   ├── process/process.py                  # ProcessTab logic lives here
│   ├── settings/settings.ui               # defines settingsPage subtree
│   ├── settings/settings.py               # SettingsTab logic lives here
│   ├── download/download.ui              # defines downloadPage subtree
│   └── download/download.py                        # DownloadTab logic lives here
│
├── widgets/                            # reusable Qt components
│   ├── __init__.py
│   ├── QTabNavigationBar.py             # leftMenuContainer nav buttons
│   ├── RenderProgressControls.py        # start/pause/kill + progressBar strip
│   ├── InfoRow.py                       # label + help-icon + optional widget row pattern
│   ├── VideoInfoPanel.py                # inputVideoInfoReadOnlyTextEdit block
│   └── ...                              # any other repeatedly-used component pulled from the ui
│
└── lib/                                # shared auxiliary modules not tied to one page
    ├── __init__.py
    ├── AnimationHandler.py              # moved from ui/
    ├── QTcustom.py                      # moved from ui/ (popups, overlays)
    ├── QTstyle.py                       # QPalette/Palette styling helper
    ├── RenderQueue.py                   # render queue state/logic
    ├── GPUDetect.py                     # gpu detection
    └── Updater.py                       # app updater
```

## Key Principles / Rules

1. One page ⇒ one self-contained folder under `pages/`; both the `.ui` and its
   corresponding Python controller inside it. The `.py` controllers keep using
   `self.parent.<widgetName>` binding (validated against their own page's `.ui`).
2. Repeatedly-used widget patterns pulled from the single ui into `widgets/`
   as dedicated reusable subclasses. Examples identified in the source:
   - The "label + help-icon spacer + control" **InfoRow** block, which recurs
     many times across process/settings/download pages (labels `label_N`, the
     25px info pixmap, spacers).
   - The left nav button set (`homeBtn`/`processBtn`/`settingsBtn`/`downloadBtn`).
   - The render control strip (start/pause/kill icons + `progressBar`).
   - The read-only videoinfo `inputVideoInfoTextEdit` block.
3. Business-logic modules that aren't tied to a single page sit in `lib/`
   (moved from `ui/`), keeping per-page pages free of cross-cutting concerns.
4. App-level helpers (`openLink`, `networkCheck`, etc.) move into `utils/`.
5. The main `main.window.ui` is the entry point: it no longer inlines page UI —
   each page widget references/embeds its own `.ui` file from its tab folder,
   assembled at runtime by `MainWindow`.
6. Keep `dist/mainwindow.py` generation working after refactor (update `build.py`
   source references and the `resources_rc` include).

## Refactor Tasks

Task list (work these in order):

1. **Create target scaffolding**
   - Create folders: `pages/home`, `pages/process`, `pages/settings`,
     `pages/download`, `widgets`, `utils`, `lib`.
   - Add `__init__.py` to `widgets`, `utils`, `lib`.

2. **Split the monolithic ui into per-page .ui files**
   - Extract `homePage` subtree → `pages/home/home.ui` (class `Home`).
   - Extract `procPage` subtree → `pages/process/process.ui` (class `Process`).
   - Extract `settingsPage` subtree → `pages/settings/settings.ui` (class
     `Settings`, carrying the 3 sub-pages).
   - Extract `downloadPage` subtree → `pages/download/download.ui` (class
     `Download`, carrying backends/backend-selector section).
   - Move everything else (the `QMainWindow`, `leftMenuContainer`,
     `stackedWidget`, global stylesheet) into the new entry `main.window.ui`.

3. **Move per-page logic**
   - `HomeTab.py` → `pages/home/home.py`.
   - `ProcessTab.py` → `pages/process/process.py`.
   - `SettingsTab.py` → `pages/settings/settings.py`.
   - `DownloadTab.py` → `pages/download/download.py`.
   Update all relative/non-relative imports accordingly.

4. **Move shared modules into `lib/`**
   - `AnimationHandler.py`, `QTcustom.py`, `QTstyle.py`, `RenderQueue.py`,
     `GPUDetect.py`, `Updater.py` → `lib/`.
   - Update their internal relative imports (`from .RenderQueue import ...` etc.)

5. **Extract reusable components into `widgets/`**
   - Pull the identified repeated patterns (InfoRow, nav bar, render-control strip,
     video-info panel) from ui + consolidate logic in widgets as Qt subclasses.
   - Update per-page controllers to use these widgets instead of recreating them.

6. **Split `Util.py` into `utils/`**
   - Decompose the monolithic `apps/gui/Util.py` into smaller modules (see the
     Target Structure tree): `FileHandler.py`, `Log.py`, `SystemInfo.py`,
     `network.py`, `Permissions.py`, `subprocess.py`, and a small `misc.py`.
   - Preserve every public symbol so existing callers don't break: `FileHandler`
     (class + its methods), `log`, `networkCheck`, `subprocess_popen_without_terminal`,
     `currentDirectory`, `checkForWritePermissions`, `open_folder`. Verify against
     current users: `DiscordRPC`, `GpuHardware`, `VideoInfo`, and the `FileHandler`
     usages in `Backendhandler`, `DownloadDeps`, `REAL-Video-Enhancer.py`, and
     `ui/DownloadTab`, `ui/SettingsTab`, `ui/Updater`.
   - Add `utils/__init__.py` that re-exports those symbols (e.g.
     `from .FileHandler import FileHandler`). Tab/controller files keep their
     existing imports unchanged; they just resolve via the re-export package.

7. **Decompose DownloadModels.py, BuiltInTorchVersions.py, GpuHardware.py, DownloadDeps.py into utils/** (by concern as requested)
   - Split each file by its concern into the `utils/` tree above; keep public symbols and re-export so callers do not change:
     - `GpuHardware.py` → static vendor/backend/provider tables (`FAMILY_HARDWARE`, `HARDWARE_FAMILIES`, compatibility/vendor tables) → `BackendsTables.py`; runtime detect/health functions (`detect_gpu_hardware()`, `get_preferred_backend()`, `check_backend_health()` and the private `_check_*` helpers, plus any per-backend accessor like `get_ncnn_*`) → `BackendDetect.py`. Keep both together in `utils/backends/` since they are interdependent.
     - `BuiltInTorchVersions.py` → static version tables/dataclasses (`TorchVersion`, `Torch2_15`, `Torch2_14`, and any remaining torch-version constants) into `torch_versions.py`; if any runtime logic exists keep it in the same file next to the tables.
     - `DownloadDeps.py` (898 lines) → one class per concern, no renames: shared `Dependency` base → `Dependency.py`; per-backend install/uninstall/download classes (`Backend`, `Python`, `FFMpeg`, `VCRedList`) → matching files in `utils/backends/`; the orchestration class → `DownloadDependencies.py`. All share the `Dependency` base.
     - `DownloadModels.py` → extract the single `DownloadModel` service (download a model from url, install for pytorch/ncnn) into `BackendsModels.py` under `utils/backends/`.
   - Preserve every symbol imported anywhere and verify each caller: importers are `apps/gui/REAL-Video-Enhancer.py`, `Backendhandler.py`, `ModelHandler.py` (DownloadDeps / DownloadModels / GpuHardware symbols) and `ui/` modules (`ui/UpdateBackend.py`, etc.). Add re-export entries in `utils/__init__.py` so any code that had done `from apps.gui.util import <symbol>` keeps resolving.

8. **Make main.window.ui the entry point + update `MainWindow`**
   - New `main.window.ui` embeds/references each page's `.ui`.
   - Update `apps/gui/REAL-Video-Enhancer.py`: change
     `from mainwindow import Ui_MainWindow` to reflect the build path, and switch
     tab imports from `apps.gui.ui.XTab` → per-page modules under their folders.

9. **Re-point build / resources**
   - Ensure `build.py` compiles `main.window.ui` (and page ui files if needed into
     `dist/`) and that `resources_rc` include resolves.
   - Add `__init__.py`/package adjustments so imports resolve after move.

10. **Validate**
   - Run the existing 178-test suite (`uv run pytest`) in `tests/`.
   - Launch the GUI to confirm all four pages render and nav switching works.
   - Fix any import/styled-path breakage until green.

## Verification

- `uv run pytest` passes (currently 178 tests).
- GUI opens, `main.window.ui` shows the left nav, clicking home/process/settings/
  download swaps the stacked widget correctly.
- Each page renders its widgets with the correct stylesheet (inherited via the
  main ui + stylesheet propagation).
- No dangling references to the old `apps/gui/ui/<Tab>` or single-ui paths.

## Notes / Risks

- `mainwindow.py` is auto-generated from the `.ui` and is git-ignored (`dist/`);
  keep the generator flow intact so it isn't hand-maintained.
- The name mismatch (`testRVEInterface.ui` vs the header in the generated file,
  and `mainwindow.py`) should be disambiguated as part of the main-window entry
  split.
- Because tab logic binds to `self.parent.<widgetName>`, the widget objectNames in
  every extracted `.ui` MUST match exactly what the controller expects; keep an
  inventory when extracting.
- Risks: stylesheet propagation across page boundaries (the global qss on the
  QMainWindow vs per-page styleSheets), and `resources_rc.py` path resolution
  after move—verify at runtime.
