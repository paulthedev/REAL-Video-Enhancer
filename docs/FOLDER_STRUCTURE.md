# REAL-Video-Enhancer - Folder Structure

## Root Level

```
REAL-Video-Enhancer/
├── AGENTS.md              # Agent guidelines and rules (this file)
├── FOLDER_STRUCTURE.md    # This file
├── README.md              # Project documentation
├── CHANGELOG.md           # Version history
├── LICENSE                # MIT License
├── build.py               # Main build script
├── pyproject.toml         # Python dependencies (single source of truth)
├── .gitignore             # Git ignore rules
├── .github/               # GitHub workflows and templates
├── dist/                  # Build output (git ignored)
├── logs/                  # Runtime logs (git ignored)
├── apps/                  # Application source code
│   ├── backend/           # Backend processing code
│   └── gui/               # GUI application code
├── docs/                  # Documentation
├── icons/                 # (legacy, moved to apps/gui/icons/)
├── installers/            # Installer scripts and assets
├── scripts/               # Utility scripts
└── tests/                 # Test suite
```

## Apps Directory

### Backend (`apps/backend/`)

```
apps/backend/
├── rve-backend.py          # Backend CLI entry point (spawned by the GUI)
├── constants.py            # platform info + provider-dir activation
├── version.py
├── backends/               # backend-agnostic loader/runner abstractions
│   ├── base.py
│   └── onnx/               # loader.py + runner.py (used by tests/tooling)
├── render/                 # the shipping render path
│   ├── RenderVideo.py      # render loop (ONNX only)
│   ├── FFmpeg.py
│   ├── FFmpegBuffers.py
│   └── onnx/               # InterpolateONNX.py, UpscaleONNX.py, session.py
├── models/                 # model definitions + registry (converter tooling)
│   ├── base.py
│   ├── registry.py
│   ├── interpolate/        # rife, gimm, gmfss, ifrnet
│   ├── upscale/            # span, tspan, animesr
│   ├── restoration/        # fbcnn, nafnet
│   ├── denoise/            # dncnn
│   └── scene_detect/       # onnx + maxvit
├── pytorch/                # PyTorch reference implementations (dev/convert only)
├── converters/             # torch_to_onnx.py (dev tooling)
├── utils/
│   ├── BorderDetect.py
│   ├── Colors.py
│   ├── Encoders.py
│   ├── FileHandler.py
│   ├── Frame.py
│   ├── GetFFMpeg.py        # imageio-ffmpeg fallback (no downloads)
│   ├── OnnxLoader.py
│   ├── PySceneDetectUtils.py
│   ├── SceneDetect.py
│   ├── Util.py
│   └── VideoInfo.py
└── tests/
    ├── benchmarks/          # ONNX + torch conversion benchmarks
    ├── integration_tests/   # ONNX e2e render, ffmpeg round-trip, conversion
    └── unit_tests/          # runtime, registry, provider-upgrade, CLI tests
```

`apps/backend/pytorch/` and the model loaders in `apps/backend/models/` are
**dev/conversion tooling** only: the shipped render path is
`render/onnx/*` + `render/RenderVideo.py` and never imports torch or ncnn.

### GUI (`apps/gui/`)

```
apps/gui/
├── REAL-Video-Enhancer.py  # Main entry point (also dispatches `-m pip`)
├── constants.py
├── version.py
├── mainwindow.ui
├── resources.qrc
├── icons/                  # Application icons
├── lib/                    # BackendHandler, ModelHandler, QTcustom,
│                           # QTstyle, ProviderUpgradeUI, RenderQueue, ...
├── pages/                  # home / process / settings / download pages
│   └── */                  # each with its .ui source
├── util/                   # shared GUI helpers
├── utils/                  # VideoInfo, backend helpers
│   └── backends/           # BackendDetect, ProviderUpgrade, model downloads
├── widgets/                # QTabNavigation, PreviewCompare, ...
└── tests/                  # offscreen smoke + render-queue tests
```

## Installers (`installers/`)

```
installers/
├── REAL-Video-Enhancer.nsi      # Windows NSIS installer script
├── build-appimage.sh            # Linux AppImage builder
└── appimage/                    # AppImage assets
    ├── AppRun
    ├── REAL-Video-Enhancer.desktop
    ├── REAL-Video-Enhancer-2.4.2-linux-x86_64.AppImage
    └── icons/
        └── hicolor/
            ├── 16x16/apps/
            ├── 24x24/apps/
            ├── 32x32/
            ├── 48x48/
            ├── 64x64/
            ├── 128x128/apps/
            ├── 256x256/apps/
            └── 512x512/apps/
```

## Build Output (`dist/`) - Git Ignored

```
dist/
├── mainwindow.py          # Compiled from apps/gui/mainwindow.ui
├── resources_rc.py        # Compiled from apps/gui/resources.qrc
└── pages/                 # Compiled page UIs (home/process/settings/download)
```

Packaged bundles and installers go to `packages/` (also git ignored).

## Logs (`logs/`) - Git Ignored

```
logs/
└── log.txt                # Runtime application logs
```

## Tests (`apps/backend/tests/` and `apps/gui/tests/`)

```
apps/backend/tests/
├── benchmarks/            # ONNX + torch-conversion benchmarks
├── integration_tests/     # ONNX e2e render, ffmpeg round-trip, conversion
└── unit_tests/            # runtime, provider upgrade, registry, CLI tests
apps/gui/tests/            # offscreen smoke + render-queue tests
```

## Documentation (`docs/`)

```
docs/
├── onnx-only-migration-plan.md
├── TRACKER.md
├── BUG_TRACKER.md
├── UI_REDESIGN_TRACKER.md
├── FOLDER_STRUCTURE.md
└── archived/              # completed plans and migration notes
```

## GitHub (`.github/`)

```
.github/
├── ISSUE_TEMPLATE/
│   └── bug_report.md
└── workflows/
    ├── tests.yml          # dev suite + torch-free runtime smoke
    └── prerelease.yml
```

## Key Points

- **Single source of truth**: `pyproject.toml` for dependencies
- **Dev UI output**: compiled modules in `dist/` (git ignored); packages in `packages/`
- **Logs**: runtime logs in `logs/` (git ignored)
- **UI source**: `apps/gui/mainwindow.ui` + `apps/gui/pages/*/*.ui`; resources in `apps/gui/resources.qrc`
- **Icons**: SVG icons in `apps/gui/icons/`
- **Installers**: all installer scripts in `installers/`
- **Backend**: ONNX runtime + processing in `apps/backend/` (`render/onnx/` is the shipping path)
- **GUI**: application code in `apps/gui/`

## Links

- [AGENTS.md](../AGENTS.md) - Agent guidelines and rules
- [README.md](../README.md) - Project documentation
