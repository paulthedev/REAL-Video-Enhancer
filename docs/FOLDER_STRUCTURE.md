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
├── README.md
├── requirements.txt
├── rve-backend.py
├── src/
│   ├── constants.py
│   ├── FFmpeg.py
│   ├── FFmpegBuffers.py
│   ├── RenderVideo.py
│   ├── version.py
│   ├── ncnn/              # NCNN backend
│   │   ├── __init__.py
│   │   ├── InterpolateNCNN.py
│   │   ├── NCNNEfficientNetSC.py
│   │   └── UpscaleNCNN.py
│   ├── onnx/              # ONNX backend
│   │   ├── __init__.py
│   │   ├── InterpolateONNX.py
│   │   └── UpscaleONNX.py
│   ├── pytorch/           # PyTorch backend
│   │   ├── __init__.py
│   │   ├── BaseInterpolate.py
│   │   ├── InterpolateGIMM.py
│   │   ├── InterpolateGMFSS.py
│   │   ├── InterpolateIFRNET.py
│   │   ├── InterpolateRIFE.py
│   │   ├── InterpolateTorch.py
│   │   ├── TensorRTHandler.py
│   │   ├── TorchUtils.py
│   │   ├── tuning.py
│   │   ├── UpscaleModelWrapper.py
│   │   ├── UpscaleTorch.py
│   │   ├── DRBA/
│   │   ├── InterpolateArchs/
│   │   ├── scenechangedetect/
│   │   ├── spandrel/
│   │   └── VSRArchs/
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── BackendDetect.py
│       ├── BorderDetect.py
│       ├── Colors.py
│       ├── Encoders.py
│       ├── FileHandler.py
│       ├── Frame.py
│       ├── GetFFMpeg.py
│       ├── PySceneDetectUtils.py
│       ├── SceneDetect.py
│       ├── SSIM.py
│       ├── Util.py
│       └── VideoInfo.py
└── tests/
    ├── benchmarks/
    │   ├── bench_torch_interpolate.py
    │   └── bench_torch_upscale.py
    └── unit_tests/
        ├── test_frame.py
        └── test_pytorch_optimizations.py
```

### GUI (`apps/gui/`)

```
apps/gui/
├── Backendhandler.py
├── BuiltInTorchVersions.py
├── constants.py
├── DiscordRPC.py
├── DownloadDeps.py
├── DownloadModels.py
├── GenerateFFMpegCommand.py
├── GpuHardware.py
├── ModelHandler.py
├── Util.py
├── version.py
├── VideoInfo.py
├── REAL-Video-Enhancer.py  # Main entry point
├── icons/                  # Application icons (290+ SVG files)
│   ├── logo-v2.svg
│   ├── activity.svg
│   ├── airplay.svg
│   └── ...
└── ui/                     # UI source files
    ├── testRVEInterface.ui
    ├── resources.qrc
    └── colors.txt
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
├── mainwindow.py          # Compiled from testRVEInterface.ui
├── resources_rc.py        # Compiled from resources.qrc
└── REAL-Video-Enhancer    # Windows executable (if built)
```

## Logs (`logs/`) - Git Ignored

```
logs/
└── log.txt                # Runtime application logs
```

## Tests (`tests/`)

```
tests/
├── benchmarks/
│   ├── bench_torch_interpolate.py
│   └── bench_torch_upscale.py
└── unit_tests/
    ├── test_frame.py
    └── test_pytorch_optimizations.py
```

## Documentation (`docs/`)

```
docs/
└── pytorch-optimizations.md
```

## GitHub (` .github/`)

```
.github/
├── ISSUE_TEMPLATE/
│   └── bug_report.md
└── workflows/
    └── prerelease.yml
```

## Key Points

- **Single source of truth**: `pyproject.toml` for dependencies
- **Build output**: All artifacts go to `dist/` (git ignored)
- **Logs**: Runtime logs in `logs/` (git ignored)
- **UI source**: `.ui` and `.qrc` files in `apps/gui/ui/`
- **Icons**: SVG icons in `apps/gui/icons/`
- **Installers**: All installer scripts in `installers/`
- **Backend**: Processing code in `apps/backend/`
- **GUI**: Application code in `apps/gui/`

## Links

- [AGENTS.md](./AGENTS.md) - Agent guidelines and rules
- [README.md](./README.md) - Project documentation
