# REAL-Video-Enhancer - Agent Guidelines

## Repository Overview

REAL-Video-Enhancer is a cross-platform video enhancement application that provides frame interpolation and upscaling using various AI models. ONNX Runtime is the only inference backend; it runs on Windows, macOS, and Linux.

## Project Structure

See [FOLDER_STRUCTURE.md](./docs/FOLDER_STRUCTURE.md) for detailed folder layout.

## Documentation

All documentation files must be placed in the `docs/` folder. This includes:
- FOLDER_STRUCTURE.md
- pytorch-optimizations.md
- Any other documentation files

## Code Organization

### Backend (`apps/backend/`)
- Core video processing logic
- Model inference (ONNX Runtime only; `apps/backend/render/onnx/`)
- PyTorch sources under `apps/backend/pytorch/` + `apps/backend/models/` are
  dev-only tooling for producing new ONNX models (`converters/torch_to_onnx.py`)
- FFmpeg integration
- Utility functions

### GUI (`apps/gui/`)
- PySide6-based user interface
- UI source files in `ui/`
- Icons in `icons/`
- Build output in `dist/`

### Build System
- `build.py` - Main build script
- `pyproject.toml` - Python dependencies (single source of truth)
- `uv` - Package manager for running Python
- Output goes to `dist/` folder

### Installers
- `installers/REAL-Video-Enhancer.nsi` - Windows NSIS installer
- `installers/build-appimage.sh` - Linux AppImage builder
- `installers/appimage/` - AppImage assets

## Development Rules

### Code Reuse Principles
**Always consider code reuse before writing new code or refactoring.**

1. **Check for existing utilities first**: Before implementing new functionality, search `apps/backend/utils/` for existing utilities that might already solve the problem.

2. **Extract common patterns**: When you notice similar code patterns in multiple places, extract them into shared utilities in `apps/backend/utils/`.

3. **Reusable utilities location**: All shared utilities should be placed in `apps/backend/utils/` and exported from `apps/backend/utils/__init__.py`.

4. **Examples of reusable utilities**:
   - `OnnxLoader.py` - Reusable ONNX model loading with automatic provider selection
   - `Util.py` - General utility functions
   - `Frame.py` - Frame handling utilities
   - `VideoInfo.py` - Video information utilities
   - `Encoders.py` - Encoding utilities

5. **Refactoring rule**: When refactoring code, look for opportunities to:
   - Replace duplicated logic with calls to shared utilities
   - Extract common patterns into utility functions
   - Use existing utilities instead of reimplementing functionality

6. **Before writing new code**:
   - Search for existing utilities using zvec grep with semantic queries (see [Semantic Code Search](#semantic-code-search))
   - Check if the functionality already exists in `apps/backend/utils/`
   - If not, consider whether it should be added as a reusable utility

### Dependencies
- Use `pyproject.toml` for all dependencies
- Run Python with `uv run python` instead of direct `python`
- Use `uv pip install -e .` for editable installs
- Remove any `requirements.txt` files (redundant with pyproject.toml)

### Building
- Dev UI modules compile to `dist/` (git ignored); package builds go to `packages/` (git ignored)
- Run `python build.py` for dev modules, `python build.py --build appimage` (or `cx_freeze`/`pyinstaller`) to package
- Dev output: `dist/mainwindow.py`, `dist/pages/`, `dist/resources_rc.py`; packaged bundles + installers: `packages/`

### Testing
- Tests in `apps/backend/tests/` directory
- Run with `uv run --extra convert --extra dev pytest` (the `convert` extra
  supplies torch for the model/converter tests; the shipping runtime does not
  need it)
- 322 tests passing, 5 warnings (CI also runs a torch-free runtime-smoke job)

### Logging
- Runtime logs go to `logs/log.txt`
- Logs are git-ignored
- Created by `apps/gui/Util.py`

### Git Workflow
- Commit often with descriptive messages
- Push the current working branch (`main` for releases; feature branches for
  larger changes)
- Use conventional commit messages

## Key Technologies
- **PySide6** - GUI framework
- **ONNX Runtime** - The inference backend (bundled; DirectML on Windows,
  runtime-upgraded CUDA/TensorRT/OpenVINO/ROCm builds per GPU — see
  docs/onnx-only-migration-plan.md)
- **imageio-ffmpeg** - Bundled static ffmpeg (no binary downloads)
- **FFmpeg** - Video processing
- **cx_Freeze/PyInstaller** - Windows builds
- **NSIS** - Windows installer
- **AppImage** - Linux portable app

## Common Tasks

### Adding a New Model
1. Produce an ONNX model (see `apps/backend/converters/torch_to_onnx.py` for
   the dev-only conversion path)
2. Add it to the models repo (or import it through the Models page, which also
   converts `.pth`/`.safetensors` when the optional torch extra is installed)
3. For a built-in option: add the ONNX entry to
   `apps/gui/lib/ModelHandler.py` (file name + scale) and the download table
4. Update documentation

### Building for Release
1. Run `python build.py` (dev UI modules → `dist/`)
2. Run `python build.py --build cx_freeze --copy_backend` (Linux) or
   `--build pyinstaller --copy_backend` (Windows/macOS) → `packages/`
3. Test the generated bundle/installers
4. Update `version` in `pyproject.toml` / `apps/gui/version.py`

### Debugging
- Check `logs/log.txt` for runtime errors
- Run tests with `uv run pytest`
- Check CI/CD in `.github/workflows/`

## Semantic Search (zvec-grep)

A semantic code search index is available via MCP. Use `zvec_grep_search` for
**conceptual/meaning-based** queries (e.g. "where do we handle auth token refresh")
and `zvec_grep_rg` for **exact string/regex** matches.

### When to use which

| Intent | Tool |
|---|---|
| "Find code that does X" (fuzzy, semantic) | `zvec_grep_search` |
| "Find all occurrences of `fooBar`" | `zvec_grep_rg` |
| "What files implement retry logic?" | `zvec_grep_search` |
| "Where is `TODO: fix` commented?" | `zvec_grep_rg` |

### Index management

- Embedding model: `local/potion-code-16m-v2`
- Index location: `.zvec-grep/` (gitignored)
- Rebuild after large refactors: ask the user to run `zg index --rebuild`
- **Never** create, rebuild, or drop the index silently — always confirm with the user first
- Check status with `zvec_grep_index_status` before assuming the index is stale

### Tips

- Prefer `zvec_grep_search` as the **first** lookup when exploring unfamiliar code
- Combine with `zvec_grep_rg` to narrow results (semantic → exact)
- If search returns nothing, the index may be stale — suggest a rebuild   
