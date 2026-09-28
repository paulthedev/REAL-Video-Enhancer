# REAL-Video-Enhancer - Agent Guidelines

## Repository Overview

REAL-Video-Enhancer is a cross-platform video enhancement application that provides frame interpolation and upscaling using various AI models. It supports multiple backends (PyTorch, ONNX, NCNN) and runs on Windows, macOS, and Linux.

## Project Structure

See [FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md) for detailed folder layout.

## Code Organization

### Backend (`apps/backend/`)
- Core video processing logic
- Model inference (PyTorch, ONNX, NCNN)
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

### Dependencies
- Use `pyproject.toml` for all dependencies
- Run Python with `uv run python` instead of direct `python`
- Use `uv pip install -e .` for editable installs
- Remove any `requirements.txt` files (redundant with pyproject.toml)

### Building
- GUI files compile to `dist/` (git ignored)
- Run `python build.py` to build
- Generated files: `dist/mainwindow.py`, `dist/resources_rc.py`

### Testing
- Tests in `tests/` directory
- Run with `uv run pytest`
- 178 tests passing

### Logging
- Runtime logs go to `logs/log.txt`
- Logs are git-ignored
- Created by `apps/gui/Util.py`

### Git Workflow
- Commit often with descriptive messages
- Push to `update_dependencies` branch for development
- Use conventional commit messages

## Key Technologies
- **PySide6** - GUI framework
- **PyTorch** - Primary ML backend
- **ONNX/NCNN** - Alternative backends
- **FFmpeg** - Video processing
- **cx_Freeze/PyInstaller** - Windows builds
- **NSIS** - Windows installer
- **AppImage** - Linux portable app

## Common Tasks

### Adding a New Model
1. Add model files to `apps/backend/models/`
2. Update model registry in backend
3. Add UI options in `apps/gui/`
4. Update documentation

### Building for Release
1. Run `python build.py`
2. Test generated artifacts in `dist/`
3. Build installers if needed
4. Update version in `version.py`

### Debugging
- Check `logs/log.txt` for runtime errors
- Run tests with `uv run pytest`
- Check CI/CD in `.github/workflows/`

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit pull request

## Links
- [GitHub Repository](https://github.com/paulthedev/REAL-Video-Enhancer)
- [Discord Community](https://discord.gg/hwGHXga8ck)
- [Steam Page](https://store.steampowered.com/app/4087640/)
