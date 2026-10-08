# REAL-Video-Enhancer - Agent Guidelines

## Repository Overview

REAL-Video-Enhancer is a cross-platform video enhancement application that provides frame interpolation and upscaling using various AI models. It supports multiple backends (PyTorch, ONNX, NCNN) and runs on Windows, macOS, and Linux.

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

### Semantic Code Search

This repository is wired for **[zvec grep](https://github.com/zvec-ai/zvec-grep)**
(`zg`) — a local-first search tool that unifies ripgrep, BM25, and vector
search — through a project-scoped MCP server. The index lives in
`.zvec-grep/` (git-ignored local cache, never commit it). Prefer semantic
search over reading files blind or grepping through raw dumps when you are
looking for behavior or concepts.

**Setup — no global install needed.** This repo ships an MCP config at
`.zcode/mcp.json` that launches the server on demand via npx (requires
Node 22+):

```json
{
  "mcpServers": {
    "zvec-grep": {
      "command": "npx",
      "args": ["-y", "@zvec/zvec-grep", "--server", "--stdio"],
      "env": { "ZVEC_GREP_MCP_TOOLSET": "search" }
    }
  }
}
```

Agents connected to this MCP server get a `zvec_grep_search` tool that takes
a natural-language `query`, optional `fts` keyword terms, `globs`, and
`fuse: true` to blend semantic + keyword results. The default `search`
toolset is intentional — agents should not silently create, rebuild, or
delete the index (see
[docs/03-mcp.md](https://github.com/zvec-ai/zvec-grep/blob/main/docs/03-mcp.md)
for the `full` toolset and transport options).

**Indexing** — the search tool needs the `.zvec-grep/` index to exist. Build
or refresh it (local `potion-code` embedding model — everything stays
on-machine):

```bash
npx -y @zvec/zvec-grep index --embedding local/potion-code-16m-v2
```

Re-run this after large file moves or renames. `ZVEC_GREP_MCP_TOOLSET=full`
exposes index-management tools to agents instead, but the CLI form above is
the default workflow.

**CLI equivalent** (same package, no MCP): `npx -y @zvec/zvec-grep query
--human "<query>" --limit 5`, or `npx -y @zvec/zvec-grep --rg -F "symbol" src`
for exact search. Diagnose index problems with
`npx -y @zvec/zvec-grep status --mode direct --debug`.

**When to reach for it**:
- Before writing any new code (find existing utilities that already solve it)
- When refactoring (find duplicated patterns worth extracting into `apps/backend/utils/`)
- When tracing behavior that spans backend and GUI (render pipeline, settings, encoders)
- When onboarding to an unfamiliar subsystem

**When plain grep is still better**: exact symbol names, error strings, file
paths, or listing every call site of a known function.

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
- Run with `uv run pytest`
- 293 tests passing, 5 skipped

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
- [FOLDER_STRUCTURE.md](./docs/FOLDER_STRUCTURE.md) - Detailed folder layout
- [Steam Page](https://store.steampowered.com/app/4087640/)
