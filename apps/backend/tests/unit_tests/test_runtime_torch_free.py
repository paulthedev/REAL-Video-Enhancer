"""The shipping ONNX runtime must never pull in torch or ncnn.

These run in the `runtime-smoke` CI job, which installs only the base
dependencies (no `convert` extra). Each test also passes in a dev venv
because it asserts on *imports*, not on installation.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("RVE_RUN_BACKEND", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_render_pipeline_does_not_import_torch_or_ncnn():
    result = _run(
        "import sys\n"
        "import apps.backend.render.RenderVideo\n"
        "import apps.backend.render.onnx.session\n"
        "assert 'torch' not in sys.modules, 'torch imported by the render path'\n"
        "assert 'ncnn' not in sys.modules, 'ncnn imported by the render path'\n"
    )
    assert result.returncode == 0, result.stderr


def test_frozen_pip_dispatch():
    """The frozen binary doubles as `python -m pip` for provider installs."""
    script = os.path.join(REPO_ROOT, "apps", "gui", "REAL-Video-Enhancer.py")
    env = dict(os.environ)
    env.pop("RVE_RUN_BACKEND", None)
    result = subprocess.run(
        [sys.executable, script, "-m", "pip", "--version"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("pip ")


def test_backend_cli_is_onnx_only():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "apps", "backend", "rve-backend.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "only 'onnx' is available" in " ".join(result.stdout.split())


def test_render_module_has_no_dead_backend_branches():
    result = _run(
        "import inspect\n"
        "import apps.backend.render.RenderVideo as rv\n"
        "src = inspect.getsource(rv)\n"
        "for bad in ('UpscaleNCNN', 'InterpolateTorch', 'UpscalePytorch'):\n"
        "    assert bad not in src, bad\n"
    )
    assert result.returncode == 0, result.stderr
