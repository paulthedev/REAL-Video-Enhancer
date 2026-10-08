"""
Subprocess tests for the rve-backend CLI surface.

The GUI drives the backend exclusively through this CLI (backend health
check via --list_backends, version lockstep via --version), so the flags
are a public contract.
"""

import os
import subprocess
import sys

from apps.backend.version import __version__ as backend_version

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir
    )
)
BACKEND_SCRIPT = os.path.join(REPO_ROOT, "apps", "backend", "rve-backend.py")


def run_backend(*args, timeout=120):
    return subprocess.run(
        [sys.executable, BACKEND_SCRIPT, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


class TestVersion:
    def test_version_flag_prints_backend_version(self):
        result = run_backend("--version")
        assert result.returncode == 0
        assert result.stdout.strip() == backend_version

    def test_backend_and_gui_versions_stay_in_lockstep(self):
        # a mismatch makes the GUI think the backend is outdated and
        # triggers spurious update downloads at startup
        from apps.gui.version import backend_dev_version

        assert backend_version == backend_dev_version


class TestListBackends:
    def test_list_backends_exits_zero(self):
        result = run_backend("--list_backends")
        assert result.returncode == 0

    def test_onnx_is_the_only_backend(self):
        result = run_backend("--list_backends")
        assert "Available Backends: ['onnx']" in result.stdout

    def test_reports_onnx_runtime_and_providers(self):
        result = run_backend("--list_backends")
        assert "ONNX Runtime Version:" in result.stdout
        assert "ONNX Providers:" in result.stdout
