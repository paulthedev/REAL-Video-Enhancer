"""
Unit tests for the GPU-aware onnxruntime provider upgrade manager.

These are pure-function tests: no network, no pip, no GPU required.
"""

import json
import os
import sys

import pytest

from apps.gui.utils.backends import ProviderUpgrade
from apps.gui.utils.backends.ProviderUpgrade import (
    MARKER_FILE,
    PROVIDERS_DIR_NAME,
    activate_providers_dir,
    get_recommended_provider,
    install_recommended_provider,
    provider_upgrade_status,
    providers_dir,
)


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    """Isolated XDG data dir so tests never touch the real providers dir."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("RVE_ONNX_PROVIDERS_DIR", raising=False)
    return tmp_path


def _write_marker(data_home, payload):
    marker_dir = data_home / "REAL-Video-Enhancer" / "providers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / MARKER_FILE).write_text(json.dumps(payload))
    return marker_dir


class TestProvidersDir:
    def test_linux_uses_xdg_data_home(self, data_home):
        assert providers_dir() == str(
            data_home / "REAL-Video-Enhancer" / "providers"
        )

    def test_linux_falls_back_to_home(self, data_home, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME")
        expected = os.path.join(
            os.path.expanduser("~/.local/share"), PROVIDERS_DIR_NAME
        )
        assert providers_dir() == expected

    def test_windows_uses_appdata(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(data_home))
        assert providers_dir() == str(
            data_home / "REAL-Video-Enhancer" / "providers"
        )


class TestUpgradeStatus:
    def test_no_marker_means_bundled(self, data_home):
        assert provider_upgrade_status() == {
            "provider": "bundled",
            "package": None,
        }

    def test_marker_is_read_back(self, data_home):
        _write_marker(
            data_home,
            {"provider": "onnxruntime-rocm", "package": "onnxruntime-rocm"},
        )
        status = provider_upgrade_status()
        assert status["provider"] == "onnxruntime-rocm"

    def test_corrupt_marker_falls_back_to_bundled(self, data_home):
        marker_dir = data_home / "REAL-Video-Enhancer" / "providers"
        marker_dir.mkdir(parents=True)
        (marker_dir / MARKER_FILE).write_text("{not json")
        assert provider_upgrade_status()["provider"] == "bundled"


class TestRecommendedProvider:
    def _no_hardware(self, monkeypatch):
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: [],
        )

    def test_windows_bundled_directml_covers_non_nvidia(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["amd", "intel"],
        )
        assert get_recommended_provider() is None

    def test_windows_nvidia_gets_cuda_build(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["nvidia"],
        )
        monkeypatch.setattr(
            ProviderUpgrade, "_nvidia_compute_capability", lambda: (8, 9)
        )
        recommendation = get_recommended_provider()
        assert recommendation["package"] == "onnxruntime-gpu"
        assert recommendation["tensorrt"] is True

    def test_macos_bundled(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert get_recommended_provider() is None

    def test_nvidia_maps_to_cuda_build(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["nvidia"],
        )
        monkeypatch.setattr(
            ProviderUpgrade, "_nvidia_compute_capability", lambda: (8, 6)
        )
        recommendation = get_recommended_provider()
        assert recommendation["package"] == "onnxruntime-gpu"
        assert recommendation["index_url"] is None
        assert recommendation["compute_capability"] == (8, 6)
        assert recommendation["tensorrt"] is True

    def test_nvidia_below_sm75_skips_tensorrt(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["nvidia"],
        )
        monkeypatch.setattr(
            ProviderUpgrade, "_nvidia_compute_capability", lambda: (7, 0)
        )
        assert get_recommended_provider()["tensorrt"] is False

    def test_intel_maps_to_openvino_build(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["intel"],
        )
        assert get_recommended_provider() == {
            "package": "onnxruntime-openvino",
            "index_url": None,
        }

    def test_amd_maps_to_rocm_build_from_amd_index(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["amd"],
        )
        monkeypatch.setattr(
            ProviderUpgrade,
            "_newest_supported_rocm_release",
            lambda gfx=None: "rocm-rel-7.2.4",
        )
        monkeypatch.setattr(ProviderUpgrade, "_detect_amd_gfx", lambda: "gfx1100")
        recommendation = get_recommended_provider()
        assert recommendation["package"] == "onnxruntime-rocm"
        assert recommendation["index_url"] == (
            ProviderUpgrade.AMD_ROCM_INDEX + "rocm-rel-7.2.4/"
        )

    def test_amd_without_compatible_release_gets_bundled(
        self, data_home, monkeypatch
    ):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["amd"],
        )
        monkeypatch.setattr(
            ProviderUpgrade, "_newest_supported_rocm_release", lambda gfx=None: None
        )
        monkeypatch.setattr(ProviderUpgrade, "_detect_amd_gfx", lambda: None)
        assert get_recommended_provider() is None

    def test_no_gpu_gets_bundled(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        self._no_hardware(monkeypatch)
        assert get_recommended_provider() is None

    def test_nvidia_wins_over_other_vendors(self, data_home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "apps.gui.utils.backends.BackendDetect.detect_gpu_hardware",
            lambda: ["amd", "nvidia"],
        )
        assert get_recommended_provider()["package"] == "onnxruntime-gpu"


class FakeProcess:
    def __init__(self, returncode=0, lines=()):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.killed = False

    def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


class TestInstallProvider:
    def test_no_recommendation_is_a_noop(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade, "get_recommended_provider", lambda: None
        )
        assert install_recommended_provider() is False
        assert not (data_home / "REAL-Video-Enhancer" / "providers").exists()

    def test_pip_command_targets_providers_dir(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {"package": "onnxruntime-gpu", "index_url": None},
        )
        commands = []

        def fake_popen(cmd, **kwargs):
            commands.append(cmd)
            return FakeProcess(lines=["done"])

        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal", fake_popen
        )
        assert install_recommended_provider() is True
        target = str(data_home / "REAL-Video-Enhancer" / "providers")
        assert commands == [
            [sys.executable, "-m", "pip",
             "install", "--upgrade", "--only-binary=:all:",
             "--no-warn-script-location",
             "--target", target, "onnxruntime-gpu"]
        ]

    def test_rocm_command_includes_amd_index(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {
                "package": "onnxruntime-rocm",
                "index_url": ProviderUpgrade.AMD_ROCM_INDEX + "rocm-rel-7.2.4/",
            },
        )
        commands = []
        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal",
            lambda cmd, **kw: commands.append(cmd) or FakeProcess(),
        )
        assert install_recommended_provider() is True
        cmd = commands[0]
        # --extra-index-url / --trusted-host come after the shared pip flags
        assert cmd[7:11] == [
            "--extra-index-url",
            ProviderUpgrade.AMD_ROCM_INDEX + "rocm-rel-7.2.4/",
            "--trusted-host",
            "repo.radeon.com",
        ]
        assert cmd[-1] == "onnxruntime-rocm"

    def test_success_writes_marker(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {"package": "onnxruntime-gpu", "index_url": None},
        )
        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal",
            lambda cmd, **kw: FakeProcess(),
        )
        install_recommended_provider()
        status = provider_upgrade_status()
        assert status["provider"] == "onnxruntime-gpu"

    def test_pip_failure_leaves_bundled_active(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {"package": "onnxruntime-gpu", "index_url": None},
        )
        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal",
            lambda cmd, **kw: FakeProcess(returncode=1, lines=["boom"]),
        )
        assert install_recommended_provider() is False
        assert provider_upgrade_status()["provider"] == "bundled"

    def test_cancel_kills_pip_and_writes_no_marker(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {"package": "onnxruntime-gpu", "index_url": None},
        )
        process = FakeProcess(lines=["line1", "line2"])

        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal",
            lambda cmd, **kw: process,
        )
        assert install_recommended_provider(cancel_check=lambda: True) is False
        assert process.killed
        assert provider_upgrade_status()["provider"] == "bundled"

    def test_progress_receives_pip_output_lines(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {"package": "onnxruntime-gpu", "index_url": None},
        )
        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal",
            lambda cmd, **kw: FakeProcess(lines=["Downloading", "Installed"]),
        )
        seen = []
        install_recommended_provider(progress=seen.append)
        assert seen == ["Downloading", "Installed"]

    def test_tensorrt_tier_installs_libs_and_records_marker(
        self, data_home, monkeypatch
    ):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {
                "package": "onnxruntime-gpu",
                "index_url": None,
                "tensorrt": True,
                "compute_capability": (8, 6),
            },
        )
        commands = []
        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal",
            lambda cmd, **kw: commands.append(cmd) or FakeProcess(),
        )
        assert install_recommended_provider() is True
        assert len(commands) == 2
        assert commands[1][-1] == ProviderUpgrade.TENSORRT_REQUIREMENT
        status = provider_upgrade_status()
        assert status["tensorrt"] is True
        assert status["compute_capability"] == [8, 6]  # JSON round-trip

    def test_tensorrt_failure_keeps_cuda_install(self, data_home, monkeypatch):
        monkeypatch.setattr(
            ProviderUpgrade,
            "get_recommended_provider",
            lambda: {
                "package": "onnxruntime-gpu",
                "index_url": None,
                "tensorrt": True,
            },
        )

        def fake_popen(cmd, **kwargs):
            return FakeProcess(
                returncode=0 if "onnxruntime-gpu" in cmd else 1,
                lines=["out"],
            )

        monkeypatch.setattr(
            "apps.gui.util.subprocess_popen_without_terminal", fake_popen
        )
        assert install_recommended_provider() is True
        status = provider_upgrade_status()
        assert status["package"] == "onnxruntime-gpu"
        assert status["tensorrt"] is False


class TestPruneShadowedPackages:
    def test_bundled_packages_are_removed_from_target(self, tmp_path):
        target = tmp_path / "providers"
        for name in ("numpy", "numpy.libs", "google", "onnxruntime"):
            (target / name).mkdir(parents=True)
        (target / "numpy-2.0.0.dist-info").mkdir()
        (target / "onnxruntime_gpu-1.30.0.dist-info").mkdir()

        ProviderUpgrade._prune_shadowed_packages(str(target))

        assert not (target / "numpy").exists()
        assert not (target / "numpy.libs").exists()
        assert not (target / "google").exists()
        assert not (target / "numpy-2.0.0.dist-info").exists()
        assert (target / "onnxruntime").exists()
        assert (target / "onnxruntime_gpu-1.30.0.dist-info").exists()


class TestAmdGfxGating:
    def test_new_arch_requires_newer_rocm(self):
        assert not ProviderUpgrade._release_supports_gfx(
            "rocm-rel-5.7.1", "gfx1200"
        )
        assert ProviderUpgrade._release_supports_gfx(
            "rocm-rel-6.4.1", "gfx1200"
        )

    def test_known_arch_accepts_supported_release(self):
        assert ProviderUpgrade._release_supports_gfx(
            "rocm-rel-6.0.0", "gfx1100"
        )

    def test_unknown_arch_is_rejected(self):
        assert not ProviderUpgrade._release_supports_gfx(
            "rocm-rel-7.0.0", "gfx9999"
        )


class TestActivateProvidersDir:
    def test_marker_prepends_providers_dir_to_sys_path(
        self, tmp_path, monkeypatch
    ):
        providers = _write_marker(
            tmp_path, {"package": "onnxruntime-rocm"}
        )
        monkeypatch.setenv("RVE_ONNX_PROVIDERS_DIR", str(providers))
        try:
            assert activate_providers_dir() is True
            assert sys.path[0] == str(providers)
        finally:
            if str(providers) in sys.path:
                sys.path.remove(str(providers))

    def test_no_marker_activates_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RVE_ONNX_PROVIDERS_DIR", str(tmp_path / "absent"))
        sys_path_before = list(sys.path)
        assert activate_providers_dir() is False
        assert sys.path == sys_path_before

    def test_marker_without_package_activates_nothing(
        self, tmp_path, monkeypatch
    ):
        providers = _write_marker(tmp_path, {"provider": "bundled"})
        monkeypatch.setenv("RVE_ONNX_PROVIDERS_DIR", str(providers))
        sys_path_before = list(sys.path)
        assert activate_providers_dir() is False
        assert sys.path == sys_path_before

    def test_tensorrt_marker_exports_cache_dir(self, tmp_path, monkeypatch):
        providers = _write_marker(
            tmp_path,
            {"package": "onnxruntime-gpu", "tensorrt": True},
        )
        monkeypatch.setenv("RVE_ONNX_PROVIDERS_DIR", str(providers))
        monkeypatch.delenv("RVE_ONNX_TRT_CACHE_DIR", raising=False)
        try:
            assert activate_providers_dir() is True
            cache = os.environ["RVE_ONNX_TRT_CACHE_DIR"]
            assert cache == str(providers / ProviderUpgrade.TRT_CACHE_DIR_NAME)
            assert os.path.isdir(cache)
            assert os.environ["RVE_ONNX_PROVIDERS_DIR"] == str(providers)
        finally:
            if str(providers) in sys.path:
                sys.path.remove(str(providers))
            monkeypatch.delenv("RVE_ONNX_TRT_CACHE_DIR", raising=False)
