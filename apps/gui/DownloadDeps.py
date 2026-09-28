from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from PySide6.QtWidgets import QMessageBox

from src.constants import (
    PLATFORM,
    PYTHON_DIRECTORY,
    PYTHON_EXECUTABLE_PATH,
    PYTHON_VERSION,
    FFMPEG_PATH,
    BACKEND_PATH,
    TEMP_DOWNLOAD_PATH,
    CWD,
    CPU_ARCH,
    USE_LOCAL_BACKEND,
    IS_STEAM
)
from src.version import version, backend_dev_version
from src.GpuHardware import (
    detect_gpu_hardware,
    FAMILY_HARDWARE,
    HARDWARE_FAMILIES,
)
from src.Util import (
    FileHandler,
    log,
    extractTarGZ,
    removeFolder,
    subprocess_popen_without_terminal
)
from apps.gui.ui.QTcustom import (
    DownloadProgressPopup,
    DisplayCommandOutputPopup,
    RegularQTPopup,
    needs_network_else_exit,
)
import os
import re
import subprocess


def run_executable(exe_path):
    try:
        # Run the executable and wait for it to complete
        result = subprocess.run(exe_path, check=True, capture_output=True, text=True)

        # Print the output of the executable
        print("STDOUT:", result.stdout)

        # Print any error messages
        print("STDERR:", result.stderr)

        # Print the exit code
        print("Exit Code:", result.returncode)

    except subprocess.CalledProcessError as e:
        print("An error occurred while running the executable.")
        print("Exit Code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print("The specified executable was not found.")
        return False
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        return False
    return True

@dataclass
class Dependency(ABC):
    updatable: bool
    download_path:str
    installed_path:str

    def __init__(self):
        FileHandler.createDirectory(os.path.dirname(self.download_path))

    @abstractmethod
    def get_download_link(self) -> str: ...
        
    @abstractmethod
    def download(self) -> None: ...

    def get_if_update_available(self) -> bool: ...
    def update_if_updates_available(self) -> None: ...

class Backend(Dependency):
    updatable: bool = True
    is_update_available: bool = False
    download_path = os.path.join(CWD, "backend.tar.gz")
    installed_path = BACKEND_PATH


    def get_download_link(self) -> str:
        backend_url = f"https://github.com/TNTwise/REAL-Video-Enhancer/releases/download/RVE-{version}/backend-v{version}.tar.gz"
        return backend_url
    
    def download(self):
        if USE_LOCAL_BACKEND:
            return
        needs_network_else_exit()
        download_link = self.get_download_link()
        DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading Backend")
        extractTarGZ(self.download_path)
    
    def get_if_update_available(self) -> bool:
        try:
            process = subprocess_popen_without_terminal(
                [PYTHON_EXECUTABLE_PATH, os.path.join(BACKEND_PATH, "rve-backend.py"), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
                )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
            output = stdout.strip() # this extracts the version number from the output
            log(f"\nBackend Version: {output}\n")
            update_available = not output == backend_dev_version
            self.is_update_available = update_available
            return update_available
        except subprocess.CalledProcessError as e: # if the backend is not found
            log("Backend not found, downloading..." + str(e))
            self.download()
            self.is_update_available = False
            return False
    
    def update_if_updates_available(self) -> None:
        if self.is_update_available:
            needs_network_else_exit()
            FileHandler.removeFolder(BACKEND_PATH) # remove the old backend directory
            self.download()


class Python(Dependency):
    download_path = os.path.join(CWD, "python", "python.tar.gz")
    installed_path = PYTHON_DIRECTORY
    is_update_available: bool

    def get_download_link(self) -> str:
        link = f"https://github.com/TNTwise/REAL-Video-Enhancer-models/releases/download/models/cpython-{PYTHON_VERSION}+20250317-"
       
        match PLATFORM:
            case "linux":
                link += "x86_64-unknown-linux-gnu-install_only.tar.gz" if CPU_ARCH == "x86_64" else "aarch64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link += "x86_64-pc-windows-msvc-install_only.tar.gz"
            case "darwin":
                link += "x86_64-apple-darwin-install_only.tar.gz" if CPU_ARCH == "x86_64" else "aarch64-apple-darwin-install_only.tar.gz"

        return link

    def download(self):
        needs_network_else_exit()
        download_link = self.get_download_link()
        FileHandler.createDirectory(os.path.dirname(self.download_path))
        DownloadProgressPopup(link = download_link, downloadLocation=self.download_path, title = f"Downloading Python {PYTHON_VERSION}")
        extractTarGZ(self.download_path)
    
    def get_version(self):
        return subprocess.run([PYTHON_EXECUTABLE_PATH, "--version"], check=True, capture_output=True, text=True).stdout.strip().split(" ")[1] # this extracts the version number from the output
    
    def get_if_update_available(self) -> bool:
        try:
            output = self.get_version()
        except subprocess.CalledProcessError: # if python is not found
            self.download()
            return False
        
        is_update = not output == PYTHON_VERSION

        if is_update:
            reply = QMessageBox.question(
                None,
                "Update Python?",
                "The installed version of Python is older than the current version of RVE. Update?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,  # type: ignore
            )
            if reply == QMessageBox.Yes:  # type: ignore
                self.is_update_available = True
                print("Updating Python")
            else:
                is_update = False
        self.is_update_available = is_update
        return self.is_update_available

    def update_if_updates_available(self) -> None:

        if self.is_update_available:
            removeFolder(PYTHON_DIRECTORY)
            self.download()

class FFMpeg(Dependency):
    download_path = os.path.join(CWD, "ffmpeg")
    installed_path = FFMPEG_PATH

    def get_download_link(self) -> str:
        link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
        match PLATFORM:
            case "linux":
                link += "ffmpeg" if CPU_ARCH == "x86_64" else "ffmpeg-linux-arm64"
            case "win32":
                link += "ffmpeg.exe" if CPU_ARCH == "x86_64" else "ffmpeg-windows-arm64.exe"
            case "darwin":
                link += "ffmpeg-macos-bin" if CPU_ARCH == "x86_64" else "ffmpeg-macos-arm"
        return link

    def download(self):
        
        needs_network_else_exit()

        download_link = self.get_download_link()
        DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading FFMpeg")
        FileHandler.createDirectory(os.path.dirname(self.installed_path))
        FileHandler.moveFile(self.download_path, self.installed_path)
        FileHandler.makeExecutable(self.installed_path)

class VCRedList(Dependency):
    updatable = False
    download_path = os.path.join(CWD, "bin", "VC_redist.x64.exe")
    installed_path = download_path

    def get_download_link(self) -> str:
        return "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    
    def download(self):
        if PLATFORM == 'win32':
            needs_network_else_exit()

            download_link = self.get_download_link()
            DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading VCRedist")
            
            # Use ShellExecute to properly handle admin elevation
            import ctypes
            try:
                result = ctypes.windll.shell32.ShellExecuteW(
                    None,                          # hwnd
                    "runas",                       # operation (runas = run as admin)
                    self.download_path,            # file
                    "/install /norestart /quiet",         # parameters
                    None,                          # directory
                    1                              # show command (1 = normal window)
                )
                if result <= 32:  # Error codes are <= 32
                    RegularQTPopup(
                        "Failed to launch VCRedist installer. Please run it manually."
                    )
            except Exception as e:
                RegularQTPopup(
                    f"Error launching VCRedist installer: {str(e)}\nThe installer will now close."
                )




class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """
    def __init__(self, use_torch_nightly:bool = False):
        self.use_torch_nightly = use_torch_nightly
    def download_all_deps(self):
        for dep in Dependency.__subclasses__():
            d = dep()
            d.download()

    def get_torch_versions(self) -> dict:
        """Return the installed torch/torchvision versions (local suffix
        included), or None for packages that are not installed."""
        versions = {"torch": None, "torchvision": None}
        try:
            result = subprocess.run(
                [PYTHON_EXECUTABLE_PATH, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            for line in result.stdout.splitlines():
                match = re.match(r"^(torchvision|torch)==(.+)$", line.strip())
                if match:
                    versions[match.group(1)] = match.group(2)
        except Exception as e:
            log(f"Could not query installed torch versions: {e}")
        return versions

    def check_torch_consistency(self, settings) -> bool:
        """Detect a torch/torchvision mismatch left behind by an older RVE
        release (e.g. an old torch with a newly installed torchvision) and
        offer to reinstall a matching pair. Returns True if a reinstall was
        performed."""
        from .BuiltInTorchVersions import TorchVersion

        installed = self.get_torch_versions()
        torch_ver = installed["torch"]
        torchvision_ver = installed["torchvision"]
        if torch_ver is None and torchvision_ver is None:
            return False  # nothing installed yet, nothing to check

        def base(v: Optional[str]) -> str:
            return v.split("+")[0] if v else ""

        # Valid (torch, torchvision) pairs as defined by the built-in matrix.
        # Nightly versions carry a date stamp (e.g. 2.15.0.dev20260925), so
        # compare only the version prefix.
        valid_pairs = [
            (v.torch_version, v.torchvision_version)
            for v in TorchVersion.__subclasses__()
        ]
        mismatch = not any(
            base(torch_ver).startswith(base(t))
            and base(torchvision_ver).startswith(base(tv))
            for t, tv in valid_pairs
        )
        if not mismatch:
            return False

        log(f"torch/torchvision mismatch detected: torch={torch_ver}, torchvision={torchvision_ver}")
        reply = QMessageBox.question(
            None,
            "Outdated PyTorch detected",
            "The installed torch and torchvision versions do not match:\n"
            f"torch: {torch_ver}\n"
            f"torchvision: {torchvision_ver}\n"
            "\nReinstall a matching version pair?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        self._reinstall_torch(settings)
        return True

    def _resolve_torch_version(self, settings):
        """The version class selected in the settings, falling back to the
        latest stable built-in version."""
        from .BuiltInTorchVersions import TorchVersion

        requested_version = settings.settings.get("pytorch_version", "")
        for v in TorchVersion.__subclasses__():
            if v.torch_version == requested_version:
                return v
        stable = [v for v in TorchVersion.__subclasses__() if not v.is_nightly]
        return max(stable or TorchVersion.__subclasses__(), key=lambda v: v.torch_version)

    def _reinstall_torch(self, settings, backend_family: str = None) -> None:
        """Reinstall the selected torch/torchvision pair for the given
        backend family (cuda/rocm/xpu/mps), defaulting to the one selected
        in the settings."""
        backend_family = (
            backend_family
            or self.backend_family_from_setting(
                settings.settings.get("pytorch_backend", "CUDA")
            )
        )
        version_class = self._resolve_torch_version(settings)
        if backend_family == "rocm":
            backend_suffix = version_class.rocm_version
        elif backend_family == "xpu":
            backend_suffix = version_class.xpu_version
        elif backend_family == "mps":
            backend_suffix = version_class.mps_version
        else:
            backend_suffix = version_class.cuda_version
        self.downloadPythonDeps(
            "torch",
            version_class.torch_version,
            version_class.torchvision_version,
            backend_suffix.lower(),
            True,
            version_class.is_nightly,
        )
        if backend_suffix.lower().lstrip("+").startswith("rocm"):
            self.ensure_rocm_targets(backend_suffix.lower(), is_nightly=version_class.is_nightly)

    def check_torch_hardware(self, settings) -> bool:
        """Detect a change of GPU hardware since the installed torch build
        was made (e.g. a ROCm install on a machine that now has an Nvidia
        GPU) and offer to reinstall torch for the new hardware. Returns
        True if a reinstall was performed."""
        installed = self.get_torch_versions()
        torch_ver = installed["torch"]
        if torch_ver is None:
            return False  # nothing installed, the backend setup flow handles it

        # The local version tag (everything after "+") identifies the
        # backend family of the installed build (e.g. 2.14.0+rocm7.14 ->
        # rocm, 2.14.0+cu132 -> cuda).
        local_tag = ""
        if "+" in torch_ver:
            local_tag = torch_ver.split("+", 1)[1].lower()
        installed_family = None
        if local_tag.startswith("rocm"):
            installed_family = "rocm"
        elif local_tag.startswith("cu") and not local_tag.startswith("cpu"):
            installed_family = "cuda"
        elif local_tag.startswith("xpu"):
            installed_family = "xpu"
        elif local_tag.startswith("mps"):
            installed_family = "mps"
        if installed_family is None:
            return False  # generic/CPU build, nothing hardware-specific to check

        hardware = detect_gpu_hardware()
        current_family = self.backend_family_from_setting(
            settings.settings.get("pytorch_backend", "")
        )
        # Only offer a switch when the installed build's hardware is not
        # present anymore (a machine with several vendors can
        # legitimately use any of them). When nothing could be detected
        # there is nothing to compare against, so stay silent.
        installed_hardware = FAMILY_HARDWARE.get(installed_family)
        if not hardware or installed_hardware in hardware:
            return False
        log(
            f"GPU hardware changed: torch={torch_ver} installed for "
            f"'{installed_family}', hardware present: {hardware}"
        )
        reply = QMessageBox.question(
            None,
            "GPU change detected",
            "The installed PyTorch build does not match the hardware "
            f"currently present in this system.\n"
            f"Installed torch: {torch_ver}\n"
            f"Detected hardware: {', '.join(hardware) if hardware else 'unknown'}\n"
            "\nReinstall PyTorch for the current hardware?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        # Switch to the backend matching the hardware; when several
        # vendors are present, keep the one selected in the settings.
        target = current_family
        if len(hardware) == 1:
            target = HARDWARE_FAMILIES.get(hardware[0], target)
        settings.writeSetting("pytorch_backend", self.backend_setting_from_family(target))
        self._reinstall_torch(settings, backend_family=target)
        return True

    @staticmethod
    def backend_setting_from_family(family: str) -> str:
        """The combo box text used in the settings for a torch backend
        family (cuda/rocm/xpu/mps)."""
        return {
            "cuda": "CUDA",
            "rocm": "ROCm",
            "xpu": "xpu",
            "mps": "MPS (Apple Silicon)",
        }.get(family, "CUDA")

    def backend_family_from_setting(self, backend_text: str) -> str:
        """Map the backend selected in the settings (combo box text) to a
        torch backend family (cuda/rocm/xpu/mps)."""
        text = backend_text.lower()
        if "mps" in text or "apple" in text:
            return "mps"
        if "rocm" in text or "amd" in text:
            return "rocm"
        if "xpu" in text or "intel" in text:
            return "xpu"
        return "cuda"

    def detect_rocm_targets(self) -> list:
        """Detect the gfx targets of the AMD GPUs in this system.

        Sources, in order:
          1. The AMDGPU_TARGETS environment variable, which distro ROCm
             packages set to the targets of the installed GPUs.
          2. rocm-smi / rocminfo output, when a ROCm userspace is present.
        Returns [] when no target could be determined, in which case the
        caller falls back to the full device set."""
        targets = []
        env_targets = os.environ.get("AMDGPU_TARGETS", "")
        for target in re.findall(r"gfx\d+", env_targets):
            if target not in targets:
                targets.append(target)
        if targets:
            log(f"Detected ROCm targets from AMDGPU_TARGETS: {targets}")
            return targets
        if PLATFORM == "linux":
            for tool in (["rocm-smi", "--showproductname"], ["rocminfo"]):
                try:
                    output = subprocess.run(
                        tool, capture_output=True, text=True, timeout=30
                    ).stdout
                    for target in re.findall(r"gfx\d+", output):
                        if target not in targets:
                            targets.append(target)
                    if targets:
                        log(f"Detected ROCm targets from {tool[0]}: {targets}")
                        return targets
                except Exception:
                    continue
        return targets

    def get_torch_runtime_deps(self) -> list:
        """Read torch's runtime requirements from its installed metadata,
        skipping the `rocm` meta package (which is installed separately,
        restricted to the targets of the GPUs actually present) and any
        optional extras."""
        from importlib.metadata import PackageNotFoundError, requires
        try:
            pkg_requires = requires("torch") or []
        except PackageNotFoundError:
            return []
        deps = []
        for req in pkg_requires:
            base = req.split(";", 1)[0].strip()
            marker = req[len(base):]
            if "extra" in marker:
                continue
            if re.match(r"rocm(\[.*\])?", base, re.IGNORECASE):
                continue
            if base and base not in deps:
                deps.append(base)
        return deps

    def install_rocm_meta(self, rocm_version: str, targets: list, is_nightly: bool = False) -> int:
        """Install the `rocm` meta package with only the device extras needed
        for the detected GPUs (instead of the device-all set, which ships
        SDKs for every architecture)."""
        extras = ["libraries"]
        for target in targets:
            extras.append(target if target.startswith("device-") else f"device-{target}")
        dep = f"rocm[{','.join(extras)}]"
        if rocm_version:
            dep += f"=={rocm_version}.*"
        return self.pip([dep], True, is_nightly=is_nightly)

    def ensure_rocm_targets(self, torch_backend: str, is_nightly: bool = False) -> None:
        """Make sure device packages are installed for every AMD GPU present.
        This also self-heals machines that got a new/second GPU after the
        initial install."""
        rocm_version = torch_backend.lstrip("+")[4:]
        targets = self.detect_rocm_targets()
        if not targets:
            log("Could not detect ROCm targets, falling back to the full device set")
            return self.install_rocm_meta(rocm_version, ["device-all"], is_nightly=is_nightly)
        installed = set()
        try:
            result = subprocess.run(
                [PYTHON_EXECUTABLE_PATH, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            installed = set(re.findall(r"^(rocm-sdk-device-\w+)==", result.stdout, re.MULTILINE))
        except Exception as e:
            log(f"Could not query installed ROCm device packages: {e}")
        missing = [target for target in targets if f"rocm-sdk-device-{target}" not in installed]
        if not missing:
            log(f"ROCm device packages for {targets} already installed")
            return
        log(f"Installing ROCm device packages for: {missing}")
        extras = ",".join(f"device-{target}" for target in missing)
        return self.pip([f"rocm[{extras}]=={rocm_version}.*"], True, is_nightly=is_nightly)

    def pip(
        self,
        deps: list,
        install: bool = True,
        is_nightly: bool = False,
    ):  # going to have to make this into a qt module pop up
        command = []
        if PLATFORM == "linux" and IS_STEAM:
            # Define the list of environment variables known to conflict with Steam/Linux Runtime
            conflict_vars = [
                'LD_LIBRARY_PATH',
                'STEAM_RUNTIME',
                'SYSTEM_LD_LIBRARY_PATH',
                'PRESSURE_VESSEL_RUNTIME',
                'PRESSURE_VESSEL_RUNTIME_BASE',
                'CUDA_PATH', # Might conflict with GPU-aware packages
            ]

            log("Cleaning up conflicting Steam environment variables...")

            for var in conflict_vars:
                if var in os.environ:
                    log(f"Unsetting {var}: {os.environ[var]}")
                    del os.environ[var] # This removes the variable for the current process
        command += [
            PYTHON_EXECUTABLE_PATH,
            "-m",
            "pip",
            "install" if install else "uninstall",
        ]
        origTemp = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = TEMP_DOWNLOAD_PATH
        if install:
            command += [
                "--no-warn-script-location",
                "--isolated",
            ]
            if is_nightly:
                command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/nightly/",
                ]
            else:
                command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/test/", 
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/", # search this first, needs to be last in the list 
                ]
            # ROCm wheels depend on a version-specific `rocm` meta package
            # (e.g. rocm==7.14.*) that only exists under the version-specific
            # index (https://download.pytorch.org/whl/rocm7.14/), not the flat
            # /whl/ index. Derive the version from the dep strings (either a
            # torch==...+rocm<ver> wheel or a rocm[...]==<ver> pin) and add
            # that index so pip can resolve the meta package.
            rocm_match = re.search(
                r"(?:torch==\S*\+rocm|rocm(?:\[[^\]]*\])?==)(\d[\d.]*)([+\s=.]|\$)",
                " ".join(deps),
            )
            if rocm_match:
                command += [
                    "--extra-index-url",
                    f"https://download.pytorch.org/whl/rocm{rocm_match.group(1)}/",
                ]
            command += [
                "--trusted-host",
                "download.pytorch.org",
            ]
        else:
            command += ["-y"]
        command += deps
        # totalDeps = self.get_total_dependencies(deps)
        totalDeps = len(deps)
        log("Downloading Deps: " + str(command))
        log("Total Dependencies: " + str(totalDeps))

        d = DisplayCommandOutputPopup(
            command=command,
            title="Download Dependencies",
            progressBarLength=totalDeps,
        )

        return_code = d.get_return_code()

        command = [
            PYTHON_EXECUTABLE_PATH,
            "-m",
            "pip",
            "cache",
            "purge",
        ]
        DisplayCommandOutputPopup(
            command=command,
            title="Purging Cache",
            progressBarLength=1,
        )
        if origTemp:
            os.environ["TMPDIR"] = str(origTemp)
        return return_code

    def _resolve_nightly_version(self, package: str, base_version: str, local_suffix: str) -> str:
        """
        Query the nightly index and return the latest dev base version of
        `package` that matches `base_version` and ships with `local_suffix`
        (e.g. "+cu134"). Returns the version WITHOUT the local suffix, e.g.
        "2.15.0.dev20260925" — the caller appends the suffix itself.
        Falls back to `base_version` if the index can't be reached.
        """
        import requests
        url = f"https://download.pytorch.org/whl/nightly/{package}/"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            # Wheel filenames look like: torch-2.15.0.dev20260925%2Bcu134-cp312-...whl
            # The '+' is URL-encoded as '%2B'. Match the base version, capture the
            # date, and require our local suffix.
            suffix = local_suffix.lstrip("+")
            # base_version already includes the ".dev" prefix (e.g. "2.15.0.dev"),
            # so we only need to capture the date that follows it.
            pattern = re.compile(
                rf"{re.escape(package)}-{re.escape(base_version)}(\d+)%2B{re.escape(suffix)}-cp"
            )
            best_date = None
            for m in pattern.finditer(resp.text):
                date = m.group(1)
                if best_date is None or date > best_date:
                    best_date = date
            if best_date:
                # base_version already ends in ".dev" (e.g. "2.15.0.dev"), so the
                # resolved form is just base + date: "2.15.0.dev20260925".
                resolved = f"{base_version}{best_date}"
                log(f"Resolved nightly {package} {base_version} -> {resolved}{local_suffix}")
                return resolved
        except Exception as e:
            log(f"Could not resolve nightly version for {package}: {e}")
        # Fallback: use the base version (pip will error if no exact match)
        return base_version

    def getPlatformIndependentDeps(self):
        platformIndependentdeps = [
            "testresources==2.0.1",
            "requests==2.32.3",
            "opencv-python-headless==4.11.0.86",
            "pypresence==4.3.0",
            "scenedetect==0.6.5.2",
            "numpy==2.2.2",
            "sympy",
            "tqdm==4.67.1",
            "typing_extensions==4.12.2",
            "packaging==24.2",
            "mpmath==1.3.0",
            "pillow==11.1.0",
        ]
        return platformIndependentdeps
    
    def downloadPythonDeps(self, backend, torch_version: Optional[str] = "2.7.0", torchvision_version: Optional[str] = "0.22.0", torch_backend: Optional[str] = "cu126", install: bool = True, is_nightly: bool = False):
        deps = []
        log("Downloading Python Deps for " + backend)
        log("Torch Version: " + torch_version)
        log("Torch Backend: " + torch_backend)
        log("Torchvision Version: " + torchvision_version)
        

        if install: # dont uninstall platform independent deps
            deps = self.getPlatformIndependentDeps()
            
        return_codes = []
        match backend:
            case "ncnn":
                deps += [
                    "rife-ncnn-vulkan-python-tntwise==1.4.5",
                    "upscale_ncnn_py==1.3.0",
                    "ncnn==1.0.20260526",
                    "numpy==2.2.2",
                ]
                return_code = self.pip(deps, install)
                return_codes.append(return_code)
            case "torch" | "tensorrt":
                # Nightly wheels carry a date stamp (e.g. 2.15.0.dev20260925) that
                # can't be expressed with a plain == pin, so resolve the latest
                # dev build from the nightly index at install time.
                if is_nightly:
                    torch_version = self._resolve_nightly_version("torch", torch_version, torch_backend)
                    torchvision_version = self._resolve_nightly_version("torchvision", torchvision_version, torch_backend)

                # The ROCm torch wheels depend on the `rocm` meta package with
                # the device-all extra, which ships SDKs for every GPU
                # architecture. Install the meta package first, restricted to
                # the targets of the GPUs actually present.
                is_rocm = torch_backend.lstrip("+").startswith("rocm")
                if install and is_rocm:
                    rocm_version = torch_backend.lstrip("+")[4:]
                    targets = self.detect_rocm_targets()
                    if not targets:
                        log("Could not detect ROCm targets, falling back to the full device set")
                        targets = ["device-all"]
                    log(f"Installing ROCm meta package with targets: {targets}")
                    return_code = self.install_rocm_meta(rocm_version, targets, is_nightly=is_nightly)
                    return_codes.append(return_code)

                deps += [
                    f"torch=={torch_version}{torch_backend}",  #
                    "safetensors==0.5.3",
                    "einops==0.8.1",
                ]
                deps += ["cupy-cuda12x==13.3.0"] if "cu" in backend else []
                # For ROCm, install torch with --no-deps: its declared
                # `rocm[device-all,libraries]` requirement would make pip
                # pull SDKs for every architecture despite the restricted
                # meta package being installed. Its other runtime deps are
                # installed separately from the wheel's own metadata.
                if install and is_rocm:
                    deps = ["--no-deps"] + deps
                return_code = self.pip(deps, install, is_nightly=is_nightly)
                return_codes.append(return_code)

                if install and is_rocm:
                    torch_deps = self.get_torch_runtime_deps()
                    log(f"Installing torch runtime dependencies: {torch_deps}")
                    if torch_deps:
                        return_code = self.pip(torch_deps, install)
                        return_codes.append(return_code)
                
                if install:
                    deps = [
                        "--no-deps",
                        f"torchvision=={torchvision_version}{torch_backend}",
                    ]
                    return_code = self.pip(deps, install, is_nightly=is_nightly)

                return_codes.append(return_code)

                if backend == "tensorrt":
                    # torch-tensorrt 2.14.0 (the latest release) pins
                    # tensorrt>=11.1.0,<11.2.0, so 11.1.0.106 is the newest
                    # compatible version.
                    trt_ver = "11.1.0.106"
                    deps = [
                        f"tensorrt=={trt_ver}",
                        f"tensorrt_cu12=={trt_ver}",
                        f"tensorrt-cu12_libs=={trt_ver}",
                        f"tensorrt_cu12_bindings=={trt_ver}",
                        
                    ]
                    if install:
                        
                        torch_version = torch_version[:-1] + "0" # remove the last character (2.7.1 -> 2.7.0), torch tensorrt doesnt release a new version for every new pytorch minor release
                        deps += ["--no-deps","dllist",f"torch-tensorrt=={torch_version}{torch_backend}"]

                    return_code = self.pip(deps, install)
                    return_codes.append(return_code)
        
        for return_code in return_codes:
            if return_code != 0:
                return return_code
        return 0