from abc import  abstractmethod
import os
import subprocess
import sys
import os
import subprocess
import sys
import shutil
import argparse
import platform

PLATFORM = sys.platform
CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()
OUTPUT_FOLDER = os.path.join("apps", "gui", "dist")
print(f"Platform: {PLATFORM}")
print(f"CPU Arch: {CPU_ARCH}")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")


def get_libxcb_cursor_binary():
    # The PySide6 wheel does not bundle libxcb-cursor, but Qt's xcb
    # platform plugin needs it at runtime, so we vendor a copy per arch
    # and copy it into the Qt lib directory at build time.
    if CPU_ARCH == "x86_64":
        input_file = os.path.join("resources", "libxcb-cursor-x86_64.so.0")
    else:
        input_file = os.path.join("resources", "libxcb-cursor-arm64.so.0")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"libxcb-cursor binary not found at {input_file}!")
    return input_file
    
class PythonManager:

    PYTHON_VENV_PATH = "venv\\Scripts\\python.exe" if PLATFORM == "win32" else "venv/bin/python3"
    VENV_DIR = "venv"
    # The venv is built with uv, which can fetch a standalone CPython
    # automatically. Pin the version so local and CI builds match.
    PYTHON_VERSION = "3.14"

    def __init__(self):
        if not os.path.exists(self.VENV_DIR):
            self.setup_python()
    
    @classmethod
    def run_venv_python(cls, command: str):
        command = [cls.PYTHON_VENV_PATH,] + command.split()
        subprocess.run(command)
    
    @classmethod
    def pip_install_package_in_venv(cls, package: str):
        command = [
            "uv",
            "pip",
            "install",
            "--python",
            cls.PYTHON_VENV_PATH,
            package,
        ]
        subprocess.run(command)

    def setup_python(self):
        self.__create_venv()
        self.__install_requirements_in_venv()

    def __create_venv(self):
        print("Creating virtual environment with uv")
        command = [
            "uv",
            "venv",
            self.VENV_DIR,
            "--python",
            self.PYTHON_VERSION,
        ]
        subprocess.run(command)

    def __install_requirements_in_venv(self):
        print("Installing requirements in virtual environment")
        # Read dependencies from pyproject.toml to avoid duplication
        import tomllib
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        deps = pyproject.get("project", {}).get("dependencies", [])
        # Also add build tools
        deps.extend(["cx-freeze==8.7.1"])
        command = [
            "uv",
            "pip",
            "install",
            "--python",
            self.PYTHON_VENV_PATH,
        ] + deps
        subprocess.run(command)
        

    def get_venv_site_packages(self):
        command = [
            self.PYTHON_VENV_PATH,
            "-c",
            'import site; print("\\n".join(site.getsitepackages()))',
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        site_packages = result.stdout.strip().split('\n')[0]
        if os.path.exists(site_packages):
            return site_packages
        site_packages = site_packages.replace('dist','site')
        if os.path.exists(site_packages):
            return site_packages
        print(site_packages)
        raise FileNotFoundError("Unable to locate site packages for python venv!")
    


class BuildManager:
    def __init__(self):
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        self.python_manager = PythonManager()

    @abstractmethod
    def build(self):
        ...
    
    # (source .ui file, generated module name inside dist/pages/)
    PAGE_UIS = [
        ("apps/gui/ui/mainwindow.ui", "_mainwindow"),
        ("apps/gui/pages/home/home.ui", "home"),
        ("apps/gui/ui/more_page.ui", "more"),
        ("apps/gui/pages/process/process.ui", "process"),
        ("apps/gui/pages/settings/settings.ui", "settings"),
        ("apps/gui/pages/download/download.ui", "download"),
    ]

    def _uic(self, ui_source: str) -> str:
        if PLATFORM == "win32":
            uic = r".\venv\Lib\site-packages\PySide6\uic.exe"
        else:
            uic = f"{self.python_manager.get_venv_site_packages()}/PySide6/Qt/libexec/uic"
        return f"{uic} -g python {ui_source}"

    def build_gui(self):
        print("Building GUI")
        pages_dir = os.path.join(OUTPUT_FOLDER, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        with open(os.path.join(pages_dir, "__init__.py"), "w"):
            pass
        for ui_source, name in self.PAGE_UIS:
            out = (
                f"{OUTPUT_FOLDER}/mainwindow.py"
                if name == "_mainwindow"
                else os.path.join(pages_dir, f"{name}.py")
            )
            command = self._uic(ui_source)
            print(f"  uic {ui_source} -> {out}")
            result = os.system(f"{command} > {out}")
            if result != 0:
                raise RuntimeError(f"uic failed for {ui_source}: {result}")
            self._inject_widget_imports(out)

    #: Custom (promoted) widgets referenced by the ``.ui`` files. PySide6's
    #: ``uic`` does not auto-import promoted widgets into the generated module,
    #: so after each compile we add the import if the widget is actually used.
    WIDGET_IMPORTS = {
        "HelpIconLabel": "from apps.gui.widgets import HelpIconLabel",
        "QTabNavigation": "from apps.gui.widgets import QTabNavigation",
    }

    @staticmethod
    def _inject_widget_imports(path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        lines = text.splitlines(keepends=True)
        needed = [
            name for name in BuildManager.WIDGET_IMPORTS if f" {name}(" in text
        ]
        if not needed:
            return
        # imports always precede the generated ``class Ui_...`` definition, so
        # inserting right before it places us after the whole import block.
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("class "):
                insert_at = i
                break
        imports_needed = [BuildManager.WIDGET_IMPORTS[n] for n in needed]
        for stmt in reversed(imports_needed):
            if stmt in text:
                continue
            lines.insert(insert_at, stmt + "\n")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(lines))

    def build_resources(self):
        print("Building resources.rc")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.python_manager.get_venv_site_packages()}/PySide6/Qt/libexec/rcc -g python apps/gui/ui/resources.qrc > {OUTPUT_FOLDER}/resources_rc.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\rcc.exe -g python apps\gui\ui\resources.qrc > {OUTPUT_FOLDER}/resources_rc.py"
            )

    
    def copy_backend(self):
        print("Copying backend")
        if "pyinstaller" in self.__str__().lower():
            backend_dir = os.path.join(f"{OUTPUT_FOLDER}/REAL-Video-Enhancer/backend")
            try:
                shutil.copytree("apps/backend", backend_dir)
            except Exception:
                raise FileNotFoundError("Backend failed to copy!")
            if not os.path.exists(backend_dir):
                raise FileNotFoundError("Backend failed to copy!")
        else:
            shutil.copytree("apps/backend", f"{OUTPUT_FOLDER}/backend")

    @abstractmethod
    def patch_for_xcbcursor(self):
        ...

class PyInstaller(BuildManager):
    pyinstaller_version = "pyinstaller==6.16.0"

  
    def build(self):
        print("Building executable")

        PythonManager.pip_install_package_in_venv(self.pyinstaller_version)
        PythonManager.run_venv_python(
            (
              "-m PyInstaller" 
            + " apps/gui/REAL-Video-Enhancer.py" 
            + " --icon=icons/logo-v2.ico" 
            + " --noconfirm"
            + " --noupx" 
            + " --noconsole" # i think this fixes weird macos dir shit
            + " --distpath"
            + f" {OUTPUT_FOLDER}"
            )
        )

    def patch_for_xcbcursor(self):
        if PLATFORM == "linux":
            input_file = get_libxcb_cursor_binary()
            print("Copying libcursor to qt lib directory")
            shutil.copy(input_file, f"{OUTPUT_FOLDER}/REAL-Video-Enhancer/_internal/PySide6/Qt/lib/")
            
class CxFreeze(BuildManager):

    # 8.x is pure Python, so it works on Python 3.14 without building from source.
    cx_freeze_version = "cx_freeze==8.7.1"

    def build(self):
        print("Building executable")

        PythonManager.pip_install_package_in_venv(self.cx_freeze_version)
        PythonManager.run_venv_python(
            (
              " -m"
            + " cx_Freeze"
            + " --script apps/gui/REAL-Video-Enhancer.py"
            + " --target-dir"
            + f" {OUTPUT_FOLDER}"
            + " build_exe"
            )
        )

    def patch_for_xcbcursor(self):
        if PLATFORM == "linux":
            input_file = get_libxcb_cursor_binary()
            qt_lib_dir = f"{OUTPUT_FOLDER}/lib/PySide6/Qt/lib"
            os.makedirs(qt_lib_dir, exist_ok=True)
            print("Copying libcursor to qt lib directory")
            # The loader expects the plain name, not the vendored per-arch name.
            shutil.copy(input_file, os.path.join(qt_lib_dir, "libxcb-cursor.so.0"))


def build_appimage(args):
    print("Packaging AppImage")
    appimage_script = os.path.join("installers", "build-appimage.sh")
    command = [
        appimage_script,
        f"{OUTPUT_FOLDER}",
        args.appimage_version,
        args.appimage_arch,
    ]
    subprocess.run(command, check=True)


def cleanup():
    """Remove build byproducts that are not part of the final output."""
    print("Cleaning up build artifacts")
    # cx_Freeze 8.x (freeze-core) drops an UNKNOWN.egg-info dir somewhere in
    # the tree when freezing a standalone script.
    for root, dirs, _files in os.walk("."):
        rel = root.split(os.sep)
        if OUTPUT_FOLDER in rel or "venv" in rel:
            continue
        for d in [d for d in dirs if d.endswith(".egg-info")]:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        dirs[:] = [d for d in dirs if not d.endswith(".egg-info")]
        # Left over from AppImage extraction (e.g. --appimage-extract).
        if root == "." and "squashfs-root" in dirs:
            shutil.rmtree("squashfs-root", ignore_errors=True)
        # Python bytecode caches.
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--run", help="Run the application", action="store_true")
    args.add_argument("--run_backend", help="Run the backend", action="store_true")
    args.add_argument("--build", help="Build the application with a specific builder.", default="gui", choices=["pyinstaller", "cx_freeze", "appimage", "gui"])
    args.add_argument("--appimage_version", help="Version for the AppImage build.", default="2.4.2")
    args.add_argument("--appimage_arch", help="Architecture for the AppImage build.", default="x86_64")
    args.add_argument("--copy_backend", help="Copy the backend to the build directory", action="store_true")    
    args = args.parse_args()

    # NOTE: a single BuildManager instance is reused below — each constructor run
    # wipes OUTPUT_FOLDER (dist/), so separate instances would delete previously
    # generated artifacts (e.g. mainwindow.py) before writing resources_rc.py.
    build_manager = BuildManager()

    if not os.path.exists("venv") or not args.build == "gui":
        build_manager.python_manager.setup_python()
    build_manager.build_gui()
    if args.run:
        PythonManager.run_venv_python("apps/gui/REAL-Video-Enhancer.py")
    elif args.run_backend:
        PythonManager.run_venv_python("apps/backend/rve-backend.py")
    else:
        build_manager.build_resources()
        
        
        match args.build:
            case "pyinstaller":
                builder = PyInstaller()
            case "cx_freeze":
                builder = CxFreeze()
            case "appimage":
                # AppImages are built on top of the cx_Freeze output.
                builder = CxFreeze()
            case "gui":
                exit()
            case _:
                raise ValueError("Invalid build option")
        builder.build()
        builder.patch_for_xcbcursor()
        if args.copy_backend or args.build == "appimage":
            builder.copy_backend()
        if args.build == "appimage":
            build_appimage(args)
        print("Build complete")
        cleanup()
    
    
