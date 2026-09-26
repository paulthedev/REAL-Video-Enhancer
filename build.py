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
OUTPUT_FOLDER = "dist"
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
        if not os.path.isfile("requirements.txt"):
            raise FileNotFoundError("No requirements.txt in current directory!")
        command = [
            "uv",
            "pip",
            "install",
            "--python",
            self.PYTHON_VENV_PATH,
            "-r",
            "requirements.txt",
        ]
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
    
    def build_gui(self):
        print("Building GUI")
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.python_manager.get_venv_site_packages()}/PySide6/Qt/libexec/uic -g python testRVEInterface.ui > mainwindow.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py"
            )
    
    def build_resources(self):
        print("Building resources.rc")
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.python_manager.get_venv_site_packages()}/PySide6/Qt/libexec/rcc -g python resources.qrc > resources_rc.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py"
            )

    
    def copy_backend(self):
        print("Copying backend")
        if "pyinstaller" in self.__str__().lower():
            backend_dir = os.path.join(f"{OUTPUT_FOLDER}/REAL-Video-Enhancer/backend")
            try:
                shutil.copytree("backend",backend_dir)
            except Exception:
                raise FileNotFoundError("Backend failed to copy!")
            if not os.path.exists(backend_dir):
                raise FileNotFoundError("Backend failed to copy!")
        else:
            shutil.copytree("backend", f"{OUTPUT_FOLDER}/backend")

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
            + " REAL-Video-Enhancer.py" 
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
            + " REAL-Video-Enhancer.py"
            + " --target-dir"
            + f" {OUTPUT_FOLDER}"
            )
        )

    def patch_for_xcbcursor(self):
        if PLATFORM == "linux":
            input_file = get_libxcb_cursor_binary()
            qt_lib_dir = f"{OUTPUT_FOLDER}/lib/PySide6/Qt/lib"
            os.makedirs(qt_lib_dir, exist_ok=True)
            print("Copying libcursor to qt lib directory")
            shutil.copy(input_file, qt_lib_dir)


def build_appimage(args):
    print("Packaging AppImage")
    appimage_script = os.path.join("appimage", "build-appimage.sh")
    command = [
        appimage_script,
        f"{OUTPUT_FOLDER}",
        args.appimage_version,
        args.appimage_arch,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--run", help="Run the application", action="store_true")
    args.add_argument("--run_backend", help="Run the backend", action="store_true")
    args.add_argument("--build", help="Build the application with a specific builder.", default="gui", choices=["pyinstaller", "cx_freeze", "appimage", "gui"])
    args.add_argument("--appimage_version", help="Version for the AppImage build.", default="2.4.2")
    args.add_argument("--appimage_arch", help="Architecture for the AppImage build.", default="x86_64")
    args.add_argument("--copy_backend", help="Copy the backend to the build directory", action="store_true")    
    args = args.parse_args()
    if not os.path.exists("venv") or not args.build == "gui":
        BuildManager().python_manager.setup_python()
    BuildManager().build_gui()
    if args.run:
        PythonManager.run_venv_python("REAL-Video-Enhancer.py")
    elif args.run_backend:
        PythonManager.run_venv_python("backend/rve-backend.py")
    else:
        BuildManager().build_resources()
        
        
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
    
    
