"""Shared base class for RVE installable dependencies.

Subclasses (``Backend``, ``Python``, ``FFmpeg``, ``VcRedlist``) represent
platform-specific artifacts that can be packaged and installed by the GUI
into the RVE environment; they all derive from this base.
"""

from abc import ABC, abstractmethod

import os
import subprocess

from apps.gui.util import FileHandler


def run_executable(exe_path):
    try:
        # Run the executable and wait for it to complete
        result = subprocess.run(exe_path, check=True, capture_output=True, text=True)

        # Print the output and the exit code
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
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


class Dependency(ABC):
    updatable: bool = True
    download_path: str
    installed_path: str

    def __init__(self) -> None:
        FileHandler.createDirectory(os.path.dirname(self.download_path))

    @abstractmethod
    def get_download_link(self) -> str: ...

    @abstractmethod
    def download(self) -> None: ...

    def get_if_update_available(self) -> bool: ...

    def update_if_updates_available(self) -> None: ...
