"""
Assorted small helpers (open folder, open link, timing decorator).

Extracted from apps/gui/Util.py as part of the GUI refactor.
"""

import os
import subprocess
import time
import webbrowser
from functools import wraps

from apps.gui.constants import PLATFORM


def openLink(link: str):
    """
    Opens a link in the default web browser.

    :param link: The link to open.
    :type link: str
    """
    webbrowser.open(link)


def open_folder(folder):
    if PLATFORM == "win32":
        os.startfile(folder)
    elif PLATFORM == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])


def print_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} : {end - start:.6f} seconds")
        return result
    return wrapper
