"""
Subprocess wrappers that isolate the GUI from child processes.

Extracted from apps/gui/Util.py as part of the GUI refactor.
"""

import subprocess

from apps.gui.constants import PLATFORM


class subprocess_popen_without_terminal(subprocess.Popen):
    """
    A class that allows you to run a subprocess without opening a terminal window.
    """

    def __init__(self, *args, **kwargs):
        if PLATFORM == "win32":
            kwargs["startupinfo"] = subprocess.STARTUPINFO()
            kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        super().__init__(*args, **kwargs)


def create_independent_process(target_func, *args, **kwargs):
    """
    Creates a completely independent process that won't interfere with PyQt's main application
    """
    import os
    import sys
    from multiprocessing import Process, set_start_method

    # Force the 'spawn' method on all platforms for complete process isolation
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set, ignore
        pass

    # Set environment variables to avoid Qt conflicts
    env = os.environ.copy()
    env['QT_PLUGIN_PATH'] = ''
    env['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

    # Create process with isolated environment
    process = Process(target=target_func, args=args, kwargs=kwargs)

    # Set process to daemon so it terminates with main process
    process.daemon = True

    return process
