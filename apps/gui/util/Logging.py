import os
import sys
import warnings

from apps.gui.constants import CWD


def log(message: str):
    """
    Append a message to the runtime log file (<CWD>/logs/log.txt) created by the GUI.

    :param message: The message to write.
    """
    try:
        os.makedirs(os.path.join(CWD, "logs"), exist_ok=True)
        with open(os.path.join(CWD, "logs", "log.txt"), "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"An error occurred while logging: {e}", file=sys.stderr)


def warnAndLog(message: str):
    """
    Emit a Python warning AND log it.

    :param message: The message to warn about and log.
    """
    warnings.warn(message)
    log("WARN: " + message)


def errorAndLog(message: str):
    """
    Log an error message AND raise an os.error with the same text.

    :param message: The error message.
    """
    log("ERROR: " + message)
    raise os.error("ERROR: " + message)
