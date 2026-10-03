"""
Network utilities.

`Util.networkCheck` and `constants.networkCheck` shared the same underlying
check previously; this module keeps that public surface while adding a small
timeout parameter for reuse.
"""

import urllib.request

from apps.gui.util.Logging import log


def networkCheck(hostname="https://raw.githubusercontent.com", timeout=1) -> bool:
    """
    Checks network availability against a url, default url: raw.githubusercontent.com.

    :param hostname: The URL to query.
    :param timeout: HTTP timeout in seconds.
    """
    try:
        req = urllib.request.Request(
            hostname, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        return True
    except Exception as e:
        log(str(e))
        log("No internet connection available.")
    return False
