"""Assembled pages.

``build.py`` compiles the main window and each page's ``.ui`` file into
generated ``Ui_*`` classes under ``apps/gui/dist/`` (the main window's
stacked widget contains empty placeholder slots named ``homePage`` /
``morePage`` / ``procPage`` / ``settingsPage`` / ``downloadPage``).

This package provides the page widgets:

- instantiate one to get a self-contained page, e.g. ``home = Home()``;
- call :func:`assemble_pages` in ``MainWindow`` to fill the main window's
  placeholder slots with each page's compiled content.
"""

import os
import sys

# generated page modules live in apps/gui/dist (written by build.py) which
# is already on sys.path via the entry point's dist/ insert.
_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist")
for _p in (_DIST, os.path.join(_DIST, "pages")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtWidgets import QWidget


class _PageBase(QWidget):
    """A page widget filled by a generated ``Ui_*`` class."""

    #: objectName of the placeholder slot inside the main window stack
    CONTAINER_NAME = None

    @classmethod
    def _ui_instance(cls):
        """Return an instantiated ``Ui_*`` object whose ``setupUi`` can be called."""
        raise NotImplementedError

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(self.CONTAINER_NAME)
        self._ui_instance().setupUi(self)

    @classmethod
    def populate(cls, container: QWidget) -> None:
        """Fill an existing (empty) slot widget with this page's content."""
        if not container.objectName():
            container.setObjectName(cls.CONTAINER_NAME)
        cls._ui_instance().setupUi(container)


class Home(_PageBase):
    """homePage — changelog, logo and support buttons."""

    CONTAINER_NAME = "homePage"

    @classmethod
    def _ui_instance(cls):
        from home import Ui_HomePage

        return Ui_HomePage()


class More(_PageBase):
    """morePage — placeholder slot between home and process."""

    CONTAINER_NAME = "morePage"

    @classmethod
    def _ui_instance(cls):
        from more import Ui_MorePage

        return Ui_MorePage()


class Process(_PageBase):
    """procPage — preview, render controls and settings tabs."""

    CONTAINER_NAME = "procPage"

    @classmethod
    def _ui_instance(cls):
        from process import Ui_ProcPage

        return Ui_ProcPage()


class SettingsPage(_PageBase):
    """settingsPage — output / render / RVE settings."""

    CONTAINER_NAME = "settingsPage"

    @classmethod
    def _ui_instance(cls):
        from settings import Ui_SettingsPage

        return Ui_SettingsPage()


class Download(_PageBase):
    """downloadPage — application updates and backend selector."""

    CONTAINER_NAME = "downloadPage"

    @classmethod
    def _ui_instance(cls):
        from download import Ui_DownloadPage

        return Ui_DownloadPage()


#: Every page, used by :func:`assemble_pages` / ``MainWindow``.
PAGES = (Home, More, Process, SettingsPage, Download)


def assemble_pages(stacked_widget: QWidget) -> None:
    """Populate each page placeholder in ``stackedWidget`` in place."""
    for cls in PAGES:
        slot = stacked_widget.findChild(
            QWidget, cls.CONTAINER_NAME
        )
        if slot is not None:
            cls.populate(slot)


__all__ = ["Home", "More", "Process", "SettingsPage", "Download", "PAGES", "assemble_pages"]
