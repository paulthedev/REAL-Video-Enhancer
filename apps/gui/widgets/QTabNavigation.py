"""Reusable left-side navigation bar for the REAL-Video-Enhancer GUI.

This widget owns the four sidebar ``QPushButton`` instances (home / process /
settings / download) that used to be built inside ``mainwindow.ui`` and exposed
as bare attributes on the :class:`MainWindow` controller. It encapsulates the
icon sizing, checkable behaviour and layout so any window can embed a nav bar
by dropping a single ``QTabNavigation`` (built here or promoted in its own
``.ui``) into its interface.

Navigation signals are emitted by :meth:`buttonClicked`; the widget also exposes
the same buttons as ``homeBtn`` / ``processBtn`` / ``settingsBtn`` /
``downloadBtn`` attributes so a parent controller that previously referenced
those names keeps resolving -- see :meth:`bind_controlled_attributes`.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy
from PySide6.QtGui import QIcon
from PySide6.QtCore import Signal, QSize

# (resource path, tooltip) keyed by the icon name. These paths resolve at
# runtime once ``resources.qrc`` has been compiled/imported with the window
# (``dist/resources_rc.py``).
NAV_ICONS = {
    "home":     (":/icons/icons/home.svg",           "Home"),
    "process":  (":/icons/icons/cpu.svg",            "Process"),
    "settings": (":/icons/icons/settings.svg",       "Settings"),
    "download": (":/icons/icons/download.svg",       "Download"),
}


class QTabNavigation(QWidget):
    """Left sidebar nav that switches the application's visible page.

    Each button is a checkbox-style icon button sized and iconified exactly as
    designed in the original ``mainwindow.ui``; clicking one emits
    :attr:`buttonClicked` with the zero-based index of the active tab (ordered
    home, process, settings, download).
    """

    #: Emitted ``(int index)`` when a navigation button is clicked.
    buttonClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(u"")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Home button is checked by default so the app lands on the home tab.
        self.buttons = []
        for icon_key in NAV_ICONS:
            btn = QPushButton(self)
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
            btn.setMaximumSize(QSize(55, 16777215))
            btn.setIcon(QIcon(NAV_ICONS[icon_key][0]))
            btn.setIconSize(QSize(35, 35))
            btn.setToolTip(NAV_ICONS[icon_key][1])
            btn.clicked.connect(self._make_button_clicked_handler(icon_key))
            if icon_key == "home":
                btn.setChecked(True)
            setattr(self, icon_key, btn)
            # Re-expose the "*Button" names a parent controller used to rely on.
            setattr(self, "%sBtn" % icon_key, btn)
            self.buttons.append(btn)

    # Bound method: ``self`` is available here (the previous @staticmethod could
    # not emit ``self.buttonClicked``). Clicking one tab emits its zero-based index.
    def _make_button_clicked_handler(self, icon_key):
        index = {key: i for i, key in enumerate(NAV_ICONS)}.get(icon_key)

        def handler(is_checked=False):
            self.buttonClicked.emit(index)

        return handler

    def bind_controlled_attributes(self, controller):
        """Point the MainWindow controller at the four nav buttons by their
        original "*Button" object names. The buttons are owned (and signal-bound)
        by this widget; re-pointing the controller's ``homeBtn`` / ``processBtn``
        / ``settingsBtn`` / ``downloadBtn`` objects onto the same instances keeps
        every existing binding (QConnect, setButtonsUnchecked, no-backends fallback)
        working. Called from :meth:MainWindow.setupUi once the promoted widget is up."""
        for icon_key in NAV_ICONS:
            setattr(controller, "%sBtn" % icon_key, getattr(self, "%sBtn" % icon_key))

    def current_index(self) -> int:
        """Index (0-3) of the currently checked button."""
        checked = next((b for b in self.buttons if b.isChecked()), None)
        return self.buttons.index(checked) if checked is not None else -1


__all__ = ["QTabNavigation"]
