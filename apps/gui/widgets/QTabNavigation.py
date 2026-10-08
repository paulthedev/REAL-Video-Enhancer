"""Reusable left-side navigation bar for the REAL-Video-Enhancer GUI.

This widget owns the four sidebar nav buttons (home / process / settings /
download) that used to be built inside ``mainwindow.ui`` and exposed as bare
attributes on the :class:`MainWindow` controller. It encapsulates the icon +
label sizing, checkable behaviour and layout so any window can embed a nav
bar by dropping a single ``QTabNavigation`` into its interface.

Navigation signals are emitted by :meth:`buttonClicked`; the widget also
exposes the same buttons as ``homeBtn`` / ``processBtn`` / ``settingsBtn`` /
``downloadBtn`` attributes so a parent controller that previously referenced
those names keeps resolving -- see :meth:`bind_controlled_attributes`.

Visual spec (Studio Polish redesign): icon with a caption underneath, active
item is an accent-tinted pill with a left accent bar; the look lives in the
global stylesheet scoped to the ``navButton`` dynamic property.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QToolButton,
    QSizePolicy,
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Signal, QSize, Qt

# (resource path, caption, tooltip) keyed by the icon name. These paths
# resolve at runtime once ``resources.qrc`` has been compiled/imported with
# the window (``dist/resources_rc.py``).
NAV_ICONS = {
    "home":     (":/icons/icons/home.svg",     "Home",     "Home"),
    "process":  (":/icons/icons/cpu.svg",      "Enhance",  "Video processing"),
    "settings": (":/icons/icons/settings.svg", "Settings", "Settings"),
    "download": (":/icons/icons/download.svg", "Models",   "Model & app downloads"),
}


class QTabNavigation(QWidget):
    """Left sidebar nav that switches the application's visible page.

    Each item is a checkable tool button (icon + caption underneath) sized
    for the 88px rail; clicking one emits :attr:`buttonClicked` with the
    zero-based index of the active tab (ordered home, process, settings,
    download).
    """

    #: Emitted ``(int index)`` when a navigation button is clicked.
    buttonClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(4)

        # Home button is checked by default so the app lands on the home tab.
        self.buttons = []
        for icon_key in NAV_ICONS:
            btn = QToolButton(self)
            btn.setCheckable(True)
            btn.setProperty("navButton", True)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(56)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setIcon(QIcon(NAV_ICONS[icon_key][0]))
            btn.setIconSize(QSize(22, 22))
            btn.setText(NAV_ICONS[icon_key][1])
            btn.setToolTip(NAV_ICONS[icon_key][2])
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(self._make_button_clicked_handler(icon_key))
            if icon_key == "home":
                btn.setChecked(True)
            setattr(self, icon_key, btn)
            # Re-expose the "*Button" names a parent controller used to rely on.
            setattr(self, "%sBtn" % icon_key, btn)
            self.buttons.append(btn)
            # without this the layout stays empty, the widget's size hint is 0
            # and the main window's grid collapses the sidebar to zero width
            layout.addWidget(btn)

        layout.addStretch(1)
        self.setFixedWidth(88)

    # Bound method: ``self`` is available here (the previous @staticmethod could
    # not emit ``self.buttonClicked``). Clicking one tab emits its index.
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
        working. Called from :meth:`MainWindow.setupUi once the promoted widget is up."""
        for icon_key in NAV_ICONS:
            setattr(controller, "%sBtn" % icon_key, getattr(self, "%sBtn" % icon_key))

    def current_index(self) -> int:
        """Index (0-3) of the currently checked button."""
        checked = next((b for b in self.buttons if b.isChecked()), None)
        return self.buttons.index(checked) if checked is not None else -1


__all__ = ["QTabNavigation"]
