"""Reusable "info / help" components extracted from the page ``.ui`` files.

Background
----------
Across ``process.ui``, ``settings.ui`` and ``download.ui`` the same help
affordance recurs dozens of times: a small 25x25 ``QLabel`` showing the
``:/icons/icons/info.svg`` pixmap with an HTML ``toolTip`` explaining the option
it sits next to. Previously every instance inlined the full property set::

    <widget class="QLabel" name="label_17">
        <property name="maximumSize"><size><width>25</width>...
        <property name="toolTip"><string>&lt;html&gt;...</string></property>
        <property name="text"><string /></property>
        <property name="pixmap">...:/icons/icons/info.svg...
        <property name="scaledContents"><bool>true</bool></property>
    </widget>

That bulk now lives once, in :class:`HelpIconLabel`. The ``.ui`` files promote
the widget to that class and only carry the per-instance ``toolTip``::

    <widget class="HelpIconLabel" name="label_17">
        <property name="toolTip"><string>...</string></property>
    </widget>

Behaviour / access
------------------
- The generated ``setupUi`` still instantiates it as a direct attribute of the
  page root (``self.label_17 = HelpIconLabel(...)``), and any other named widget
  inside a promoted container is flattened to the root as well. Page controllers
  therefore keep binding to their controls exactly as before — nothing about
  them sees these labels.
- Only the ``toolTip`` remains in the ``.ui``; it is applied by the generated
  ``retranslateUi`` call, so translations still work.

The module is named after the plan's "InfoRow" pattern (see
``docs/gui-refactor-plan.md``); the reusable leaf actually extracted from those
rows is the help icon itself, hence :class:`HelpIconLabel`.
"""

from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

#: Icon used by every help affordance in the app (see ``resources.qrc``).
INFO_ICON_RESOURCE = ":/icons/icons/info.svg"


class HelpIconLabel(QLabel):
    """A 25x25 info-icon :class:`QLabel` with an explanatory HTML toolTip.

    The size, pixmap and scaled-contents behaviour are fixed here; instances
    only provide the ``toolTip`` text (set via ``retranslateUi`` or directly).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        size = QSize(25, 25)
        self.setMaximumSize(size)
        pixmap = QPixmap(INFO_ICON_RESOURCE)
        if not pixmap.isNull():
            self.setPixmap(pixmap)
        self.setScaledContents(True)
