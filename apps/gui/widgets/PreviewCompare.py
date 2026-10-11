"""Live preview canvas with before/after comparison modes.

Modes:
    enhanced    the streamed processed frame only
    side_by_side original | enhanced, with captions
    slider      original/enhanced split by a draggable divider

The widget is fed by the render pipeline: ``set_enhanced(QImage)`` receives
the streamed processed frame (from ``UpdateGUIThread``), ``set_source(QImage)``
receives the matching original frame (from the GUI-side source-frame reader).
A compare mode only paints a source frame when one has been provided; until
then it falls back to the enhanced frame and shows a hint.

The mode toolbar lives *below* the display (product decision) and is exposed
as :attr:`toolbar` so the process page can place it.
"""

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QPixmap, QColor, QFont, QPen, QPainterPath
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QButtonGroup,
    QSizePolicy,
)


class ModeButton(QToolButton):
    """Checkable view-mode chip; the look lives in the global QSS under the
    ``previewMode`` dynamic property (checked = accent-filled)."""

    def __init__(self, text, mode_id, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setProperty("previewMode", True)
        self.setCursor(Qt.PointingHandCursor)
        self._mode_id = mode_id

    @property
    def mode_id(self):
        return self._mode_id


class PreviewCompare(QWidget):
    """Preview display + compare-mode toolbar. See module docstring."""

    #: emitted when the user picks a view mode ("enhanced"/"side_by_side"/"slider")
    modeChanged = Signal(str)

    MODE_ENHANCED = "enhanced"
    MODE_SIDE_BY_SIDE = "side_by_side"
    MODE_SLIDER = "slider"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        self._enhanced = None  # QPixmap
        self._source = None  # QPixmap
        self._split = 0.5  # slider divider position 0..1
        self._dragging = False
        self._mode = self.MODE_ENHANCED
        self._hint = "Start a render to see the live preview"
        self._compare_hint = "Compare needs the source video — using enhanced only"

        self._mode_buttons = {}
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 10, 0, 0)
        toolbar.setSpacing(6)
        toolbar.addStretch(1)
        for text, mode_id in (
            ("Enhanced", self.MODE_ENHANCED),
            ("Side by Side", self.MODE_SIDE_BY_SIDE),
            ("Before ⟷ After", self.MODE_SLIDER),
        ):
            btn = ModeButton(text, mode_id)
            btn.clicked.connect(
                lambda checked=False, m=mode_id: self._on_mode_clicked(m)
            )
            self._button_group.addButton(btn)
            self._mode_buttons[mode_id] = btn
            toolbar.addWidget(btn)
        self._enhanced_button = self._mode_buttons[self.MODE_ENHANCED]
        self._enhanced_button.setChecked(True)
        toolbar.addStretch(1)  # second stretch -> chips centered

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._display = _PreviewCanvas(self)
        self._root.addWidget(self._display, 1)
        self._root.addLayout(toolbar)

    # ---------- public API ----------

    @property
    def toolbar(self) -> QHBoxLayout:
        """The mode-toolbar layout (already inserted below the display)."""
        return self._root.itemAt(1).layout()

    def set_enhanced(self, image):
        """Feed the streamed processed frame (QImage or None)."""
        if image is None or image.isNull():
            self._enhanced = None
        else:
            self._enhanced = QPixmap.fromImage(image)
        self._display.update()

    def set_source(self, image):
        """Feed the matching original frame (QImage or None)."""
        if image is None or image.isNull():
            self._source = None
        else:
            self._source = QPixmap.fromImage(image)
        self._display.update()

    def set_hint(self, text: str):
        self._hint = text
        self._display.update()

    def set_compare_available(self, available: bool):
        """Enable/disable the compare modes (e.g. URL or unreadable input)."""
        for mode_id in (self.MODE_SIDE_BY_SIDE, self.MODE_SLIDER):
            self._mode_buttons[mode_id].setEnabled(available)
        if not available and self._mode != self.MODE_ENHANCED:
            self.set_mode(self.MODE_ENHANCED)

    def set_mode(self, mode_id: str):
        self._mode = mode_id
        self._mode_buttons[mode_id].setChecked(True)
        self._display.update()
        self.modeChanged.emit(mode_id)

    @property
    def mode(self) -> str:
        return self._mode

    def clear(self):
        self._enhanced = None
        self._source = None
        self._display.update()

    # ---------- internals ----------

    def _on_mode_clicked(self, mode_id):
        if self._mode != mode_id:
            self.set_mode(mode_id)

    def _split_positions(self):
        """Split x in local canvas coords for slider mode."""
        return int(self._display.width() * self._split)

    def mousePressEvent(self, event):
        if self._mode == self.MODE_SLIDER and self._enhanced is not None:
            self._dragging = True
            self._set_split_from_x(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._set_split_from_x(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _set_split_from_x(self, x):
        self._split = min(max(x / max(self._display.width(), 1), 0.02), 0.98)
        self._display.update()


class _PreviewCanvas(QWidget):
    """The painting surface; owns nothing, delegates hit-testing up."""

    def __init__(self, owner: PreviewCompare):
        super().__init__(owner)
        self._owner = owner
        self.setMinimumHeight(200)

    def paintEvent(self, event):
        o = self._owner
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        radius = 10.0
        w, h = self.width(), self.height()

        # card background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#141920"))
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
        p.drawPath(path)

        enhanced = o._enhanced
        source = o._source
        compare = o._mode != o.MODE_ENHANCED and source is not None

        if enhanced is None:
            self._paint_empty(p, w, h)
            p.end()
            return

        if not compare:
            self._draw_fitted(p, enhanced, QRectF(0, 0, w, h), radius)
            if o._mode != o.MODE_ENHANCED and source is None:
                self._paint_caption(p, o._compare_hint, w, h, bottom=True)
            p.end()
            return

        if o._mode == o.MODE_SIDE_BY_SIDE:
            half = w / 2
            self._draw_fitted(p, source, QRectF(0, 0, half, h), radius, left=True)
            self._draw_fitted(
                p, enhanced, QRectF(half, 0, half, h), radius, left=False
            )
            self._paint_divider(p, half, h)
            self._paint_tag(p, "Original", 14, 14)
            self._paint_tag(p, "Enhanced", half + 14, 14)
        else:  # slider
            x = int(w * o._split)
            self._draw_split(p, source, enhanced, x, radius)
            self._paint_divider(p, x, h)
            self._paint_tag(p, "Original", 14, 14)
            self._paint_tag(p, "Enhanced", x + 14, 14)
            self._paint_handle(p, x, h)
        p.end()

    # ---------- paint helpers ----------

    def _paint_empty(self, p: QPainter, w, h):
        # placeholder art: dashed rounded frame + icon circle + copy
        p.setPen(QPen(QColor("#313945"), 1, Qt.DashLine))
        path = QPainterPath()
        path.addRoundedRect(QRectF(14, 14, w - 28, h - 28), 12, 12)
        p.drawPath(path)

        cx, cy = w / 2, h / 2 - 26
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(34, 211, 238, 26))
        p.drawEllipse(QPointF(cx, cy), 30, 30)
        p.setBrush(QColor("#22D3EE"))
        tri = QPainterPath(QPointF(cx - 7, cy - 10))
        tri.lineTo(cx - 7, cy + 10)
        tri.lineTo(cx + 12, cy)
        tri.closeSubpath()
        p.drawPath(tri)

        f = p.font()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#e6edf3"))
        p.drawText(QRectF(0, cy + 44, w, 24), Qt.AlignCenter, "Live Preview")
        f.setPointSize(9)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor("#67717e"))
        p.drawText(
            QRectF(0, cy + 68, w, 20), Qt.AlignCenter, self._owner._hint
        )

    def _draw_fitted(self, p: QPainter, pixmap: QPixmap, rect: QRectF, radius, left=None):
        """Draw pixmap aspect-fitted inside rect, clipped to rounded corners.
        With ``left`` set, align to that half instead of centering."""
        if left is True:
            target = self._half_rect(pixmap, rect, align_left=True)
        elif left is False:
            target = self._half_rect(pixmap, rect, align_left=False)
        else:
            target = self._fitted_rect(pixmap, rect)
        p.save()
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        p.setClipPath(path)
        p.drawPixmap(target, pixmap)
        p.restore()

    def _draw_split(self, p: QPainter, source: QPixmap, enhanced: QPixmap, x, radius):
        w, h = self.width(), self.height()
        p.save()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
        p.setClipPath(path)
        src_rect = self._fitted_rect(source, QRectF(0, 0, w, h))
        enh_rect = self._fitted_rect(enhanced, QRectF(0, 0, w, h))
        p.drawPixmap(src_rect, source)
        p.setClipRect(QRectF(x, 0, w - x, h))
        p.drawPixmap(enh_rect, enhanced)
        p.restore()

    def _fitted_rect(self, pixmap: QPixmap, rect: QRectF) -> QRectF:
        if pixmap.isNull():
            return rect
        scale = min(rect.width() / pixmap.width(), rect.height() / pixmap.height())
        pw, ph = pixmap.width() * scale, pixmap.height() * scale
        return QRectF(
            rect.x() + (rect.width() - pw) / 2,
            rect.y() + (rect.height() - ph) / 2,
            pw,
            ph,
        )

    def _half_rect(self, pixmap: QPixmap, rect: QRectF, align_left: bool) -> QRectF:
        """Aspect-fit into the half-width rect, centered on the visible half."""
        if pixmap.isNull():
            return rect
        scale = min(rect.width() / pixmap.width(), rect.height() / pixmap.height())
        pw, ph = pixmap.width() * scale, pixmap.height() * scale
        cx = rect.center().x() + (rect.width() / 4 if align_left else -rect.width() / 4)
        return QRectF(cx - pw / 2, rect.y() + (rect.height() - ph) / 2, pw, ph)

    def _paint_divider(self, p: QPainter, x, h):
        pen = QPen(QColor("#22D3EE"), 2)
        p.setPen(pen)
        p.drawLine(int(x), 0, int(x), int(h))

    def _paint_handle(self, p: QPainter, x, h):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#22D3EE"))
        p.drawEllipse(QPointF(x, h / 2), 11, 11)
        p.setBrush(QColor("#0f1216"))
        p.drawEllipse(QPointF(x, h / 2), 5, 5)

    def _paint_tag(self, p: QPainter, text, x, y):
        f = p.font()
        f.setPointSize(8)
        f.setBold(True)
        p.setFont(f)
        metrics = p.fontMetrics()
        tw = metrics.horizontalAdvance(text) + 12
        th = metrics.height() + 4
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 18, 22, 190))
        p.drawRoundedRect(int(x), int(y), tw, th, 4, 4)
        p.setPen(QColor("#e6edf3"))
        p.drawText(int(x) + 6, int(y) + th - 5, text)

    def _paint_caption(self, p: QPainter, text, w, h, bottom=False):
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor("#9aa4b2"))
        rect = QRectF(0, h - 26, w, 22) if bottom else QRectF(0, 0, w, 22)
        p.drawText(rect, Qt.AlignCenter, text)


__all__ = ["PreviewCompare"]
