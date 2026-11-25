"""Small runtime UI fixes applied after the pages are assembled.

These are intentionally conservative: hide controls that no longer have an
implementation, give the tiny checkbox rows a full-label click target, and
prevent layout stretches from spreading a few rows across a whole card.
"""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QCheckBox, QLabel, QScrollArea, QTabWidget, QWidget


class _LabelToggleFilter(QObject):
    """Clicking a row's label toggles its checkbox."""

    def __init__(self, checkbox: QCheckBox, parent=None):
        super().__init__(parent)
        self._checkbox = checkbox

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        ):
            self._checkbox.toggle()
            return True
        return False


def _row_label_for(checkbox: QCheckBox) -> QLabel | None:
    parent = checkbox.parentWidget()
    if parent is None or parent.layout() is None:
        return None
    outer = parent.layout()
    row = outer if outer.indexOf(checkbox) >= 0 else None
    if row is None:
        for i in range(outer.count()):
            candidate = outer.itemAt(i).layout()
            if candidate is not None and candidate.indexOf(checkbox) >= 0:
                row = candidate
                break
    if row is None:
        return None
    for i in range(row.count()):
        widget = row.itemAt(i).widget()
        if isinstance(widget, QLabel) and widget.text().strip():
            return widget
    return None


def _make_checkbox_labels_clickable(root: QWidget) -> None:
    for checkbox in root.findChildren(QCheckBox):
        label = _row_label_for(checkbox)
        if label is None:
            continue
        label.setBuddy(checkbox)
        label.setCursor(Qt.PointingHandCursor)
        filt = _LabelToggleFilter(checkbox, label)
        label.installEventFilter(filt)
        label._rve_toggle_filter = filt  # keep the filter alive


def _hide_dead_controls(window) -> None:
    # ONNX is the only backend — the combo has no alternatives and is never read.
    for name in ("backendComboBox", "label_12"):
        widget = getattr(window, name, None)
        if widget is not None:
            widget.setVisible(False)
    backend_row = getattr(window, "widget_3", None)
    if backend_row is not None:
        backend_row.setVisible(False)

    # UHD interpolation mode has no ONNX implementation.
    for name in ("uhd_mode", "label_22", "uhdHelp"):
        widget = getattr(window, name, None)
        if widget is not None:
            widget.setVisible(False)

    # Leftover help icon for the removed torch.compile toggle.
    label_90 = getattr(window, "label_90", None)
    if label_90 is not None:
        label_90.setVisible(False)

    # Section titles inside the scroll areas were clipped at the top edge.
    for name in ("label_74", "label_41"):
        label = getattr(window, name, None)
        if label is not None:
            label.setContentsMargins(0, 6, 0, 0)

    # Ko-fi is hidden in this fork; don't leave its caption behind.
    label_13 = getattr(window, "label_13", None)
    if label_13 is not None:
        label_13.setVisible(False)


def _settings_content_layouts(page):
    """Yield the innermost layouts that hold settings rows."""
    layout = page.layout() if page is not None else None
    if layout is None:
        return
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is None:
            continue
        if isinstance(widget, QScrollArea):
            content = widget.widget()
            if content is not None and content.layout() is not None:
                yield content.layout()
            continue
        inner_layout = widget.layout()
        if inner_layout is None:
            continue
        found_scroll = False
        for j in range(inner_layout.count()):
            inner = inner_layout.itemAt(j).widget()
            if isinstance(inner, QScrollArea):
                content = inner.widget()
                if content is not None and content.layout() is not None:
                    yield content.layout()
                    found_scroll = True
                    break
        if not found_scroll:
            yield inner_layout


def _compact_settings_tabs(window) -> None:
    tabs = getattr(window, "tabWidget", None)
    if not isinstance(tabs, QTabWidget):
        return
    for i in range(tabs.count()):
        page = tabs.widget(i)
        for layout in _settings_content_layouts(page):
            # Put the slack at the bottom instead of spreading a handful of
            # rows over the whole card height.
            layout.addStretch(1)


def _fix_scroll_margins(window) -> None:
    for name in ("scrollArea_4", "scrollArea_5", "scrollArea_7"):
        area = getattr(window, name, None)
        content = area.widget() if area is not None else None
        layout = content.layout() if content is not None else None
        if layout is None:
            continue
        margins = layout.contentsMargins()
        # Section titles sat flush against the card edge and were clipped.
        layout.setContentsMargins(
            margins.left(), max(margins.top(), 10), margins.right(), margins.bottom()
        )
        layout.addStretch(1)

    # The Enhancements tab's scroll area lost its object name during the
    # studio re-layout; keep its rows compact too.
    tabs = getattr(window, "processSettingsContainer", None)
    if tabs is not None:
        area = tabs.findChild(QScrollArea)
        content = area.widget() if area is not None else None
        layout = content.layout() if content is not None else None
        if layout is not None and not getattr(window, "_enhancements_stretch_added", False):
            layout.addStretch(1)
            window._enhancements_stretch_added = True


def _enlarge_checkboxes(window) -> None:
    for checkbox in window.findChildren(QCheckBox):
        checkbox.setMinimumHeight(24)


def _add_placeholders(window) -> None:
    for name, text in (
        ("inputFileText", "No input selected"),
        ("outputFileText", "Select an output folder"),
        ("renderOutput", "Render logs will appear here."),
    ):
        widget = getattr(window, name, None)
        if widget is not None and hasattr(widget, "setPlaceholderText"):
            widget.setPlaceholderText(text)


def _fix_home_hero(window) -> None:
    # The logo label stretched to half the page, pushing the title into the
    # middle of the window. Constrain it and left-align the title beside it.
    logo = getattr(window, "label_2", None)
    if logo is not None:
        logo.setMaximumSize(220, 220)
    title = getattr(window, "label", None)
    if title is not None:
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 600;")


def _fix_source_info(window) -> None:
    info = getattr(window, "inputVideoInfoTextEdit", None)
    if info is not None:
        # The .ui set min-height 190 / max-height 180 and disabled scrolling,
        # so a short window clipped the frame/encoder lines.
        info.setMinimumHeight(100)
        info.setMaximumHeight(190)
        info.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


def _fix_encoder_command(window) -> None:
    field = getattr(window, "EncoderCommand", None)
    if field is None or getattr(window, "_encoder_label_added", False):
        return
    layout = field.parentWidget().layout() if field.parentWidget() else None
    if layout is None:
        return
    label = QLabel("Encoder Command")
    layout.insertWidget(0, label)
    field.setMinimumWidth(0)
    window._encoder_label_added = True


def _regroup_settings(window) -> None:
    # Scene detection is a render setting; it was in the Output tab.
    row = getattr(window, "widget_7", None)
    target = getattr(window, "renderSettingsContainer", None)
    target_layout = target.layout() if target is not None else None
    if row is None or target_layout is None:
        return
    insert_at = target_layout.count()
    last = target_layout.itemAt(insert_at - 1) if insert_at else None
    if last is not None and last.spacerItem() is not None:
        insert_at -= 1
    target_layout.insertWidget(insert_at, row)


def _add_advanced_caption(window) -> None:
    heading = getattr(window, "label_41", None)
    if heading is None or getattr(window, "_advanced_caption_added", False):
        return
    layout = heading.parentWidget().layout() if heading.parentWidget() else None
    if layout is None:
        return
    caption = QLabel("Optional behavior for diagnostics and benchmarking.")
    caption.setProperty("caption", True)
    layout.insertWidget(layout.indexOf(heading) + 1, caption)
    window._advanced_caption_added = True


def _add_settings_captions(window) -> None:
    def add(layout, text, flag):
        if layout is None or getattr(window, flag, False):
            return
        caption = QLabel(text)
        caption.setProperty("caption", True)
        caption.setWordWrap(True)
        layout.insertWidget(0, caption)
        setattr(window, flag, True)

    output_area = getattr(window, "scrollArea_3", None)
    output_content = output_area.widget() if output_area is not None else None
    if output_content is not None:
        add(
            output_content.layout(),
            "Where rendered files are written by default.",
            "_output_caption_added",
        )

    render_container = getattr(window, "renderSettingsContainer", None)
    if render_container is not None:
        add(
            render_container.layout(),
            "Model precision, HDR handling and the GPU used for inference.",
            "_render_caption_added",
        )


def _polish_rve_tab(window) -> None:
    from apps.gui.constants import CWD

    preview = getattr(window, "previewEnabledContainer", None)
    if preview is not None and preview.parentWidget() is not None:
        layout = preview.parentWidget().layout()
        if layout is not None and not getattr(window, "_rve_caption_added", False):
            caption = QLabel(
                "Application preferences. Settings, models and caches live in "
                "the data folder below."
            )
            caption.setProperty("caption", True)
            caption.setWordWrap(True)
            layout.insertWidget(0, caption)
            window._rve_caption_added = True

    button = getattr(window, "openRVEFolderBtn", None)
    if button is not None and button.parentWidget() is not None:
        layout = button.parentWidget().layout()
        if layout is not None and not getattr(window, "_rve_path_added", False):
            info = QLabel(f"Data folder: {CWD}")
            info.setProperty("caption", True)
            info.setWordWrap(True)
            layout.insertWidget(layout.indexOf(button) + 1, info)
            window._rve_path_added = True


def _add_shortcuts_and_accessibility(window) -> None:
    shortcuts = [
        ("Ctrl+1", window.homeBtn.click),
        ("Ctrl+2", window.processBtn.click),
        ("Ctrl+3", window.settingsBtn.click),
        ("Ctrl+4", window.downloadBtn.click),
        ("Ctrl+O", window.openInputFile),
        ("Ctrl+Shift+O", window.openBatchFiles),
        ("Ctrl+Return", window.addToRenderQueue),
        ("Ctrl+Shift+Return", window.startRender),
        ("Ctrl+P", window.renderPreview),
    ]
    window._rve_shortcuts = []
    for key, slot in shortcuts:
        shortcut = QShortcut(QKeySequence(key), window)
        shortcut.activated.connect(slot)
        window._rve_shortcuts.append(shortcut)

    tooltips = {
        "inputFileSelectButton": "Select input file (Ctrl+O)",
        "batchSelectButton": "Select an input folder (Ctrl+Shift+O)",
        "addToRenderQueueButton": "Add the current setup to the render queue (Ctrl+Enter)",
        "startRenderButton": "Start rendering the queue (Ctrl+Shift+Enter)",
        "pauseRenderButton": "Pause rendering",
        "killRenderButton": "Stop rendering",
        "githubBtn": "Open the GitHub page",
        "uninstallAppBtn": "Uninstall REAL Video Enhancer",
    }
    for name, tooltip in tooltips.items():
        widget = getattr(window, name, None)
        if widget is None:
            continue
        widget.setToolTip(tooltip)
        widget.setAccessibleName(tooltip)


def _hide_idle_render_controls(window) -> None:
    for name in ("pauseRenderButton", "killRenderButton"):
        widget = getattr(window, name, None)
        if widget is not None:
            widget.setVisible(False)


def apply_polish(window) -> None:
    """Apply the conservative UI fixes once the main window is assembled."""
    _hide_dead_controls(window)
    _make_checkbox_labels_clickable(window)
    _compact_settings_tabs(window)
    _fix_scroll_margins(window)
    _enlarge_checkboxes(window)
    _add_placeholders(window)
    _fix_home_hero(window)
    _fix_source_info(window)
    _fix_encoder_command(window)
    _regroup_settings(window)
    _add_advanced_caption(window)
    _add_settings_captions(window)
    _polish_rve_tab(window)
    _add_shortcuts_and_accessibility(window)
    _hide_idle_render_controls(window)
