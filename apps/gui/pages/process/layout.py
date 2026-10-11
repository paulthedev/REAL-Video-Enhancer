"""Studio Polish re-layout for the process page.

Re-homes the widgets generated from ``process.ui`` into the approved
structure — preview card on top (with compare modes), Source / Models /
Queue cards below, the remaining option tabs under them, and a full-width
action bar at the bottom — WITHOUT renaming anything: the MainWindow flat
``self.<widget>`` contract is preserved because widgets are stolen from
their generated containers and re-added here by reference.

Legacy containers that no longer fit (the old preview container, the
General and Render Queue tabs) are emptied and hidden, not deleted.
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QStackedLayout,
    QWidget,
    QTabWidget,
)

from apps.gui.widgets.PreviewCompare import PreviewCompare
from apps.gui.widgets.QueuePanel import QueuePanel


def _steal(widget: QWidget) -> QWidget:
    """Detach ``widget`` from its current layout without destroying it."""
    parent = widget.parentWidget()
    if parent is not None:
        lay = parent.layout()
        if lay is not None:
            lay.removeWidget(widget)
    widget.setParent(None)
    return widget


def _steal_layout(layout) -> None:
    """Detach a generated child QLayout from its parent so it can be
    re-inserted elsewhere without the 'already has a parent' warning."""
    owner = layout.parentWidget()
    if owner is not None:
        lay = owner.layout()
        if lay is not None:
            lay.removeItem(layout)


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(10)
    header = QLabel(title)
    header.setProperty("cardTitle", True)
    lay.addWidget(header)
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setObjectName("cardSeparator")
    lay.addWidget(separator)
    return card, lay


def apply_studio_layout(win) -> tuple[PreviewCompare, QueuePanel, QStackedLayout]:
    """Rebuild the process page layout on ``win`` (the MainWindow).

    Returns ``(preview_compare, queue_panel, preview_stack)`` for ProcessTab
    to wire up. Idempotent: a second call steals from the already-built
    scaffold cleanly (widgets are found by attribute, containers hidden).
    """
    proc_page = win.procPage

    # ---- preview card (top) -------------------------------------------
    preview_compare = PreviewCompare()
    preview_card = QFrame()
    preview_card.setObjectName("card")
    preview_card.setMinimumHeight(280)
    preview_stack = QStackedLayout(preview_card)
    preview_stack.setContentsMargins(10, 10, 10, 10)
    preview_stack.addWidget(preview_compare)

    # button affordances (feather icons shipped in resources)
    def _icon(name):
        return QIcon(f":/icons/icons/{name}.svg")

    win.inputFileSelectButton.setIcon(_icon("folder"))
    win.batchSelectButton.setIcon(_icon("layers"))
    win.outputFileSelectButton.setIcon(_icon("folder"))
    win.outputFileSelectButton.setText("Output Folder")
    win.openOutputFolderButton.setIcon(_icon("external-link"))
    win.openOutputFolderButton.setText("Open Folder")
    win.addToRenderQueueButton.setIcon(_icon("plus"))

    # post-render playback page keeps the existing video widget + controls
    video_preview = _steal(win.VideoPreview)
    rendered_controls = _steal(win.RenderedPreviewControlsContainer)
    playback_page = QWidget()
    playback_lay = QHBoxLayout(playback_page)
    playback_lay.setContentsMargins(0, 0, 0, 0)
    playback_lay.addWidget(video_preview, 3)
    playback_lay.addWidget(rendered_controls, 1)
    preview_stack.addWidget(playback_page)
    video_preview.setVisible(True)

    win.previewLabel.hide()  # replaced by PreviewCompare

    # ---- source card ---------------------------------------------------
    source_card, source_lay = _card("Source")
    for name in ("inputFileContainer", "inputFileText", "outputFileContainer",
                 "outputFileText"):
        source_lay.addWidget(_steal(getattr(win, name)))
    input_info = _steal(win.widget_26)  # inputVideoInfoTextEdit container
    source_lay.addWidget(input_info, 1)

    add_btn = _steal(win.addToRenderQueueButton)
    add_btn.setProperty("variant", "primary")
    source_lay.addWidget(add_btn, 0, Qt.AlignLeft)

    # ---- models card ----------------------------------------------------
    models_card, models_lay = _card("Enhancements")
    for name in (
        "widget_3",               # backend combo row
        "widget_38",              # upscale toggle + scale/model containers
        "widget_35",              # interpolate toggle + model/factor containers
        "deblurCheckBoxContainer", "deblurContainer",
        "denoiseCheckBoxContainer", "denoiseContainer",
        "decompressModelContainer_2", "decompressContainer",
    ):
        models_lay.addWidget(_steal(getattr(win, name)))

    tiling_box = win.tilingCheckBox.parentWidget()
    while tiling_box is not None and tiling_box is proc_page:
        tiling_box = None
    if tiling_box is not None and tiling_box not in (proc_page,):
        models_lay.addWidget(_steal(tiling_box))

    models_scroll = _wrap_in_scroll(models_card)
    models_scroll.setMinimumHeight(0)

    # ---- queue & status card --------------------------------------------
    queue_card, queue_lay = _card("Queue & Status")
    queue_panel = QueuePanel(win.renderQueue)
    queue_lay.addWidget(queue_panel, 1)

    status_row = QHBoxLayout()  # transport strip (placed under the preview)
    status_row.setSpacing(8)
    for name in ("STATUS", "ETA", "FPS"):
        label = _steal(getattr(win, name))
        label.setStyleSheet("")  # drop the inline box style from the .ui
        label.setObjectName("statusPill")
        status_row.addWidget(label)
    status_row.addStretch(1)

    # ---- body: preview column LEFT, control rail RIGHT (Topaz-style) ------
    left_col = QVBoxLayout()
    left_col.setSpacing(8)
    left_col.addWidget(preview_card, 1)
    # ---- tabs: drop General + Render Queue (content re-homed above) -----
    tabs: QTabWidget = win.processSettingsContainer
    for tab_widget in (win.generalSettings, win.renderQueueTab):
        idx = tabs.indexOf(tab_widget)
        if idx >= 0:
            tabs.removeTab(idx)

    # Enhancements becomes the first tab of the left column (user request)
    tabs.insertTab(0, models_scroll, QIcon(":/icons/icons/sliders.svg"), "Enhancements")
    tabs.setCurrentIndex(0)

    left_col.addLayout(status_row)
    left_col.addWidget(tabs, 0)

    rail = QVBoxLayout()
    rail.setContentsMargins(0, 0, 0, 0)
    rail.setSpacing(10)
    rail.addWidget(source_card)
    rail.addWidget(queue_card)
    rail_widget = QWidget()
    rail_widget.setLayout(rail)
    rail_widget.setFixedWidth(360)

    body_row = QHBoxLayout()
    body_row.setSpacing(10)
    body_row.addLayout(left_col, 1)
    body_row.addWidget(rail_widget)

    tabs.setTabIcon(tabs.indexOf(win.tab_6), QIcon(":/icons/icons/film.svg"))
    tabs.setTabIcon(tabs.indexOf(win.advancedSettings), QIcon(":/icons/icons/settings.svg"))
    tabs.setTabIcon(tabs.indexOf(win.tab_5), QIcon(":/icons/icons/file-text.svg"))

    # ---- action bar (bottom) ----------------------------------------------
    action_bar = QHBoxLayout()
    action_bar.setSpacing(10)
    start_btn = _steal(win.startRenderButton)
    start_btn.setProperty("variant", "primary")
    start_btn.setText("Enhance")
    start_btn.setIcon(_icon("zap"))
    # the .ui caps it at 59x55 icon-only — undo that for the CTA treatment
    start_btn.setMinimumSize(200, 44)
    start_btn.setMaximumSize(16777215, 16777215)
    start_btn.setIconSize(QSize(20, 20))
    action_bar.addWidget(start_btn)

    # pause/stop: compact square ghost buttons matching the CTA height
    for name in ("pauseRenderButton", "killRenderButton"):
        btn = _steal(getattr(win, name))
        btn.setProperty("variant", "ghostIcon")
        btn.setIcon(_icon("pause" if "pause" in name else "square"))
        btn.setFixedSize(44, 44)
        btn.setIconSize(QSize(20, 20))
        btn.setText("")
        action_bar.addWidget(btn)

    progress_col = QVBoxLayout()
    progress_col.setContentsMargins(0, 12, 8, 12)
    progress = _steal(win.progressBar)
    progress.setFixedHeight(10)
    progress_col.addWidget(progress)
    action_bar.addLayout(progress_col, 1)

    # ---- assemble onto the page -------------------------------------------
    content = QVBoxLayout()
    content.setContentsMargins(16, 16, 16, 16)
    content.setSpacing(12)
    content.addLayout(body_row, 1)
    content.addLayout(action_bar)

    old_grid = proc_page.layout()
    # detach the legacy top-level containers from the grid before installing
    # the new column (they stay alive, hidden, holding nothing we need)
    for legacy_name in ("processInfoContainer", "processContainer"):
        legacy = win.findChild(QWidget, legacy_name)
        if legacy is not None and old_grid is not None:
            old_grid.removeWidget(legacy)
            legacy.hide()
    if old_grid is not None:
        old_grid.addLayout(content, 0, 0, 1, 1)

    return preview_compare, queue_panel, preview_stack


def _wrap_in_scroll(inner: QWidget) -> QWidget:
    from PySide6.QtWidgets import QScrollArea

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setMinimumHeight(120)
    return scroll
