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

from PySide6.QtCore import Qt
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
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(8)
    header = QLabel(title)
    header.setProperty("cardTitle", True)
    lay.addWidget(header)
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
    preview_stack = QStackedLayout(preview_card)
    preview_stack.setContentsMargins(8, 8, 8, 8)
    preview_stack.addWidget(preview_compare)

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

    status_row = QHBoxLayout()
    status_row.setSpacing(12)
    for name in ("STATUS", "ETA", "FPS"):
        label = _steal(getattr(win, name))
        label.setStyleSheet("")  # drop the inline box style from the .ui
        status_row.addWidget(label)
    status_row.addStretch(1)
    queue_lay.addLayout(status_row)

    # ---- options row ------------------------------------------------------
    options_row = QHBoxLayout()
    options_row.setSpacing(10)
    options_row.addWidget(source_card, 3)
    options_row.addWidget(models_scroll, 3)
    options_row.addWidget(queue_card, 3)

    # ---- tabs: drop General + Render Queue (content re-homed above) -----
    tabs: QTabWidget = win.processSettingsContainer
    for tab_widget in (win.generalSettings, win.renderQueueTab):
        idx = tabs.indexOf(tab_widget)
        if idx >= 0:
            tabs.removeTab(idx)

    # ---- action bar (bottom) ----------------------------------------------
    action_bar = QHBoxLayout()
    action_bar.setSpacing(10)
    start_btn = _steal(win.startRenderButton)
    start_btn.setProperty("variant", "primary")
    start_btn.setText("Enhance")
    # the .ui caps it at 59x55 icon-only — undo that for the CTA treatment
    start_btn.setMinimumSize(220, 44)
    start_btn.setMaximumSize(16777215, 16777215)
    start_btn.setIconSize(start_btn.icon().actualSize(start_btn.iconSize()))
    action_bar.addWidget(start_btn)

    render_controls = getattr(win, "onRenderButtonsContiainer")
    if isinstance(render_controls, QWidget):
        action_bar.addWidget(_steal(render_controls))
    else:  # generated as a bare QHBoxLayout (pause/kill button row)
        _steal_layout(render_controls)
        action_bar.addLayout(render_controls)

    progress_row = QVBoxLayout()
    progress_row.addWidget(_steal(win.progressBar))
    action_bar.addLayout(progress_row, 1)

    # ---- assemble onto the page -------------------------------------------
    content = QVBoxLayout()
    content.setContentsMargins(12, 12, 12, 12)
    content.setSpacing(10)
    content.addWidget(preview_card, 5)
    content.addLayout(options_row, 4)
    content.addWidget(tabs, 2)
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
