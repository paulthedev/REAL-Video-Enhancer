"""Design tokens + global stylesheet for the Studio Polish redesign.

One source of truth: every color/radius lives in TOKENS, the stylesheet and
the QPalette are both derived from it, and pages pick up the look through
type/objectName/property selectors — no widget renames required.

Button variants (set in code, picked up by QSS):
    btn.setProperty("variant", "primary")   accent-filled CTA
    btn.setProperty("variant", "ghost")     flat transparent
    btn.setProperty("variant", "danger")    destructive
    (unset)                                  standard field-colored button
After changing a variant at runtime call btn.style().unpolish/polish.
"""

from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt, QPropertyAnimation
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

TOKENS = {
    # accent
    "accent": "#22D3EE",
    "accent_hover": "#67E8F9",
    "accent_pressed": "#0EA5C9",
    "accent_dim": "#0E7490",
    "accent_text": "#062a30",  # dark text on accent fills
    # background layers (window -> sidebar -> card -> field)
    "window": "#0f1216",
    "sidebar": "#16191d",
    "card": "#1c2128",
    "field": "#262c35",
    "field_hover": "#2d3542",
    # lines & text
    "border": "#313945",
    "border_subtle": "#272e38",
    "text": "#f4f6f8",
    "text_dim": "#9aa4b2",
    "text_faint": "#67717e",
    # status
    "danger": "#f87171",
    "success": "#34d399",
    "warning": "#fbbf24",
    # radii
    "radius_sm": "6px",
    "radius_md": "8px",
    "radius_lg": "12px",
    # type
    "font_family": '"Segoe UI", "Inter", "Ubuntu", "Noto Sans", sans-serif',
}


def styleSheet() -> str:
    t = TOKENS
    return f"""
* {{
    font-family: {t["font_family"]};
    font-size: 10pt;
}}
QLabel[caption="true"], QLabel[dim="true"] {{
    font-size: 9pt;
}}

QMainWindow, #centralwidget {{
    background-color: {t["window"]};
    color: {t["text"]};
}}

#leftMenuSubContainer, #bottomMenuSubContainer {{
    background-color: {t["sidebar"]};
    border-radius: {t["radius_lg"]};
    border: 1px solid {t["border_subtle"]};
}}

QStackedWidget {{
    background-color: transparent;
}}

/* ---------- text ---------- */
QLabel {{
    background: transparent;
    color: {t["text"]};
    border: none;
}}
QLabel[heading="true"] {{
    font-size: 13pt;
    font-weight: 600;
}}
QLabel[caption="true"], QLabel[dim="true"] {{
    color: {t["text_dim"]};
}}

/* ---------- buttons ---------- */
QPushButton {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_sm"]};
    padding: 6px 14px;
    min-height: 24px;
    text-align: center;
}}
QPushButton:hover {{
    background-color: {t["field_hover"]};
    border-color: {t["border"]};
}}
QPushButton:pressed {{
    background-color: {t["card"]};
}}
QPushButton:disabled {{
    color: {t["text_faint"]};
    background-color: {t["card"]};
    border-color: {t["border_subtle"]};
}}
QPushButton:checked {{
    background-color: {t["accent_dim"]};
    border-color: {t["accent"]};
    color: {t["text"]};
}}

QPushButton[variant="primary"], #startRenderButton {{
    background-color: {t["accent"]};
    color: {t["accent_text"]};
    border: none;
    font-weight: 600;
    padding: 10px 22px;
    min-height: 24px;
}}
QPushButton[variant="primary"]:hover, #startRenderButton:hover {{
    background-color: {t["accent_hover"]};
}}
QPushButton[variant="primary"]:pressed, #startRenderButton:pressed {{
    background-color: {t["accent_pressed"]};
}}
QPushButton[variant="primary"]:disabled, #startRenderButton:disabled {{
    background-color: {t["accent_dim"]};
    color: {t["text_faint"]};
}}

QPushButton[variant="ghost"] {{
    background: transparent;
    border: 1px solid {t["border"]};
}}
QPushButton[variant="ghost"]:hover {{
    background-color: {t["field"]};
    border-color: {t["accent"]};
}}

QPushButton[variant="danger"] {{
    background: transparent;
    color: {t["danger"]};
    border: 1px solid {t["danger"]};
}}
QPushButton[variant="danger"]:hover {{
    background-color: rgba(248, 113, 113, 0.12);
}}

/* ---------- inputs ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_sm"]};
    padding: 5px 10px;
    min-height: 22px;
    selection-background-color: {t["accent_dim"]};
    selection-color: {t["text"]};
}}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {t["border"]};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {t["accent"]};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {t["text_faint"]};
    background-color: {t["card"]};
}}
QPlainTextEdit, QTextEdit {{
    padding: 10px 12px;
}}

QComboBox {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_sm"]};
    padding: 5px 30px 5px 10px;
    min-height: 22px;
}}
QComboBox:hover {{
    border-color: {t["border"]};
}}
QComboBox:focus {{
    border: 1px solid {t["accent"]};
}}
QComboBox:disabled {{
    color: {t["text_faint"]};
    background-color: {t["card"]};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: url(:/icons/icons/down-arrow.png);
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: {t["radius_sm"]};
    selection-background-color: {t["accent_dim"]};
    selection-color: {t["text"]};
    outline: none;
}}

/* ---------- tabs ---------- */
QTabWidget::pane {{
    background-color: {t["card"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_md"]};
    top: -1px;
    padding: 8px;
}}
QTabBar::tab {{
    background: transparent;
    color: {t["text_dim"]};
    border: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: {t["radius_sm"]};
    border-top-right-radius: {t["radius_sm"]};
}}
QTabBar::tab:hover {{
    color: {t["text"]};
    background-color: rgba(255, 255, 255, 0.04);
}}
QTabBar::tab:selected {{
    color: {t["accent"]};
    background-color: {t["card"]};
    border-bottom: 2px solid {t["accent"]};
}}

/* ---------- selection controls ---------- */
QCheckBox, QRadioButton {{
    color: {t["text"]};
    spacing: 8px;
    background: transparent;
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {t["text_faint"]};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {t["border"]};
    background-color: {t["field"]};
}}
QCheckBox::indicator {{
    border-radius: 4px;
}}
QCheckBox::indicator:hover {{
    border-color: {t["accent"]};
}}
QCheckBox::indicator:checked {{
    background-color: {t["accent"]};
    border-color: {t["accent"]};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QRadioButton::indicator:checked {{
    border: 5px solid {t["accent"]};
    background-color: {t["window"]};
}}

/* ---------- progress ---------- */
QProgressBar {{
    background-color: {t["field"]};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: {t["text"]};
}}
QProgressBar::chunk {{
    background-color: {t["accent"]};
    border-radius: 4px;
}}

/* ---------- sliders ---------- */
QSlider::groove:horizontal {{
    height: 4px;
    background: {t["field"]};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {t["accent"]};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t["accent"]};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {t["accent_hover"]};
}}

/* ---------- lists / tables ---------- */
QListWidget, QTreeView, QTableView, QTreeWidget {{
    background-color: {t["card"]};
    color: {t["text"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_md"]};
    outline: none;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: {t["radius_sm"]};
}}
QListWidget::item:hover {{
    background-color: rgba(255, 255, 255, 0.04);
}}
QListWidget::item:selected {{
    background-color: {t["accent_dim"]};
    color: {t["text"]};
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t["border"]};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t["text_faint"]};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {t["border"]};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t["text_faint"]};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ---------- sidebar nav (QTabNavigation items) ---------- */
QToolButton[navButton="true"] {{
    background: transparent;
    color: {t["text_dim"]};
    border: none;
    border-left: 3px solid transparent;
    border-radius: {t["radius_sm"]};
    padding: 8px 4px;
    font-size: 9pt;
}}
QToolButton[navButton="true"]:hover {{
    background-color: rgba(255, 255, 255, 0.05);
    color: {t["text"]};
}}
QToolButton[navButton="true"]:checked {{
    background-color: rgba(34, 211, 238, 0.12);
    border-left: 3px solid {t["accent"]};
    color: {t["accent"]};
    font-weight: 600;
}}
QToolButton[navButton="true"]:disabled {{
    color: {t["text_faint"]};
}}

/* ---------- square ghost icon buttons (render controls) ---------- */
QPushButton[variant="ghostIcon"] {{
    background: transparent;
    border: 1px solid {t["border"]};
    border-radius: {t["radius_sm"]};
    padding: 6px;
}}
QPushButton[variant="ghostIcon"]:hover {{
    background-color: {t["field"]};
    border-color: {t["accent"]};
}}
QPushButton[variant="ghostIcon"]:pressed {{
    background-color: {t["card"]};
}}
QPushButton[variant="ghostIcon"]:disabled {{
    color: {t["text_faint"]};
    border-color: {t["border_subtle"]};
}}

/* ---------- status pills ---------- */
QLabel#statusPill {{
    color: {t["text_dim"]};
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid {t["border_subtle"]};
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 9pt;
}}

/* ---------- preview mode chips (PreviewCompare) ---------- */
QToolButton[previewMode="true"] {{
    background-color: {t["field"]};
    color: {t["text_dim"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: 14px;
    padding: 5px 14px;
    font-size: 9pt;
}}
QToolButton[previewMode="true"]:hover {{
    border-color: {t["accent"]};
    color: {t["text"]};
}}
QToolButton[previewMode="true"]:checked {{
    background-color: rgba(34, 211, 238, 0.15);
    border-color: {t["accent"]};
    color: {t["accent"]};
    font-weight: 600;
}}
QToolButton[previewMode="true"]:disabled {{
    color: {t["text_faint"]};
}}

/* ---------- cards (Studio Polish page scaffold) ---------- */
QFrame#card {{
    background-color: {t["card"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_lg"]};
}}
QFrame#cardSeparator {{
    color: {t["border_subtle"]};
    max-height: 1px;
}}
QLabel[cardTitle="true"] {{
    color: {t["text_dim"]};
    font-size: 10pt;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#statusChip {{
    font-size: 8pt;
}}

QFrame#dropZone {{
    background-color: {t["card"]};
    border: 2px dashed {t["border"]};
    border-radius: {t["radius_lg"]};
}}
QFrame#dropZone:hover {{
    border-color: {t["accent"]};
    background-color: rgba(34, 211, 238, 0.05);
}}

/* ---------- misc ---------- */
QToolTip {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: {t["radius_sm"]};
    padding: 6px 8px;
}}
QMenu {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: {t["radius_sm"]};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {t["accent_dim"]};
}}
QGroupBox {{
    background-color: {t["card"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: {t["radius_lg"]};
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    color: {t["text"]};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: {t["text_dim"]};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {t["border_subtle"]};
}}
QSplitter::handle {{
    background-color: {t["border_subtle"]};
}}
QSplitter::handle:hover {{
    background-color: {t["accent_dim"]};
}}
"""


def Palette() -> QPalette:
    """QPalette derived from the same tokens as the stylesheet, so anything
    drawing outside QSS (native dialogs, disabled palettes) matches."""
    t = TOKENS
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(t["window"]))
    palette.setColor(QPalette.WindowText, QColor(t["text"]))
    palette.setColor(QPalette.Base, QColor(t["card"]))
    palette.setColor(QPalette.AlternateBase, QColor(t["field"]))
    palette.setColor(QPalette.ToolTipBase, QColor(t["field"]))
    palette.setColor(QPalette.ToolTipText, QColor(t["text"]))
    palette.setColor(QPalette.Text, QColor(t["text"]))
    palette.setColor(QPalette.Button, QColor(t["field"]))
    palette.setColor(QPalette.ButtonText, QColor(t["text"]))
    palette.setColor(QPalette.BrightText, QColor(t["danger"]))
    palette.setColor(QPalette.Link, QColor(t["accent"]))
    palette.setColor(QPalette.LinkVisited, QColor(t["accent_dim"]))
    palette.setColor(QPalette.Highlight, QColor(t["accent_dim"]))
    palette.setColor(QPalette.HighlightedText, QColor(t["text"]))

    disabled_text = QColor(t["text_faint"])
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Base, QColor(t["card"]))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(t["card"]))
    palette.setColor(QPalette.Disabled, QPalette.Window, QColor(t["sidebar"]))
    return palette


def fadeAnimation(qObject: QWidget):
    opacity_effect = QGraphicsOpacityEffect()
    qObject.setGraphicsEffect(opacity_effect)

    animation = QPropertyAnimation(opacity_effect, b"opacity")
    animation.setDuration(200)
    animation.setStartValue(0.3)
    animation.setEndValue(1)
    animation.start()


__all__ = ["TOKENS", "styleSheet", "Palette", "fadeAnimation"]
