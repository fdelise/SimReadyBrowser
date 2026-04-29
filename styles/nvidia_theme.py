"""
NVIDIA SimReady Browser – NVIDIA-style dark theme stylesheet.
Colours match the NVIDIA brand palette: #76b900 green on #1a1a1a dark.
"""

# ── Colour tokens ──────────────────────────────────────────────────────────────
COLOR_BG_WINDOW     = "#1a1a1a"
COLOR_BG_PANEL      = "#242424"
COLOR_BG_WIDGET     = "#2d2d2d"
COLOR_BG_HOVER      = "#363636"
COLOR_BG_PRESSED    = "#1e1e1e"
COLOR_BG_SELECTED   = "#2a3d1a"
COLOR_BG_HEADER     = "#1e1e1e"

COLOR_ACCENT        = "#76b900"   # NVIDIA Green
COLOR_ACCENT_HOVER  = "#88d300"
COLOR_ACCENT_DIM    = "#4a7500"

COLOR_BORDER        = "#3a3a3a"
COLOR_BORDER_FOCUS  = "#76b900"

COLOR_TEXT_PRIMARY  = "#e8e8e8"
COLOR_TEXT_SECONDARY= "#a0a0a0"
COLOR_TEXT_DISABLED = "#555555"
COLOR_TEXT_ACCENT   = "#76b900"

COLOR_SCROLLBAR     = "#3a3a3a"
COLOR_SCROLLBAR_H   = "#76b900"

COLOR_VIEWPORT_BG   = "#0d0d0d"
COLOR_STATUS_BG     = "#111111"

FONT_FAMILY = "Segoe UI, Barlow, Arial, sans-serif"
FONT_SIZE_NORMAL = "12px"
FONT_SIZE_SMALL  = "10px"
FONT_SIZE_LARGE  = "14px"
FONT_SIZE_TITLE  = "16px"


def get_stylesheet() -> str:
    return f"""
/* ─── Global ───────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {COLOR_BG_WINDOW};
    color: {COLOR_TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_NORMAL};
    selection-background-color: {COLOR_BG_SELECTED};
    selection-color: {COLOR_TEXT_ACCENT};
}}

QMainWindow {{
    background-color: {COLOR_BG_WINDOW};
}}

/* ─── Panels / Frames ───────────────────────────────────────────────────────── */
QFrame, QGroupBox {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
}}

QGroupBox {{
    border-top: 2px solid {COLOR_ACCENT};
    padding-top: 18px;
    margin-top: 8px;
    font-weight: bold;
    color: {COLOR_ACCENT};
    font-size: {FONT_SIZE_LARGE};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 8px;
    color: {COLOR_ACCENT};
    background-color: transparent;
}}

/* ─── Labels ────────────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    border: none;
    color: {COLOR_TEXT_PRIMARY};
}}

QLabel#section_header {{
    color: {COLOR_ACCENT};
    font-weight: bold;
    font-size: {FONT_SIZE_LARGE};
    border-bottom: 1px solid {COLOR_ACCENT_DIM};
    padding-bottom: 4px;
}}

QLabel#asset_name {{
    color: {COLOR_TEXT_PRIMARY};
    font-size: {FONT_SIZE_SMALL};
    background: transparent;
}}

QLabel#status_label {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: {FONT_SIZE_SMALL};
    background: transparent;
}}

/* ─── Line Edit / Search ────────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    color: {COLOR_TEXT_PRIMARY};
    font-size: {FONT_SIZE_NORMAL};
}}

QLineEdit:focus {{
    border: 1px solid {COLOR_BORDER_FOCUS};
    background-color: {COLOR_BG_HOVER};
}}

QLineEdit::placeholder {{
    color: {COLOR_TEXT_DISABLED};
}}

/* ─── Push Buttons ──────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {COLOR_BG_WIDGET};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: {FONT_SIZE_NORMAL};
}}

QPushButton:hover {{
    background-color: {COLOR_BG_HOVER};
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT};
}}

QPushButton:pressed {{
    background-color: {COLOR_BG_PRESSED};
    border-color: {COLOR_ACCENT};
}}

QPushButton:disabled {{
    color: {COLOR_TEXT_DISABLED};
    border-color: {COLOR_BORDER};
}}

QPushButton#accent_btn {{
    background-color: {COLOR_ACCENT};
    color: #000000;
    font-weight: bold;
    border: none;
}}

QPushButton#accent_btn:hover {{
    background-color: {COLOR_ACCENT_HOVER};
}}

QPushButton#accent_btn:pressed {{
    background-color: {COLOR_ACCENT_DIM};
}}

/* ─── ComboBox ──────────────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 5px 10px;
    color: {COLOR_TEXT_PRIMARY};
    min-width: 80px;
}}

QComboBox:hover {{
    border-color: {COLOR_ACCENT};
}}

QComboBox:focus {{
    border-color: {COLOR_ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {COLOR_ACCENT};
    width: 0;
    height: 0;
    margin-right: 6px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_ACCENT_DIM};
    selection-background-color: {COLOR_BG_SELECTED};
    selection-color: {COLOR_ACCENT};
    outline: none;
}}

/* ─── Sliders ───────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {COLOR_BG_WIDGET};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLOR_ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLOR_ACCENT_HOVER};
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_ACCENT};
    border-radius: 2px;
}}

QSlider::add-page:horizontal {{
    background: {COLOR_BG_WIDGET};
    border-radius: 2px;
}}

QSlider::groove:vertical {{
    width: 4px;
    background: {COLOR_BG_WIDGET};
    border-radius: 2px;
}}

QSlider::handle:vertical {{
    background: {COLOR_ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
}}

QSlider::sub-page:vertical {{
    background: {COLOR_BG_WIDGET};
    border-radius: 2px;
}}

QSlider::add-page:vertical {{
    background: {COLOR_ACCENT};
    border-radius: 2px;
}}

/* ─── Spin Box ──────────────────────────────────────────────────────────────── */
QDoubleSpinBox, QSpinBox {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {COLOR_TEXT_PRIMARY};
}}

QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {COLOR_ACCENT};
}}

QDoubleSpinBox::up-button, QSpinBox::up-button {{
    background-color: {COLOR_BG_HOVER};
    border: none;
    border-radius: 2px;
}}

QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background-color: {COLOR_BG_HOVER};
    border: none;
    border-radius: 2px;
}}

/* ─── Splitter ──────────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {COLOR_BORDER};
    width: 2px;
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {COLOR_ACCENT};
}}

/* ─── Scrollbar ─────────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {COLOR_BG_PANEL};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_SCROLLBAR};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLOR_ACCENT_DIM};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {COLOR_BG_PANEL};
    height: 8px;
    border-radius: 4px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLOR_SCROLLBAR};
    border-radius: 4px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLOR_ACCENT_DIM};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ─── Scroll Area ───────────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: transparent;
}}

QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

/* ─── List Widget ───────────────────────────────────────────────────────────── */
QListWidget {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 4px 8px;
    border-radius: 3px;
}}

QListWidget::item:selected {{
    background-color: {COLOR_BG_SELECTED};
    color: {COLOR_ACCENT};
}}

QListWidget::item:hover {{
    background-color: {COLOR_BG_HOVER};
}}

/* ─── Tree Widget ───────────────────────────────────────────────────────────── */
QTreeWidget {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    outline: none;
}}

QTreeWidget::item {{
    padding: 3px 4px;
}}

QTreeWidget::item:selected {{
    background-color: {COLOR_BG_SELECTED};
    color: {COLOR_ACCENT};
}}

QTreeWidget::item:hover {{
    background-color: {COLOR_BG_HOVER};
}}

QHeaderView::section {{
    background-color: {COLOR_BG_HEADER};
    color: {COLOR_TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 5px 8px;
    font-size: {FONT_SIZE_SMALL};
}}

/* ─── Status Bar ────────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {COLOR_STATUS_BG};
    color: {COLOR_TEXT_SECONDARY};
    font-size: {FONT_SIZE_SMALL};
    border-top: 1px solid {COLOR_BORDER};
}}

QStatusBar::item {{
    border: none;
}}

/* ─── Menu / Toolbar ────────────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {COLOR_BG_HEADER};
    color: {COLOR_TEXT_PRIMARY};
    border-bottom: 1px solid {COLOR_BORDER};
}}

QMenuBar::item:selected {{
    background-color: {COLOR_BG_HOVER};
}}

QMenu {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_ACCENT_DIM};
    color: {COLOR_TEXT_PRIMARY};
}}

QMenu::item:selected {{
    background-color: {COLOR_BG_SELECTED};
    color: {COLOR_ACCENT};
}}

QMenu::separator {{
    height: 1px;
    background: {COLOR_BORDER};
    margin: 4px 8px;
}}

QToolBar {{
    background-color: {COLOR_BG_HEADER};
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    spacing: 4px;
    padding: 2px 4px;
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    color: {COLOR_TEXT_PRIMARY};
}}

QToolButton:hover {{
    background-color: {COLOR_BG_HOVER};
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT};
}}

QToolButton:checked {{
    background-color: {COLOR_BG_SELECTED};
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT};
}}

/* ─── Progress Bar ──────────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    text-align: center;
    color: {COLOR_TEXT_PRIMARY};
    font-size: {FONT_SIZE_SMALL};
}}

QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 3px;
}}

/* ─── Tab Widget ────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    background-color: {COLOR_BG_PANEL};
}}

QTabBar::tab {{
    background-color: {COLOR_BG_WIDGET};
    color: {COLOR_TEXT_SECONDARY};
    padding: 6px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    border: 1px solid {COLOR_BORDER};
    border-bottom: none;
}}

QTabBar::tab:selected {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_ACCENT};
    border-bottom: 2px solid {COLOR_ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLOR_BG_HOVER};
    color: {COLOR_TEXT_PRIMARY};
}}

/* ─── CheckBox ──────────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {COLOR_TEXT_PRIMARY};
    spacing: 6px;
    background: transparent;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {COLOR_ACCENT};
}}

/* ─── Tooltip ───────────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {COLOR_BG_WIDGET};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_ACCENT_DIM};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: {FONT_SIZE_SMALL};
}}

/* ─── Viewport frame ────────────────────────────────────────────────────────── */
QLabel#viewport_label {{
    background-color: {COLOR_VIEWPORT_BG};
    border: 2px solid {COLOR_BORDER};
    border-radius: 4px;
}}

QLabel#viewport_label:hover {{
    border-color: {COLOR_ACCENT_DIM};
}}

/* ─── Asset card ────────────────────────────────────────────────────────────── */
QFrame#asset_card {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 4px;
}}

QFrame#asset_card:hover {{
    border-color: {COLOR_ACCENT};
    background-color: {COLOR_BG_HOVER};
}}

QFrame#asset_card[selected="true"] {{
    border: 2px solid {COLOR_ACCENT};
    background-color: {COLOR_BG_SELECTED};
}}

/* ─── Section divider ───────────────────────────────────────────────────────── */
QFrame#divider {{
    background-color: {COLOR_BORDER};
    max-height: 1px;
    border: none;
}}
"""


VIEWPORT_OVERLAY_STYLE = f"""
    QLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: 11px;
        background: rgba(0,0,0,0.5);
        border-radius: 4px;
        padding: 4px 8px;
    }}
"""

ACCENT_COLOR = COLOR_ACCENT
BG_DARK = COLOR_BG_WINDOW
