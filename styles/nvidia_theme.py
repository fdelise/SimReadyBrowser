"""
Modern dark theme for the NVIDIA SimReady Browser.

The palette keeps the NVIDIA lime accent, but moves the app toward a
marketplace-style UI: deep charcoal surfaces, soft borders, pill controls,
and high-contrast selected states.
"""

# Color tokens
COLOR_BG_WINDOW     = "#10110f"
COLOR_BG_PANEL      = "#161815"
COLOR_BG_WIDGET     = "#1d201c"
COLOR_BG_HOVER      = "#252a22"
COLOR_BG_PRESSED    = "#0b0c0a"
COLOR_BG_SELECTED   = "#b7ff00"
COLOR_BG_HEADER     = "#0b0c0b"

COLOR_ACCENT        = "#b7ff00"
COLOR_ACCENT_HOVER  = "#d7ff3f"
COLOR_ACCENT_DIM    = "#6f8f14"

COLOR_BORDER        = "#343a32"
COLOR_BORDER_FOCUS  = "#b7ff00"

COLOR_TEXT_PRIMARY  = "#f5f7f1"
COLOR_TEXT_SECONDARY= "#a9b0a3"
COLOR_TEXT_DISABLED = "#646b60"
COLOR_TEXT_ACCENT   = "#10110f"

COLOR_SCROLLBAR     = "#3a4238"
COLOR_SCROLLBAR_H   = "#b7ff00"

COLOR_VIEWPORT_BG   = "#070807"
COLOR_STATUS_BG     = "#0b0c0b"

FONT_FAMILY = "Segoe UI, Barlow, Arial, sans-serif"
FONT_SIZE_NORMAL = "12px"
FONT_SIZE_SMALL  = "10px"
FONT_SIZE_LARGE  = "14px"
FONT_SIZE_TITLE  = "16px"


def get_stylesheet() -> str:
    return f"""
/* Global */
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

/* Panels / Frames */
QFrame, QGroupBox {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}

QGroupBox {{
    margin-top: 8px;
    padding-top: 18px;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
    font-size: {FONT_SIZE_LARGE};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {COLOR_ACCENT};
    background-color: transparent;
}}

/* Labels */
QLabel {{
    background: transparent;
    border: none;
    color: {COLOR_TEXT_PRIMARY};
}}

QLabel#section_header {{
    color: {COLOR_TEXT_PRIMARY};
    font-weight: 800;
    font-size: {FONT_SIZE_LARGE};
    border: none;
    padding: 2px 0 4px 0;
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

/* Line Edit / Search */
QLineEdit {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 15px;
    padding: 7px 14px;
    color: {COLOR_TEXT_PRIMARY};
    font-size: {FONT_SIZE_NORMAL};
    min-height: 18px;
}}

QLineEdit:hover {{
    border-color: {COLOR_ACCENT_DIM};
}}

QLineEdit:focus {{
    border: 1px solid {COLOR_BORDER_FOCUS};
    background-color: {COLOR_BG_HOVER};
}}

QLineEdit::placeholder {{
    color: {COLOR_TEXT_DISABLED};
}}

/* Push Buttons */
QPushButton {{
    background-color: {COLOR_BG_WIDGET};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 7px 14px;
    font-size: {FONT_SIZE_NORMAL};
    font-weight: 600;
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
    background-color: {COLOR_BG_PANEL};
    border-color: {COLOR_BORDER};
}}

QPushButton#accent_btn {{
    background-color: {COLOR_ACCENT};
    color: #090a08;
    font-weight: 800;
    border: none;
}}

QPushButton#accent_btn:hover {{
    background-color: {COLOR_ACCENT_HOVER};
    color: #090a08;
}}

QPushButton#accent_btn:pressed {{
    background-color: {COLOR_ACCENT_DIM};
}}

/* ComboBox */
QComboBox {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 6px 28px 6px 12px;
    color: {COLOR_TEXT_PRIMARY};
    min-width: 80px;
    min-height: 18px;
}}

QComboBox:hover {{
    background-color: {COLOR_BG_HOVER};
    border-color: {COLOR_ACCENT_DIM};
}}

QComboBox:focus {{
    border-color: {COLOR_ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {COLOR_ACCENT};
    width: 0;
    height: 0;
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    color: {COLOR_TEXT_PRIMARY};
    padding: 4px;
    selection-background-color: {COLOR_BG_HOVER};
    selection-color: {COLOR_TEXT_PRIMARY};
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    color: {COLOR_TEXT_PRIMARY};
    padding: 5px 8px;
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {COLOR_BG_HOVER};
    color: {COLOR_TEXT_PRIMARY};
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 6px;
    background: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLOR_ACCENT};
    border: 2px solid #090a08;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLOR_ACCENT_HOVER};
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_ACCENT};
    border-radius: 3px;
}}

QSlider::add-page:horizontal {{
    background: {COLOR_BG_WIDGET};
    border-radius: 3px;
}}

QSlider::groove:vertical {{
    width: 6px;
    background: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background: {COLOR_ACCENT};
    border: 2px solid #090a08;
    width: 16px;
    height: 16px;
    margin: 0 -6px;
    border-radius: 8px;
}}

QSlider::sub-page:vertical {{
    background: {COLOR_BG_WIDGET};
    border-radius: 3px;
}}

QSlider::add-page:vertical {{
    background: {COLOR_ACCENT};
    border-radius: 3px;
}}

/* Spin Box */
QDoubleSpinBox, QSpinBox {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 5px 8px;
    color: {COLOR_TEXT_PRIMARY};
    min-height: 18px;
}}

QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: {COLOR_ACCENT_DIM};
}}

QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {COLOR_ACCENT};
}}

QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background-color: {COLOR_BG_HOVER};
    border: none;
    width: 14px;
}}

/* Splitter */
QSplitter::handle {{
    background-color: {COLOR_BORDER};
    width: 2px;
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {COLOR_ACCENT};
}}

/* Scrollbar */
QScrollBar:vertical {{
    background: {COLOR_BG_WINDOW};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_SCROLLBAR};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLOR_ACCENT_DIM};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {COLOR_BG_WINDOW};
    height: 10px;
    border-radius: 5px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLOR_SCROLLBAR};
    border-radius: 5px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLOR_ACCENT_DIM};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* Scroll Area */
QScrollArea {{
    border: none;
    background: transparent;
}}

QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

/* List / Tree Widgets */
QListWidget, QTreeWidget {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    outline: none;
    alternate-background-color: {COLOR_BG_PANEL};
}}

QListWidget::item, QTreeWidget::item {{
    padding: 4px 6px;
    border-radius: 4px;
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {COLOR_ACCENT};
    color: #090a08;
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {COLOR_BG_HOVER};
}}

QHeaderView::section {{
    background-color: {COLOR_BG_HEADER};
    color: {COLOR_TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 6px 8px;
    font-size: {FONT_SIZE_SMALL};
    font-weight: 700;
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLOR_STATUS_BG};
    color: {COLOR_TEXT_SECONDARY};
    font-size: {FONT_SIZE_SMALL};
    border-top: 1px solid {COLOR_BORDER};
}}

QStatusBar::item {{
    border: none;
}}

/* Menu / Toolbar */
QMenuBar {{
    background-color: {COLOR_BG_HEADER};
    color: {COLOR_TEXT_PRIMARY};
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 2px 8px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 10px;
}}

QMenuBar::item:selected {{
    background-color: {COLOR_BG_HOVER};
    color: {COLOR_ACCENT};
}}

QMenu {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    color: {COLOR_TEXT_PRIMARY};
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 18px;
    border-radius: 6px;
}}

QMenu::item:selected {{
    background-color: {COLOR_ACCENT};
    color: #090a08;
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
    spacing: 6px;
    padding: 6px 10px;
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 14px;
    padding: 6px 10px;
    color: {COLOR_TEXT_PRIMARY};
}}

QToolButton:hover {{
    background-color: {COLOR_BG_HOVER};
    border-color: {COLOR_ACCENT_DIM};
    color: {COLOR_ACCENT};
}}

QToolButton:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
    color: #090a08;
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLOR_BG_WIDGET};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    text-align: center;
    color: {COLOR_TEXT_PRIMARY};
    font-size: {FONT_SIZE_SMALL};
}}

QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 4px;
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    background-color: {COLOR_BG_PANEL};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {COLOR_BG_WIDGET};
    color: {COLOR_TEXT_SECONDARY};
    padding: 7px 12px;
    border-radius: 14px;
    margin: 0 4px 6px 0;
    border: 1px solid {COLOR_BORDER};
}}

QTabBar::tab:selected {{
    background-color: {COLOR_ACCENT};
    color: #090a08;
    border-color: {COLOR_ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLOR_BG_HOVER};
    color: {COLOR_TEXT_PRIMARY};
    border-color: {COLOR_ACCENT_DIM};
}}

/* CheckBox */
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
    border-radius: 4px;
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {COLOR_ACCENT};
}}

/* Tooltip */
QToolTip {{
    background-color: {COLOR_BG_WIDGET};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 6px 8px;
    font-size: {FONT_SIZE_SMALL};
}}

/* Viewport frame */
QLabel#viewport_label {{
    background-color: {COLOR_VIEWPORT_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}

QLabel#viewport_label:hover {{
    border-color: {COLOR_ACCENT_DIM};
}}

/* Asset card */
QFrame#asset_card {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 0;
}}

QFrame#asset_card:hover {{
    border-color: {COLOR_ACCENT_DIM};
    background-color: {COLOR_BG_HOVER};
}}

QFrame#asset_card[selected="true"] {{
    border: 2px solid {COLOR_ACCENT};
    background-color: #171f10;
}}

/* Section divider */
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
        background: rgba(7, 8, 7, 190);
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 6px 10px;
    }}
"""

ACCENT_COLOR = COLOR_ACCENT
BG_DARK = COLOR_BG_WINDOW
