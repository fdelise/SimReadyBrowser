"""
NVIDIA SimReady Browser – entry point.

Run via:
    python main.py
or via the launch.bat launcher which handles all dependency installation.
"""

import sys
import os
from pathlib import Path

# ── Ensure project root is on sys.path so absolute imports work ───────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── High-DPI support ──────────────────────────────────────────────────────────
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap

from styles.nvidia_theme import get_stylesheet, COLOR_ACCENT, COLOR_BG_WINDOW


def _make_splash() -> QSplashScreen:
    """Create a simple NVIDIA-branded splash screen."""
    w, h = 600, 300
    px = QPixmap(w, h)
    px.fill(QColor(COLOR_BG_WINDOW))

    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)

    # Green accent strip at top
    p.fillRect(0, 0, w, 4, QColor(COLOR_ACCENT))

    # NVIDIA wordmark
    p.setFont(QFont("Segoe UI", 36, QFont.Bold))
    p.setPen(QColor(COLOR_ACCENT))
    p.drawText(0, 0, w, h // 2, Qt.AlignCenter, "NVIDIA")

    # App name
    p.setFont(QFont("Segoe UI", 16))
    p.setPen(QColor("#c0c0c0"))
    p.drawText(0, h // 2 - 10, w, 40, Qt.AlignCenter, "SimReady Browser")

    # Version / sub-text
    p.setFont(QFont("Segoe UI", 10))
    p.setPen(QColor("#606060"))
    p.drawText(0, h - 40, w, 30, Qt.AlignCenter, "Powered by OVRTX  |  Initialising…")

    # Bottom accent strip
    p.fillRect(0, h - 3, w, 3, QColor(COLOR_ACCENT))

    p.end()

    splash = QSplashScreen(px, Qt.WindowStaysOnTopHint)
    return splash


def main():
    # ── Application setup ─────────────────────────────────────────────────────
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("NVIDIA SimReady Browser")
    app.setOrganizationName("NVIDIA Corporation")
    app.setStyle("Fusion")   # Fusion base for reliable dark-theme rendering

    # Apply NVIDIA stylesheet
    app.setStyleSheet(get_stylesheet())

    # ── Splash screen ─────────────────────────────────────────────────────────
    splash = _make_splash()
    splash.show()
    app.processEvents()

    # ── Main window ───────────────────────────────────────────────────────────
    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    splash.finish(window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
