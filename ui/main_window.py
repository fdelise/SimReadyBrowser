"""
Main application window – NVIDIA SimReady Browser.

Layout
------
┌─────────────────────────────────────────────────────────────┐
│  MenuBar                                                      │
│  ToolBar (S3 URL | Refresh | GPU info)                       │
├────────────────┬──────────────────────────┬─────────────────┤
│  AssetBrowser  │      ViewportWidget      │  ControlsPanel  │
│  (300 px fixed)│   (stretch, OVRTX RTX)   │  (260 px fixed) │
└────────────────┴──────────────────────────┴─────────────────┘
│  StatusBar                                                    │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.s3_client import AssetInfo, S3Client, S3_URI_ROOT
from styles import nvidia_theme as theme
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_BG_HEADER,
    COLOR_BG_WINDOW,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)
from ui.asset_browser import AssetBrowserPanel
from ui.controls_panel import ControlsPanel
from ui.viewport_widget import ViewportWidget

APP_NAME    = "NVIDIA SimReady Browser"
APP_VERSION = "1.0.0"


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()

        # ── Core objects ────────────────────────────────────────────────────────
        cache_dir = Path.home() / ".cache" / "simready_browser"
        self._s3  = S3Client(cache_dir=cache_dir)
        self._closing = False

        # ── Window setup ────────────────────────────────────────────────────────
        self.setWindowTitle(APP_NAME)
        self.resize(1600, 900)
        self.setMinimumSize(1024, 600)
        self._set_window_icon()

        # ── Build UI ────────────────────────────────────────────────────────────
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._wire_signals()

        # ── Start S3 discovery ──────────────────────────────────────────────────
        QTimer.singleShot(200, self._s3.refresh)

    # ── Window icon ────────────────────────────────────────────────────────────

    def _set_window_icon(self):
        # Draw a simple NVIDIA-green "N" icon programmatically
        from PyQt5.QtGui import QPainter, QColor, QFont
        px = QPixmap(32, 32)
        px.fill(QColor("#000000"))
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor(COLOR_ACCENT))
        p.setFont(QFont("Arial Black", 20, QFont.Bold))
        p.drawText(px.rect(), Qt.AlignCenter, "N")
        p.end()
        self.setWindowIcon(QIcon(px))

    # ── Menu ────────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet(
            f"QMenuBar {{ background: {COLOR_BG_HEADER}; color: {COLOR_TEXT_PRIMARY}; "
            f"border-bottom: 1px solid #3a3a3a; font-size: 12px; }}"
            f"QMenuBar::item:selected {{ background: #363636; }}"
        )

        # File menu
        file_menu = mb.addMenu("&File")
        act_refresh = QAction("&Refresh S3 Bucket", self)
        act_refresh.setShortcut("Ctrl+R")
        act_refresh.triggered.connect(lambda: self._s3.refresh(force_network=True))
        file_menu.addAction(act_refresh)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # View menu
        view_menu = mb.addMenu("&View")
        self._act_toggle_browser = QAction("Show Asset Browser", self, checkable=True, checked=True)
        self._act_toggle_browser.triggered.connect(self._toggle_browser)
        view_menu.addAction(self._act_toggle_browser)

        self._act_toggle_controls = QAction("Show Controls", self, checkable=True, checked=True)
        self._act_toggle_controls.triggered.connect(self._toggle_controls)
        view_menu.addAction(self._act_toggle_controls)

        # Help menu
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ── Toolbar ─────────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setStyleSheet(
            f"QToolBar {{ background: {COLOR_BG_HEADER}; border-bottom: 1px solid #3a3a3a; "
            f"spacing: 6px; padding: 4px 8px; }}"
        )
        self.addToolBar(tb)

        # NVIDIA logo label
        logo = QLabel()
        logo.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 18px; font-weight: bold; "
                           f"letter-spacing: 2px; padding: 0 8px;")
        logo.setText("NVIDIA")
        tb.addWidget(logo)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: #3a3a3a;")
        tb.addWidget(sep)

        # App name
        app_label = QLabel(APP_NAME)
        app_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 13px; padding: 0 6px;")
        tb.addWidget(app_label)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        # GPU status
        self._gpu_label = QLabel("Detecting GPU…")
        self._gpu_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; padding: 0 8px;"
        )
        tb.addWidget(self._gpu_label)
        QTimer.singleShot(500, self._detect_gpu)

        # FPS display
        self._fps_label = QLabel("-- fps")
        self._fps_label.setStyleSheet(
            f"color: {COLOR_ACCENT}; font-size: 11px; font-weight: bold; padding: 0 8px;"
        )
        tb.addWidget(self._fps_label)

    # ── Central layout ─────────────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        central.setStyleSheet(f"background: {COLOR_BG_WINDOW};")
        self.setCentralWidget(central)

        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Main splitter
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStyleSheet(
            "QSplitter::handle { background: #2a2a2a; width: 2px; }"
            "QSplitter::handle:hover { background: #76b900; }"
        )

        # Left: asset browser
        self._browser = AssetBrowserPanel(self._s3)
        self._splitter.addWidget(self._browser)

        # Centre: OVRTX viewport
        self._viewport = ViewportWidget()
        self._splitter.addWidget(self._viewport)

        # Right: controls
        self._controls = ControlsPanel()
        self._splitter.addWidget(self._controls)

        self._splitter.setSizes([320, 1000, 260])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        h_layout.addWidget(self._splitter)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background: #111; color: {COLOR_TEXT_SECONDARY}; "
            f"font-size: 11px; border-top: 1px solid #2a2a2a; }}"
        )

        self._status_left = QLabel("Ready")
        sb.addWidget(self._status_left)

        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedSize(180, 10)
        self._loading_bar.setStyleSheet(
            f"QProgressBar {{ background: #252525; border: 1px solid #3a3a3a; "
            f"border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 2px; }}"
        )
        self._loading_bar.hide()
        sb.addWidget(self._loading_bar)

        sb.addPermanentWidget(_StatusSep())

        self._status_s3 = QLabel(f"S3: {S3_URI_ROOT}")
        self._status_s3.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 10px;")
        sb.addPermanentWidget(self._status_s3)

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _wire_signals(self):
        # Browser → controls & viewport
        self._browser.asset_selected.connect(self._on_asset_selected)
        self._browser.asset_activated.connect(self._load_asset)
        self._browser.status_message.connect(self._set_status)

        # Viewport → status / FPS
        self._viewport.status_msg.connect(self._set_status)
        self._viewport.loading_changed.connect(self._set_loading)
        self._viewport.fps_updated.connect(
            lambda fps: self._fps_label.setText(f"{fps:.1f} fps")
        )
        self._viewport.physics_status_changed.connect(self._controls.set_physics_status)
        self._viewport.physics_running_changed.connect(self._controls.set_physics_running)

        # Controls → viewport
        self._controls.dome_intensity_changed.connect(self._viewport.set_dome_intensity)
        self._controls.dir_light_changed.connect(
            lambda i, az, el: self._viewport.set_directional_light(i, az, el)
        )
        self._controls.reset_camera_requested.connect(self._viewport.reset_camera)
        self._controls.load_asset_requested.connect(self._load_asset)
        self._controls.physics_play_changed.connect(self._viewport.set_physics_playing)
        self._controls.physics_step_requested.connect(self._viewport.step_physics)
        self._controls.physics_restart_requested.connect(self._viewport.restart_physics)
        self._controls.physics_base_scene_changed.connect(self._viewport.set_physics_base_scene)
        self._controls.physics_collision_vis_changed.connect(self._viewport.set_physics_collision_overlay)
        self._controls.physics_grab_force_changed.connect(self._viewport.set_physics_grab_force)
        self._controls.set_physics_status(self._viewport.physics_status)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_asset_selected(self, asset: AssetInfo):
        self._controls.update_asset_info(asset)

    def _load_asset(self, asset: AssetInfo):
        """Load the selected USD directly from S3."""
        self._set_status(f"Loading {asset.display_name} in OVRTX...")
        self._viewport.load_usd(asset.usd_url)

    def _set_status(self, msg: str):
        self._status_left.setText(msg)

    def _set_loading(self, active: bool, msg: str):
        self._loading_bar.hide()
        if msg:
            self._set_status(msg)

    def _toggle_browser(self, checked: bool):
        self._browser.setVisible(checked)

    def _toggle_controls(self, checked: bool):
        self._controls.setVisible(checked)

    def _detect_gpu(self):
        """Try to read GPU name via nvidia-smi or pynvml."""
        gpu_name = "RTX GPU"
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).decode().strip().split("\n")[0]
            if out:
                gpu_name = out
        except Exception:
            try:
                import pynvml                       # type: ignore
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(h).decode()
            except Exception:
                pass

        self._gpu_label.setText(f"GPU: {gpu_name}")

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h2 style='color:#76b900'>NVIDIA SimReady Browser</h2>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>Browse and preview NVIDIA SimReady USD assets from the "
            f"Omniverse content production S3 bucket using the OVRTX renderer.</p>"
            f"<p>Powered by <b>NVIDIA OVRTX</b> and <b>PyQt5</b>.</p>"
            f"<p style='color:#888'>© 2026 NVIDIA Corporation</p>",
        )

    # ── Close ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._closing:
            event.ignore()
            return

        self._closing = True
        self._set_status("Closing renderer and background workers...")
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            self._s3.shutdown(timeout_ms=10000)
            viewport_ok = self._viewport.shutdown(timeout_ms=20000)
        finally:
            QApplication.restoreOverrideCursor()

        if not viewport_ok:
            self._closing = False
            self.setEnabled(True)
            self._set_status("OVRTX is still shutting down; retrying close...")
            QTimer.singleShot(500, self.close)
            event.ignore()
            return

        event.accept()


class _StatusSep(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setStyleSheet("color: #3a3a3a;")
        self.setFixedWidth(1)
