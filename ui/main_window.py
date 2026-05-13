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

import os
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
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

from core.cad2usd_bridge import (
    CAD2USD_ROOT_ENV,
    CAD_FILE_FILTER,
    find_cad2usd_root,
    is_supported_cad_file,
    make_cad_usd_output_path,
)
from core.s3_client import AssetInfo, S3Client
from styles import nvidia_theme as theme
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_BG_HEADER,
    COLOR_BG_HOVER,
    COLOR_BG_WIDGET,
    COLOR_BG_WINDOW,
    COLOR_BORDER,
    COLOR_STATUS_BG,
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
        self._last_open_dir = Path.home()
        self._last_cad_open_dir = Path.home()
        self._cad2usd_root = find_cad2usd_root(Path(__file__).resolve().parents[1])
        self._cad2usd_process: Optional[QProcess] = None
        self._cad2usd_output_path: Optional[Path] = None
        self._cad2usd_last_line = ""
        self._cad2usd_log_lines: list[str] = []
        self._cad2usd_log_path: Optional[Path] = None
        self._cad2usd_job_id = 0
        self._cad2usd_base_env = QProcessEnvironment.systemEnvironment()
        self._browser_fullscreen = False
        self._browser_fullscreen_state = None
        self._default_splitter_applied = False

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
            f"border-bottom: 1px solid {COLOR_BORDER}; font-size: 12px; padding: 2px 8px; }}"
            "QMenuBar::item { background: transparent; padding: 5px 10px; border-radius: 10px; }"
            f"QMenuBar::item:selected {{ background: {COLOR_BG_HOVER}; color: {COLOR_ACCENT}; }}"
        )

        # File menu
        file_menu = mb.addMenu("&File")
        act_open = QAction("&Open USD File...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_usd_file)
        file_menu.addAction(act_open)

        act_open_cad = QAction("Open &CAD File...", self)
        act_open_cad.setShortcut("Ctrl+Shift+O")
        act_open_cad.triggered.connect(self._open_cad_file)
        file_menu.addAction(act_open_cad)

        file_menu.addSeparator()

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

        self._act_browser_fullscreen = QAction("Asset Browser Full Screen", self, checkable=True)
        self._act_browser_fullscreen.setShortcut("F11")
        self._act_browser_fullscreen.triggered.connect(self._toggle_browser_fullscreen)
        view_menu.addAction(self._act_browser_fullscreen)

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
            f"QToolBar {{ background: {COLOR_BG_HEADER}; border-bottom: 1px solid {COLOR_BORDER}; "
            f"spacing: 6px; padding: 6px 10px; }}"
        )
        self.addToolBar(tb)

        # NVIDIA logo label
        logo = QLabel()
        logo.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 18px; font-weight: bold; "
                           f"letter-spacing: 0px; padding: 0 8px;")
        logo.setText("NVIDIA")
        tb.addWidget(logo)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER};")
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
            f"QSplitter::handle {{ background: {COLOR_BORDER}; width: 2px; }}"
            f"QSplitter::handle:hover {{ background: {COLOR_ACCENT}; }}"
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

        self._splitter.setSizes([self._browser.maximumWidth(), 900, 340])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        h_layout.addWidget(self._splitter)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._default_splitter_applied:
            self._default_splitter_applied = True
            QTimer.singleShot(0, self._apply_default_splitter_sizes)

    def _apply_default_splitter_sizes(self):
        if self._browser_fullscreen:
            return
        browser_width = self._browser.maximumWidth()
        controls_width = min(340, self._controls.maximumWidth())
        viewport_width = max(1, self._splitter.width() - browser_width - controls_width)
        self._splitter.setSizes([browser_width, viewport_width, controls_width])

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background: {COLOR_STATUS_BG}; color: {COLOR_TEXT_SECONDARY}; "
            f"font-size: 11px; border-top: 1px solid {COLOR_BORDER}; }}"
        )

        self._status_left = QLabel("Ready")
        sb.addWidget(self._status_left)

        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedSize(180, 10)
        self._loading_bar.setStyleSheet(
            f"QProgressBar {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 5px; }}"
            f"QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 4px; }}"
        )
        self._loading_bar.hide()
        sb.addWidget(self._loading_bar)

        sb.addPermanentWidget(_StatusSep())

        self._status_s3 = QLabel(self._s3_status_text())
        self._status_s3.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 10px;")
        sb.addPermanentWidget(self._status_s3)

    def _s3_status_text(self) -> str:
        count = len(self._s3.locations)
        if count == 1:
            return f"S3: {self._s3.locations[0].root_uri}"
        return f"S3: {count} locations"

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _wire_signals(self):
        # Browser → controls & viewport
        self._browser.asset_selected.connect(self._on_asset_selected)
        self._browser.asset_activated.connect(self._load_asset)
        self._browser.load_selected_requested.connect(self._load_assets)
        self._browser.status_message.connect(self._set_status)
        self._browser.fullscreen_requested.connect(self._toggle_browser_fullscreen)
        self._s3.locations_changed.connect(lambda _locations: self._status_s3.setText(self._s3_status_text()))

        # Viewport → status / FPS
        self._viewport.status_msg.connect(self._set_status)
        self._viewport.loading_changed.connect(self._set_loading)
        self._viewport.fps_updated.connect(
            lambda fps: self._fps_label.setText(f"{fps:.1f} fps")
        )
        self._viewport.physics_status_changed.connect(self._controls.set_physics_status)
        self._viewport.physics_running_changed.connect(self._controls.set_physics_running)
        self._viewport.scene_tree_changed.connect(self._controls.set_scene_tree)
        self._viewport.scene_part_selection_changed.connect(self._controls.set_selected_scene_part)

        # Controls → viewport
        self._controls.dome_intensity_changed.connect(self._viewport.set_dome_intensity)
        self._controls.dome_environment_changed.connect(self._viewport.set_dome_environment)
        self._controls.dir_light_changed.connect(
            lambda i, az, el: self._viewport.set_directional_light(i, az, el)
        )
        self._controls.reset_camera_requested.connect(self._viewport.reset_camera)
        self._controls.load_asset_requested.connect(self._load_asset)
        self._controls.physics_play_changed.connect(self._viewport.set_physics_playing)
        self._controls.physics_step_requested.connect(self._viewport.step_physics)
        self._controls.physics_restart_requested.connect(self._viewport.restart_physics)
        self._controls.physics_drop_requested.connect(self._viewport.drop_physics)
        self._controls.physics_base_scene_changed.connect(self._viewport.set_physics_base_scene)
        self._controls.physics_collision_vis_changed.connect(self._viewport.set_physics_collision_overlay)
        self._controls.physics_grab_force_changed.connect(self._viewport.set_physics_grab_force)
        self._controls.physics_drop_options_changed.connect(self._viewport.set_physics_drop_options)
        self._controls.physics_ccd_changed.connect(self._viewport.set_physics_ccd_enabled)
        self._controls.physics_steps_changed.connect(self._viewport.set_physx_steps_per_second)
        self._controls.physics_substeps_changed.connect(self._viewport.set_physx_substeps)
        self._controls.physics_device_changed.connect(self._viewport.set_physx_device_mode)
        self._controls.physics_settings_reset_requested.connect(self._viewport.reset_physx_settings)
        self._controls.physics_engine_restart_requested.connect(self._viewport.restart_physics_engine)
        self._controls.scene_part_selected.connect(self._viewport.select_scene_part)
        self._controls.scene_part_property_changed.connect(self._viewport.set_scene_part_property)
        self._controls.scene_explorer_refresh_requested.connect(self._viewport.refresh_scene_explorer)
        physics_settings = self._controls.physics_settings()
        self._viewport.set_physics_ccd_enabled(bool(physics_settings["ccd_enabled"]))
        self._viewport.set_physx_steps_per_second(int(physics_settings["steps_per_second"]))
        self._viewport.set_physx_substeps(int(physics_settings["substeps"]))
        self._viewport.set_physx_device_mode(str(physics_settings["device_mode"]))
        self._controls.set_physics_status(self._viewport.physics_status)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_asset_selected(self, asset: AssetInfo):
        self._controls.update_asset_info(asset)

    def _load_asset(self, asset: AssetInfo):
        """Load the selected USD directly from S3."""
        self._set_status(f"Loading {asset.display_name} in OVRTX...")
        self._viewport.load_usd(asset.usd_url)

    def _load_assets(self, assets):
        selected = [asset for asset in assets if isinstance(asset, AssetInfo)]
        if not selected:
            return
        if len(selected) == 1:
            self._load_asset(selected[0])
            return

        self._set_status(f"Loading {len(selected)} selected assets in OVRTX...")
        self._viewport.load_usds(
            [
                {"name": asset.display_name, "source": asset.usd_url, "key": asset.asset_id}
                for asset in selected
            ]
        )

    def _open_usd_file(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open USD File",
            str(self._last_open_dir),
            "USD files (*.usd *.usda *.usdc *.usdz);;All files (*.*)",
        )
        if not path:
            return

        file_path = Path(path)
        self._last_open_dir = file_path.parent
        self._set_status(f"Loading local USD {file_path.name} in OVRTX...")
        self._viewport.load_usd(str(file_path), auto_cook_physics=False)

    def _open_cad_file(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open CAD File",
            str(self._last_cad_open_dir),
            CAD_FILE_FILTER,
        )
        if not path:
            return

        cad_path = Path(path)
        self._last_cad_open_dir = cad_path.parent
        if not is_supported_cad_file(cad_path):
            QMessageBox.warning(
                self,
                "Unsupported CAD File",
                f"CAD2USD does not list {cad_path.suffix or 'this file type'} as a supported input format.",
            )
            return

        self._convert_cad_file(cad_path)

    def _convert_cad_file(self, cad_path: Path):
        if self._cad2usd_process and self._cad2usd_process.state() != QProcess.NotRunning:
            QMessageBox.information(
                self,
                "CAD Conversion Running",
                "A CAD2USD conversion is already running. Wait for it to finish before starting another.",
            )
            return

        root = self._ensure_cad2usd_root()
        if root is None:
            return

        output_dir = Path(__file__).resolve().parents[1] / "cache" / "cad2usd"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = make_cad_usd_output_path(cad_path, output_dir)
        log_path = output_path.with_suffix(".log")

        process = QProcess(self)
        process.setWorkingDirectory(str(root))
        process.setProcessEnvironment(self._cad2usd_process_environment(root))

        self._cad2usd_process = process
        self._cad2usd_output_path = output_path
        self._cad2usd_last_line = ""
        self._cad2usd_log_lines = []
        self._cad2usd_log_path = log_path
        self._cad2usd_job_id += 1
        job_id = self._cad2usd_job_id

        process.readyReadStandardOutput.connect(
            lambda proc=process, job=job_id: self._read_cad2usd_output(proc, job, False)
        )
        process.readyReadStandardError.connect(
            lambda proc=process, job=job_id: self._read_cad2usd_output(proc, job, True)
        )
        process.errorOccurred.connect(
            lambda error, proc=process, job=job_id: self._on_cad2usd_error(proc, job, error)
        )
        process.finished.connect(
            lambda exit_code, exit_status, proc=process, job=job_id, out=output_path: self._on_cad2usd_finished(
                proc, job, out, exit_code, exit_status
            )
        )

        convert_bat = root / "convert.bat"
        program = os.environ.get("ComSpec") or "cmd.exe"
        args = ["/d", "/c", "call", str(convert_bat), str(cad_path), str(output_path)]
        self._record_cad2usd_log(f"START root={root}", stderr=False)
        self._record_cad2usd_log(f"INPUT {cad_path}", stderr=False)
        self._record_cad2usd_log(f"OUTPUT {output_path}", stderr=False)
        self._record_cad2usd_log(f"COMMAND {program} {' '.join(args)}", stderr=False)
        self._set_status(f"Converting CAD with CAD2USD: {cad_path.name}...")
        process.start(program, args)

    def _cad2usd_process_environment(self, root: Path) -> QProcessEnvironment:
        env = QProcessEnvironment(self._cad2usd_base_env)
        for name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PXR_PLUGINPATH_NAME",
            "USD_PLUGIN_PATH",
            "CARB_APP_PATH",
            "CARB_PLUGIN_PATH",
            "OMNI_KIT_PATH",
        ):
            env.remove(name)
        env.insert("OMNI_KIT_ACCEPT_EULA", "yes")
        env.insert(CAD2USD_ROOT_ENV, str(root))
        return env

    def _ensure_cad2usd_root(self) -> Optional[Path]:
        root = self._cad2usd_root
        if root and (root / "convert.bat").is_file():
            return root

        root = find_cad2usd_root(Path(__file__).resolve().parents[1])
        if root and (root / "convert.bat").is_file():
            self._cad2usd_root = root
            return root

        QMessageBox.information(
            self,
            "Locate CAD2USD",
            "Select the CAD2USD convert.bat file. You can also set CAD2USD_ROOT to the CAD2USD checkout.",
        )
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Locate CAD2USD convert.bat",
            str(Path.home()),
            "CAD2USD converter (convert.bat);;Batch files (*.bat);;All files (*.*)",
        )
        if not path:
            self._set_status("CAD2USD converter not configured.")
            return None

        convert_bat = Path(path)
        if convert_bat.name.lower() != "convert.bat":
            QMessageBox.warning(self, "CAD2USD", "Please select CAD2USD's convert.bat file.")
            self._set_status("CAD2USD converter not configured.")
            return None

        self._cad2usd_root = convert_bat.parent
        return self._cad2usd_root

    def _read_cad2usd_output(self, process: QProcess, job_id: int, stderr: bool):
        if job_id != self._cad2usd_job_id or process is not self._cad2usd_process:
            return
        data = process.readAllStandardError() if stderr else process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            self._cad2usd_last_line = line
            self._record_cad2usd_log(line, stderr=stderr)
            prefix = "CAD2USD warning" if stderr else "CAD2USD"
            self._set_status(f"{prefix}: {line[-180:]}")

    def _on_cad2usd_error(self, process: QProcess, job_id: int, _error):
        if job_id != self._cad2usd_job_id or process is not self._cad2usd_process:
            process.deleteLater()
            return
        detail = self._cad2usd_last_line or "CAD2USD process could not start."
        self._record_cad2usd_log(f"PROCESS ERROR {detail}", stderr=True)
        self._set_status(f"CAD conversion failed: {detail}")
        if process and process.state() == QProcess.NotRunning:
            process.deleteLater()
            self._cad2usd_process = None
            self._cad2usd_output_path = None

    def _on_cad2usd_finished(self, process: QProcess, job_id: int, output_path: Path, exit_code: int, exit_status):
        if job_id != self._cad2usd_job_id or process is not self._cad2usd_process:
            process.deleteLater()
            return

        self._read_cad2usd_output(process, job_id, False)
        self._read_cad2usd_output(process, job_id, True)

        self._cad2usd_process = None
        self._cad2usd_output_path = None
        process.deleteLater()
        self._record_cad2usd_log(f"EXIT code={exit_code} status={exit_status}", stderr=exit_code != 0)

        if (
            exit_code == 0
            and exit_status == QProcess.NormalExit
            and output_path
            and output_path.is_file()
            and output_path.stat().st_size > 0
        ):
            self._last_open_dir = output_path.parent
            self._set_status(f"CAD converted to {output_path.name}; loading in OVRTX...")
            self._viewport.load_usd(str(output_path), auto_cook_physics=False)
            return

        detail = self._cad2usd_last_line or f"Process exited with code {exit_code}."
        self._set_status(f"CAD conversion failed: {detail}")
        tail = "\n".join(self._cad2usd_log_lines[-12:])
        log_text = f"\n\nLog: {self._cad2usd_log_path}" if self._cad2usd_log_path else ""
        tail_text = f"\n\nLast CAD2USD output:\n{tail}" if tail else ""
        QMessageBox.warning(
            self,
            "CAD Conversion Failed",
            f"CAD2USD did not create a USD file.\n\n{detail}{log_text}{tail_text}",
        )

    def _record_cad2usd_log(self, line: str, stderr: bool = False):
        text = f"[CAD2USD {'STDERR' if stderr else 'STDOUT'}] {line}"
        print(text, file=sys.stderr if stderr else sys.stdout, flush=True)
        self._cad2usd_log_lines.append(text)
        log_path = self._cad2usd_log_path
        if log_path is None:
            return
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass

    def _set_status(self, msg: str):
        self._status_left.setText(msg)

    def _set_loading(self, active: bool, msg: str):
        self._loading_bar.hide()
        if msg:
            self._set_status(msg)

    def _toggle_browser(self, checked: bool):
        if self._browser_fullscreen:
            return
        self._browser.setVisible(checked)

    def _toggle_controls(self, checked: bool):
        if self._browser_fullscreen:
            return
        self._controls.setVisible(checked)

    def _toggle_browser_fullscreen(self, checked: bool):
        if checked == self._browser_fullscreen:
            self._browser.set_fullscreen_mode(checked)
            self._set_action_checked(self._act_browser_fullscreen, checked)
            return

        if checked:
            self._browser_fullscreen_state = {
                "sizes": self._splitter.sizes(),
                "browser_visible": self._browser.isVisible(),
                "controls_visible": self._controls.isVisible(),
            }
            self._browser_fullscreen = True
            self._browser.set_fullscreen_mode(True)
            self._browser.show()
            self._viewport.hide()
            self._controls.hide()
            self._set_action_checked(self._act_browser_fullscreen, True)
            self._set_action_checked(self._act_toggle_browser, True)
            self._set_action_checked(self._act_toggle_controls, False)
            self._act_toggle_browser.setEnabled(False)
            self._act_toggle_controls.setEnabled(False)
            self._splitter.setSizes([max(1, self.width()), 0, 0])
            self._set_status("Asset browser full screen. Press F11 to restore.")
            return

        state = self._browser_fullscreen_state or {}
        self._browser_fullscreen = False
        self._browser.set_fullscreen_mode(False)
        self._viewport.show()

        browser_visible = state.get("browser_visible", True)
        controls_visible = state.get("controls_visible", True)
        self._browser.setVisible(browser_visible)
        self._controls.setVisible(controls_visible)

        self._act_toggle_browser.setEnabled(True)
        self._act_toggle_controls.setEnabled(True)
        self._set_action_checked(self._act_browser_fullscreen, False)
        self._set_action_checked(self._act_toggle_browser, browser_visible)
        self._set_action_checked(self._act_toggle_controls, controls_visible)

        sizes = state.get("sizes")
        if sizes:
            self._splitter.setSizes(sizes)
        self._browser_fullscreen_state = None
        self._set_status("Viewport restored.")

    @staticmethod
    def _set_action_checked(action: QAction, checked: bool):
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

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
            f"<h2 style='color:{COLOR_ACCENT}'>NVIDIA SimReady Browser</h2>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>Browse and preview NVIDIA SimReady USD assets from the "
            f"Omniverse content production S3 bucket using the OVRTX renderer.</p>"
            f"<p>Powered by <b>NVIDIA OVRTX</b> and <b>PyQt5</b>.</p>"
            f"<p style='color:#888'>© 2026 NVIDIA Corporation</p>",
        )

    # ── Close ──────────────────────────────────────────────────────────────────

    def _stop_cad2usd_process(self):
        process = self._cad2usd_process
        if not process or process.state() == QProcess.NotRunning:
            return
        self._cad2usd_job_id += 1
        process.terminate()
        if not process.waitForFinished(3000):
            process.kill()
            process.waitForFinished(3000)
        self._cad2usd_process = None
        self._cad2usd_output_path = None

    def closeEvent(self, event):
        if self._closing:
            event.ignore()
            return

        self._closing = True
        self._set_status("Closing renderer and background workers...")
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            self._stop_cad2usd_process()
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
        self.setStyleSheet(f"color: {COLOR_BORDER};")
        self.setFixedWidth(1)
