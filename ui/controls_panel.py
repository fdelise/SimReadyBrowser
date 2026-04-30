"""
Controls Panel – right-hand side of the SimReady Browser.

Provides:
  • Lighting controls  (dome intensity, directional light intensity/direction)
  • Material overrides (roughness, metallic – write USD attributes via OVRTX)
  • Asset info display
  • Viewport settings  (resolution preset, reset camera)
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.s3_client import AssetInfo
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_BG_PANEL,
    COLOR_BORDER,
    COLOR_TEXT_DISABLED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)


class ControlsPanel(QWidget):
    """
    Right-side panel: lighting, materials, asset info.

    Signals
    -------
    dome_intensity_changed(float)
    dir_intensity_changed(float, float, float)   – intensity, azimuth, elevation
    material_changed(float, float)               – roughness, metallic
    reset_camera_requested()
    """

    dome_intensity_changed = pyqtSignal(float)
    dir_light_changed      = pyqtSignal(float, float, float)
    material_changed       = pyqtSignal(float, float)
    reset_camera_requested = pyqtSignal()
    load_asset_requested   = pyqtSignal(object)  # AssetInfo
    physics_play_changed   = pyqtSignal(bool)
    physics_step_requested = pyqtSignal()
    physics_restart_requested = pyqtSignal()
    physics_drop_requested = pyqtSignal(int)
    physics_base_scene_changed = pyqtSignal(str)
    physics_collision_vis_changed = pyqtSignal(bool)
    physics_grab_force_changed = pyqtSignal(float)
    physics_ccd_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_asset: Optional[AssetInfo] = None
        self._build_ui()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(240)
        self.setMaximumWidth(300)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 4, 0)
        inner_layout.setSpacing(10)

        inner_layout.addWidget(self._build_lighting_group())
        inner_layout.addWidget(self._build_physics_group())
        inner_layout.addWidget(self._build_camera_group())
        inner_layout.addWidget(self._build_asset_info_group())
        inner_layout.addStretch(1)

        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ── Lighting ───────────────────────────────────────────────────────────────

    def _build_lighting_group(self) -> QGroupBox:
        grp = QGroupBox("Lighting")
        lay = QFormLayout(grp)
        lay.setSpacing(10)
        lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.setFormAlignment(Qt.AlignTop)
        lay.setContentsMargins(10, 30, 10, 10)

        # Dome intensity
        self._dome_slider = _make_slider(0, 500, 100)
        self._dome_slider.valueChanged.connect(
            lambda v: self.dome_intensity_changed.emit(v / 100.0)
        )
        dome_row = _slider_with_value(self._dome_slider, fmt=lambda v: f"{v/100:.2f}")
        lay.addRow("Dome:", dome_row)

        self._dir_enabled = QCheckBox("Direct light")
        self._dir_enabled.setChecked(True)
        self._dir_enabled.toggled.connect(self._emit_dir_light)
        lay.addRow("Enable:", self._dir_enabled)

        # Directional intensity
        self._dir_int_slider = _make_slider(0, 500, 80)
        self._dir_int_slider.valueChanged.connect(self._emit_dir_light)
        dir_row = _slider_with_value(self._dir_int_slider, fmt=lambda v: f"{v/100:.2f}")
        lay.addRow("Sun:", dir_row)

        # Directional azimuth
        self._dir_az_slider = _make_slider(0, 360, 45)
        self._dir_az_slider.valueChanged.connect(self._emit_dir_light)
        az_row = _slider_with_value(self._dir_az_slider, fmt=lambda v: f"{v}°")
        lay.addRow("Azimuth:", az_row)

        # Directional elevation
        self._dir_el_slider = _make_slider(0, 90, 60)
        self._dir_el_slider.valueChanged.connect(self._emit_dir_light)
        el_row = _slider_with_value(self._dir_el_slider, fmt=lambda v: f"{v}°")
        lay.addRow("Elevation:", el_row)

        # Preset buttons
        preset_row = QHBoxLayout()
        for name, az, el in [("Studio", 45, 60), ("Noon", 0, 90), ("Sunset", 270, 10)]:
            btn = _small_button(name)
            az_, el_ = az, el
            btn.clicked.connect(lambda _, a=az_, e=el_: self._apply_light_preset(a, e))
            preset_row.addWidget(btn)
        preset_w = QWidget()
        preset_w.setLayout(preset_row)
        preset_w.setStyleSheet("background: transparent;")
        lay.addRow("Preset:", preset_w)

        return grp

    def _emit_dir_light(self):
        intensity = self._dir_int_slider.value() / 100.0
        if not self._dir_enabled.isChecked():
            intensity = 0.0

        self.dir_light_changed.emit(
            intensity,
            float(self._dir_az_slider.value()),
            float(self._dir_el_slider.value()),
        )

    def _apply_light_preset(self, azimuth: int, elevation: int):
        self._dir_az_slider.setValue(azimuth)
        self._dir_el_slider.setValue(elevation)

    # Physics

    def _build_physics_group(self) -> QGroupBox:
        grp = QGroupBox("Physics")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(10, 30, 10, 10)
        lay.setSpacing(8)

        self._physics_status = QLabel("Load an asset, then Play or Restart physics.")
        self._physics_status.setWordWrap(True)
        self._physics_status.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        lay.addWidget(self._physics_status)

        grab_hint = QLabel("Grab: hold Shift + Left Mouse Button on the asset, drag to pull, release to drop or throw.")
        grab_hint.setWordWrap(True)
        grab_hint.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        lay.addWidget(grab_hint)

        scene_row = QHBoxLayout()
        scene_label = QLabel("Base:")
        scene_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        scene_row.addWidget(scene_label)

        self._physics_base_scene = QComboBox()
        self._physics_base_scene.addItem("Plane", "plane")
        self._physics_base_scene.addItem("Ramp", "ramp")
        self._physics_base_scene.addItem("Obstacles", "obstacles")
        self._physics_base_scene.currentIndexChanged.connect(
            lambda _idx: self.physics_base_scene_changed.emit(self._physics_base_scene.currentData())
        )
        scene_row.addWidget(self._physics_base_scene, 1)

        scene_widget = QWidget()
        scene_widget.setStyleSheet("background: transparent;")
        scene_widget.setLayout(scene_row)
        lay.addWidget(scene_widget)

        force_row = QHBoxLayout()
        force_label = QLabel("Grab force:")
        force_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        force_row.addWidget(force_label)

        self._physics_grab_force = QDoubleSpinBox()
        self._physics_grab_force.setRange(0.25, 100.0)
        self._physics_grab_force.setDecimals(2)
        self._physics_grab_force.setSingleStep(1.0)
        self._physics_grab_force.setValue(2.0)
        self._physics_grab_force.setSuffix("x")
        self._physics_grab_force.setToolTip("Higher values pull harder and need less mouse travel while grabbing.")
        self._physics_grab_force.valueChanged.connect(
            lambda value: self.physics_grab_force_changed.emit(float(value))
        )
        force_row.addWidget(self._physics_grab_force, 1)

        force_widget = QWidget()
        force_widget.setStyleSheet("background: transparent;")
        force_widget.setLayout(force_row)
        lay.addWidget(force_widget)

        self._physics_play = QCheckBox("Play physics")
        self._physics_play.toggled.connect(self.physics_play_changed)
        lay.addWidget(self._physics_play)

        self._physics_collision_vis = QCheckBox("Collision wire overlay")
        self._physics_collision_vis.toggled.connect(self.physics_collision_vis_changed)
        lay.addWidget(self._physics_collision_vis)

        self._physics_ccd = QCheckBox("CCD continuous collision")
        self._physics_ccd.setToolTip(
            "Authors scene-level PhysX continuous collision detection on the next physics scene start."
        )
        self._physics_ccd.toggled.connect(self.physics_ccd_changed)
        lay.addWidget(self._physics_ccd)

        drop_count_row = QHBoxLayout()
        drop_count_label = QLabel("Drop count:")
        drop_count_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        drop_count_row.addWidget(drop_count_label)

        self._physics_drop_count = QSpinBox()
        self._physics_drop_count.setRange(1, 100)
        self._physics_drop_count.setValue(1)
        self._physics_drop_count.setToolTip("Drop multiple copies of the loaded asset with small random offsets.")
        drop_count_row.addWidget(self._physics_drop_count, 1)

        drop_count_widget = QWidget()
        drop_count_widget.setStyleSheet("background: transparent;")
        drop_count_widget.setLayout(drop_count_row)
        lay.addWidget(drop_count_widget)

        button_row = QHBoxLayout()
        restart_btn = _small_button("Restart")
        restart_btn.clicked.connect(self.physics_restart_requested)
        button_row.addWidget(restart_btn)

        drop_btn = _small_button("Drop")
        drop_btn.clicked.connect(lambda: self.physics_drop_requested.emit(int(self._physics_drop_count.value())))
        button_row.addWidget(drop_btn)

        step_btn = _small_button("Step")
        step_btn.clicked.connect(self.physics_step_requested)
        button_row.addWidget(step_btn)

        row_widget = QWidget()
        row_widget.setStyleSheet("background: transparent;")
        row_widget.setLayout(button_row)
        lay.addWidget(row_widget)

        return grp

    # ── Materials ──────────────────────────────────────────────────────────────

    def _build_material_group(self) -> QGroupBox:
        grp = QGroupBox("Material Override")
        lay = QFormLayout(grp)
        lay.setSpacing(10)
        lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.setFormAlignment(Qt.AlignTop)
        lay.setContentsMargins(10, 30, 10, 10)

        self._roughness_slider = _make_slider(0, 100, 50)
        self._roughness_slider.valueChanged.connect(self._emit_material)
        lay.addRow(
            "Roughness:",
            _slider_with_value(self._roughness_slider, fmt=lambda v: f"{v/100:.2f}")
        )

        self._metallic_slider = _make_slider(0, 100, 0)
        self._metallic_slider.valueChanged.connect(self._emit_material)
        lay.addRow(
            "Metallic:",
            _slider_with_value(self._metallic_slider, fmt=lambda v: f"{v/100:.2f}")
        )

        preset_row = QHBoxLayout()
        for name, r, m in [("Plastic", 70, 0), ("Metal", 20, 100), ("Rubber", 90, 0), ("Chrome", 5, 100)]:
            btn = _small_button(name)
            r_, m_ = r, m
            btn.clicked.connect(lambda _, rv=r_, mv=m_: self._apply_mat_preset(rv, mv))
            preset_row.addWidget(btn)
        preset_w = QWidget()
        preset_w.setLayout(preset_row)
        preset_w.setStyleSheet("background: transparent;")
        lay.addRow("Preset:", preset_w)

        return grp

    def _emit_material(self):
        self.material_changed.emit(
            self._roughness_slider.value() / 100.0,
            self._metallic_slider.value() / 100.0,
        )

    def _apply_mat_preset(self, roughness: int, metallic: int):
        self._roughness_slider.setValue(roughness)
        self._metallic_slider.setValue(metallic)

    # ── Camera ─────────────────────────────────────────────────────────────────

    def _build_camera_group(self) -> QGroupBox:
        grp = QGroupBox("Camera")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(10, 30, 10, 10)
        lay.setSpacing(8)

        hint = QLabel(
            "LMB drag – Tumble\n"
            "MMB drag – Pan\n"
            "RMB / Scroll – Zoom\n"
            "F or Dbl-click – Reset"
        )
        hint.setText(
            "Alt+LMB drag - Tumble\n"
            "Alt+MMB drag - Pan\n"
            "Alt+RMB drag - Dolly\n"
            "RMB + WASD/QE - Fly\n"
            "F - Frame extents"
        )
        hint.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; "
            f"background: transparent; line-height: 1.6;"
        )
        lay.addWidget(hint)

        reset_btn = QPushButton("Reset Camera")
        reset_btn.setObjectName("accent_btn")
        reset_btn.clicked.connect(self.reset_camera_requested)
        lay.addWidget(reset_btn)

        return grp

    # ── Asset info ─────────────────────────────────────────────────────────────

    def _build_asset_info_group(self) -> QGroupBox:
        grp = QGroupBox("Asset Info")
        lay = QFormLayout(grp)
        lay.setSpacing(10)
        lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.setFormAlignment(Qt.AlignTop)
        lay.setContentsMargins(10, 30, 10, 10)

        def _info_label(text="—"):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {COLOR_TEXT_PRIMARY}; font-size: 11px; background: transparent;"
            )
            lbl.setWordWrap(True)
            return lbl

        self._info_name     = _info_label()
        self._info_category = _info_label()
        self._info_tags     = _info_label()
        self._info_path     = _info_label()
        self._info_path.setStyleSheet(
            f"color: {COLOR_TEXT_DISABLED}; font-size: 9px; background: transparent;"
        )

        lay.addRow(_label("Name:"),     self._info_name)
        lay.addRow(_label("Category:"), self._info_category)
        lay.addRow(_label("Tags:"),     self._info_tags)
        lay.addRow(_label("Key:"),      self._info_path)

        self._load_btn = QPushButton("Load in Viewport")
        self._load_btn.setObjectName("accent_btn")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._request_load)
        lay.addRow(self._load_btn)

        return grp

    # ── Public ─────────────────────────────────────────────────────────────────

    def update_asset_info(self, asset: AssetInfo):
        self._current_asset = asset
        self._info_name.setText(asset.display_name)
        self._info_category.setText(asset.category)
        self._info_tags.setText(", ".join(asset.tags) if asset.tags else "—")
        self._info_path.setText(asset.usd_key)
        self._load_btn.setEnabled(True)

    def set_physics_status(self, text: str):
        self._physics_status.setText(text)

    def set_physics_running(self, running: bool):
        old = self._physics_play.blockSignals(True)
        self._physics_play.setChecked(bool(running))
        self._physics_play.blockSignals(old)

    def _request_load(self):
        if self._current_asset:
            self.load_asset_requested.emit(self._current_asset)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_slider(lo: int, hi: int, default: int) -> QSlider:
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    s.setValue(default)
    return s


def _slider_with_value(slider: QSlider, fmt=None) -> QWidget:
    """Wrap a slider with a live value label beside it."""
    w   = QWidget()
    w.setStyleSheet("background: transparent;")
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(slider, 1)

    val_label = QLabel()
    val_label.setFixedWidth(36)
    val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    val_label.setStyleSheet(
        f"color: {COLOR_ACCENT}; font-size: 10px; background: transparent;"
    )

    def _update(v):
        val_label.setText(fmt(v) if fmt else str(v))

    _update(slider.value())
    slider.valueChanged.connect(_update)
    row.addWidget(val_label)
    return w


def _small_button(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(22)
    btn.setStyleSheet(
        f"QPushButton {{ background: #2d2d2d; border: 1px solid {COLOR_BORDER}; "
        f"border-radius: 3px; padding: 2px 6px; font-size: 10px; color: #c0c0c0; }}"
        f"QPushButton:hover {{ border-color: {COLOR_ACCENT}; color: {COLOR_ACCENT}; }}"
    )
    return btn


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
    )
    return lbl
