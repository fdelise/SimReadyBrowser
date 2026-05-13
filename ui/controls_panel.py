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

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
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
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.s3_client import AssetInfo
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_DIM,
    COLOR_BG_HEADER,
    COLOR_BG_HOVER,
    COLOR_BG_PANEL,
    COLOR_BG_WIDGET,
    COLOR_BORDER,
    COLOR_TEXT_DISABLED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)


PHYSX_DEFAULT_CCD = False
PHYSX_DEFAULT_STEPS_PER_SECOND = 60
PHYSX_MIN_STEPS_PER_SECOND = 15
PHYSX_MAX_STEPS_PER_SECOND = 240
PHYSX_DEFAULT_SUBSTEPS = 4
PHYSX_MIN_SUBSTEPS = 1
PHYSX_MAX_SUBSTEPS = 16
PHYSX_DEFAULT_DEVICE = "gpu"


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
    dome_environment_changed = pyqtSignal(str)
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
    physics_steps_changed = pyqtSignal(int)
    physics_substeps_changed = pyqtSignal(int)
    physics_device_changed = pyqtSignal(str)
    physics_settings_reset_requested = pyqtSignal()
    physics_engine_restart_requested = pyqtSignal()
    physics_drop_options_changed = pyqtSignal(float, float)
    scene_part_selected = pyqtSignal(str)
    scene_part_property_changed = pyqtSignal(str, str, object)
    scene_explorer_refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_asset: Optional[AssetInfo] = None
        self._scene_nodes_by_path: dict[str, dict] = {}
        self._scene_items_by_path: dict[str, QTreeWidgetItem] = {}
        self._scene_default_properties_by_path: dict[str, dict] = {}
        self._scene_edit_properties_by_path: dict[str, dict] = {}
        self._selected_scene_path = ""
        self._updating_scene_ui = False
        self._build_ui()
        self._load_fast_physx_defaults()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {COLOR_BORDER}; border-radius: 8px; "
            f"background: {COLOR_BG_PANEL}; top: -1px; }}"
            f"QTabBar::tab {{ background: {COLOR_BG_WIDGET}; color: {COLOR_TEXT_SECONDARY}; "
            f"padding: 7px 10px; border: 1px solid {COLOR_BORDER}; border-radius: 14px; "
            f"margin: 0 4px 6px 0; }}"
            f"QTabBar::tab:selected {{ color: #090a08; background: {COLOR_ACCENT}; "
            f"border-color: {COLOR_ACCENT}; }}"
            f"QTabBar::tab:hover:!selected {{ background: {COLOR_BG_HOVER}; "
            f"border-color: {COLOR_ACCENT_DIM}; color: {COLOR_TEXT_PRIMARY}; }}"
        )
        self._tabs.addTab(self._build_app_settings_tab(), "App Settings")
        self._tabs.addTab(self._build_scene_explorer_tab(), "Scene Explorer")
        root.addWidget(self._tabs)

    def _build_app_settings_tab(self) -> QWidget:
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
        return scroll

    def _build_scene_explorer_tab(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        header_row = QHBoxLayout()
        self._scene_status = QLabel("Load an asset to inspect the USD scene.")
        self._scene_status.setWordWrap(True)
        self._scene_status.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        header_row.addWidget(self._scene_status, 1)

        refresh_btn = _small_button("Refresh")
        refresh_btn.setToolTip("Refresh scene parts from the loaded USD asset.")
        refresh_btn.clicked.connect(self.scene_explorer_refresh_requested)
        header_row.addWidget(refresh_btn)

        header_widget = QWidget()
        header_widget.setStyleSheet("background: transparent;")
        header_widget.setLayout(header_row)
        lay.addWidget(header_widget)

        self._scene_tree = QTreeWidget()
        self._scene_tree.setHeaderLabels(["Prim", "Type"])
        self._scene_tree.setRootIsDecorated(True)
        self._scene_tree.setUniformRowHeights(True)
        self._scene_tree.setAlternatingRowColors(False)
        self._scene_tree.setColumnWidth(0, 185)
        self._scene_tree.setStyleSheet(
            f"QTreeWidget {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 8px; color: {COLOR_TEXT_PRIMARY}; font-size: 10px; }}"
            f"QTreeWidget::item {{ padding: 4px 5px; border-radius: 4px; }}"
            f"QTreeWidget::item:hover {{ background: {COLOR_BG_HOVER}; }}"
            f"QTreeWidget::item:selected {{ background: {COLOR_ACCENT}; color: #090a08; }}"
            f"QHeaderView::section {{ background: {COLOR_BG_HEADER}; color: {COLOR_TEXT_SECONDARY}; "
            f"border: none; border-bottom: 1px solid {COLOR_BORDER}; padding: 5px; }}"
        )
        self._scene_tree.itemSelectionChanged.connect(self._on_scene_tree_selection_changed)
        lay.addWidget(self._scene_tree, 1)

        self._scene_properties_group = QGroupBox("Selected Prim")
        prop_lay = QFormLayout(self._scene_properties_group)
        prop_lay.setSpacing(8)
        prop_lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prop_lay.setFormAlignment(Qt.AlignTop)
        prop_lay.setContentsMargins(10, 30, 10, 10)

        self._scene_name_value = self._scene_value_label()
        self._scene_type_value = self._scene_value_label()
        self._scene_path_value = self._scene_value_label(font_size=9, color=COLOR_TEXT_DISABLED)
        self._scene_physics_value = self._scene_value_label(font_size=9, color=COLOR_TEXT_DISABLED)
        prop_lay.addRow(_label("Name:"), self._scene_name_value)
        prop_lay.addRow(_label("Type:"), self._scene_type_value)
        prop_lay.addRow(_label("Path:"), self._scene_path_value)
        prop_lay.addRow(_label("Physics:"), self._scene_physics_value)

        self._scene_visible = QCheckBox("Visible")
        self._scene_visible.setToolTip("Authors a viewport visibility override on the selected prim.")
        self._scene_visible.toggled.connect(lambda checked: self._emit_scene_scalar_property("visible", bool(checked)))
        prop_lay.addRow(_label("Vis:"), self._scene_visible)

        self._scene_translate_spins, translate_row = self._scene_vector_row(-100000.0, 100000.0, 0.1, 3)
        self._scene_rotate_spins, rotate_row = self._scene_vector_row(-3600.0, 3600.0, 1.0, 2)
        self._scene_scale_spins, scale_row = self._scene_vector_row(0.001, 1000.0, 0.05, 3)
        for spin in self._scene_translate_spins:
            spin.valueChanged.connect(lambda _value: self._emit_scene_vector_property("translate"))
        for spin in self._scene_rotate_spins:
            spin.valueChanged.connect(lambda _value: self._emit_scene_vector_property("rotate"))
        for spin in self._scene_scale_spins:
            spin.valueChanged.connect(lambda _value: self._emit_scene_vector_property("scale"))
        prop_lay.addRow(_label("T:"), translate_row)
        prop_lay.addRow(_label("R:"), rotate_row)
        prop_lay.addRow(_label("S:"), scale_row)

        reset_part_btn = _small_button("Reset Prim Edits")
        reset_part_btn.setToolTip("Restore the selected prim to the transform values discovered for this scene load.")
        reset_part_btn.clicked.connect(self._reset_scene_part_edits)
        prop_lay.addRow(reset_part_btn)

        self._scene_usd_tree = QTreeWidget()
        self._scene_usd_tree.setHeaderLabels(["USD Property", "Type", "Value"])
        self._scene_usd_tree.setRootIsDecorated(True)
        self._scene_usd_tree.setUniformRowHeights(True)
        self._scene_usd_tree.setMinimumHeight(180)
        self._scene_usd_tree.setStyleSheet(
            f"QTreeWidget {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 8px; color: {COLOR_TEXT_PRIMARY}; font-size: 10px; }}"
            f"QTreeWidget::item {{ padding: 4px 5px; border-radius: 4px; }}"
            f"QTreeWidget::item:hover {{ background: {COLOR_BG_HOVER}; }}"
            f"QTreeWidget::item:selected {{ background: {COLOR_ACCENT}; color: #090a08; }}"
            f"QHeaderView::section {{ background: {COLOR_BG_HEADER}; color: {COLOR_TEXT_SECONDARY}; "
            f"border: none; border-bottom: 1px solid {COLOR_BORDER}; padding: 5px; }}"
        )
        self._scene_usd_tree.setColumnWidth(0, 135)
        self._scene_usd_tree.setColumnWidth(1, 74)
        prop_lay.addRow(self._scene_usd_tree)

        self._scene_properties_group.setEnabled(False)
        lay.addWidget(self._scene_properties_group)

        self.clear_scene_tree()
        return page

    def _scene_value_label(self, text: str = "-", font_size: int = 10, color: str = COLOR_TEXT_PRIMARY) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {color}; font-size: {font_size}px; background: transparent;")
        return lbl

    def _scene_vector_row(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
    ) -> tuple[list[QDoubleSpinBox], QWidget]:
        row_widget = QWidget()
        row_widget.setStyleSheet("background: transparent;")
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        spins: list[QDoubleSpinBox] = []
        for axis in ("X", "Y", "Z"):
            spin = QDoubleSpinBox()
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(decimals)
            spin.setSingleStep(float(step))
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spin.setMinimumWidth(58)
            spin.setToolTip(axis)
            row.addWidget(spin)
            spins.append(spin)
        return spins, row_widget

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

        self._dome_environment = QComboBox()
        self._dome_environment.addItem("Flat dome", "flat")
        self._dome_environment.addItem("Studio softbox HDRI", "blurry_studio")
        self._dome_environment.addItem("Automotive show HDRI", "automotive_show")
        self._dome_environment.addItem("Outdoor day HDRI", "outdoor_day")
        self._dome_environment.setToolTip("Use a lat-long environment texture on the dome light for reflections.")
        self._dome_environment.currentIndexChanged.connect(self._emit_dome_environment)
        lay.addRow("Env:", self._dome_environment)

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

    def _emit_dome_environment(self):
        self.dome_environment_changed.emit(str(self._dome_environment.currentData() or "flat"))

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
        self._physics_base_scene.addItem("No Ground", "none")
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
        self._physics_grab_force.setToolTip("Higher values pull harder toward the cursor while grabbing.")
        self._physics_grab_force.valueChanged.connect(
            lambda value: self.physics_grab_force_changed.emit(float(value))
        )
        force_row.addWidget(self._physics_grab_force, 1)

        force_widget = QWidget()
        force_widget.setStyleSheet("background: transparent;")
        force_widget.setLayout(force_row)
        lay.addWidget(force_widget)

        device_row = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        device_row.addWidget(device_label)

        self._physics_device = QComboBox()
        self._physics_device.addItem("Auto", "auto")
        self._physics_device.addItem("CPU", "cpu")
        self._physics_device.addItem("GPU", "gpu")
        self._physics_device.setToolTip("Selects the OVPhysX device mode used when the PhysX worker starts.")
        self._physics_device.currentIndexChanged.connect(lambda _idx: self._on_physx_device_changed())
        device_row.addWidget(self._physics_device, 1)

        device_widget = QWidget()
        device_widget.setStyleSheet("background: transparent;")
        device_widget.setLayout(device_row)
        lay.addWidget(device_widget)

        steps_row = QHBoxLayout()
        steps_label = QLabel("Steps/s:")
        steps_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        steps_row.addWidget(steps_label)

        self._physics_steps = QSpinBox()
        self._physics_steps.setRange(PHYSX_MIN_STEPS_PER_SECOND, PHYSX_MAX_STEPS_PER_SECOND)
        self._physics_steps.setValue(PHYSX_DEFAULT_STEPS_PER_SECOND)
        self._physics_steps.setSingleStep(5)
        self._physics_steps.setToolTip("Authors physxScene:timeStepsPerSecond and drives the simulation dt.")
        self._physics_steps.valueChanged.connect(lambda value: self._on_physx_steps_changed(int(value)))
        steps_row.addWidget(self._physics_steps, 1)

        substeps_label = QLabel("Sub:")
        substeps_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        steps_row.addWidget(substeps_label)

        self._physics_substeps = QSpinBox()
        self._physics_substeps.setRange(PHYSX_MIN_SUBSTEPS, PHYSX_MAX_SUBSTEPS)
        self._physics_substeps.setValue(PHYSX_DEFAULT_SUBSTEPS)
        self._physics_substeps.setSingleStep(1)
        self._physics_substeps.setToolTip("Runs this many PhysX substeps for each displayed simulation step.")
        self._physics_substeps.valueChanged.connect(lambda value: self._on_physx_substeps_changed(int(value)))
        steps_row.addWidget(self._physics_substeps, 1)

        steps_widget = QWidget()
        steps_widget.setStyleSheet("background: transparent;")
        steps_widget.setLayout(steps_row)
        lay.addWidget(steps_widget)

        self._physics_play = QCheckBox("Play physics")
        self._physics_play.toggled.connect(self.physics_play_changed)
        lay.addWidget(self._physics_play)

        self._physics_collision_vis = QCheckBox("Collision wire overlay")
        self._physics_collision_vis.toggled.connect(self.physics_collision_vis_changed)
        lay.addWidget(self._physics_collision_vis)

        self._physics_ccd = QCheckBox("CCD continuous collision")
        self._physics_ccd.setToolTip(
            "Globally authors scene-level and rigid-body PhysX continuous collision detection; changing it restarts the active physics scene."
        )
        self._physics_ccd.setChecked(PHYSX_DEFAULT_CCD)
        self._physics_ccd.toggled.connect(self._on_physx_ccd_changed)
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

        drop_options_row = QHBoxLayout()
        spacing_label = QLabel("Spacing:")
        spacing_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        drop_options_row.addWidget(spacing_label)

        self._physics_drop_spacing = QDoubleSpinBox()
        self._physics_drop_spacing.setRange(0.00, 5.00)
        self._physics_drop_spacing.setDecimals(2)
        self._physics_drop_spacing.setSingleStep(0.10)
        self._physics_drop_spacing.setValue(0.20)
        self._physics_drop_spacing.setSuffix("x")
        self._physics_drop_spacing.setToolTip("0 stacks tightly; higher values push dropped assets farther apart.")
        self._physics_drop_spacing.valueChanged.connect(lambda _value: self._emit_drop_options())
        drop_options_row.addWidget(self._physics_drop_spacing, 1)

        random_label = QLabel("Random:")
        random_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
        )
        drop_options_row.addWidget(random_label)

        self._physics_drop_random = QDoubleSpinBox()
        self._physics_drop_random.setRange(0.00, 6.00)
        self._physics_drop_random.setDecimals(2)
        self._physics_drop_random.setSingleStep(0.25)
        self._physics_drop_random.setValue(0.25)
        self._physics_drop_random.setSuffix("x")
        self._physics_drop_random.setToolTip("0 is ordered; higher values scatter dropped assets much more aggressively.")
        self._physics_drop_random.valueChanged.connect(lambda _value: self._emit_drop_options())
        drop_options_row.addWidget(self._physics_drop_random, 1)

        drop_options_widget = QWidget()
        drop_options_widget.setStyleSheet("background: transparent;")
        drop_options_widget.setLayout(drop_options_row)
        lay.addWidget(drop_options_widget)

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

        engine_row = QHBoxLayout()
        reset_settings_btn = _small_button("Reset Defaults")
        reset_settings_btn.setToolTip("Restore PhysX defaults: GPU device, 60 steps/s, 4 substeps, CCD off.")
        reset_settings_btn.clicked.connect(self._reset_physx_settings)
        engine_row.addWidget(reset_settings_btn)

        restart_engine_btn = _small_button("Restart Engine")
        restart_engine_btn.setToolTip("Stops the current OVPhysX worker and starts a new one with the selected settings.")
        restart_engine_btn.clicked.connect(self.physics_engine_restart_requested)
        engine_row.addWidget(restart_engine_btn)

        engine_widget = QWidget()
        engine_widget.setStyleSheet("background: transparent;")
        engine_widget.setLayout(engine_row)
        lay.addWidget(engine_widget)

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

    def clear_scene_tree(self, message: str = "Load an asset to inspect the USD scene.") -> None:
        self.set_scene_tree({"status": message, "roots": []})

    def set_scene_tree(self, payload) -> None:
        data = payload if isinstance(payload, dict) else {}
        roots = data.get("roots", []) if isinstance(data.get("roots", []), list) else []
        status = str(data.get("status") or "Load an asset to inspect the USD scene.")

        self._updating_scene_ui = True
        try:
            self._scene_tree.clear()
            self._scene_nodes_by_path.clear()
            self._scene_items_by_path.clear()
            self._scene_default_properties_by_path.clear()
            self._scene_edit_properties_by_path.clear()
            self._selected_scene_path = ""
            self._scene_status.setText(status)

            if not roots:
                placeholder = QTreeWidgetItem(["No scene loaded", ""])
                placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsSelectable)
                self._scene_tree.addTopLevelItem(placeholder)
                self._clear_scene_properties()
                return

            for root in roots:
                if isinstance(root, dict):
                    self._add_scene_tree_node(root)
            self._scene_tree.expandToDepth(1)
            self._scene_tree.resizeColumnToContents(1)
            self._clear_scene_properties()
        finally:
            self._updating_scene_ui = False

    def set_selected_scene_part(self, path: str) -> None:
        text = str(path or "").strip()
        item = self._scene_items_by_path.get(text)
        if item is None:
            return
        self._updating_scene_ui = True
        try:
            self._scene_tree.setCurrentItem(item)
            node = self._scene_nodes_by_path.get(text, {})
            self._selected_scene_path = text
            self._show_scene_properties(node)
        finally:
            self._updating_scene_ui = False

    def physics_settings(self) -> dict:
        return {
            "ccd_enabled": bool(self._physics_ccd.isChecked()),
            "steps_per_second": int(self._physics_steps.value()),
            "substeps": int(self._physics_substeps.value()),
            "device_mode": str(self._physics_device.currentData() or PHYSX_DEFAULT_DEVICE),
        }

    def _load_fast_physx_defaults(self) -> None:
        self._clear_persistent_physx_settings()
        self._set_check_silently(self._physics_ccd, PHYSX_DEFAULT_CCD)
        self._set_spin_silently(self._physics_steps, PHYSX_DEFAULT_STEPS_PER_SECOND)
        self._set_spin_silently(self._physics_substeps, PHYSX_DEFAULT_SUBSTEPS)
        self._set_device_silently(PHYSX_DEFAULT_DEVICE)

    @staticmethod
    def _clear_persistent_physx_settings() -> None:
        settings = QSettings("NVIDIA Corporation", "NVIDIA SimReady Browser")
        settings.remove("physics")
        settings.sync()

    def _on_physx_ccd_changed(self, enabled: bool) -> None:
        self.physics_ccd_changed.emit(bool(enabled))

    def _on_physx_steps_changed(self, value: int) -> None:
        self.physics_steps_changed.emit(int(value))

    def _on_physx_substeps_changed(self, value: int) -> None:
        self.physics_substeps_changed.emit(int(value))

    def _on_physx_device_changed(self) -> None:
        mode = str(self._physics_device.currentData() or PHYSX_DEFAULT_DEVICE)
        self.physics_device_changed.emit(mode)

    def _reset_physx_settings(self) -> None:
        self._set_check_silently(self._physics_ccd, PHYSX_DEFAULT_CCD)
        self._set_spin_silently(self._physics_steps, PHYSX_DEFAULT_STEPS_PER_SECOND)
        self._set_spin_silently(self._physics_substeps, PHYSX_DEFAULT_SUBSTEPS)
        self._set_device_silently(PHYSX_DEFAULT_DEVICE)
        self._clear_persistent_physx_settings()
        self.physics_settings_reset_requested.emit()

    def _emit_drop_options(self):
        self.physics_drop_options_changed.emit(
            float(self._physics_drop_spacing.value()),
            float(self._physics_drop_random.value()),
        )

    @staticmethod
    def _set_spin_silently(widget, value: int) -> None:
        old = widget.blockSignals(True)
        widget.setValue(int(value))
        widget.blockSignals(old)

    @staticmethod
    def _set_check_silently(widget, value: bool) -> None:
        old = widget.blockSignals(True)
        widget.setChecked(bool(value))
        widget.blockSignals(old)

    def _set_device_silently(self, value: str) -> None:
        mode = str(value or PHYSX_DEFAULT_DEVICE).strip().lower()
        if mode.startswith("cuda"):
            mode = "gpu"
        index = self._physics_device.findData(mode)
        if index < 0:
            index = self._physics_device.findData(PHYSX_DEFAULT_DEVICE)
        old = self._physics_device.blockSignals(True)
        self._physics_device.setCurrentIndex(max(0, index))
        self._physics_device.blockSignals(old)

    def _request_load(self):
        if self._current_asset:
            self.load_asset_requested.emit(self._current_asset)

    def _add_scene_tree_node(self, node: dict, parent: Optional[QTreeWidgetItem] = None) -> QTreeWidgetItem:
        name = str(node.get("name") or node.get("path") or "Prim")
        type_name = str(node.get("type") or "Xform")
        path = str(node.get("path") or "")
        item = QTreeWidgetItem([name, type_name])
        item.setData(0, Qt.UserRole, path)
        if parent is None:
            self._scene_tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        if path:
            self._scene_nodes_by_path[path] = node
            self._scene_items_by_path[path] = item
            props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
            self._scene_default_properties_by_path[path] = dict(props)
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                self._add_scene_tree_node(child, item)
        return item

    def _on_scene_tree_selection_changed(self) -> None:
        if self._updating_scene_ui:
            return
        items = self._scene_tree.selectedItems()
        if not items:
            self._clear_scene_properties()
            return
        path = str(items[0].data(0, Qt.UserRole) or "")
        node = self._scene_nodes_by_path.get(path, {})
        self._selected_scene_path = path
        self._show_scene_properties(node)
        if path:
            self.scene_part_selected.emit(path)

    def _show_scene_properties(self, node: dict) -> None:
        if not node:
            self._clear_scene_properties()
            return
        path = str(node.get("path") or "")
        props = self._scene_properties_for_path(path)
        self._scene_properties_group.setEnabled(bool(path))
        self._scene_name_value.setText(str(node.get("name") or "-"))
        self._scene_type_value.setText(str(node.get("type") or "-"))
        self._scene_path_value.setText(path or "-")
        self._scene_physics_value.setText(str(node.get("physics_path") or "-"))
        self._set_check_silently(self._scene_visible, bool(props.get("visible", True)))
        self._set_scene_vector_silently(self._scene_translate_spins, props.get("translate"), [0.0, 0.0, 0.0])
        self._set_scene_vector_silently(self._scene_rotate_spins, props.get("rotate"), [0.0, 0.0, 0.0])
        self._set_scene_vector_silently(self._scene_scale_spins, props.get("scale"), [1.0, 1.0, 1.0])
        self._populate_usd_inspector(node)

    def _clear_scene_properties(self) -> None:
        self._selected_scene_path = ""
        self._scene_properties_group.setEnabled(False)
        self._scene_name_value.setText("-")
        self._scene_type_value.setText("-")
        self._scene_path_value.setText("-")
        self._scene_physics_value.setText("-")
        self._set_check_silently(self._scene_visible, True)
        self._set_scene_vector_silently(self._scene_translate_spins, None, [0.0, 0.0, 0.0])
        self._set_scene_vector_silently(self._scene_rotate_spins, None, [0.0, 0.0, 0.0])
        self._set_scene_vector_silently(self._scene_scale_spins, None, [1.0, 1.0, 1.0])
        self._scene_usd_tree.clear()

    def _populate_usd_inspector(self, node: dict) -> None:
        self._scene_usd_tree.clear()
        usd = node.get("usd") if isinstance(node.get("usd"), dict) else {}
        properties = node.get("usd_properties") if isinstance(node.get("usd_properties"), list) else []

        summary = self._add_inspector_group("Prim Summary", f"{len(properties)} properties")
        for key in ("type_name", "specifier", "kind", "active", "defined", "loaded", "instance", "instanceable", "prototype"):
            value = usd.get(key)
            if value not in ("", None, [], {}):
                self._add_inspector_row(summary, key, "", value)
        metadata = usd.get("metadata") if isinstance(usd.get("metadata"), dict) else {}
        if metadata:
            meta_group = self._add_inspector_group("Metadata", f"{len(metadata)} items")
            self._add_mapping_rows(meta_group, metadata)

        schemas = list(usd.get("applied_schemas") or [])
        schema_group = self._add_inspector_group("Schemas", f"{len(schemas)} applied")
        concrete = usd.get("type_name")
        if concrete:
            self._add_inspector_row(schema_group, "Concrete", "schema", concrete)
        for schema in schemas:
            self._add_inspector_row(schema_group, str(schema), "apiSchema", "")

        for title, key in (("Geometry", "geometry"), ("Materials", "materials"), ("Physics", "physics")):
            data = usd.get(key) if isinstance(usd.get(key), dict) else {}
            group = self._add_inspector_group(title, f"{len(data)} items")
            self._add_mapping_rows(group, data)
            if not data:
                group.setExpanded(False)

        props_group = self._add_inspector_group("USD Properties", f"{len(properties)} attrs/rels")
        for prop in properties:
            if not isinstance(prop, dict):
                continue
            flags = []
            if prop.get("authored"):
                flags.append("authored")
            if prop.get("custom"):
                flags.append("custom")
            if prop.get("time_samples"):
                flags.append(f"{prop.get('time_samples')} samples")
            item = self._add_inspector_row(
                props_group,
                str(prop.get("name") or ""),
                str(prop.get("type") or prop.get("kind") or ""),
                str(prop.get("value") or ""),
            )
            if flags:
                self._add_inspector_row(item, "flags", "", ", ".join(flags))
            for label, values in (("connections", prop.get("connections")), ("targets", prop.get("targets"))):
                if values:
                    self._add_inspector_row(item, label, "", ", ".join(str(value) for value in values))

        for index in range(self._scene_usd_tree.topLevelItemCount()):
            self._scene_usd_tree.topLevelItem(index).setExpanded(index < 4)

    def _add_mapping_rows(self, parent: QTreeWidgetItem, mapping: dict) -> None:
        for key in sorted(mapping.keys()):
            value = mapping.get(key)
            if isinstance(value, dict):
                group = self._add_inspector_row(parent, str(key), "dict", f"{len(value)} items")
                self._add_mapping_rows(group, value)
            elif isinstance(value, (list, tuple)):
                self._add_inspector_row(parent, str(key), "list", ", ".join(str(item) for item in value))
            else:
                self._add_inspector_row(parent, str(key), "", value)

    def _add_inspector_group(self, name: str, value: str = "") -> QTreeWidgetItem:
        item = QTreeWidgetItem([str(name), "", str(value)])
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        self._scene_usd_tree.addTopLevelItem(item)
        return item

    @staticmethod
    def _add_inspector_row(parent: QTreeWidgetItem, name: str, type_name: str, value) -> QTreeWidgetItem:
        text = str(value if value is not None else "")
        item = QTreeWidgetItem([str(name), str(type_name or ""), text])
        parent.addChild(item)
        return item

    def _emit_scene_scalar_property(self, property_name: str, value) -> None:
        if self._updating_scene_ui or not self._selected_scene_path:
            return
        props = self._scene_properties_for_path(self._selected_scene_path)
        props[str(property_name)] = value
        self._scene_edit_properties_by_path[self._selected_scene_path] = props
        self.scene_part_property_changed.emit(self._selected_scene_path, str(property_name), value)

    def _emit_scene_vector_property(self, property_name: str) -> None:
        if self._updating_scene_ui or not self._selected_scene_path:
            return
        spins = {
            "translate": self._scene_translate_spins,
            "rotate": self._scene_rotate_spins,
            "scale": self._scene_scale_spins,
        }.get(property_name)
        if not spins:
            return
        values = [float(spin.value()) for spin in spins]
        props = self._scene_properties_for_path(self._selected_scene_path)
        props[property_name] = values
        self._scene_edit_properties_by_path[self._selected_scene_path] = props
        self.scene_part_property_changed.emit(
            self._selected_scene_path,
            property_name,
            values,
        )

    def _reset_scene_part_edits(self) -> None:
        if not self._selected_scene_path:
            return
        self._scene_edit_properties_by_path.pop(self._selected_scene_path, None)
        node = self._scene_nodes_by_path.get(self._selected_scene_path, {})
        self._show_scene_properties(node)
        props = self._scene_properties_for_path(self._selected_scene_path)
        self.scene_part_property_changed.emit(self._selected_scene_path, "reset", dict(props))

    def _scene_properties_for_path(self, path: str) -> dict:
        text = str(path or "")
        if text in self._scene_edit_properties_by_path:
            return dict(self._scene_edit_properties_by_path[text])
        if text in self._scene_default_properties_by_path:
            return dict(self._scene_default_properties_by_path[text])
        node = self._scene_nodes_by_path.get(text, {})
        props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        return dict(props)

    @staticmethod
    def _set_scene_vector_silently(spins: list[QDoubleSpinBox], value, default: list[float]) -> None:
        try:
            values = list(value if value is not None else default)
        except TypeError:
            values = list(default)
        values = (values + list(default))[:3]
        for spin, item in zip(spins, values):
            old = spin.blockSignals(True)
            try:
                spin.setValue(float(item))
            except Exception:
                spin.setValue(float(default[0]))
            spin.blockSignals(old)


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
    btn.setFixedHeight(24)
    btn.setStyleSheet(
        f"QPushButton {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
        f"border-radius: 12px; padding: 3px 8px; font-size: 10px; "
        f"font-weight: 700; color: {COLOR_TEXT_SECONDARY}; }}"
        f"QPushButton:hover {{ background: {COLOR_BG_HOVER}; border-color: {COLOR_ACCENT_DIM}; "
        f"color: {COLOR_ACCENT}; }}"
    )
    return btn


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; background: transparent;"
    )
    return lbl
