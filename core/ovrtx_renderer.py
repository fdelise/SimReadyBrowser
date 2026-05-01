"""Qt-friendly wrapper around the NVIDIA OVRTX Python API.

OVRTX renders named USD RenderProduct prims. This wrapper loads a selected USD
asset, injects a small review layer containing a camera, lights, and a render
product, then maps the LdrColor output back to a QImage for the Qt viewport.
"""

from __future__ import annotations

import math
import gc
import hashlib
import json
import os
import random
import re
import struct
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QCoreApplication, QObject, QProcess, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from core.camera_controller import SphericalCamera

ovrtx = None  # type: ignore
Device = PrimMode = Renderer = RendererConfig = Semantic = None  # type: ignore
OVRTX_AVAILABLE: Optional[bool] = None
OVRTX_IMPORT_ERROR: Optional[BaseException] = None


REVIEW_ROOT = "/SimReadyReview"
ASSET_ROOT = "/SimReadyAsset"
ASSET_ROOT_NAME = "SimReadyAsset"
PHYSICS_ASSET_ROOT = "/World/Asset"
PHYSICS_ASSET_ROOT_NAME = "Asset"
STAGE_SETTINGS_PATH = "/SimReadyStageSettings"
CAMERA_PATH = f"{REVIEW_ROOT}/Camera"
DOME_LIGHT_PATH = f"{REVIEW_ROOT}/DomeLight"
KEY_LIGHT_PATH = f"{REVIEW_ROOT}/KeyLight"
RENDER_PRODUCT_PATH = f"{REVIEW_ROOT}/Render/Viewport"
BASE_RAMP_PATH = f"{REVIEW_ROOT}/Ramp"
BASE_OBSTACLE_PATHS = [
    f"{REVIEW_ROOT}/ObstacleA",
    f"{REVIEW_ROOT}/ObstacleB",
    f"{REVIEW_ROOT}/ObstacleC",
]
COLLISION_ASSET_PATH = f"{REVIEW_ROOT}/CollisionAssetProxy"
COLLISION_ASSET_OVERLAY_ROOT = f"{REVIEW_ROOT}/CollisionAssetOverlay"
COLLISION_GROUND_PATH = f"{REVIEW_ROOT}/CollisionGround"
COLLISION_RAMP_PATH = f"{REVIEW_ROOT}/CollisionRamp"
COLLISION_OBSTACLE_PATHS = [
    f"{REVIEW_ROOT}/CollisionObstacleA",
    f"{REVIEW_ROOT}/CollisionObstacleB",
    f"{REVIEW_ROOT}/CollisionObstacleC",
]
DEFAULT_FRAME_BOUNDS = {"center": [0.0, 0.0, 0.0], "extent": 1.0}
STAGE_UP_AXIS = "Z"
STAGE_METERS_PER_UNIT = 1
GROUND_PLANE_HALF_SIZE = 20.0
GROUND_PLANE_Z = -0.005
GROUND_COLLIDER_THICKNESS = 0.5
BASE_SCENES = {"plane", "ramp", "obstacles"}
HIDDEN_BASE_Z = -10000.0
MAX_ASSET_INSTANCES = 100
ENABLE_OVRTX_DEBUG_BOUNDS = os.environ.get("SIMREADY_OVRTX_DEBUG_BOUNDS") == "1"
USD_DISCOVERY_PYTHON_ENV = "SIMREADY_USD_PYTHON"


def _ensure_ovrtx_available() -> bool:
    """Import OVRTX only when the viewport actually needs RTX rendering."""
    global ovrtx, Device, PrimMode, Renderer, RendererConfig, Semantic
    global OVRTX_AVAILABLE, OVRTX_IMPORT_ERROR

    if OVRTX_AVAILABLE is not None:
        return OVRTX_AVAILABLE

    try:
        import ovrtx as ovrtx_module  # type: ignore
        from ovrtx import Device as DeviceClass  # type: ignore
        from ovrtx import PrimMode as PrimModeClass  # type: ignore
        from ovrtx import Renderer as RendererClass  # type: ignore
        from ovrtx import RendererConfig as RendererConfigClass  # type: ignore
        from ovrtx import Semantic as SemanticClass  # type: ignore

        ovrtx = ovrtx_module
        Device = DeviceClass
        PrimMode = PrimModeClass
        Renderer = RendererClass
        RendererConfig = RendererConfigClass
        Semantic = SemanticClass
        OVRTX_AVAILABLE = True
    except Exception as exc:
        OVRTX_IMPORT_ERROR = exc
        OVRTX_AVAILABLE = False

    return OVRTX_AVAILABLE


class OVRTXRenderer(QObject):
    """Owns a single OVRTX renderer on a background Qt thread."""

    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    bounds_ready = pyqtSignal(object)
    loading_started = pyqtSignal(str)
    loading_progress = pyqtSignal(int, str)
    loading_finished = pyqtSignal(bool, str)

    _load_stage_requested = pyqtSignal(str)
    _load_stage_items_requested = pyqtSignal(object)
    _resolution_requested = pyqtSignal(int, int)
    _camera_transform_requested = pyqtSignal(object)
    _asset_transform_requested = pyqtSignal(object)
    _asset_instance_count_requested = pyqtSignal(int)
    _physics_body_transforms_requested = pyqtSignal(object)
    _base_scene_requested = pyqtSignal(str)
    _collision_overlay_requested = pyqtSignal(bool)
    _collision_bounds_requested = pyqtSignal(object)
    _dome_intensity_requested = pyqtSignal(float)
    _dir_light_requested = pyqtSignal(float, float, float)
    _render_requested = pyqtSignal()
    _start_realtime_requested = pyqtSignal(int)
    _stop_realtime_requested = pyqtSignal()
    _shutdown_requested = pyqtSignal()

    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720

    def __init__(self, parent=None):
        super().__init__(None)

        self._renderer = None
        self._stage_loaded = False
        self._pending_stage: Optional[str] = None
        self._pending_stage_items: Optional[list[dict]] = None
        self._render_timer: Optional[QTimer] = None
        self._placeholder_timer: Optional[QTimer] = None
        self._placeholder_frame = 0
        self._shutdown_started = False

        self._width = self.DEFAULT_WIDTH
        self._height = self.DEFAULT_HEIGHT
        self._delta_time = 1.0 / 60.0

        self._camera_transform = SphericalCamera().get_transform()
        self._camera_dirty = True
        self._asset_transform = np.eye(4, dtype=np.float64)
        self._asset_transform_dirty = True
        self._asset_layout_transforms: list[np.ndarray] = []
        self._stage_item_sources: list[str] = []
        self._asset_transform_warning_shown = False
        self._asset_instance_count = 1
        self._loaded_asset_instance_count = 1
        self._asset_instance_warning_shown = False
        self._physics_body_transforms: list[tuple[str, np.ndarray]] = []
        self._physics_body_transform_dirty = False
        self._physics_body_reset_paths: set[str] = set()
        self._physics_body_transform_warning_shown = False
        self._base_scene = "plane"
        self._base_scene_dirty = True
        self._collision_overlay_enabled = False
        self._collision_proxy_bounds: Optional[dict] = None
        self._collision_asset_overlay_loaded = False
        self._collision_asset_overlay_path: Optional[Path] = None
        self._collision_overlay_process: Optional[QProcess] = None
        self._collision_overlay_process_asset: Optional[str] = None
        self._collision_overlay_process_path: Optional[Path] = None
        self._collision_overlay_stats_path: Optional[Path] = None
        self._collision_overlay_buffer = ""
        self._collision_overlay_error = ""
        self._collision_overlay_build_announced = False
        self._collision_overlay_dirty = True
        self._collision_overlay_warning_shown = False
        self._current_usd_source: Optional[str] = None
        self._dome_intensity = 1.0
        self._dir_intensity = 0.8
        self._dir_azimuth = 45.0
        self._dir_elevation = 60.0

        self._frame_times: list[float] = []
        self._startup_timer_start = time.perf_counter()

        self._thread = QThread()
        self._connect_requests()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._init_renderer, Qt.QueuedConnection)
        self._thread.start()

    def _connect_requests(self) -> None:
        self._load_stage_requested.connect(self._load_stage, Qt.QueuedConnection)
        self._load_stage_items_requested.connect(self._load_stage_items, Qt.QueuedConnection)
        self._resolution_requested.connect(self._set_resolution, Qt.QueuedConnection)
        self._camera_transform_requested.connect(self._set_camera_transform, Qt.QueuedConnection)
        self._asset_transform_requested.connect(self._set_asset_transform, Qt.QueuedConnection)
        self._asset_instance_count_requested.connect(self._set_asset_instance_count, Qt.QueuedConnection)
        self._physics_body_transforms_requested.connect(self._set_physics_body_transforms, Qt.QueuedConnection)
        self._base_scene_requested.connect(self._set_base_scene, Qt.QueuedConnection)
        self._collision_overlay_requested.connect(self._set_collision_overlay_enabled, Qt.QueuedConnection)
        self._collision_bounds_requested.connect(self._set_collision_proxy_bounds, Qt.QueuedConnection)
        self._dome_intensity_requested.connect(self._set_dome_intensity, Qt.QueuedConnection)
        self._dir_light_requested.connect(self._set_directional_light, Qt.QueuedConnection)
        self._render_requested.connect(self._render_one, Qt.QueuedConnection)
        self._start_realtime_requested.connect(self._start_timer, Qt.QueuedConnection)
        self._stop_realtime_requested.connect(self._stop_timer, Qt.QueuedConnection)
        self._shutdown_requested.connect(self._shutdown_renderer, Qt.QueuedConnection)

    def load_stage(self, usd_source: str | Path) -> None:
        if self._shutdown_started:
            return
        self._load_stage_requested.emit(str(usd_source))

    def load_stage_items(self, items) -> None:
        if self._shutdown_started:
            return
        self._load_stage_items_requested.emit(self._normalize_stage_items(items))

    def set_resolution(self, width: int, height: int) -> None:
        if self._shutdown_started:
            return
        self._resolution_requested.emit(max(1, int(width)), max(1, int(height)))

    def set_camera_transform(self, matrix: np.ndarray) -> None:
        if self._shutdown_started:
            return
        self._camera_transform_requested.emit(np.array(matrix, dtype=np.float64, copy=True))

    def set_asset_transform(self, matrix: np.ndarray) -> None:
        if self._shutdown_started:
            return
        try:
            arr = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
        except Exception:
            return
        if not np.all(np.isfinite(arr)):
            return
        self._asset_transform_requested.emit(arr)

    def set_asset_instance_count(self, count: int) -> None:
        if self._shutdown_started:
            return
        try:
            value = int(count)
        except Exception:
            value = 1
        self._asset_instance_count_requested.emit(max(1, min(MAX_ASSET_INSTANCES, value)))

    def set_physics_body_transforms(self, bodies) -> None:
        if self._shutdown_started:
            return
        self._physics_body_transforms_requested.emit(bodies)

    def set_base_scene(self, scene_id: str) -> None:
        if self._shutdown_started:
            return
        self._base_scene_requested.emit(str(scene_id or "plane"))

    def set_collision_overlay_enabled(self, enabled: bool) -> None:
        if self._shutdown_started:
            return
        self._collision_overlay_requested.emit(bool(enabled))

    def set_collision_proxy_bounds(self, bounds: dict) -> None:
        if self._shutdown_started:
            return
        self._collision_bounds_requested.emit(dict(bounds or {}))

    def set_dome_intensity(self, value: float) -> None:
        if self._shutdown_started:
            return
        self._dome_intensity_requested.emit(float(value))

    def set_directional_light(self, intensity: float, azimuth: float, elevation: float) -> None:
        if self._shutdown_started:
            return
        self._dir_light_requested.emit(float(intensity), float(azimuth), float(elevation))

    def request_render(self) -> None:
        if self._shutdown_started:
            return
        self._render_requested.emit()

    def start_realtime(self, target_fps: int = 60) -> None:
        self._start_realtime_requested.emit(max(1, int(target_fps)))

    def stop_realtime(self) -> None:
        self._stop_realtime_requested.emit()

    def shutdown(self, timeout_ms: int = 20000) -> bool:
        if not self._thread.isRunning():
            return True
        self._shutdown_started = True
        self._shutdown_requested.emit()
        if self._thread.wait(timeout_ms):
            return True
        return False

    def _init_renderer(self) -> None:
        if not _ensure_ovrtx_available():
            detail = f" ({OVRTX_IMPORT_ERROR})" if OVRTX_IMPORT_ERROR else ""
            msg = f"ovrtx is not installed; showing preview mode.{detail}"
            self.status_changed.emit(msg)
            if self._pending_stage or self._pending_stage_items:
                self.loading_finished.emit(False, msg)
                self._pending_stage = None
                self._pending_stage_items = None
            self._start_placeholder_timer()
            return

        try:
            self.status_changed.emit("Initializing OVRTX renderer. First launch can compile shaders.")
            cfg = RendererConfig(
                active_cuda_gpus="0",
                log_level="warn",
                keep_system_alive=True,
            )
            self._renderer = Renderer(cfg)
            elapsed = max(0.0, time.perf_counter() - self._startup_timer_start)
            self.status_changed.emit(f"OVRTX renderer ready in {elapsed:.2f}s; waiting for first frame.")
            if self._pending_stage_items:
                pending_items = self._pending_stage_items
                self._pending_stage_items = None
                self._pending_stage = None
                self._load_stage_items(pending_items)
            elif self._pending_stage:
                pending_stage = self._pending_stage
                self._pending_stage = None
                self._load_stage(pending_stage)
        except Exception as exc:
            if self._pending_stage or self._pending_stage_items:
                msg = f"OVRTX init failed before loading asset: {exc}"
                self.loading_finished.emit(False, msg)
                self._pending_stage = None
                self._pending_stage_items = None
            self.error_occurred.emit(f"OVRTX init failed: {exc}")
            self._start_placeholder_timer()

    def _load_stage(self, usd_source: str) -> None:
        self._load_stage_items([{"source": str(usd_source)}])

    def _load_stage_items(self, items) -> None:
        if self._shutdown_started:
            return

        stage_items = self._normalize_stage_items(items)
        if not stage_items:
            msg = "No USD assets were selected."
            self.error_occurred.emit(msg)
            self.loading_finished.emit(False, msg)
            return

        count = len(stage_items)
        display_name = stage_items[0]["name"] if count == 1 else f"{count} selected assets"
        if not self._renderer:
            if OVRTX_AVAILABLE is not False:
                self._pending_stage_items = stage_items
                self._pending_stage = stage_items[0]["source"] if count == 1 else None
                msg = "Waiting for OVRTX renderer to finish initializing..."
                self.loading_started.emit(f"Loading {display_name} in OVRTX...")
                self.loading_progress.emit(0, msg)
                self.status_changed.emit(msg)
            else:
                msg = "OVRTX renderer is not available."
                self.error_occurred.emit(msg)
                self.loading_finished.emit(False, msg)
            return

        try:
            self._stage_loaded = False
            self._current_usd_source = stage_items[0]["source"] if count == 1 else None
            self._stage_item_sources = [item["source"] for item in stage_items]
            self._asset_transform = np.eye(4, dtype=np.float64)
            self._asset_transform_dirty = True
            self._asset_layout_transforms = []
            self._asset_transform_warning_shown = False
            self._asset_instance_count = count
            self._loaded_asset_instance_count = 0
            self._asset_instance_warning_shown = False
            self._physics_body_transforms = []
            self._physics_body_transform_dirty = False
            self._physics_body_reset_paths = set()
            self._physics_body_transform_warning_shown = False
            self._release_collision_overlay_process()
            self._collision_asset_overlay_loaded = False
            self._collision_asset_overlay_path = None
            self._collision_proxy_bounds = None
            self._collision_overlay_build_announced = False
            self._base_scene_dirty = True
            self._collision_overlay_dirty = True
            self._collision_overlay_warning_shown = False
            self.loading_started.emit(f"Loading {display_name} in OVRTX...")
            self.loading_progress.emit(5, "Preparing OVRTX stage...")
            self.status_changed.emit(f"Loading {display_name}...")

            self.loading_progress.emit(15, "Clearing previous stage...")
            self._renderer.reset_stage()
            self._renderer.reset(time=0.0)

            self.loading_progress.emit(25, "Applying Isaac Z-up stage settings...")
            self._renderer.add_usd_layer(self._stage_settings_layer())

            for index, item in enumerate(stage_items):
                progress = 35 + int(25 * index / max(1, count))
                self.loading_progress.emit(progress, f"Streaming {index + 1} of {count}: {item['name']}")
                self._renderer.add_usd(item["source"], path_prefix=self._asset_render_root(index))
                self._loaded_asset_instance_count = index + 1

            self.loading_progress.emit(60, "Adding review camera and lights...")
            self._renderer.add_usd_layer(self._review_layer(), path_prefix=REVIEW_ROOT)

            bounds_entries: list[dict] = []
            if count > 1:
                self.loading_progress.emit(68, "Reading selected asset bounds...")
                for item in stage_items:
                    bounds_entries.append(self._read_stage_bounds(item["source"]) or DEFAULT_FRAME_BOUNDS.copy())
                self._asset_layout_transforms = self._layout_asset_transforms(bounds_entries)

            self._stage_loaded = True
            self._camera_dirty = True
            self.loading_progress.emit(76, "Applying viewport settings...")
            self._apply_resolution()
            self._apply_asset_transform()
            self._apply_physics_body_transforms()
            self._apply_base_scene()
            self._apply_collision_overlay()
            self._apply_dome_light()
            self._apply_directional_light()

            self.loading_progress.emit(86, "Framing asset bounds...")
            if count == 1:
                bounds = dict(self._read_stage_bounds(stage_items[0]["source"]) or DEFAULT_FRAME_BOUNDS.copy())
                bounds["_usd_source"] = stage_items[0]["source"]
            else:
                bounds = self._combined_layout_bounds(bounds_entries, self._asset_layout_transforms)
                bounds["_multi_asset"] = True
                bounds["_asset_count"] = count
                bounds["_asset_sources"] = [item["source"] for item in stage_items]
                bounds["_asset_names"] = [item["name"] for item in stage_items]
                bounds["_asset_bounds"] = bounds_entries
                bounds["_asset_layout_transforms"] = [
                    np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()
                    for matrix in self._asset_layout_transforms
                ]

            self._frame_initial_camera(bounds)
            self.bounds_ready.emit(bounds)

            self.loading_progress.emit(95, "Rendering first frame...")
            self._render_one()
            self.loading_progress.emit(100, f"Loaded {display_name}")
            self.status_changed.emit(f"Loaded {display_name}")
            self.loading_finished.emit(True, f"Loaded {display_name}")
        except Exception as exc:
            msg = f"Stage load error: {exc}"
            self.error_occurred.emit(msg)
            self.loading_finished.emit(False, msg)

    def _stage_settings_layer(self) -> str:
        return f"""#usda 1.0
(
    defaultPrim = "SimReadyStageSettings"
    metersPerUnit = {STAGE_METERS_PER_UNIT}
    upAxis = "{STAGE_UP_AXIS}"
)

def Scope "SimReadyStageSettings"
{{
}}
"""

    def _base_visual_layer(self) -> str:
        return f"""
    def Mesh "Ramp" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        uniform bool doubleSided = true
        int[] faceVertexCounts = [4, 4, 4, 3, 3]
        int[] faceVertexIndices = [0, 1, 2, 3, 0, 3, 5, 4, 1, 4, 5, 2, 0, 4, 1, 3, 2, 5]
        rel material:binding = </SimReadyReview/Materials/RampMat>
        point3f[] points = [
            (-1.7, -1.1, 0.0),
            (1.7, -1.1, 0.0),
            (1.7, 1.1, 0.0),
            (-1.7, 1.1, 0.0),
            (1.7, -1.1, 1.9),
            (1.7, 1.1, 1.9)
        ]
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, {HIDDEN_BASE_Z}, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
        uniform token subdivisionScheme = "none"
    }}

{self._base_box_mesh("ObstacleA")}

{self._base_box_mesh("ObstacleB")}

{self._base_box_mesh("ObstacleC")}
"""

    @staticmethod
    def _base_box_mesh(name: str) -> str:
        return f"""    def Mesh "{name}" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        uniform bool doubleSided = true
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [
            0, 1, 2, 3,
            4, 7, 6, 5,
            0, 4, 5, 1,
            1, 5, 6, 2,
            2, 6, 7, 3,
            3, 7, 4, 0
        ]
        rel material:binding = </SimReadyReview/Materials/ObstacleMat>
        point3f[] points = [
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5)
        ]
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, {HIDDEN_BASE_Z}, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
        uniform token subdivisionScheme = "none"
    }}"""

    def _collision_visual_layer(self) -> str:
        return f"""
    over "CollisionAssetOverlay" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    )
    {{
        bool omni:rtx:wireframe:enabled = 1
        token omni:rtx:wireframe:mode = "emissive"
        bool omni:rtx:wireframe:shading:enabled = 0
        bool omni:rtx:wireframe:perPrimThicknessWorldSpace = 1
        float omni:rtx:wireframe:thickness = 0.006
    }}

{self._edge_box_curves("CollisionAssetProxy")}

{self._edge_box_curves("CollisionGround")}

{self._edge_prism_curves("CollisionRamp")}

{self._edge_box_curves("CollisionObstacleA")}

{self._edge_box_curves("CollisionObstacleB")}

{self._edge_box_curves("CollisionObstacleC")}
"""

    @classmethod
    def _edge_box_curves(cls, name: str) -> str:
        corners = {
            "000": (-0.5, -0.5, -0.5),
            "100": (0.5, -0.5, -0.5),
            "110": (0.5, 0.5, -0.5),
            "010": (-0.5, 0.5, -0.5),
            "001": (-0.5, -0.5, 0.5),
            "101": (0.5, -0.5, 0.5),
            "111": (0.5, 0.5, 0.5),
            "011": (-0.5, 0.5, 0.5),
        }
        edges = [
            (corners["000"], corners["100"]),
            (corners["100"], corners["110"]),
            (corners["110"], corners["010"]),
            (corners["010"], corners["000"]),
            (corners["001"], corners["101"]),
            (corners["101"], corners["111"]),
            (corners["111"], corners["011"]),
            (corners["011"], corners["001"]),
            (corners["000"], corners["001"]),
            (corners["100"], corners["101"]),
            (corners["110"], corners["111"]),
            (corners["010"], corners["011"]),
        ]
        return cls._edge_curves(name, edges, thickness=0.012)

    @classmethod
    def _edge_prism_curves(cls, name: str) -> str:
        p0 = (-1.7, -1.1, 0.0)
        p1 = (1.7, -1.1, 0.0)
        p2 = (1.7, 1.1, 0.0)
        p3 = (-1.7, 1.1, 0.0)
        p4 = (1.7, -1.1, 1.9)
        p5 = (1.7, 1.1, 1.9)
        edges = [
            (p0, p1),
            (p1, p4),
            (p4, p0),
            (p3, p2),
            (p2, p5),
            (p5, p3),
            (p0, p3),
            (p1, p2),
            (p4, p5),
        ]
        return cls._edge_curves(name, edges, thickness=0.025)

    @classmethod
    def _edge_curves(cls, name: str, edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]], thickness: float) -> str:
        point_lines = ",\n            ".join(
            f"({cls._usd_float(coord[0])}, {cls._usd_float(coord[1])}, {cls._usd_float(coord[2])})"
            for edge in edges
            for coord in edge
        )
        curve_counts = ", ".join("2" for _ in edges)
        width = max(float(thickness), 0.0005)

        return f"""    def BasisCurves "{name}" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding = </SimReadyReview/Materials/CollisionMat>
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [{curve_counts}]
        point3f[] points = [
            {point_lines}
        ]
        float[] widths = [{cls._usd_float(width)}] (
            interpolation = "constant"
        )
        color3f[] primvars:displayColor = [(0.46, 0.95, 0)] (
            interpolation = "constant"
        )
        float[] primvars:displayOpacity = [1] (
            interpolation = "constant"
        )
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, {HIDDEN_BASE_Z}, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}"""

    @staticmethod
    def _usd_float(value: float) -> str:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        return f"{value:.9g}"

    def _review_layer(self) -> str:
        return f"""#usda 1.0
(
    defaultPrim = "SimReadyReview"
    metersPerUnit = {STAGE_METERS_PER_UNIT}
    upAxis = "{STAGE_UP_AXIS}"
)

def Xform "SimReadyReview"
{{
    def Scope "Materials"
    {{
        def Material "GroundMat"
        {{
            token outputs:surface.connect = </SimReadyReview/Materials/GroundMat/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.42, 0.42, 0.39)
                float inputs:metallic = 0
                float inputs:roughness = 0.82
                token outputs:surface
            }}
        }}

        def Material "RampMat"
        {{
            token outputs:surface.connect = </SimReadyReview/Materials/RampMat/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.48, 0.50, 0.47)
                float inputs:metallic = 0
                float inputs:roughness = 0.78
                token outputs:surface
            }}
        }}

        def Material "ObstacleMat"
        {{
            token outputs:surface.connect = </SimReadyReview/Materials/ObstacleMat/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.30, 0.36, 0.34)
                float inputs:metallic = 0
                float inputs:roughness = 0.8
                token outputs:surface
            }}
        }}

        def Material "CollisionMat"
        {{
            token outputs:surface.connect = </SimReadyReview/Materials/CollisionMat/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.46, 0.73, 0.0)
                color3f inputs:emissiveColor = (0.46, 0.95, 0.0)
                float inputs:metallic = 0
                float inputs:roughness = 0.25
                float inputs:opacity = 1
                token outputs:surface
            }}
        }}
    }}

    def Mesh "GroundPlane" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        uniform bool doubleSided = true
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </SimReadyReview/Materials/GroundMat>
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
        point3f[] points = [
            ({-GROUND_PLANE_HALF_SIZE}, {-GROUND_PLANE_HALF_SIZE}, {GROUND_PLANE_Z}),
            ({GROUND_PLANE_HALF_SIZE}, {-GROUND_PLANE_HALF_SIZE}, {GROUND_PLANE_Z}),
            ({GROUND_PLANE_HALF_SIZE}, {GROUND_PLANE_HALF_SIZE}, {GROUND_PLANE_Z}),
            ({-GROUND_PLANE_HALF_SIZE}, {GROUND_PLANE_HALF_SIZE}, {GROUND_PLANE_Z})
        ]
        uniform token subdivisionScheme = "none"
    }}

{self._base_visual_layer()}

{self._collision_visual_layer()}

    def Camera "Camera"
    {{
        float2 clippingRange = (0.01, 1000000)
        float focalLength = 24
        float focusDistance = 10
        float fStop = 0
        float horizontalAperture = 20.955
        token projection = "perspective"
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (3, 2, 3, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def DomeLight "DomeLight"
    {{
        color3f inputs:color = (1, 1, 1)
        float inputs:intensity = 1000
        token inputs:texture:format = "latlong"
    }}

    def DistantLight "KeyLight"
    {{
        float angle = 0.53
        color3f color = (1, 1, 1)
        float intensity = 2500
        color3f inputs:color = (1, 1, 1)
        float inputs:angle = 0.53
        float inputs:intensity = 2500
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def "Render"
    {{
        def RenderProduct "Viewport"
        {{
            rel camera = <{CAMERA_PATH}>
            rel orderedVars = <{REVIEW_ROOT}/Render/Vars/LdrColor>
            token omni:rtx:background:source:type = "domeLight"
            token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
            uniform int2 resolution = ({self._width}, {self._height})
        }}

        def "Vars"
        {{
            def RenderVar "LdrColor"
            {{
                uniform string sourceName = "LdrColor"
            }}
        }}
    }}
}}
"""

    def _set_resolution(self, width: int, height: int) -> None:
        if self._shutdown_started:
            return
        if self._width == width and self._height == height:
            return
        self._width = width
        self._height = height
        self._apply_resolution()

    def _set_camera_transform(self, matrix: np.ndarray) -> None:
        if self._shutdown_started:
            return
        self._camera_transform = np.array(matrix, dtype=np.float64, copy=True)
        self._camera_dirty = True

    def _set_asset_transform(self, matrix: np.ndarray) -> None:
        if self._shutdown_started:
            return
        try:
            transform = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
        except Exception:
            return
        if not np.all(np.isfinite(transform)):
            return
        self._asset_transform = transform
        self._asset_transform_dirty = True
        self._collision_overlay_dirty = True
        self._apply_asset_transform()
        self._apply_collision_overlay()

    def _set_asset_instance_count(self, count: int) -> None:
        if self._shutdown_started:
            return
        try:
            value = int(count)
        except Exception:
            value = 1
        self._asset_instance_count = max(1, min(MAX_ASSET_INSTANCES, value))
        self._asset_transform_dirty = True
        self._physics_body_transform_dirty = True
        self._ensure_asset_instances()
        self._apply_asset_transform()
        self._apply_physics_body_transforms()
        self._apply_collision_overlay()

    def _set_physics_body_transforms(self, bodies) -> None:
        if self._shutdown_started:
            return
        self._physics_body_transforms = self._normalize_physics_body_transforms(bodies)
        self._physics_body_transform_dirty = True
        self._collision_overlay_dirty = True
        self._apply_physics_body_transforms()
        self._apply_collision_overlay()

    def _set_base_scene(self, scene_id: str) -> None:
        if self._shutdown_started:
            return
        scene = str(scene_id or "plane").lower()
        if scene not in BASE_SCENES:
            scene = "plane"
        self._base_scene = scene
        self._base_scene_dirty = True
        self._collision_overlay_dirty = True
        self._apply_base_scene()
        self._apply_collision_overlay()

    def _set_collision_overlay_enabled(self, enabled: bool) -> None:
        if self._shutdown_started:
            return
        self._collision_overlay_enabled = bool(enabled)
        self._collision_overlay_dirty = True
        self._apply_collision_overlay()

    def _set_collision_proxy_bounds(self, bounds: dict) -> None:
        if self._shutdown_started:
            return
        self._collision_proxy_bounds = self._normalize_collision_bounds(bounds)
        self._collision_overlay_dirty = True
        self._apply_collision_overlay()

    def _set_dome_intensity(self, value: float) -> None:
        if self._shutdown_started:
            return
        self._dome_intensity = value
        self._apply_dome_light()

    def _set_directional_light(self, intensity: float, azimuth: float, elevation: float) -> None:
        if self._shutdown_started:
            return
        self._dir_intensity = intensity
        self._dir_azimuth = azimuth
        self._dir_elevation = elevation
        self._apply_directional_light()

    def _apply_resolution(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return
        try:
            self._renderer.write_attribute(
                prim_paths=[RENDER_PRODUCT_PATH],
                attribute_name="resolution",
                tensor=np.array([[self._width, self._height]], dtype=np.int32),
                prim_mode=PrimMode.MUST_EXIST,
            )
        except Exception as exc:
            self.status_changed.emit(f"Resolution update skipped: {exc}")

    def _apply_camera(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return
        self._renderer.write_attribute(
            prim_paths=[CAMERA_PATH],
            attribute_name="omni:xform",
            tensor=self._camera_transform.reshape(1, 4, 4),
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._camera_dirty = False

    def _ensure_asset_instances(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return
        target = max(1, min(MAX_ASSET_INSTANCES, int(self._asset_instance_count)))
        while self._loaded_asset_instance_count < target:
            index = self._loaded_asset_instance_count
            source = self._asset_source_for_instance(index)
            if not source:
                return
            try:
                self._renderer.add_usd(source, path_prefix=self._asset_render_root(index))
                self._loaded_asset_instance_count += 1
            except Exception as exc:
                if not self._asset_instance_warning_shown:
                    self._asset_instance_warning_shown = True
                    self.status_changed.emit(f"Additional asset instance load skipped: {exc}")
                break

    def _asset_source_for_instance(self, index: int) -> str:
        if self._current_usd_source:
            return self._current_usd_source
        if not self._stage_item_sources:
            return ""
        try:
            source_index = max(0, int(index)) % len(self._stage_item_sources)
        except Exception:
            source_index = 0
        return str(self._stage_item_sources[source_index] or "")

    def _apply_asset_transform(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return
        try:
            self._ensure_asset_instances()
            paths = []
            matrices = []
            hidden = self._hidden_base_transform()
            for index in range(max(1, self._loaded_asset_instance_count)):
                paths.append(self._asset_render_root(index))
                if index < self._asset_instance_count and index < len(self._asset_layout_transforms):
                    matrices.append(np.array(self._asset_layout_transforms[index], dtype=np.float64, copy=True).reshape(4, 4))
                elif index == 0 and self._asset_instance_count <= 1:
                    matrices.append(np.array(self._asset_transform, dtype=np.float64, copy=True).reshape(4, 4))
                elif index < self._asset_instance_count:
                    matrices.append(np.eye(4, dtype=np.float64))
                else:
                    matrices.append(hidden)
            self._renderer.write_attribute(
                prim_paths=paths,
                attribute_name="omni:xform",
                tensor=np.stack(matrices, axis=0),
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )
            self._asset_transform_dirty = False
        except Exception as exc:
            self._asset_transform_dirty = False
            if not self._asset_transform_warning_shown:
                self._asset_transform_warning_shown = True
                self.status_changed.emit(f"Asset transform update skipped: {exc}")

    def _apply_physics_body_transforms(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return

        self._ensure_asset_instances()
        entries = []
        for physics_path, matrix in self._physics_body_transforms:
            render_path = self._render_path_from_physics_path(physics_path)
            if not render_path:
                continue
            if render_path == ASSET_ROOT and self._asset_instance_count <= 1:
                continue
            entries.append((render_path, np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)))

        if not entries:
            self._physics_body_transform_dirty = False
            return

        paths = [path for path, _matrix in entries]
        matrices = np.stack([matrix for _path, matrix in entries], axis=0)
        new_reset_paths = [path for path in paths if path not in self._physics_body_reset_paths]

        try:
            if new_reset_paths:
                self._renderer.write_attribute(
                    prim_paths=new_reset_paths,
                    attribute_name="omni:resetXformStack",
                    tensor=np.ones(len(new_reset_paths), dtype=np.bool_),
                    prim_mode=PrimMode.CREATE_NEW,
                )
                self._physics_body_reset_paths.update(new_reset_paths)

            self._renderer.write_attribute(
                prim_paths=paths,
                attribute_name="omni:xform",
                tensor=matrices,
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )
            self._physics_body_transform_dirty = False
        except Exception as exc:
            applied = 0
            last_exc = exc
            for path, matrix in entries:
                try:
                    if path not in self._physics_body_reset_paths:
                        self._renderer.write_attribute(
                            prim_paths=[path],
                            attribute_name="omni:resetXformStack",
                            tensor=np.ones(1, dtype=np.bool_),
                            prim_mode=PrimMode.CREATE_NEW,
                        )
                        self._physics_body_reset_paths.add(path)
                    self._renderer.write_attribute(
                        prim_paths=[path],
                        attribute_name="omni:xform",
                        tensor=np.array(matrix, dtype=np.float64, copy=True).reshape(1, 4, 4),
                        semantic=Semantic.XFORM_MAT4x4,
                        prim_mode=PrimMode.MUST_EXIST,
                    )
                    applied += 1
                except Exception as item_exc:
                    last_exc = item_exc
            self._physics_body_transform_dirty = False
            if applied <= 0 and not self._physics_body_transform_warning_shown:
                self._physics_body_transform_warning_shown = True
                self.status_changed.emit(f"Per-body physics transform update skipped: {last_exc}")

    def _apply_base_scene(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return

        ramp_transform = self._hidden_base_transform()
        obstacle_transforms = [self._hidden_base_transform() for _ in BASE_OBSTACLE_PATHS]

        if self._base_scene == "ramp":
            ramp_transform = np.eye(4, dtype=np.float64)
        elif self._base_scene == "obstacles":
            obstacle_transforms = [
                self._box_transform((-2.4, -1.4, 0.35), (1.2, 1.2, 0.7)),
                self._box_transform((1.8, 1.2, 0.6), (1.0, 1.6, 1.2)),
                self._box_transform((0.0, -2.8, 0.25), (2.0, 0.6, 0.5)),
            ]

        try:
            paths = [BASE_RAMP_PATH] + BASE_OBSTACLE_PATHS
            transforms = [ramp_transform] + obstacle_transforms
            for path, transform in zip(paths, transforms):
                self._renderer.write_attribute(
                    prim_paths=[path],
                    attribute_name="omni:xform",
                    tensor=np.array(transform, dtype=np.float64, copy=True).reshape(1, 4, 4),
                    semantic=Semantic.XFORM_MAT4x4,
                    prim_mode=PrimMode.MUST_EXIST,
                )
            self._base_scene_dirty = False
        except Exception as exc:
            self._base_scene_dirty = False
            self.status_changed.emit(f"Base scene visual update skipped: {exc}")

    def _apply_collision_overlay(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return

        hidden = self._hidden_base_transform()
        asset_transform = hidden
        authored_asset_transform = hidden
        ground_transform = hidden
        ramp_transform = hidden
        obstacle_transforms = [hidden for _ in COLLISION_OBSTACLE_PATHS]

        if self._collision_overlay_enabled:
            if self._ensure_collision_asset_overlay():
                authored_asset_transform = self._expanded_collision_asset_transform()
            ground_transform = self._box_transform(
                (0.0, 0.0, -GROUND_COLLIDER_THICKNESS * 0.5),
                (GROUND_PLANE_HALF_SIZE * 2.0, GROUND_PLANE_HALF_SIZE * 2.0, GROUND_COLLIDER_THICKNESS),
            )
            if self._base_scene == "ramp":
                ramp_transform = np.eye(4, dtype=np.float64)
            elif self._base_scene == "obstacles":
                obstacle_transforms = [
                    self._box_transform((-2.4, -1.4, 0.35), (1.2, 1.2, 0.7)),
                    self._box_transform((1.8, 1.2, 0.6), (1.0, 1.6, 1.2)),
                    self._box_transform((0.0, -2.8, 0.25), (2.0, 0.6, 0.5)),
                ]

        try:
            paths = [COLLISION_ASSET_PATH, COLLISION_GROUND_PATH, COLLISION_RAMP_PATH] + COLLISION_OBSTACLE_PATHS
            transforms = [asset_transform, ground_transform, ramp_transform] + obstacle_transforms
            if self._collision_asset_overlay_loaded:
                paths.append(COLLISION_ASSET_OVERLAY_ROOT)
                transforms.append(authored_asset_transform)
            for path, transform in zip(paths, transforms):
                self._renderer.write_attribute(
                    prim_paths=[path],
                    attribute_name="omni:xform",
                    tensor=np.array(transform, dtype=np.float64, copy=True).reshape(1, 4, 4),
                    semantic=Semantic.XFORM_MAT4x4,
                    prim_mode=PrimMode.MUST_EXIST,
                )
            self._collision_overlay_dirty = False
        except Exception as exc:
            self._collision_overlay_dirty = False
            if not self._collision_overlay_warning_shown:
                self._collision_overlay_warning_shown = True
                self.status_changed.emit(f"Collision overlay update skipped: {exc}")

    def _ensure_collision_asset_overlay(self) -> bool:
        if self._collision_asset_overlay_loaded:
            return True
        if not self._renderer or not self._stage_loaded:
            return False
        if not self._current_usd_source:
            return False

        try:
            overlay_path, stats_path = self._collision_wire_overlay_paths(self._current_usd_source)
            if overlay_path.exists() and overlay_path.stat().st_size > 0:
                stats = self._read_collision_wire_overlay_stats(stats_path)
                return self._load_collision_wire_overlay(overlay_path, stats)
            if self._collision_overlay_process is not None:
                return False
            return self._start_collision_wire_overlay_build(self._current_usd_source, overlay_path, stats_path)
        except Exception as exc:
            if not self._collision_overlay_warning_shown:
                self._collision_overlay_warning_shown = True
                self.status_changed.emit(f"Authored collision mesh overlay unavailable: {exc}")
            return False

    def _load_collision_wire_overlay(self, overlay_path: Path, stats: dict) -> bool:
        if not self._renderer:
            return False
        self._renderer.add_usd(str(overlay_path))
        self._collision_asset_overlay_path = overlay_path
        self._collision_asset_overlay_loaded = True
        collider_count = int(stats.get("collider_count", 0) or 0)
        edge_count = int(stats.get("edge_count", 0) or 0)
        if collider_count > 0 or edge_count > 0:
            self.status_changed.emit(
                f"Authored collision wire overlay ready ({collider_count} colliders, {edge_count} edges)."
            )
        else:
            self.status_changed.emit("Authored collision wire overlay ready.")
        return True

    def _start_collision_wire_overlay_build(self, usd_source: str, output_path: Path, stats_path: Path) -> bool:
        helper_python = self._usd_discovery_python()
        if not helper_python:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        process = QProcess(self)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.readyReadStandardOutput.connect(self._on_collision_overlay_stdout)
        process.readyReadStandardError.connect(self._on_collision_overlay_stderr)
        process.finished.connect(self._on_collision_overlay_finished)
        self._collision_overlay_process = process
        self._collision_overlay_process_asset = str(usd_source)
        self._collision_overlay_process_path = output_path
        self._collision_overlay_buffer = ""
        self._collision_overlay_error = ""
        self._collision_overlay_stats_path = stats_path
        if not self._collision_overlay_build_announced:
            self._collision_overlay_build_announced = True
            self.status_changed.emit("Building authored collision wire overlay in background...")
        process.start(
            helper_python,
            [
                "-u",
                "-m",
                "core.usd_collision_discovery",
                "--wire-usd",
                str(usd_source),
                str(output_path),
                "/World/Asset",
            ],
        )
        if not process.waitForStarted(1000):
            error = process.errorString()
            self._release_collision_overlay_process()
            raise RuntimeError(f"OpenUSD wire overlay helper failed to start: {error}")
        return False

    def _on_collision_overlay_stdout(self) -> None:
        process = self._collision_overlay_process
        if process is None:
            return
        self._collision_overlay_buffer += bytes(process.readAllStandardOutput()).decode("utf-8", "replace")

    def _on_collision_overlay_stderr(self) -> None:
        process = self._collision_overlay_process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", "replace").strip()
        if text:
            self._collision_overlay_error = text

    def _on_collision_overlay_finished(self, exit_code: int, _exit_status) -> None:
        process = self._collision_overlay_process
        asset_source = self._collision_overlay_process_asset
        output_path = self._collision_overlay_process_path
        stats_path = getattr(self, "_collision_overlay_stats_path", None)
        self._collision_overlay_process = None
        self._collision_overlay_process_asset = None
        self._collision_overlay_process_path = None
        self._collision_overlay_stats_path = None
        if process is not None:
            process.deleteLater()

        if exit_code != 0 or output_path is None or not output_path.exists():
            detail = self._collision_overlay_error.splitlines()[-1] if self._collision_overlay_error else "helper failed"
            if not self._collision_overlay_warning_shown:
                self._collision_overlay_warning_shown = True
                self.status_changed.emit(f"Authored collision mesh overlay unavailable: {detail}")
            return
        if asset_source != self._current_usd_source or self._shutdown_started:
            return

        stats = self._stats_from_collision_overlay_stdout(self._collision_overlay_buffer)
        if isinstance(stats_path, Path):
            try:
                stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            except Exception:
                pass
        try:
            self._load_collision_wire_overlay(output_path, stats)
            self._collision_overlay_dirty = True
            self._apply_collision_overlay()
            self._render_one()
        except Exception as exc:
            if not self._collision_overlay_warning_shown:
                self._collision_overlay_warning_shown = True
                self.status_changed.emit(f"Authored collision mesh overlay unavailable: {exc}")

    @staticmethod
    def _stats_from_collision_overlay_stdout(text: str) -> dict:
        for line in reversed(str(text or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                continue
        return {}

    @staticmethod
    def _read_collision_wire_overlay_stats(path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _collision_wire_overlay_paths(usd_source: str) -> tuple[Path, Path]:
        digest = hashlib.sha1(str(usd_source).encode("utf-8", "replace")).hexdigest()[:16]
        output_path = Path(tempfile.gettempdir()) / "simready_browser_collision_overlay" / f"{digest}.usda"
        return output_path, output_path.with_suffix(".json")

    def _release_collision_overlay_process(self) -> None:
        process = self._collision_overlay_process
        self._collision_overlay_process = None
        self._collision_overlay_process_asset = None
        self._collision_overlay_process_path = None
        self._collision_overlay_stats_path = None
        self._collision_overlay_buffer = ""
        self._collision_overlay_error = ""
        if process is None:
            return
        try:
            try:
                process.finished.disconnect(self._on_collision_overlay_finished)
            except Exception:
                pass
            if process.state() == QProcess.Running:
                process.terminate()
                if not process.waitForFinished(1000):
                    process.kill()
                    process.waitForFinished(1000)
        finally:
            process.deleteLater()

    @staticmethod
    def _usd_discovery_python() -> Optional[str]:
        root = Path(__file__).resolve().parents[1]
        candidates = [
            os.environ.get(USD_DISCOVERY_PYTHON_ENV, ""),
            str(root / ".usd_discovery_venv" / "Scripts" / "python.exe"),
            str(root / ".usd_discovery_venv" / "bin" / "python"),
        ]
        current = Path(sys.executable).resolve() if hasattr(sys, "executable") else None
        for item in candidates:
            if not item:
                continue
            path = Path(item).expanduser()
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if not resolved.exists():
                continue
            if current is not None and resolved == current and not os.environ.get(USD_DISCOVERY_PYTHON_ENV):
                continue
            return str(resolved)
        return None

    def _expanded_collision_asset_transform(self) -> np.ndarray:
        if not self._collision_proxy_bounds:
            return np.array(self._asset_transform, dtype=np.float64, copy=True)

        center = np.array(self._collision_proxy_bounds["center"], dtype=np.float64)
        extent = float(np.linalg.norm(np.array(self._collision_proxy_bounds["size"], dtype=np.float64))) * 0.5
        grow = 1.0 + min(0.018, max(0.004, 0.01 / max(extent, 0.1)))

        translate_to_origin = np.eye(4, dtype=np.float64)
        translate_to_origin[3, :3] = -center
        scale = np.eye(4, dtype=np.float64)
        scale[0, 0] = grow
        scale[1, 1] = grow
        scale[2, 2] = grow
        translate_back = np.eye(4, dtype=np.float64)
        translate_back[3, :3] = center
        return translate_to_origin @ scale @ translate_back @ self._asset_transform

    @staticmethod
    def _normalize_physics_body_transforms(bodies) -> list[tuple[str, np.ndarray]]:
        normalized: list[tuple[str, np.ndarray]] = []
        try:
            raw_items = list(bodies or [])
        except TypeError:
            raw_items = []

        for item in raw_items[:10000]:
            if isinstance(item, dict):
                path = str(item.get("path", "") or "").strip()
                matrix_value = item.get("matrix")
            else:
                try:
                    path, matrix_value = item
                    path = str(path or "").strip()
                except Exception:
                    continue
            if not path:
                continue
            try:
                matrix = np.array(matrix_value, dtype=np.float64, copy=True).reshape(4, 4)
            except Exception:
                continue
            if not np.all(np.isfinite(matrix)):
                continue
            normalized.append((path, matrix))
        return normalized

    @staticmethod
    def _render_path_from_physics_path(path: str) -> str:
        text = str(path or "").strip()
        if not text:
            return ""
        if not text.startswith("/"):
            text = "/" + text
        if text == ASSET_ROOT or text.startswith(f"{ASSET_ROOT}/"):
            return text
        for index in range(MAX_ASSET_INSTANCES):
            physics_root = OVRTXRenderer._physics_asset_root(index)
            render_root = OVRTXRenderer._asset_render_root(index)
            if text == physics_root:
                return render_root
            if text.startswith(f"{physics_root}/"):
                return render_root + text[len(physics_root) :]
        return ""

    @staticmethod
    def _normalize_stage_items(items) -> list[dict]:
        normalized: list[dict] = []
        try:
            raw_items = list(items or [])
        except TypeError:
            raw_items = []

        for item in raw_items:
            if isinstance(item, dict):
                source = str(item.get("source") or item.get("usd_source") or item.get("path") or "").strip()
                name = str(item.get("name") or Path(source.split("?", 1)[0]).name or source).strip()
                key = str(item.get("key") or source).strip()
            else:
                source = str(item or "").strip()
                name = Path(source.split("?", 1)[0]).name or source
                key = source
            if not source:
                continue
            normalized.append({"source": source, "name": name or source, "key": key or source})

        return normalized[:MAX_ASSET_INSTANCES]

    @classmethod
    def _layout_asset_transforms(cls, bounds_entries: list[dict]) -> list[np.ndarray]:
        if not bounds_entries:
            return []

        normalized = [
            cls._normalize_collision_bounds(bounds) or {"center": (0.0, 0.0, 0.0), "size": (1.0, 1.0, 1.0)}
            for bounds in bounds_entries
        ]
        max_span = max(max(float(entry["size"][0]), float(entry["size"][1])) for entry in normalized)
        clearance = max(0.08, min(0.6, max_span * 0.08))
        rng = random.Random(time.time_ns())

        rects: list[tuple[float, float, float, float]] = []
        transforms: list[np.ndarray] = []

        for index, entry in enumerate(normalized):
            local_center = np.array(entry["center"], dtype=np.float64)
            size = np.maximum(np.array(entry["size"], dtype=np.float64), np.array([0.05, 0.05, 0.05]))
            half_xy = np.maximum(size[:2] * 0.5 + clearance * 0.5, np.array([0.05, 0.05], dtype=np.float64))
            xy = np.zeros(2, dtype=np.float64) if index == 0 else cls._find_non_overlapping_xy(
                rects,
                half_xy,
                max_span,
                clearance,
                index,
                rng,
            )
            target_center = np.array([xy[0], xy[1], size[2] * 0.5], dtype=np.float64)

            matrix = np.eye(4, dtype=np.float64)
            matrix[3, :3] = target_center - local_center
            transforms.append(matrix)
            rects.append((xy[0] - half_xy[0], xy[1] - half_xy[1], xy[0] + half_xy[0], xy[1] + half_xy[1]))

        return transforms

    @staticmethod
    def _find_non_overlapping_xy(
        rects: list[tuple[float, float, float, float]],
        half_xy: np.ndarray,
        max_span: float,
        clearance: float,
        index: int,
        rng: random.Random,
    ) -> np.ndarray:
        def candidate_rect(xy: np.ndarray) -> tuple[float, float, float, float]:
            return (xy[0] - half_xy[0], xy[1] - half_xy[1], xy[0] + half_xy[0], xy[1] + half_xy[1])

        def fits(rect: tuple[float, float, float, float]) -> bool:
            return all(not OVRTXRenderer._rects_overlap(rect, existing) for existing in rects)

        cell = max(max_span + clearance, 0.25)
        for ring in range(1, 48):
            candidates = []
            for gx in range(-ring, ring + 1):
                for gy in range(-ring, ring + 1):
                    if max(abs(gx), abs(gy)) != ring:
                        continue
                    jitter = np.array(
                        [
                            rng.uniform(-clearance, clearance) * 0.35,
                            rng.uniform(-clearance, clearance) * 0.35,
                        ],
                        dtype=np.float64,
                    )
                    xy = np.array([gx * cell, gy * cell], dtype=np.float64) + jitter
                    candidates.append((float(np.linalg.norm(xy)) + rng.uniform(0.0, 0.01), xy))
            candidates.sort(key=lambda item: item[0])
            for _score, xy in candidates:
                rect = candidate_rect(xy)
                if fits(rect):
                    return xy

        slot = 0
        columns = max(3, int(math.ceil(math.sqrt(index + 1))) * 2 + 1)
        while True:
            gx = (slot % columns) - columns // 2
            gy = slot // columns
            xy = np.array([gx * cell, gy * cell], dtype=np.float64)
            if fits(candidate_rect(xy)):
                return xy
            slot += 1

    @staticmethod
    def _rects_overlap(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    @classmethod
    def _combined_layout_bounds(cls, bounds_entries: list[dict], transforms: list[np.ndarray]) -> dict:
        if not bounds_entries:
            return DEFAULT_FRAME_BOUNDS.copy()

        mins = []
        maxs = []
        for index, bounds in enumerate(bounds_entries):
            matrix = transforms[index] if index < len(transforms) else np.eye(4, dtype=np.float64)
            entry = cls._normalize_collision_bounds(bounds) or {"center": (0.0, 0.0, 0.0), "size": (1.0, 1.0, 1.0)}
            center = np.array(entry["center"], dtype=np.float64)
            size = np.array(entry["size"], dtype=np.float64)
            transform = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
            world_center = center @ transform[:3, :3] + transform[3, :3]
            half = np.maximum(size * 0.5, np.array([0.05, 0.05, 0.05], dtype=np.float64))
            mins.append(world_center - half)
            maxs.append(world_center + half)

        if not mins:
            return DEFAULT_FRAME_BOUNDS.copy()

        lo = np.min(np.stack(mins), axis=0)
        hi = np.max(np.stack(maxs), axis=0)
        center = (lo + hi) * 0.5
        size = np.maximum(hi - lo, np.array([0.05, 0.05, 0.05], dtype=np.float64))
        extent = max(float(np.linalg.norm(size)) * 0.5, 0.1)
        return {"center": center.tolist(), "extent": extent, "size": size.tolist()}

    @staticmethod
    def _asset_render_root(index: int) -> str:
        return ASSET_ROOT if index <= 0 else f"/{ASSET_ROOT_NAME}_{index + 1:02d}"

    @staticmethod
    def _physics_asset_root(index: int) -> str:
        return PHYSICS_ASSET_ROOT if index <= 0 else f"/World/Instance_{index + 1:02d}"

    @staticmethod
    def _normalize_collision_bounds(bounds: dict) -> Optional[dict]:
        try:
            center = np.array(bounds.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        except Exception:
            center = np.zeros(3, dtype=np.float64)

        try:
            size = np.array(bounds.get("size", []), dtype=np.float64).reshape(3)
        except Exception:
            size = np.array([], dtype=np.float64)

        if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0):
            extent = float(bounds.get("extent", 1.0) or 1.0)
            if not math.isfinite(extent) or extent <= 0:
                extent = 1.0
            side = max((2.0 * extent) / math.sqrt(3.0), 0.25)
            size = np.array([side, side, side], dtype=np.float64)

        if not np.all(np.isfinite(center)):
            center = np.zeros(3, dtype=np.float64)
        size = np.maximum(np.abs(size), np.array([0.05, 0.05, 0.05], dtype=np.float64))
        return {"center": tuple(float(v) for v in center), "size": tuple(float(v) for v in size)}

    @staticmethod
    def _hidden_base_transform() -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[3, :3] = [0.0, 0.0, HIDDEN_BASE_Z]
        return matrix

    @staticmethod
    def _box_transform(center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = float(size[0])
        matrix[1, 1] = float(size[1])
        matrix[2, 2] = float(size[2])
        matrix[3, :3] = [float(center[0]), float(center[1]), float(center[2])]
        return matrix

    def _apply_dome_light(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return
        self._write_float_attribute(
            DOME_LIGHT_PATH,
            "inputs:intensity",
            self._dome_intensity * 1000.0,
            "Dome intensity",
        )

    def _apply_directional_light(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return

        intensity = self._dir_intensity * 3000.0
        self._write_float_attribute(KEY_LIGHT_PATH, "inputs:intensity", intensity, "Sun intensity")
        self._write_float_attribute(KEY_LIGHT_PATH, "intensity", intensity, "Sun legacy intensity", quiet=True)
        self._write_float_attribute(KEY_LIGHT_PATH, "inputs:angle", 0.53, "Sun angle", quiet=True)
        self._write_light_transform(self._directional_light_transform(self._dir_azimuth, self._dir_elevation))

    def _write_float_attribute(self, prim_path: str, attr_name: str, value: float, label: str, quiet: bool = False) -> bool:
        try:
            self._renderer.write_attribute(
                prim_paths=[prim_path],
                attribute_name=attr_name,
                tensor=np.array([float(value)], dtype=np.float32),
                prim_mode=PrimMode.MUST_EXIST,
            )
            return True
        except Exception as exc:
            if not quiet:
                self.status_changed.emit(f"{label} update skipped: {exc}")
            return False

    def _write_light_transform(self, matrix: np.ndarray) -> None:
        try:
            self._renderer.write_attribute(
                prim_paths=[KEY_LIGHT_PATH],
                attribute_name="omni:xform",
                tensor=np.array(matrix, dtype=np.float64, copy=True).reshape(1, 4, 4),
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )
        except Exception as exc:
            self.status_changed.emit(f"Sun direction update skipped: {exc}")

    @staticmethod
    def _directional_light_transform(azimuth: float, elevation: float) -> np.ndarray:
        az = math.radians(float(azimuth))
        el = math.radians(float(elevation))
        sun_dir = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=np.float64,
        )
        norm = np.linalg.norm(sun_dir)
        if norm < 1e-9:
            sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            sun_dir /= norm

        local_z = sun_dir
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(local_z, up_hint))) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        local_x = np.cross(up_hint, local_z)
        local_x /= max(np.linalg.norm(local_x), 1e-9)
        local_y = np.cross(local_z, local_x)

        matrix = np.eye(4, dtype=np.float64)
        matrix[0, :3] = local_x
        matrix[1, :3] = local_y
        matrix[2, :3] = local_z
        return matrix

    def _read_stage_bounds(self, usd_source: str = "") -> Optional[dict]:
        """Best-effort bounds without touching fragile OVRTX debug USD paths."""
        metadata_bounds = self._read_glb_bounds(usd_source) or self._read_metadata_bounds(usd_source)
        if metadata_bounds or not ENABLE_OVRTX_DEBUG_BOUNDS:
            return metadata_bounds or DEFAULT_FRAME_BOUNDS.copy()

        try:
            products = self._renderer.step(render_products={"ovrtx_debug_dump_stage"}, delta_time=0.0)
            with products:
                frame = products["ovrtx_debug_dump_stage"].frames[0]
                with frame.render_vars["debug"].map(device=Device.CPU) as mapping:
                    dump = mapping.tensor.to_bytes().decode("utf-8", "replace")
        except Exception:
            return metadata_bounds or DEFAULT_FRAME_BOUNDS.copy()

        mins = []
        maxs = []
        pattern = re.compile(
            r"extent\s*=\s*\[\(([^)]*)\),\s*\(([^)]*)\)\]",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(dump):
            try:
                lo = np.array([float(v.strip()) for v in match.group(1).split(",")[:3]], dtype=np.float64)
                hi = np.array([float(v.strip()) for v in match.group(2).split(",")[:3]], dtype=np.float64)
            except Exception:
                continue
            if lo.size == 3 and hi.size == 3 and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                mins.append(np.minimum(lo, hi))
                maxs.append(np.maximum(lo, hi))

        if not mins:
            return metadata_bounds or DEFAULT_FRAME_BOUNDS.copy()

        lo = np.min(np.stack(mins), axis=0)
        hi = np.max(np.stack(maxs), axis=0)
        center = (lo + hi) * 0.5
        extent = max(float(np.linalg.norm(hi - lo)) * 0.5, 0.1)
        if metadata_bounds:
            extent = max(extent, float(metadata_bounds.get("extent", extent)))
        return {"center": center.tolist(), "extent": extent, "size": (hi - lo).tolist()}

    def _read_metadata_bounds(self, usd_source: str) -> Optional[dict]:
        parsed = urllib.parse.urlparse(usd_source)
        if parsed.scheme not in {"http", "https"}:
            return None

        path = parsed.path
        directory, _, name = path.rpartition("/")
        if not directory or not name:
            return None
        stem = name.rsplit(".", 1)[0]
        candidates = [
            f"{directory}/{stem}.json",
            f"{directory}/metadata.json",
        ]

        for candidate_path in candidates:
            candidate = urllib.parse.urlunparse(parsed._replace(path=candidate_path))
            try:
                req = urllib.request.Request(candidate, headers={"User-Agent": "SimReadyBrowser/1.0"})
                with urllib.request.urlopen(req, timeout=1.25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                extent = data.get("Extent", data.get("extent"))
                if isinstance(extent, list) and len(extent) >= 3:
                    dims = np.array([float(v) for v in extent[:3]], dtype=np.float64)
                    if np.all(np.isfinite(dims)):
                        radius = max(float(np.linalg.norm(np.abs(dims))) * 0.5, 0.1)
                        return {"center": [0.0, 0.0, 0.0], "extent": radius, "size": np.abs(dims).tolist()}
            except Exception:
                continue
        return None

    def _read_glb_bounds(self, usd_source: str) -> Optional[dict]:
        parsed = urllib.parse.urlparse(usd_source)
        if parsed.scheme not in {"http", "https"}:
            return None

        path = parsed.path
        directory, _, name = path.rpartition("/")
        if not directory or not name:
            return None

        stem = name.rsplit(".", 1)[0]
        glb_path = f"{directory}/web/{stem}.glb"
        glb_url = urllib.parse.urlunparse(parsed._replace(path=glb_path))

        try:
            gltf = self._fetch_glb_json(glb_url)
            if not gltf:
                return None
            return self._bounds_from_gltf_json(gltf)
        except Exception:
            return None

    def _fetch_glb_json(self, url: str) -> Optional[dict]:
        req = urllib.request.Request(
            url,
            headers={"Range": "bytes=0-19", "User-Agent": "SimReadyBrowser/1.0"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            header = resp.read()

        if len(header) < 20 or header[:4] != b"glTF":
            return None

        _magic, version, _length = struct.unpack_from("<III", header, 0)
        json_length, chunk_type = struct.unpack_from("<II", header, 12)
        if version != 2 or chunk_type != 0x4E4F534A or json_length <= 0:
            return None

        start = 20
        end = start + json_length - 1
        req = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "SimReadyBrowser/1.0"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = resp.read()
        return json.loads(data[:json_length].decode("utf-8"))

    def _bounds_from_gltf_json(self, gltf: dict) -> Optional[dict]:
        accessors = gltf.get("accessors", [])
        meshes = gltf.get("meshes", [])
        nodes = gltf.get("nodes", [])
        if not isinstance(accessors, list) or not isinstance(meshes, list):
            return None

        mesh_bounds: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for mesh_index, mesh in enumerate(meshes):
            if not isinstance(mesh, dict):
                continue
            bounds = []
            for prim in mesh.get("primitives", []):
                if not isinstance(prim, dict):
                    continue
                attrs = prim.get("attributes", {})
                accessor_index = attrs.get("POSITION") if isinstance(attrs, dict) else None
                if not isinstance(accessor_index, int) or accessor_index >= len(accessors):
                    continue
                accessor = accessors[accessor_index]
                lo = accessor.get("min") if isinstance(accessor, dict) else None
                hi = accessor.get("max") if isinstance(accessor, dict) else None
                if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) >= 3 and len(hi) >= 3):
                    continue
                lo_arr = np.array([float(v) for v in lo[:3]], dtype=np.float64)
                hi_arr = np.array([float(v) for v in hi[:3]], dtype=np.float64)
                if np.all(np.isfinite(lo_arr)) and np.all(np.isfinite(hi_arr)):
                    bounds.append((np.minimum(lo_arr, hi_arr), np.maximum(lo_arr, hi_arr)))
            if bounds:
                mesh_bounds[mesh_index] = bounds

        if not mesh_bounds:
            return None

        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []

        def add_bounds(lo: np.ndarray, hi: np.ndarray, matrix: np.ndarray) -> None:
            corners = np.array(
                [
                    [x, y, z, 1.0]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])
                ],
                dtype=np.float64,
            )
            transformed = (matrix @ corners.T).T[:, :3]
            mins.append(np.min(transformed, axis=0))
            maxs.append(np.max(transformed, axis=0))

        visited: set[int] = set()

        def walk_node(node_index: int, parent_matrix: np.ndarray) -> None:
            if node_index in visited or node_index < 0 or node_index >= len(nodes):
                return
            visited.add(node_index)
            node = nodes[node_index]
            if not isinstance(node, dict):
                return

            matrix = parent_matrix @ self._gltf_node_matrix(node)
            mesh_index = node.get("mesh")
            if isinstance(mesh_index, int) and mesh_index in mesh_bounds:
                for lo, hi in mesh_bounds[mesh_index]:
                    add_bounds(lo, hi, matrix)

            for child in node.get("children", []):
                if isinstance(child, int):
                    walk_node(child, matrix)

        scenes = gltf.get("scenes", [])
        scene_index = gltf.get("scene", 0)
        roots = []
        if isinstance(scenes, list) and isinstance(scene_index, int) and scene_index < len(scenes):
            scene = scenes[scene_index]
            if isinstance(scene, dict):
                roots = [idx for idx in scene.get("nodes", []) if isinstance(idx, int)]

        if roots and isinstance(nodes, list):
            for root in roots:
                walk_node(root, np.eye(4, dtype=np.float64))

        if not mins:
            for bounds in mesh_bounds.values():
                for lo, hi in bounds:
                    mins.append(lo)
                    maxs.append(hi)

        lo = np.min(np.stack(mins), axis=0)
        hi = np.max(np.stack(maxs), axis=0)
        center = (lo + hi) * 0.5
        extent = max(float(np.linalg.norm(hi - lo)) * 0.5, 0.05)
        center = self._gltf_y_up_to_usd_z_up(center)
        size = np.abs(self._gltf_y_up_to_usd_z_up(hi - lo))
        return {"center": center.tolist(), "extent": extent, "size": size.tolist()}

    @staticmethod
    def _gltf_y_up_to_usd_z_up(value: np.ndarray) -> np.ndarray:
        # SimReady web previews are glTF Y-up; the USD assets are Isaac Z-up.
        return np.array([value[0], -value[2], value[1]], dtype=np.float64)

    @staticmethod
    def _gltf_node_matrix(node: dict) -> np.ndarray:
        if isinstance(node.get("matrix"), list) and len(node["matrix"]) == 16:
            return np.array([float(v) for v in node["matrix"]], dtype=np.float64).reshape(4, 4, order="F")

        matrix = np.eye(4, dtype=np.float64)
        scale = node.get("scale", [1.0, 1.0, 1.0])
        if isinstance(scale, list) and len(scale) >= 3:
            matrix = matrix @ np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0])

        rotation = node.get("rotation")
        if isinstance(rotation, list) and len(rotation) >= 4:
            x, y, z, w = [float(v) for v in rotation[:4]]
            norm = math.sqrt(x * x + y * y + z * z + w * w)
            if norm > 1e-9:
                x, y, z, w = x / norm, y / norm, z / norm, w / norm
                rot = np.array(
                    [
                        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
                        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w, 0],
                        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float64,
                )
                matrix = rot @ matrix

        translation = node.get("translation")
        if isinstance(translation, list) and len(translation) >= 3:
            translate = np.eye(4, dtype=np.float64)
            translate[:3, 3] = [float(translation[0]), float(translation[1]), float(translation[2])]
            matrix = translate @ matrix

        return matrix

    def _frame_initial_camera(self, bounds: dict) -> None:
        center = np.array(bounds.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        extent = float(bounds.get("extent", DEFAULT_FRAME_BOUNDS["extent"]))
        camera = SphericalCamera()
        camera.frame_bounds(center, extent)
        self._camera_transform = camera.get_transform()
        self._camera_dirty = True

    def _start_timer(self, target_fps: int) -> None:
        if self._shutdown_started:
            return
        self._stop_timer()
        interval_ms = max(1, 1000 // max(1, target_fps))
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_one)
        self._render_timer.start(interval_ms)

    def _stop_timer(self) -> None:
        if self._render_timer:
            self._render_timer.stop()
            self._render_timer.setParent(None)
            self._render_timer.deleteLater()
            self._render_timer = None

    def _shutdown_renderer(self) -> None:
        self._shutdown_started = True
        self._stage_loaded = False
        self._pending_stage = None
        self._pending_stage_items = None
        self._release_collision_overlay_process()
        self._stop_timer()
        if self._placeholder_timer:
            self._placeholder_timer.stop()
            self._placeholder_timer.setParent(None)
            self._placeholder_timer.deleteLater()
            self._placeholder_timer = None

        if self._renderer is not None:
            self._renderer = None
            gc.collect()

        app = QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())

        self._thread.quit()

    def _render_one(self) -> None:
        if self._shutdown_started:
            return
        if not self._renderer or not self._stage_loaded:
            return

        try:
            if self._camera_dirty:
                self._apply_camera()
            if self._asset_transform_dirty:
                self._apply_asset_transform()
            if self._physics_body_transform_dirty:
                self._apply_physics_body_transforms()
            if self._base_scene_dirty:
                self._apply_base_scene()
            if self._collision_overlay_dirty:
                self._apply_collision_overlay()

            t0 = time.perf_counter()
            products = self._renderer.step(
                render_products={RENDER_PRODUCT_PATH},
                delta_time=self._delta_time,
            )
            dt = time.perf_counter() - t0

            image = self._extract_qimage(products)
            if image is not None:
                self.frame_ready.emit(image)
                self._update_fps(dt)
        except Exception as exc:
            self.error_occurred.emit(f"Render error: {exc}")

    def _extract_qimage(self, products) -> Optional[QImage]:
        try:
            with products:
                product = products[RENDER_PRODUCT_PATH]
                for frame in reversed(product.frames):
                    render_var = frame.render_vars.get("LdrColor")
                    if not render_var:
                        continue
                    with render_var.map(device=Device.CPU) as mapping:
                        pixels = np.from_dlpack(mapping.tensor).copy()
                    return self._numpy_to_qimage(pixels)
        except Exception as exc:
            self.error_occurred.emit(f"Frame readback error: {exc}")
        return None

    def _update_fps(self, dt: float) -> None:
        self._frame_times.append(dt)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        if avg > 0:
            self.fps_updated.emit(1.0 / avg)

    @staticmethod
    def _numpy_to_qimage(arr: np.ndarray) -> QImage:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dimensions, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        channels = arr.shape[2]
        if channels >= 4:
            arr = np.ascontiguousarray(arr[:, :, :4])
            fmt = QImage.Format_RGBA8888
        elif channels == 3:
            arr = np.ascontiguousarray(arr)
            fmt = QImage.Format_RGB888
        else:
            arr = np.ascontiguousarray(np.repeat(arr[:, :, :1], 3, axis=2))
            fmt = QImage.Format_RGB888

        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, w * arr.shape[2], fmt)
        return img.copy()

    def _start_placeholder_timer(self) -> None:
        if self._placeholder_timer:
            return
        self._placeholder_timer = QTimer(self)
        self._placeholder_timer.timeout.connect(self._render_placeholder)
        self._placeholder_timer.start(33)

    def _render_placeholder(self) -> None:
        self._placeholder_frame += 1
        w, h = self._width, self._height
        arr = np.zeros((h, w, 3), dtype=np.uint8)

        t = self._placeholder_frame * 0.015
        cx = int(w / 2 + w * 0.28 * math.sin(t))
        cy = int(h / 2 + h * 0.20 * math.cos(t * 0.7))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        v = np.clip(1.0 - dist / max(w * 0.55, 1), 0.0, 1.0)

        arr[:, :, 1] = (v * 118).astype(np.uint8)
        arr[::40, :, :] = (arr[::40, :, :] * 0.5 + 10).astype(np.uint8)
        arr[:, ::40, :] = (arr[:, ::40, :] * 0.5 + 10).astype(np.uint8)

        self.frame_ready.emit(self._numpy_to_qimage(arr))
        if self._placeholder_frame % 60 == 0:
            self.fps_updated.emit(30.0)
