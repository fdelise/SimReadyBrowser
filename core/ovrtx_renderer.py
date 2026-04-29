"""Qt-friendly wrapper around the NVIDIA OVRTX Python API.

OVRTX renders named USD RenderProduct prims. This wrapper loads a selected USD
asset, injects a small review layer containing a camera, lights, and a render
product, then maps the LdrColor output back to a QImage for the Qt viewport.
"""

from __future__ import annotations

import math
import gc
import json
import os
import re
import struct
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from core.camera_controller import SphericalCamera

ovrtx = None  # type: ignore
Device = PrimMode = Renderer = RendererConfig = Semantic = None  # type: ignore
OVRTX_AVAILABLE: Optional[bool] = None
OVRTX_IMPORT_ERROR: Optional[BaseException] = None


REVIEW_ROOT = "/SimReadyReview"
ASSET_ROOT = "/SimReadyAsset"
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
DEFAULT_FRAME_BOUNDS = {"center": [0.0, 0.0, 0.0], "extent": 1.0}
STAGE_UP_AXIS = "Z"
STAGE_METERS_PER_UNIT = 1
GROUND_PLANE_HALF_SIZE = 20.0
GROUND_PLANE_Z = -0.005
BASE_SCENES = {"plane", "ramp", "obstacles"}
HIDDEN_BASE_Z = -10000.0
ENABLE_OVRTX_DEBUG_BOUNDS = os.environ.get("SIMREADY_OVRTX_DEBUG_BOUNDS") == "1"


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
    _resolution_requested = pyqtSignal(int, int)
    _camera_transform_requested = pyqtSignal(object)
    _asset_transform_requested = pyqtSignal(object)
    _base_scene_requested = pyqtSignal(str)
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
        self._asset_transform_warning_shown = False
        self._base_scene = "plane"
        self._base_scene_dirty = True
        self._dome_intensity = 1.0
        self._dir_intensity = 0.8
        self._dir_azimuth = 45.0
        self._dir_elevation = 60.0

        self._frame_times: list[float] = []

        self._thread = QThread()
        self._connect_requests()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._init_renderer, Qt.QueuedConnection)
        self._thread.start()

    def _connect_requests(self) -> None:
        self._load_stage_requested.connect(self._load_stage, Qt.QueuedConnection)
        self._resolution_requested.connect(self._set_resolution, Qt.QueuedConnection)
        self._camera_transform_requested.connect(self._set_camera_transform, Qt.QueuedConnection)
        self._asset_transform_requested.connect(self._set_asset_transform, Qt.QueuedConnection)
        self._base_scene_requested.connect(self._set_base_scene, Qt.QueuedConnection)
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
        self._asset_transform_requested.emit(np.array(matrix, dtype=np.float64, copy=True))

    def set_base_scene(self, scene_id: str) -> None:
        if self._shutdown_started:
            return
        self._base_scene_requested.emit(str(scene_id or "plane"))

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
            if self._pending_stage:
                self.loading_finished.emit(False, msg)
                self._pending_stage = None
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
            self.status_changed.emit("OVRTX renderer ready.")
            if self._pending_stage:
                pending_stage = self._pending_stage
                self._pending_stage = None
                self._load_stage(pending_stage)
        except Exception as exc:
            if self._pending_stage:
                msg = f"OVRTX init failed before loading asset: {exc}"
                self.loading_finished.emit(False, msg)
                self._pending_stage = None
            self.error_occurred.emit(f"OVRTX init failed: {exc}")
            self._start_placeholder_timer()

    def _load_stage(self, usd_source: str) -> None:
        if self._shutdown_started:
            return
        display_name = Path(usd_source).name or usd_source
        if not self._renderer:
            if OVRTX_AVAILABLE is not False:
                self._pending_stage = usd_source
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
            self._asset_transform = np.eye(4, dtype=np.float64)
            self._asset_transform_dirty = True
            self._asset_transform_warning_shown = False
            self._base_scene_dirty = True
            self.loading_started.emit(f"Loading {display_name} in OVRTX...")
            self.loading_progress.emit(5, "Preparing OVRTX stage...")
            self.status_changed.emit(f"Loading {display_name}...")

            self.loading_progress.emit(15, "Clearing previous stage...")
            self._renderer.reset_stage()
            self._renderer.reset(time=0.0)

            self.loading_progress.emit(25, "Applying Isaac Z-up stage settings...")
            self._renderer.add_usd_layer(self._stage_settings_layer())

            self.loading_progress.emit(35, "Streaming USD, payloads, and materials...")
            self._renderer.add_usd(usd_source, path_prefix=ASSET_ROOT)

            self.loading_progress.emit(60, "Adding review camera and lights...")
            self._renderer.add_usd_layer(self._review_layer(), path_prefix=REVIEW_ROOT)

            self._stage_loaded = True
            self._camera_dirty = True
            self.loading_progress.emit(72, "Applying viewport settings...")
            self._apply_resolution()
            self._apply_asset_transform()
            self._apply_base_scene()
            self._apply_dome_light()
            self._apply_directional_light()

            self.loading_progress.emit(84, "Framing asset bounds...")
            bounds = self._read_stage_bounds(usd_source)
            if bounds:
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
        self._asset_transform = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
        self._asset_transform_dirty = True
        self._apply_asset_transform()

    def _set_base_scene(self, scene_id: str) -> None:
        if self._shutdown_started:
            return
        scene = str(scene_id or "plane").lower()
        if scene not in BASE_SCENES:
            scene = "plane"
        self._base_scene = scene
        self._base_scene_dirty = True
        self._apply_base_scene()

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

    def _apply_asset_transform(self) -> None:
        if not self._renderer or not self._stage_loaded:
            return
        try:
            self._renderer.write_attribute(
                prim_paths=[ASSET_ROOT],
                attribute_name="omni:xform",
                tensor=np.array(self._asset_transform, dtype=np.float64, copy=True).reshape(1, 4, 4),
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )
            self._asset_transform_dirty = False
        except Exception as exc:
            self._asset_transform_dirty = False
            if not self._asset_transform_warning_shown:
                self._asset_transform_warning_shown = True
                self.status_changed.emit(f"Asset transform update skipped: {exc}")

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
            if self._base_scene_dirty:
                self._apply_base_scene()

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
