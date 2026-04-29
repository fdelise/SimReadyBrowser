"""Interactive Qt viewport for OVRTX rendered frames."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

from core.camera_controller import SphericalCamera
from core.physics_controller import PhysicsController
from styles.nvidia_theme import COLOR_ACCENT, COLOR_TEXT_SECONDARY, COLOR_VIEWPORT_BG

LOAD_START_DELAY_MS = 150
CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
GRAB_THROW_VELOCITY_LIMIT = 10.0
DEFAULT_GRAB_FORCE_AMOUNT = 2.0


class ViewportWidget(QWidget):
    """Interactive 3D viewport powered by OVRTX frame readback."""

    asset_loaded = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    status_msg = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)
    physics_status_changed = pyqtSignal(str)
    physics_running_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 400)
        self.setFocusPolicy(Qt.StrongFocus)

        self._camera = SphericalCamera()
        self._last_bounds: Optional[dict] = None
        self._last_pos: Optional[QPoint] = None
        self._active_button: Optional[Qt.MouseButton] = None
        self._active_mode: Optional[str] = None
        self._renderer = None
        self._pending_dome_intensity = 1.0
        self._pending_dir_light = (0.8, 45.0, 60.0)
        self._pending_base_scene = "plane"
        self._pending_collision_overlay = False
        self._load_generation = 0
        self._loading_asset = False
        self._physics_cooking_active = False
        self._physics_auto_cook_started = False
        self._current_usd_source: Optional[str] = None
        self._physics = PhysicsController(self)
        self._physics.pose_changed.connect(self._on_physics_pose)
        self._physics.status_changed.connect(self._on_physics_status)
        self._physics.running_changed.connect(self.physics_running_changed)
        self._physics.cooking_progress.connect(self._on_physics_cooking_progress)
        self._physics.cooking_finished.connect(self._on_physics_cooking_finished)
        self._physics_current_transform = np.eye(4, dtype=np.float64)
        self._physics_grabbing = False
        self._physics_drag_start: Optional[QPoint] = None
        self._physics_drag_matrix = np.eye(4, dtype=np.float64)
        self._physics_grab_anchor = np.zeros(3, dtype=np.float64)
        self._physics_grab_target_start = np.zeros(3, dtype=np.float64)
        self._physics_grab_plane_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._physics_grab_pointer_offset = np.zeros(3, dtype=np.float64)
        self._physics_grab_last_target = np.zeros(3, dtype=np.float64)
        self._physics_grab_last_time = 0.0
        self._physics_grab_velocity = np.zeros(3, dtype=np.float64)
        self._physics_grab_force_amount = DEFAULT_GRAB_FORCE_AMOUNT
        self._physics.set_grab_force_amount(self._physics_grab_force_amount)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = _RenderCanvas(self)
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._canvas.installEventFilter(self)
        self._canvas.setMouseTracking(True)
        layout.addWidget(self._canvas)

        self._loading_overlay = _ViewportLoadingOverlay(self)
        self._loading_overlay.hide()

        self._show_hint()

    def load_usd(self, path: str | Path) -> None:
        self._hide_hint()
        self.setFocus(Qt.OtherFocusReason)
        self._canvas.setFocus(Qt.OtherFocusReason)
        self._current_usd_source = str(path)
        self._last_bounds = None
        self._camera.reset()
        self._physics.clear_asset()
        self._physics_current_transform = np.eye(4, dtype=np.float64)
        self._physics_grabbing = False
        self._physics_cooking_active = False
        self._physics_auto_cook_started = False
        self._loading_asset = True
        self._load_generation += 1
        generation = self._load_generation
        self._set_loading(True, "Queued asset for OVRTX...")
        self.loading_changed.emit(True, "Queued asset for OVRTX...")
        QTimer.singleShot(
            LOAD_START_DELAY_MS,
            lambda source=str(path), gen=generation: self._load_usd_after_overlay(source, gen),
        )

    @property
    def camera(self) -> SphericalCamera:
        return self._camera

    @property
    def physics_status(self) -> str:
        return self._physics.status_text

    def reset_camera(self) -> None:
        if self._last_bounds:
            center = np.array(self._last_bounds["center"], dtype=np.float64)
            extent = float(self._last_bounds["extent"])
            self._camera.frame_bounds(center, extent)
        else:
            self._camera.reset()
        self._push_camera()

    def set_dome_intensity(self, value: float) -> None:
        self._pending_dome_intensity = value
        if self._renderer:
            self._renderer.set_dome_intensity(value)
            self._renderer.request_render()

    def set_dome_texture(self, path: Optional[str]) -> None:
        # Reserved for HDRI support. The current injected review layer uses sky lighting.
        _ = path

    def set_directional_light(self, intensity: float, azimuth: float, elevation: float) -> None:
        self._pending_dir_light = (intensity, azimuth, elevation)
        if self._renderer:
            self._renderer.set_directional_light(intensity, azimuth, elevation)
            self._renderer.request_render()

    def set_physics_playing(self, playing: bool) -> None:
        if playing and not self._physics.has_scene and not self._physics_cooking_active:
            self._begin_physics_progress("Preparing physics colliders...", 0)
        self._physics.set_playing(playing)

    def restart_physics(self) -> None:
        if self._last_bounds is None:
            self._on_physics_status("Load an asset before starting physics.")
            self.physics_running_changed.emit(False)
            return
        if not self._physics.has_scene and not self._physics_cooking_active:
            self._begin_physics_progress("Preparing physics colliders...", 0)
        if not self._physics.restart(play=True):
            self._physics_cooking_active = False
            self.loading_changed.emit(False, self._physics.status_text)
            self._set_loading(False)

    def step_physics(self) -> None:
        self._physics.step_once()

    def set_physics_base_scene(self, scene_id: str) -> None:
        self._pending_base_scene = str(scene_id or "plane")
        self._physics.set_base_scene(self._pending_base_scene)
        if self._renderer:
            self._renderer.set_base_scene(self._pending_base_scene)
            self._renderer.request_render()

    def set_physics_collision_overlay(self, enabled: bool) -> None:
        self._pending_collision_overlay = bool(enabled)
        if enabled:
            self._on_physics_status(
                "Collision wire overlay shows authored asset colliders plus base-scene box colliders."
            )
        if self._renderer:
            self._renderer.set_collision_overlay_enabled(self._pending_collision_overlay)
            if self._last_bounds:
                self._renderer.set_collision_proxy_bounds(self._last_bounds)
            self._renderer.request_render()

    def set_physics_grab_force(self, amount: float) -> None:
        self._physics_grab_force_amount = self._sanitize_grab_force_amount(amount)
        self._physics.set_grab_force_amount(self._physics_grab_force_amount)

    def shutdown(self, timeout_ms: int = 20000) -> bool:
        self._load_generation += 1
        self._loading_asset = False
        self._set_loading(False)
        self._physics.shutdown()

        if not self._renderer:
            return True

        renderer = self._renderer
        if not renderer.shutdown(timeout_ms=timeout_ms):
            return False

        self._disconnect_renderer(renderer)
        self._renderer = None
        return True

    def _disconnect_renderer(self, renderer) -> None:
        for signal, slot in (
            (renderer.frame_ready, self._on_frame),
            (renderer.status_changed, self._on_status),
            (renderer.error_occurred, self._on_error),
            (renderer.fps_updated, self.fps_updated),
            (renderer.fps_updated, self._canvas.update_fps),
            (renderer.bounds_ready, self._on_bounds_ready),
            (renderer.loading_started, self._on_loading_started),
            (renderer.loading_progress, self._on_loading_progress),
            (renderer.loading_finished, self._on_loading_finished),
        ):
            try:
                signal.disconnect(slot)
            except TypeError:
                pass

    def _ensure_renderer(self):
        if self._renderer:
            return self._renderer

        self._set_loading(True, "Starting OVRTX renderer...")
        self.loading_changed.emit(True, "Starting OVRTX renderer...")

        from core.ovrtx_renderer import OVRTXRenderer

        renderer = OVRTXRenderer()
        renderer.frame_ready.connect(self._on_frame)
        renderer.status_changed.connect(self._on_status)
        renderer.error_occurred.connect(self._on_error)
        renderer.fps_updated.connect(self.fps_updated)
        renderer.fps_updated.connect(self._canvas.update_fps)
        renderer.bounds_ready.connect(self._on_bounds_ready)
        renderer.loading_started.connect(self._on_loading_started)
        renderer.loading_progress.connect(self._on_loading_progress)
        renderer.loading_finished.connect(self._on_loading_finished)

        size = self._canvas.size()
        renderer.set_resolution(size.width(), size.height())
        renderer.set_camera_transform(self._camera.get_transform())
        renderer.set_dome_intensity(self._pending_dome_intensity)
        renderer.set_directional_light(*self._pending_dir_light)
        renderer.set_base_scene(self._pending_base_scene)
        renderer.set_collision_overlay_enabled(self._pending_collision_overlay)
        if self._last_bounds:
            renderer.set_collision_proxy_bounds(self._last_bounds)

        self._renderer = renderer
        return renderer

    def _load_usd_after_overlay(self, path: str, generation: int) -> None:
        if generation != self._load_generation:
            return
        renderer = self._ensure_renderer()
        self._push_camera()
        renderer.load_stage(path)
        self.asset_loaded.emit(path)

    def _on_frame(self, img: QImage) -> None:
        self._canvas.set_image(img)

    def _on_status(self, msg: str) -> None:
        self.status_msg.emit(msg)
        self._canvas.set_overlay_text(msg if not self._canvas.has_image else "")

    def _on_error(self, msg: str) -> None:
        text = f"Warning: {msg}"
        self._loading_asset = False
        self.status_msg.emit(text)
        self.loading_changed.emit(False, text)
        self._set_loading(False)
        self._canvas.set_overlay_text(text)

    def _on_loading_started(self, msg: str) -> None:
        self.status_msg.emit(msg)
        self.loading_changed.emit(True, msg)
        self._set_loading(True, msg)

    def _on_loading_progress(self, value: int, msg: str) -> None:
        self.status_msg.emit(msg)
        self.loading_changed.emit(True, msg)
        self._set_loading(True, msg, None if value <= 0 else value)

    def _on_loading_finished(self, ok: bool, msg: str) -> None:
        self._loading_asset = False
        self.status_msg.emit(msg)
        self.loading_changed.emit(False, msg)
        self._set_loading(False)
        if not ok:
            self._canvas.set_overlay_text(msg)
            return
        self._cook_physics_after_load()

    def _on_bounds_ready(self, bounds: dict) -> None:
        self._last_bounds = bounds
        self._physics.configure_asset(bounds, usd_source=self._current_usd_source)
        self._physics_current_transform = np.eye(4, dtype=np.float64)
        if self._renderer:
            self._renderer.set_collision_proxy_bounds(bounds)
        center = np.array(bounds.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        extent = float(bounds.get("extent", 1.0))
        self._camera.frame_bounds(center, extent)
        if not self._loading_asset:
            self._push_camera()
            self._cook_physics_after_load()

    def _cook_physics_after_load(self) -> None:
        if self._loading_asset or self._physics_auto_cook_started:
            return
        if self._last_bounds is None or not self._current_usd_source:
            return
        if self._physics.has_scene or self._physics_cooking_active:
            return

        self._physics_auto_cook_started = True
        self._begin_physics_progress("Cooking physics colliders for loaded asset...", 0)
        if not self._physics.cook_colliders():
            self._physics_cooking_active = False
            self.loading_changed.emit(False, self._physics.status_text)
            self._set_loading(False)

    def _on_physics_cooking_progress(self, value: int, msg: str) -> None:
        if value >= 100:
            return
        self._physics_cooking_active = True
        self.physics_status_changed.emit(msg)
        self.status_msg.emit(msg)
        self.loading_changed.emit(True, msg)
        self._set_loading(True, msg, value)

    def _on_physics_cooking_finished(self, ok: bool, msg: str) -> None:
        _ = ok
        self._physics_cooking_active = False
        self.physics_status_changed.emit(msg)
        self.status_msg.emit(msg)
        self.loading_changed.emit(False, msg)
        self._set_loading(False)

    def _begin_physics_progress(self, text: str, progress: int = 0) -> None:
        self._physics_cooking_active = True
        self.physics_status_changed.emit(text)
        self.status_msg.emit(text)
        self.loading_changed.emit(True, text)
        self._set_loading(True, text, progress)

    def _push_camera(self) -> None:
        if not self._renderer:
            return
        self._renderer.set_camera_transform(self._camera.get_transform())
        self._renderer.request_render()

    def _on_physics_pose(self, matrix: np.ndarray) -> None:
        self._physics_current_transform = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
        if self._renderer:
            self._renderer.set_asset_transform(self._physics_current_transform)
            self._renderer.request_render()

    def _on_physics_status(self, msg: str) -> None:
        self.physics_status_changed.emit(msg)
        self.status_msg.emit(msg)

    def mousePressEvent(self, event) -> None:
        if self._try_start_physics_grab(event):
            return
        self._last_pos = event.pos()
        self._active_button = event.button()
        self._active_mode = self._navigation_mode(event.button(), event.modifiers())
        self.setFocus()
        self._canvas.setFocus()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._physics_grabbing:
            self._update_physics_grab(event.pos())
            event.accept()
            return

        if self._last_pos is None or not self._active_mode:
            return

        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()

        if self._active_mode == "orbit":
            self._camera.orbit(dx, dy)
        elif self._active_mode == "pan":
            self._camera.pan(dx, dy)
        elif self._active_mode == "zoom":
            self._camera.zoom(-dy * 0.1)
        elif self._active_mode == "look":
            self._camera.look(dx, dy)

        self._push_camera()
        self._last_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._physics_grabbing:
            self._finish_physics_grab(drop=True, pos=event.pos())
            event.accept()
            return
        self._last_pos = None
        self._active_button = None
        self._active_mode = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.reset_camera()
            event.accept()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        if delta == 0 and not event.pixelDelta().isNull():
            delta = event.pixelDelta().y() / 240.0
        self._camera.zoom(delta)
        self._push_camera()
        event.accept()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        speed = 1.0
        if event.modifiers() & Qt.ShiftModifier:
            speed = 4.0
        elif event.modifiers() & Qt.ControlModifier:
            speed = 0.25

        if key == Qt.Key_F:
            self.reset_camera()
            event.accept()
            return
        if key == Qt.Key_R:
            self._camera.reset()
            self._push_camera()
            event.accept()
            return

        movement = {
            Qt.Key_W: (0.0, 0.0, speed),
            Qt.Key_S: (0.0, 0.0, -speed),
            Qt.Key_A: (-speed, 0.0, 0.0),
            Qt.Key_D: (speed, 0.0, 0.0),
            Qt.Key_Q: (0.0, -speed, 0.0),
            Qt.Key_E: (0.0, speed, 0.0),
        }.get(key)
        if movement:
            self._camera.fly(*movement)
            self._push_camera()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Shift and self._physics_grabbing:
            self._finish_physics_grab(drop=True)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_loading_overlay()
        if self._renderer:
            size = self._canvas.size()
            self._renderer.set_resolution(size.width(), size.height())
            self._renderer.request_render()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._canvas:
            if event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
                return True
            if event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event)
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True
            if event.type() == QEvent.MouseButtonDblClick:
                self.mouseDoubleClickEvent(event)
                return True
            if event.type() == QEvent.Wheel:
                self.wheelEvent(event)
                return True
            if event.type() == QEvent.KeyPress:
                self.keyPressEvent(event)
                return True
            if event.type() == QEvent.KeyRelease:
                self.keyReleaseEvent(event)
                return True
        return super().eventFilter(obj, event)

    def _show_hint(self) -> None:
        self._canvas.set_overlay_text(
            "Select an asset from the browser to load it\n\n"
            "Alt+LMB - Tumble   |   Alt+MMB - Pan   |   Alt+RMB - Dolly\n"
            "RMB - Look   |   WASD/QE - Fly   |   Shift+LMB - Grab   |   F - Frame"
        )

    def _hide_hint(self) -> None:
        self._canvas.set_overlay_text("")

    def _try_start_physics_grab(self, event) -> bool:
        if self._last_bounds is None:
            return False
        if event.button() != Qt.LeftButton:
            return False
        modifiers = event.modifiers()
        if not (modifiers & Qt.ShiftModifier) or modifiers & Qt.AltModifier:
            return False

        self._physics_grabbing = True
        self._physics_drag_start = event.pos()
        self._physics_drag_matrix = np.array(self._physics_current_transform, dtype=np.float64, copy=True)
        self._physics_grab_anchor = self._select_grab_anchor(event.pos())
        self._physics_grab_target_start = self._anchor_world(self._physics_drag_matrix, self._physics_grab_anchor)
        _right, _up, forward = self._camera._camera_axes()
        self._physics_grab_plane_normal = np.array(forward, dtype=np.float64, copy=True)
        self._physics_grab_pointer_offset = np.zeros(3, dtype=np.float64)
        pointer_target = self._grab_target_from_pointer(event.pos())
        self._physics_grab_pointer_offset = (
            self._physics_grab_target_start - pointer_target
            if pointer_target is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._physics_grab_last_target = np.array(self._physics_grab_target_start, dtype=np.float64, copy=True)
        self._physics_grab_last_time = time.monotonic()
        self._physics_grab_velocity = np.zeros(3, dtype=np.float64)
        if not self._physics.begin_magnet(
            self._physics_grab_anchor,
            self._physics_grab_target_start,
            self._physics_grab_velocity,
        ):
            self._physics_grabbing = False
            return False
        self.setFocus()
        self._canvas.setFocus()
        self._on_physics_status("Grabbed a physics corner. Drag to lift; flick and release to throw.")
        event.accept()
        return True

    def _update_physics_grab(self, pos: QPoint) -> None:
        if self._physics_drag_start is None:
            return

        target = self._grab_target_from_pointer(pos)
        if target is None:
            target = self._fallback_grab_target(pos)
        target = self._scaled_grab_target(target)

        now = time.monotonic()
        dt = max(now - self._physics_grab_last_time, 1.0 / 120.0)
        raw_velocity = (target - self._physics_grab_last_target) / dt
        self._physics_grab_velocity = self._clamp_vector(
            self._physics_grab_velocity * 0.6 + raw_velocity * 0.4,
            GRAB_THROW_VELOCITY_LIMIT * max(1.0, np.sqrt(self._physics_grab_force_amount)),
        )
        self._physics_grab_last_target = np.array(target, dtype=np.float64, copy=True)
        self._physics_grab_last_time = now
        self._physics.update_magnet(target, self._physics_grab_velocity)

    def _fallback_grab_target(self, pos: QPoint) -> np.ndarray:
        dx = pos.x() - self._physics_drag_start.x()
        dy = pos.y() - self._physics_drag_start.y()
        right, up, _forward = self._camera._camera_axes()
        extent = 1.0
        if self._last_bounds:
            extent = float(self._last_bounds.get("extent", 1.0))
        scale = max(self._camera.radius * 0.0018, extent * 0.003, 0.005)
        delta = (dx * right - dy * up) * scale
        return self._physics_grab_target_start + delta

    def _grab_target_from_pointer(self, pos: QPoint) -> Optional[np.ndarray]:
        ray_origin, ray_direction = self._pointer_ray(pos)
        plane_normal = np.array(self._physics_grab_plane_normal, dtype=np.float64).reshape(3)
        denom = float(np.dot(ray_direction, plane_normal))
        if abs(denom) < 1.0e-5:
            return None
        distance = float(np.dot(self._physics_grab_target_start - ray_origin, plane_normal) / denom)
        if distance <= 0.0 or not np.isfinite(distance):
            return None
        return ray_origin + ray_direction * distance + self._physics_grab_pointer_offset

    def _pointer_ray(self, pos: QPoint) -> tuple[np.ndarray, np.ndarray]:
        width = max(float(self._canvas.width()), 1.0)
        height = max(float(self._canvas.height()), 1.0)
        ndc_x = (float(pos.x()) / width) * 2.0 - 1.0
        ndc_y = 1.0 - (float(pos.y()) / height) * 2.0

        aspect = width / height
        half_horizontal = np.tan(np.arctan(CAMERA_HORIZONTAL_APERTURE_MM / (2.0 * CAMERA_FOCAL_LENGTH_MM)))
        half_vertical = half_horizontal / max(aspect, 1.0e-6)
        right, up, forward = self._camera._camera_axes()
        direction = forward + right * ndc_x * half_horizontal + up * ndc_y * half_vertical
        direction_norm = max(float(np.linalg.norm(direction)), 1.0e-9)
        return self._camera.eye.copy(), direction / direction_norm

    def _scaled_grab_target(self, target: np.ndarray) -> np.ndarray:
        amount = self._physics_grab_force_amount
        if abs(amount - 1.0) < 1.0e-6:
            return np.array(target, dtype=np.float64, copy=True)
        offset = np.asarray(target, dtype=np.float64) - self._physics_grab_target_start
        return self._physics_grab_target_start + offset * amount

    def _finish_physics_grab(self, drop: bool, pos: Optional[QPoint] = None) -> None:
        if pos is not None:
            self._update_physics_grab(pos)

        self._physics_grabbing = False
        self._physics_drag_start = None
        self._last_pos = None
        self._active_button = None
        self._active_mode = None

        if drop:
            self._physics.end_magnet(self._physics_grab_velocity)
            self._on_physics_status("Released asset into physics.")
        else:
            self._physics.end_magnet(np.zeros(3, dtype=np.float64))

    def _select_grab_anchor(self, pos: QPoint) -> np.ndarray:
        size = np.ones(3, dtype=np.float64)
        if self._last_bounds:
            try:
                size = np.array(self._last_bounds.get("size", size), dtype=np.float64).reshape(3)
            except Exception:
                size = np.ones(3, dtype=np.float64)
        half = np.maximum(np.abs(size) * 0.5, np.array([0.05, 0.05, 0.05], dtype=np.float64))

        matrix = np.array(self._physics_current_transform, dtype=np.float64, copy=True).reshape(4, 4)
        body_center = self._body_center_world(matrix)
        right, up, _forward = self._camera._camera_axes()
        eye_dir = self._camera.eye - body_center
        eye_norm = max(float(np.linalg.norm(eye_dir)), 1.0e-6)
        pointer = np.array(
            [
                (float(pos.x()) / max(float(self._canvas.width()), 1.0)) * 2.0 - 1.0,
                1.0 - (float(pos.y()) / max(float(self._canvas.height()), 1.0)) * 2.0,
            ],
            dtype=np.float64,
        )

        best_anchor = np.array([half[0], half[1], half[2]], dtype=np.float64)
        best_score = -1.0e9
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    anchor = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                    offset = anchor @ matrix[:3, :3]
                    footprint = max(float(np.linalg.norm(offset)), 1.0e-6)
                    screen_score = (np.dot(offset, right) * pointer[0] + np.dot(offset, up) * pointer[1]) / footprint
                    facing_score = np.dot(offset, eye_dir) / (footprint * eye_norm)
                    score = screen_score + facing_score * 0.4
                    if score > best_score:
                        best_score = score
                        best_anchor = anchor
        return best_anchor

    def _anchor_world(self, matrix: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return self._body_center_world(transform) + np.asarray(anchor, dtype=np.float64).reshape(3) @ transform[:3, :3]

    def _body_center_world(self, matrix: np.ndarray) -> np.ndarray:
        transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        center = np.zeros(3, dtype=np.float64)
        if self._last_bounds:
            try:
                center = np.array(self._last_bounds.get("center", center), dtype=np.float64).reshape(3)
            except Exception:
                center = np.zeros(3, dtype=np.float64)
        return center @ transform[:3, :3] + transform[3, :3]

    @staticmethod
    def _clamp_vector(vector: np.ndarray, limit: float) -> np.ndarray:
        arr = np.array(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(arr))
        if norm > max(float(limit), 1.0e-6):
            arr *= float(limit) / norm
        return arr

    @staticmethod
    def _sanitize_grab_force_amount(value: float) -> float:
        try:
            amount = float(value)
        except Exception:
            amount = DEFAULT_GRAB_FORCE_AMOUNT
        if not np.isfinite(amount):
            amount = DEFAULT_GRAB_FORCE_AMOUNT
        return max(0.25, min(5.0, amount))

    def _set_loading(self, active: bool, text: str = "", progress: Optional[int] = None) -> None:
        self._canvas.set_loading(False)
        self._loading_overlay.set_loading(active, text, progress)
        if active:
            self._position_loading_overlay()
            self._loading_overlay.raise_()
            self._loading_overlay.repaint()

    def _position_loading_overlay(self) -> None:
        self._loading_overlay.setGeometry(self._canvas.geometry())

    @staticmethod
    def _navigation_mode(button, modifiers) -> Optional[str]:
        alt_down = bool(modifiers & Qt.AltModifier)

        if alt_down:
            if button == Qt.LeftButton:
                return "orbit"
            if button == Qt.MiddleButton:
                return "pan"
            if button == Qt.RightButton:
                return "zoom"

        if button == Qt.MiddleButton:
            return "pan"
        if button == Qt.RightButton:
            return "look"
        return None


class _RenderCanvas(QLabel):
    """Paints the rendered frame and lightweight HUD overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewport_label")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {COLOR_VIEWPORT_BG};")
        self._pixmap: Optional[QPixmap] = None
        self._overlay_text = ""
        self._fps = 0.0
        self._loading = False
        self._loading_text = ""
        self._loading_progress: Optional[int] = None
        self._loading_tick = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._tick_loading)

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None

    def set_image(self, img: QImage) -> None:
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def set_overlay_text(self, text: str) -> None:
        self._overlay_text = text
        self.update()

    def set_loading(self, active: bool, text: str = "", progress: Optional[int] = None) -> None:
        self._loading = active
        self._loading_text = text
        self._loading_progress = None if progress is None else max(0, min(100, int(progress)))

        if active and not self._loading_timer.isActive():
            self._loading_timer.start(33)
        elif not active and self._loading_timer.isActive():
            self._loading_timer.stop()

        self.update()

    def update_fps(self, fps: float) -> None:
        self._fps = fps
        self.update()

    def _tick_loading(self) -> None:
        self._loading_tick = (self._loading_tick + 1) % 10000
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(COLOR_VIEWPORT_BG))

        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        if self._fps > 0:
            painter.setFont(QFont("Segoe UI", 10))
            painter.setPen(QColor(COLOR_ACCENT))
            painter.drawText(
                self.rect().adjusted(0, 8, -10, 0),
                Qt.AlignTop | Qt.AlignRight,
                f"{self._fps:.1f} fps",
            )

        if self._loading:
            self._draw_loading_overlay(painter)
        elif self._overlay_text:
            painter.setFont(QFont("Segoe UI", 11))
            painter.setPen(QColor(COLOR_TEXT_SECONDARY))
            painter.fillRect(self.rect(), QColor(0, 0, 0, 150))
            painter.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignCenter | Qt.TextWordWrap, self._overlay_text)

        if self._pixmap:
            painter.setPen(QPen(QColor(COLOR_ACCENT + "40"), 2))
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

        painter.end()

    def _draw_loading_overlay(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColor(0, 0, 0, 165))

        bar_width = min(520, max(240, self.width() - 96))
        bar_height = 10
        bar_x = (self.width() - bar_width) // 2
        bar_y = max(70, self.height() - 54)

        painter.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        painter.setPen(QColor(COLOR_TEXT_SECONDARY))
        painter.drawText(
            32,
            max(8, bar_y - 56),
            max(120, self.width() - 64),
            46,
            Qt.AlignBottom | Qt.AlignHCenter | Qt.TextWordWrap,
            self._loading_text or "Loading asset in OVRTX...",
        )

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(45, 45, 45, 230))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 3, 3)

        fill_x = bar_x
        fill_width = max(64, bar_width // 4)
        if self._loading_progress is not None:
            fill_width = max(8, int(bar_width * self._loading_progress / 100.0))
        else:
            fill_x = bar_x + ((self._loading_tick * 6) % (bar_width + fill_width)) - fill_width

        painter.save()
        painter.setClipRect(bar_x, bar_y, bar_width, bar_height)
        painter.setBrush(QColor(COLOR_ACCENT))
        painter.drawRoundedRect(fill_x, bar_y, fill_width, bar_height, 3, 3)

        highlight_width = max(56, bar_width // 5)
        highlight_x = bar_x + ((self._loading_tick * 7) % (bar_width + highlight_width)) - highlight_width
        painter.setBrush(QColor(255, 255, 255, 70))
        painter.drawRect(highlight_x, bar_y, highlight_width, bar_height)
        painter.restore()

        if self._loading_progress is not None:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(COLOR_TEXT_SECONDARY))
            painter.drawText(
                bar_x,
                bar_y + 26,
                bar_width,
                18,
                Qt.AlignCenter,
                f"{self._loading_progress}%",
            )

    def sizeHint(self) -> QSize:
        return QSize(960, 540)


class _ViewportLoadingOverlay(QWidget):
    """A real widget overlay so loading state is visible before OVRTX blocks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: rgba(0, 0, 0, 165);")

        root = QVBoxLayout(self)
        root.setContentsMargins(48, 0, 48, 30)
        root.setSpacing(8)
        root.addStretch(1)

        self._label = QLabel("Loading asset in OVRTX...")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; background: transparent; "
            "font-size: 12px; font-weight: 600;"
        )
        root.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFixedHeight(14)
        self._bar.setStyleSheet(
            "QProgressBar { background: #2d2d2d; border: 1px solid #464646; "
            "border-radius: 4px; color: #d8d8d8; text-align: center; font-size: 9px; }"
            f"QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 3px; }}"
        )
        root.addWidget(self._bar)

    def set_loading(self, active: bool, text: str = "", progress: Optional[int] = None) -> None:
        if not active:
            self.hide()
            return

        self._label.setText(text or "Loading asset in OVRTX...")
        if progress is None:
            self._bar.setRange(0, 0)
        else:
            self._bar.setRange(0, 100)
            self._bar.setValue(max(0, min(100, int(progress))))
        self.show()
