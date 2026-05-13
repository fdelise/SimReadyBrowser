"""Interactive Qt viewport for OVRTX rendered frames."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QProcess, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

from core.camera_controller import SphericalCamera
from core.physics_controller import PhysicsController
from core.scene_explorer_model import build_scene_tree
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_BG_WIDGET,
    COLOR_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_VIEWPORT_BG,
)

LOAD_START_DELAY_MS = 150
CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
GRAB_THROW_VELOCITY_LIMIT = 10.0
DEFAULT_GRAB_FORCE_AMOUNT = 2.0
MIN_GRAB_FORCE_AMOUNT = 0.25
MAX_GRAB_FORCE_AMOUNT = 100.0


class ViewportWidget(QWidget):
    """Interactive 3D viewport powered by OVRTX frame readback."""

    asset_loaded = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    status_msg = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)
    physics_status_changed = pyqtSignal(str)
    physics_running_changed = pyqtSignal(bool)
    scene_tree_changed = pyqtSignal(object)
    scene_part_selection_changed = pyqtSignal(str)

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
        self._pending_dome_environment = "flat"
        self._pending_dir_light = (0.8, 45.0, 60.0)
        self._pending_base_scene = "plane"
        self._pending_collision_overlay = False
        self._pending_asset_instance_count = 1
        self._load_generation = 0
        self._loading_asset = False
        self._physics_cooking_active = False
        self._physics_auto_cook_started = False
        self._auto_cook_physics_after_load = True
        self._current_usd_source: Optional[str] = None
        self._current_load_name = ""
        self._current_scene_items: list[dict] = []
        self._scene_explorer_process: Optional[QProcess] = None
        self._scene_explorer_buffer = ""
        self._scene_explorer_error = ""
        self._scene_explorer_refs: list[str] = []
        self._scene_explorer_generation = 0
        self._first_frame_timer_start: Optional[float] = None
        self._physics = PhysicsController(self)
        self._physics.pose_changed.connect(self._on_physics_pose)
        self._physics.status_changed.connect(self._on_physics_status)
        self._physics.running_changed.connect(self.physics_running_changed)
        self._physics.cooking_progress.connect(self._on_physics_cooking_progress)
        self._physics.cooking_finished.connect(self._on_physics_cooking_finished)
        self._physics_current_transform = np.eye(4, dtype=np.float64)
        self._physics_body_transforms: list[dict] = []
        self._physics_grabbing = False
        self._physics_grab_body_path: Optional[str] = None
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

    def load_usd(self, path: str | Path, *, auto_cook_physics: bool = True) -> None:
        self._hide_hint()
        self.setFocus(Qt.OtherFocusReason)
        self._canvas.setFocus(Qt.OtherFocusReason)
        self._auto_cook_physics_after_load = bool(auto_cook_physics)
        self._current_usd_source = str(path)
        self._current_load_name = Path(str(path).split("?", 1)[0]).name
        self._current_scene_items = [{"source": str(path), "name": self._current_load_name, "key": str(path)}]
        self._reset_scene_explorer("Loading scene explorer data...")
        self._begin_first_frame_timer()
        self._last_bounds = None
        self._camera.reset()
        self._reset_physics_interaction_state()
        self._physics.clear_asset()
        self._physics_cooking_active = False
        self._physics_auto_cook_started = False
        self._pending_asset_instance_count = 1
        if self._renderer:
            self._renderer.set_asset_instance_count(1)
            self._renderer.set_asset_transform(np.eye(4, dtype=np.float64))
            self._renderer.set_physics_body_transforms([])
        self._loading_asset = True
        self._load_generation += 1
        generation = self._load_generation
        self._set_loading(True, "Queued asset for OVRTX...")
        self.loading_changed.emit(True, "Queued asset for OVRTX...")
        QTimer.singleShot(
            LOAD_START_DELAY_MS,
            lambda source=str(path), gen=generation: self._load_usd_after_overlay(source, gen),
        )

    def load_usds(self, items) -> None:
        normalized = self._normalize_stage_items(items)
        if not normalized:
            return
        if len(normalized) == 1:
            self.load_usd(normalized[0]["source"])
            return

        self._hide_hint()
        self.setFocus(Qt.OtherFocusReason)
        self._canvas.setFocus(Qt.OtherFocusReason)
        self._auto_cook_physics_after_load = True
        self._current_usd_source = None
        self._current_load_name = ""
        self._current_scene_items = list(normalized)
        self._reset_scene_explorer("Loading scene explorer data...")
        self._begin_first_frame_timer()
        self._last_bounds = None
        self._camera.reset()
        self._reset_physics_interaction_state()
        self._physics.clear_asset()
        self._physics_cooking_active = False
        self._physics_auto_cook_started = False
        self._pending_asset_instance_count = 1
        if self._renderer:
            self._renderer.set_asset_instance_count(1)
            self._renderer.set_asset_transform(np.eye(4, dtype=np.float64))
            self._renderer.set_physics_body_transforms([])
        self._loading_asset = True
        self._load_generation += 1
        generation = self._load_generation
        msg = f"Queued {len(normalized)} selected assets for OVRTX..."
        self._set_loading(True, msg)
        self.loading_changed.emit(True, msg)
        QTimer.singleShot(
            LOAD_START_DELAY_MS,
            lambda stage_items=normalized, gen=generation: self._load_usds_after_overlay(stage_items, gen),
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

    def set_dome_environment(self, mode: str) -> None:
        value = str(mode or "flat")
        self._pending_dome_environment = value
        if self._renderer:
            self._renderer.set_dome_environment(value)
            self._renderer.request_render()

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

    def restart_physics_engine(self) -> None:
        if self._last_bounds is None:
            self._on_physics_status("Load an asset before restarting the PhysX engine.")
            self.physics_running_changed.emit(False)
            return
        self._begin_physics_progress("Restarting PhysX engine...", 0)
        if not self._physics.restart_engine(play=True):
            self._physics_cooking_active = False
            self.loading_changed.emit(False, self._physics.status_text)
            self._set_loading(False)

    def drop_physics(self, count: int = 1) -> None:
        if self._last_bounds is None:
            self._on_physics_status("Load an asset before dropping physics.")
            self.physics_running_changed.emit(False)
            return
        try:
            drop_count = int(count)
        except Exception:
            drop_count = 1
        drop_count = max(1, min(100, drop_count))
        if isinstance(self._last_bounds, dict):
            try:
                selected_count = len(list(self._last_bounds.get("_asset_sources", []) or []))
            except TypeError:
                selected_count = 0
            if selected_count > 1:
                drop_count = max(drop_count, min(100, selected_count))
        self._pending_asset_instance_count = drop_count
        if self._renderer:
            self._renderer.set_asset_instance_count(drop_count)
        if not self._physics.has_scene and not self._physics_cooking_active:
            label = "asset" if drop_count == 1 else "assets"
            self._begin_physics_progress(f"Preparing physics colliders to drop {drop_count} {label}...", 0)
        if not self._physics.drop_asset(drop_count):
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

    def set_physics_drop_options(self, spacing: float, randomness: float) -> None:
        self._physics.set_drop_options(float(spacing), float(randomness))

    def set_physics_ccd_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        previous = self._physics.ccd_enabled
        live_scene = self._physics.has_scene
        was_running = self._physics.is_running
        self._physics.set_ccd_enabled(enabled)
        if live_scene and previous != enabled:
            self._begin_physics_progress("Restarting PhysX engine with CCD setting...", 0)
            if not self._physics.restart_engine(play=was_running):
                self._physics_cooking_active = False
                self.loading_changed.emit(False, self._physics.status_text)
                self._set_loading(False)

    def set_physx_steps_per_second(self, value: int) -> None:
        self._physics.set_steps_per_second(int(value))

    def set_physx_substeps(self, value: int) -> None:
        self._physics.set_substeps(int(value))

    def set_physx_device_mode(self, mode: str) -> None:
        self._physics.set_device_mode(str(mode or "auto"))

    def reset_physx_settings(self) -> None:
        self._physics.reset_physx_settings()

    def refresh_scene_explorer(self) -> None:
        self._refresh_scene_explorer(force=True)

    def select_scene_part(self, path: str) -> None:
        text = str(path or "").strip()
        if not text:
            return
        self.scene_part_selection_changed.emit(text)
        self.status_msg.emit(f"Selected scene prim {text}")

    def set_scene_part_property(self, path: str, property_name: str, value) -> None:
        text = str(path or "").strip()
        prop = str(property_name or "").strip()
        if not text or not prop:
            return
        if self._renderer:
            self._renderer.set_scene_part_property(text, prop, value)

    def shutdown(self, timeout_ms: int = 20000) -> bool:
        self._load_generation += 1
        self._loading_asset = False
        self._set_loading(False)
        self._release_scene_explorer_process()
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
        renderer.set_dome_environment(self._pending_dome_environment)
        renderer.set_directional_light(*self._pending_dir_light)
        renderer.set_base_scene(self._pending_base_scene)
        renderer.set_collision_overlay_enabled(self._pending_collision_overlay)
        renderer.set_asset_instance_count(self._pending_asset_instance_count)
        if self._physics_body_transforms:
            renderer.set_physics_body_transforms(self._physics_body_transforms)
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

    def _load_usds_after_overlay(self, items: list[dict], generation: int) -> None:
        if generation != self._load_generation:
            return
        renderer = self._ensure_renderer()
        self._push_camera()
        renderer.load_stage_items(items)
        self.asset_loaded.emit(f"{len(items)} assets")

    def _on_frame(self, img: QImage) -> None:
        self._finish_first_frame_timer()
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
        if ok and self._current_load_name and self._current_load_name not in str(msg):
            return
        self._loading_asset = False
        self.status_msg.emit(msg)
        self.loading_changed.emit(False, msg)
        self._set_loading(False)
        if not ok:
            self._canvas.set_overlay_text(msg)
            return
        self._cook_physics_after_load()

    def _on_bounds_ready(self, bounds: dict) -> None:
        source = str(bounds.get("_usd_source", "") or "") if isinstance(bounds, dict) else ""
        if source and self._current_usd_source and source != self._current_usd_source:
            return
        self._last_bounds = bounds
        usd_source = None if bounds.get("_multi_asset") else self._current_usd_source
        self._physics.configure_asset(bounds, usd_source=usd_source)
        self._physics_current_transform = np.eye(4, dtype=np.float64)
        if self._renderer:
            self._renderer.set_collision_proxy_bounds(bounds)
        self._refresh_scene_explorer()
        center = np.array(bounds.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        extent = float(bounds.get("extent", 1.0))
        self._camera.frame_bounds(center, extent)
        if not self._loading_asset:
            self._push_camera()
            self._cook_physics_after_load()

    def _cook_physics_after_load(self) -> None:
        if not self._auto_cook_physics_after_load:
            self.physics_status_changed.emit("Physics colliders will cook on Play or Drop.")
            return
        if self._loading_asset or self._physics_auto_cook_started:
            return
        if self._last_bounds is None:
            return
        has_multi_sources = bool(self._last_bounds.get("_asset_sources")) if isinstance(self._last_bounds, dict) else False
        if not self._current_usd_source and not has_multi_sources:
            return
        if self._physics.has_scene or self._physics_cooking_active:
            return

        self._physics_auto_cook_started = True
        if has_multi_sources:
            count = len(self._last_bounds.get("_asset_sources", []) or [])
            self._begin_physics_progress(f"Cooking physics colliders for {count} selected assets...", 0)
        else:
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

    def _reset_scene_explorer(self, message: str) -> None:
        self._release_scene_explorer_process()
        self.scene_tree_changed.emit({"status": str(message or "Loading scene explorer data..."), "roots": []})

    def _refresh_scene_explorer(self, force: bool = False) -> None:
        items = list(self._current_scene_items or [])
        if not items:
            self.scene_tree_changed.emit({"status": "Load an asset to inspect the USD scene.", "roots": []})
            return

        refs = self._scene_asset_refs(items)
        if not refs:
            self.scene_tree_changed.emit(build_scene_tree(items, [], self._last_bounds or {}))
            return

        cached_payloads: list[dict] = []
        for ref in refs:
            discovery = PhysicsController._cached_discovery(ref)
            if discovery is None:
                cached_payloads.append({})
            else:
                cached_payloads.append(PhysicsController._discovery_to_payload(discovery))

        self.scene_tree_changed.emit(build_scene_tree(items, cached_payloads, self._last_bounds or {}))
        self._start_scene_explorer_discovery(refs)

    def _start_scene_explorer_discovery(self, asset_refs: list[str]) -> None:
        self._release_scene_explorer_process()
        refs = [str(ref or "").strip() for ref in asset_refs if str(ref or "").strip()]
        if not refs:
            return

        process = QProcess(self)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.readyReadStandardOutput.connect(self._on_scene_explorer_stdout)
        process.readyReadStandardError.connect(self._on_scene_explorer_stderr)
        process.finished.connect(self._on_scene_explorer_finished)
        self._scene_explorer_process = process
        self._scene_explorer_buffer = ""
        self._scene_explorer_error = ""
        self._scene_explorer_refs = list(refs)
        self._scene_explorer_generation = self._load_generation
        helper_python = PhysicsController._usd_discovery_python() or sys.executable
        args = ["-u", "-m", "core.usd_scene_discovery"]
        if len(refs) > 1:
            args.extend(["--multi", *refs])
        else:
            args.extend([refs[0], "/World/Asset"])
        process.start(helper_python, args)
        if not process.waitForStarted(1000):
            detail = process.errorString()
            self._release_scene_explorer_process()
            self.status_msg.emit(f"Scene explorer discovery could not start: {detail}")

    def _on_scene_explorer_stdout(self) -> None:
        process = self._scene_explorer_process
        if process is None:
            return
        self._scene_explorer_buffer += bytes(process.readAllStandardOutput()).decode("utf-8", "replace")

    def _on_scene_explorer_stderr(self) -> None:
        process = self._scene_explorer_process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", "replace").strip()
        if text:
            self._scene_explorer_error = text

    def _on_scene_explorer_finished(self, exit_code: int, _exit_status) -> None:
        process = self._scene_explorer_process
        refs = list(self._scene_explorer_refs)
        generation = self._scene_explorer_generation
        buffer = self._scene_explorer_buffer
        error = self._scene_explorer_error
        self._scene_explorer_process = None
        self._scene_explorer_refs = []
        self._scene_explorer_buffer = ""
        self._scene_explorer_error = ""
        if process is not None:
            process.deleteLater()

        if generation != self._load_generation or refs != self._scene_asset_refs(self._current_scene_items):
            return

        discoveries = self._scene_discoveries_from_stdout(buffer, len(refs))
        if exit_code != 0 or discoveries is None:
            detail = error.splitlines()[-1] if error else "no discovery payload returned"
            self.status_msg.emit(f"Scene explorer is showing asset roots only; part discovery skipped ({detail}).")
            discoveries = [{} for _ref in refs]
        else:
            for ref, payload in zip(refs, discoveries):
                try:
                    PhysicsController._store_discovery(ref, PhysicsController._discovery_from_payload(payload))
                except Exception:
                    pass

        self.scene_tree_changed.emit(build_scene_tree(self._current_scene_items, discoveries, self._last_bounds or {}))

    def _release_scene_explorer_process(self) -> None:
        process = self._scene_explorer_process
        self._scene_explorer_process = None
        self._scene_explorer_refs = []
        self._scene_explorer_buffer = ""
        self._scene_explorer_error = ""
        if process is None:
            return
        try:
            try:
                process.finished.disconnect(self._on_scene_explorer_finished)
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
    def _scene_asset_refs(items: list[dict]) -> list[str]:
        refs: list[str] = []
        try:
            raw_items = list(items or [])
        except TypeError:
            raw_items = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            if not source:
                continue
            refs.append(PhysicsController._usd_asset_reference(source))
        return refs

    @staticmethod
    def _scene_discoveries_from_stdout(text: str, expected_count: int) -> Optional[list[dict]]:
        payload = None
        for line in reversed(str(text or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            return None
        if "discoveries" in payload:
            items = payload.get("discoveries")
            if not isinstance(items, list) or len(items) != expected_count:
                return None
            return [item if isinstance(item, dict) else {} for item in items]
        return [payload] if expected_count == 1 else None

    def _begin_first_frame_timer(self) -> None:
        self._first_frame_timer_start = time.perf_counter()
        self._canvas.set_debug_timing("")

    def _finish_first_frame_timer(self) -> None:
        if self._first_frame_timer_start is None:
            return
        elapsed = max(0.0, time.perf_counter() - self._first_frame_timer_start)
        self._first_frame_timer_start = None
        self._canvas.set_debug_timing(f"First frame: {elapsed:.2f}s")

    def _push_camera(self) -> None:
        if not self._renderer:
            return
        self._renderer.set_camera_transform(self._camera.get_transform())
        self._renderer.request_render()

    def _on_physics_pose(self, payload) -> None:
        if self._loading_asset or not self._physics.has_scene:
            return

        bodies: list[dict] = []
        if isinstance(payload, dict):
            matrix = payload.get("root", np.eye(4, dtype=np.float64))
            bodies = self._normalize_body_transforms(payload.get("bodies", []))
        else:
            matrix = payload

        try:
            root_matrix = np.array(matrix, dtype=np.float64, copy=True).reshape(4, 4)
        except Exception:
            self._handle_unstable_physics("Physics returned an invalid transform; physics was stopped.")
            return
        if not np.all(np.isfinite(root_matrix)):
            self._handle_unstable_physics("Physics returned a non-finite transform; physics was stopped.")
            return

        self._physics_current_transform = root_matrix
        self._physics_body_transforms = bodies
        if self._renderer:
            self._renderer.set_asset_transform(self._physics_current_transform)
            self._renderer.set_physics_body_transforms(self._physics_body_transforms)
            self._renderer.request_render()

    def _on_physics_status(self, msg: str) -> None:
        lower = str(msg or "").lower()
        if "physics was stopped" in lower or "ovphysx worker stopped" in lower or "physics returned" in lower:
            self._reset_physics_interaction_state(clear_renderer=False)
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

        selected_body = self._select_physics_body(event.pos())
        self._physics_grabbing = True
        self._physics_drag_start = event.pos()
        if selected_body is not None:
            self._physics_grab_body_path = str(selected_body["path"])
            self._physics_drag_matrix = np.array(selected_body["matrix"], dtype=np.float64, copy=True).reshape(4, 4)
            self._physics_grab_anchor = self._select_body_grab_anchor(selected_body, event.pos())
            self._physics_grab_target_start = self._body_anchor_world(
                self._physics_drag_matrix,
                self._physics_grab_anchor,
            )
        else:
            self._physics_grab_body_path = None
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
            body_path=self._physics_grab_body_path,
        ):
            self._physics_grabbing = False
            self._physics_grab_body_path = None
            return False
        self.setFocus()
        self._canvas.setFocus()
        if self._physics_grab_body_path:
            name = self._physics_grab_body_path.rsplit("/", 1)[-1]
            self._on_physics_status(f"Grabbed physics body {name}. Drag to pull the joint; flick and release to throw.")
        else:
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
        return np.array(target, dtype=np.float64, copy=True)

    def _finish_physics_grab(self, drop: bool, pos: Optional[QPoint] = None) -> None:
        if pos is not None:
            self._update_physics_grab(pos)

        self._physics_grabbing = False
        self._physics_grab_body_path = None
        self._physics_drag_start = None
        self._last_pos = None
        self._active_button = None
        self._active_mode = None

        if drop:
            self._physics.end_magnet(self._physics_grab_velocity)
            self._on_physics_status("Released asset into physics.")
        else:
            self._physics.end_magnet(np.zeros(3, dtype=np.float64))

    def _reset_physics_interaction_state(self, clear_renderer: bool = True) -> None:
        self._physics_current_transform = np.eye(4, dtype=np.float64)
        self._physics_body_transforms = []
        self._physics_grabbing = False
        self._physics_grab_body_path = None
        self._physics_drag_start = None
        self._physics_grab_velocity = np.zeros(3, dtype=np.float64)
        self._physics_grab_last_target = np.zeros(3, dtype=np.float64)
        if clear_renderer and self._renderer:
            self._renderer.set_asset_transform(self._physics_current_transform)
            self._renderer.set_physics_body_transforms([])

    def _handle_unstable_physics(self, message: str) -> None:
        self._reset_physics_interaction_state(clear_renderer=False)
        self._physics.shutdown()
        self.physics_running_changed.emit(False)
        self._on_physics_status(message)

    @staticmethod
    def _normalize_body_transforms(items) -> list[dict]:
        bodies: list[dict] = []
        try:
            raw_items = list(items or [])
        except TypeError:
            raw_items = []
        for item in raw_items[:10000]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "") or "").strip()
            if not path:
                continue
            try:
                matrix = np.array(item.get("matrix"), dtype=np.float64, copy=True).reshape(4, 4)
            except Exception:
                continue
            if not np.all(np.isfinite(matrix)):
                continue
            bodies.append({"path": path, "matrix": matrix})
        return bodies

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
            if source:
                normalized.append({"source": source, "name": name, "key": key})
        return normalized[:100]

    def _select_physics_body(self, pos: QPoint) -> Optional[dict]:
        if not self._physics_body_transforms:
            return None

        ray_origin, ray_direction = self._pointer_ray(pos)
        extent = 1.0
        if self._last_bounds:
            try:
                extent = max(float(self._last_bounds.get("extent", extent)), 0.1)
            except Exception:
                extent = 1.0
        threshold = max(extent * 0.35, self._camera.radius * 0.08, 0.15)

        best_body = None
        best_score = float("inf")
        for body in self._physics_body_transforms:
            matrix = np.asarray(body["matrix"], dtype=np.float64).reshape(4, 4)
            center = matrix[3, :3]
            to_center = center - ray_origin
            distance_along_ray = float(np.dot(to_center, ray_direction))
            if distance_along_ray <= 0.0 or not np.isfinite(distance_along_ray):
                continue
            closest = ray_origin + ray_direction * distance_along_ray
            ray_distance = float(np.linalg.norm(center - closest))
            if ray_distance > threshold:
                continue
            score = ray_distance + distance_along_ray * 0.002
            if score < best_score:
                best_score = score
                best_body = body
        return best_body

    def _select_body_grab_anchor(self, body: dict, pos: QPoint) -> np.ndarray:
        matrix = np.asarray(body["matrix"], dtype=np.float64).reshape(4, 4)
        ray_origin, ray_direction = self._pointer_ray(pos)
        body_origin = matrix[3, :3]
        depth = max(float(np.dot(body_origin - ray_origin, ray_direction)), 0.05)
        hit_world = ray_origin + ray_direction * depth
        anchor = (hit_world - body_origin) @ matrix[:3, :3].T
        half = self._body_grab_half_extent()
        anchor = np.minimum(np.maximum(anchor, -half), half)
        if float(np.linalg.norm(anchor)) < max(0.025, float(np.min(half)) * 0.2):
            anchor = self._select_matrix_corner_anchor(pos, matrix, half)
        return anchor.astype(np.float64)

    def _body_grab_half_extent(self) -> np.ndarray:
        size = np.ones(3, dtype=np.float64)
        if self._last_bounds:
            try:
                size = np.array(self._last_bounds.get("size", size), dtype=np.float64).reshape(3)
            except Exception:
                size = np.ones(3, dtype=np.float64)
        scale = 0.4 if len(self._physics_body_transforms) > 1 else 1.0
        return np.maximum(np.abs(size) * 0.5 * scale, np.array([0.05, 0.05, 0.05], dtype=np.float64))

    def _select_matrix_corner_anchor(self, pos: QPoint, matrix: np.ndarray, half: np.ndarray) -> np.ndarray:
        body_center = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[3, :3]
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

    @staticmethod
    def _body_anchor_world(matrix: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return transform[3, :3] + np.asarray(anchor, dtype=np.float64).reshape(3) @ transform[:3, :3]

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
        return max(MIN_GRAB_FORCE_AMOUNT, min(MAX_GRAB_FORCE_AMOUNT, amount))

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
        self._debug_timing = ""
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

    def set_debug_timing(self, text: str) -> None:
        self._debug_timing = str(text or "")
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

        self._draw_debug_hud(painter)

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

    def _draw_debug_hud(self, painter: QPainter) -> None:
        lines: list[tuple[str, QColor]] = []
        if self._debug_timing:
            lines.append((self._debug_timing, QColor(COLOR_TEXT_SECONDARY)))
        if self._fps > 0:
            lines.append((f"{self._fps:.1f} fps", QColor(COLOR_ACCENT)))
        if not lines:
            return

        painter.setFont(QFont("Segoe UI", 10))
        top = 8
        for text, color in lines:
            painter.setPen(color)
            painter.drawText(
                self.rect().adjusted(0, top, -10, 0),
                Qt.AlignTop | Qt.AlignRight,
                text,
            )
            top += 17

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
            f"QProgressBar {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 5px; color: {COLOR_TEXT_PRIMARY}; text-align: center; font-size: 9px; }}"
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
