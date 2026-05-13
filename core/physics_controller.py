"""Qt-side OVPhysX controller.

OVPhysX 0.3.7 currently crashes if a PhysX instance is created after Qt has
initialized in this app, so simulation runs in a small worker subprocess. The
Qt process owns UI state and converts worker poses back into OVRTX transforms.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal


PROXY_PATH = "/World/AssetProxy"
AUTHORED_ASSET_NAME = "Asset"
AUTHORED_ASSET_PATH = "/World/Asset"
S3_BUCKET = "omniverse-content-production"
S3_HTTP_ROOT = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com"


AUTHORED_BODY_PATTERNS = [
    f"{AUTHORED_ASSET_PATH}/Geometry/*",
    f"{AUTHORED_ASSET_PATH}/Geometry/*/*",
    AUTHORED_ASSET_PATH,
]
MAX_DISCOVERED_BODY_PATTERNS = 256
USD_DISCOVERY_PYTHON_ENV = "SIMREADY_USD_PYTHON"
ENABLE_SLOW_USD_COLLIDER_DISCOVERY_ENV = "SIMREADY_ENABLE_SLOW_USD_COLLIDER_DISCOVERY"
DISCOVERY_CACHE_VERSION = 9
PHYSICS_MODE_AUTHORED = "authored"
PHYSICS_MODE_PROXY = "proxy"
GROUND_HALF_SIZE = 20.0
GROUND_THICKNESS = 0.5
GROUND_TOP_Z = 0.0
DROP_HEIGHT_MIN = 0.75
DROP_HEIGHT_MAX = 3.0
DROP_HEIGHT_EXTENT_SCALE = 0.35
DROP_HEIGHT_SIZE_SCALE = 0.75
MIN_DROP_ASSETS = 1
MAX_DROP_ASSETS = 100
DEFAULT_DROP_SPACING = 0.20
MIN_DROP_SPACING = 0.0
MAX_DROP_SPACING = 5.0
DEFAULT_DROP_RANDOMNESS = 0.25
MIN_DROP_RANDOMNESS = 0.0
MAX_DROP_RANDOMNESS = 6.0
DEFAULT_DT = 1.0 / 60.0
DEFAULT_SUBSTEPS = 4
ISAAC_PHYSICS_STEPS_PER_SECOND = 60
DEFAULT_PHYSX_STEPS_PER_SECOND = ISAAC_PHYSICS_STEPS_PER_SECOND
USD_STAGE_FRAMES_PER_SECOND = 60
MIN_PHYSX_STEPS_PER_SECOND = 15
MAX_PHYSX_STEPS_PER_SECOND = 240
MIN_PHYSX_SUBSTEPS = 1
MAX_PHYSX_SUBSTEPS = 16
DEFAULT_PHYSX_DEVICE_MODE = "gpu"
PHYSX_DEVICE_MODES = {"auto", "cpu", "gpu"}
DYNAMIC_COLLIDER_APPROXIMATION = "convexDecomposition"
BOX_COLLIDER_APPROXIMATION = "boundingCube"
BASE_SCENES = {"none", "plane", "ramp", "obstacles"}
ESTIMATED_ASSET_DENSITY_KG_M3 = 260.0
MIN_ESTIMATED_MASS_KG = 0.05
MAX_ESTIMATED_MASS_KG = 500.0
ISAAC_DYNAMIC_STATIC_FRICTION = 0.2
ISAAC_DYNAMIC_FRICTION = 1.0
ISAAC_DYNAMIC_RESTITUTION = 0.0
ISAAC_GROUND_STATIC_FRICTION = 0.5
ISAAC_GROUND_DYNAMIC_FRICTION = 0.5
ISAAC_GROUND_RESTITUTION = 0.8
ISAAC_CUBOID_CONTACT_OFFSET = 0.1
ISAAC_CUBOID_REST_OFFSET = 0.0
ISAAC_CUBOID_TORSIONAL_PATCH_RADIUS = 1.0
ISAAC_CUBOID_MIN_TORSIONAL_PATCH_RADIUS = 0.8
DEFAULT_GRAB_FORCE_AMOUNT = 2.0
MIN_GRAB_FORCE_AMOUNT = 0.25
MAX_GRAB_FORCE_AMOUNT = 100.0
MAX_SIM_POSITION = 10000.0

Z_TO_Y_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
Z_TO_Y_MATRIX = np.eye(4, dtype=np.float64)
Z_TO_Y_MATRIX[:3, :3] = Z_TO_Y_ROTATION
Y_TO_Z_MATRIX = Z_TO_Y_MATRIX.T


@dataclass(frozen=True)
class AuthoredColliderDiscovery:
    collision_overrides: str
    body_patterns: list[str]
    body_paths: list[str]
    rigid_body_paths: list[str]
    articulation_paths: list[str]
    collider_count: int
    override_count: int


@dataclass
class PendingPhysicsStart:
    asset_refs: list[str]
    initial_pose: Optional[np.ndarray]
    cook_only: bool


class PhysicsController(QObject):
    """Owns a worker process and streams visual transforms into the viewport."""

    _DISCOVERY_CACHE: dict[str, AuthoredColliderDiscovery] = {}
    _DISCOVERY_CACHE_ORDER: list[str] = []
    _DISCOVERY_CACHE_LIMIT = 64
    _PAYLOAD_TEXT_CACHE: dict[tuple[str, str], str] = {}
    _PAYLOAD_TEXT_CACHE_ORDER: list[tuple[str, str]] = []
    _PAYLOAD_TEXT_CACHE_LIMIT = 256

    pose_changed = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    cooking_progress = pyqtSignal(int, str)
    cooking_finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._send_step)

        self._process: Optional[QProcess] = None
        self._stdout_buffer = ""
        self._intentional_stop = False
        self._worker_ready = False
        self._discovery_process: Optional[QProcess] = None
        self._discovery_buffer = ""
        self._discovery_error = ""
        self._pending_discovery_start: Optional[PendingPhysicsStart] = None
        self._pending_play = False
        self._pending_step_after_start = False
        self._step_in_flight = False
        self._cooking_only = False
        self._startup_progress_active = False

        self._bounds: Optional[dict] = None
        self._usd_source: Optional[str] = None
        self._usd_sources: list[str] = []
        self._asset_source_transforms: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        self._asset_source_bounds: list[dict] = []
        self._center = np.zeros(3, dtype=np.float64)
        self._size = np.ones(3, dtype=np.float64)
        self._estimated_mass_kg = 10.0
        self._sim_time = 0.0
        self._steps_per_second = DEFAULT_PHYSX_STEPS_PER_SECOND
        self._dt = 1.0 / float(self._steps_per_second)
        self._substeps = DEFAULT_SUBSTEPS
        self._physx_device_mode = DEFAULT_PHYSX_DEVICE_MODE
        self._live_settings_dirty = False
        self._running = False
        self._status_text = "Load an asset, then use Play or Restart physics."
        self._scene_path: Optional[Path] = None
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._base_scene = "plane"
        self._pending_magnet: Optional[dict] = None
        self._physics_mode = PHYSICS_MODE_PROXY
        self._last_start_visual_transform = np.eye(4, dtype=np.float64)
        self._last_start_play = False
        self._current_body_patterns = list(AUTHORED_BODY_PATTERNS)
        self._current_body_paths: list[str] = []
        self._current_articulation_paths: list[str] = []
        self._authored_collider_count = 0
        self._authored_override_count = 0
        self._grab_force_amount = DEFAULT_GRAB_FORCE_AMOUNT
        self._drop_spacing_amount = DEFAULT_DROP_SPACING
        self._drop_randomness_amount = DEFAULT_DROP_RANDOMNESS
        self._ccd_enabled = False
        self._scene_instance_count = 1
        self._active_instance_count = 1
        self._last_start_instance_count = 1
        self._last_start_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_asset_indices = [0]
        self._pending_instance_reset = False
        self._cooked_asset_refs: tuple[str, ...] = ()
        self._cooked_base_scene = ""
        self._cooked_ccd_enabled = False
        self._cooked_steps_per_second = 0
        self._cooked_physx_device_mode = ""
        self._cooked_instance_capacity = 0
        self._cooked_contact_instances_required = False
        self._authored_contact_instances_required = False
        self._runtime_clone_source_path = ""
        self._runtime_clone_target_paths: list[str] = []
        self._runtime_clone_parent_poses: list[np.ndarray] = []
        self._runtime_clone_groups: list[dict] = []

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def ccd_enabled(self) -> bool:
        return bool(self._ccd_enabled)

    @property
    def has_scene(self) -> bool:
        return self._process is not None and self._worker_ready

    @property
    def current_visual_transform(self) -> np.ndarray:
        return np.array(self._current_visual_transform, dtype=np.float64, copy=True)

    def configure_asset(self, bounds: dict, usd_source: Optional[str] = None) -> None:
        self.shutdown()
        self._pending_magnet = None
        self._pending_discovery_start = None
        self._bounds = self._normalize_bounds(bounds)
        self._usd_sources = self._configured_asset_sources(bounds, usd_source)
        self._usd_source = self._usd_sources[0] if len(self._usd_sources) == 1 else None
        self._asset_source_transforms = self._configured_asset_transforms(bounds, len(self._usd_sources))
        self._asset_source_bounds = self._configured_asset_bounds(bounds, len(self._usd_sources))
        self._center = np.array(self._bounds["center"], dtype=np.float64)
        self._size = np.array(self._bounds["size"], dtype=np.float64)
        self._estimated_mass_kg = self._estimate_asset_mass(self._size)
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._physics_mode = PHYSICS_MODE_AUTHORED if self._usd_sources else PHYSICS_MODE_PROXY
        self._current_body_patterns = list(AUTHORED_BODY_PATTERNS)
        self._current_body_paths = []
        self._current_articulation_paths = []
        self._authored_collider_count = 0
        self._authored_override_count = 0
        self._scene_instance_count = max(1, len(self._asset_source_transforms))
        self._active_instance_count = self._scene_instance_count
        self._last_start_instance_count = self._scene_instance_count
        self._last_start_instance_transforms = [
            np.array(item, dtype=np.float64, copy=True) for item in self._asset_source_transforms
        ]
        self._authored_scene_instance_transforms = [
            np.array(item, dtype=np.float64, copy=True) for item in self._asset_source_transforms
        ]
        self._authored_scene_asset_indices = self._default_asset_indices(self._scene_instance_count)
        self._pending_instance_reset = False
        self._authored_contact_instances_required = False
        self._clear_runtime_clone_state()
        if len(self._usd_sources) > 1:
            self._set_status(f"Physics ready for {len(self._usd_sources)} selected assets. Colliders will cook after load.")
        elif self._usd_sources:
            self._set_status("Physics ready. Colliders will cook when asset loading finishes.")
        else:
            self._set_status("Physics unavailable: no USD source to inspect authored colliders.")

    def clear_asset(self) -> None:
        self.shutdown()
        self._bounds = None
        self._usd_source = None
        self._usd_sources = []
        self._asset_source_transforms = [np.eye(4, dtype=np.float64)]
        self._asset_source_bounds = []
        self._pending_magnet = None
        self._pending_discovery_start = None
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._estimated_mass_kg = 10.0
        self._physics_mode = PHYSICS_MODE_PROXY
        self._current_body_patterns = list(AUTHORED_BODY_PATTERNS)
        self._current_body_paths = []
        self._current_articulation_paths = []
        self._authored_collider_count = 0
        self._authored_override_count = 0
        self._scene_instance_count = 1
        self._active_instance_count = 1
        self._last_start_instance_count = 1
        self._last_start_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_asset_indices = [0]
        self._pending_instance_reset = False
        self._authored_contact_instances_required = False
        self._clear_runtime_clone_state()
        self._set_status("Load an asset, then use Play or Restart physics.")

    def restart(
        self,
        visual_transform: Optional[np.ndarray] = None,
        play: bool = True,
        force_rebuild: bool = False,
        instance_transforms: Optional[list[np.ndarray]] = None,
    ) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before starting physics.")
            self.running_changed.emit(False)
            return False

        if not self._usd_sources:
            self._set_status("Physics not started: this asset has no USD source with authored colliders.")
            self.running_changed.emit(False)
            return False

        visual = (
            np.array(visual_transform, dtype=np.float64, copy=True)
            if visual_transform is not None
            else np.array(self._current_visual_transform, dtype=np.float64, copy=True)
        )
        if instance_transforms is None and self._scene_instance_count > 1:
            active_existing = max(1, min(self._active_instance_count, len(self._last_start_instance_transforms)))
            instance_transforms = self._last_start_instance_transforms[:active_existing]
            if instance_transforms:
                visual = np.array(instance_transforms[0], dtype=np.float64, copy=True)
        instances = self._normalize_instance_transforms(instance_transforms, visual)
        active_count = len(instances)
        self._authored_scene_asset_indices = self._scene_asset_indices(active_count, len(self._active_asset_refs()))
        asset_refs = self._active_asset_refs()
        can_reuse_pool = self._can_reuse_live_scene(asset_refs, active_count)
        force_rebuild = bool(
            force_rebuild
            or self._live_settings_dirty
            or (active_count > self._scene_instance_count and not can_reuse_pool)
        )
        reset_instances = (
            self._pad_instance_transforms(instances, self._cooked_instance_capacity)
            if can_reuse_pool
            else instances
        )

        if self._process is not None and self._worker_ready and (can_reuse_pool or not force_rebuild):
            self._sim_time = 0.0
            self._pending_step_after_start = False
            self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
            self._active_instance_count = active_count
            self._last_start_instance_count = active_count
            self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in instances]
            self._last_start_play = bool(play)
            if active_count > 1 or len(reset_instances) > 1:
                self._set_instance_transforms(reset_instances, zero_velocity=True)
                label = "asset" if active_count == 1 else "assets"
                suffix = "" if len(reset_instances) == active_count else f" from a {len(reset_instances)}-instance cooked pool"
                self._set_status(f"Physics reset {active_count} {label} using cooked colliders{suffix}.")
            else:
                self.set_visual_transform(visual, zero_velocity=True)
                self._set_status("Physics reset using cooked colliders.")
            self.set_playing(bool(play))
            return True

        if self._process is not None and not self._worker_ready and not force_rebuild:
            self._pending_play = bool(play)
            self._pending_step_after_start = False
            self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
            self._active_instance_count = active_count
            self._last_start_instance_count = active_count
            self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in instances]
            self._last_start_play = bool(play)
            self._pending_instance_reset = active_count > 1
            self._set_status("Physics collider cook already in progress.")
            return True

        self._release_scene()
        self._sim_time = 0.0
        self._pending_play = bool(play)
        self._pending_step_after_start = False
        self._cooking_only = False
        self._startup_progress_active = True
        self._physics_mode = PHYSICS_MODE_AUTHORED
        self.cooking_progress.emit(0, "Cooking physics colliders...")
        self._set_status("Cooking authored physics colliders...")

        self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
        self._active_instance_count = active_count
        self._last_start_instance_count = active_count
        self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in instances]
        self._last_start_play = bool(play)
        self._pending_instance_reset = active_count > 1
        self._authored_scene_instance_transforms = (
            [np.eye(4, dtype=np.float64)]
            if active_count == 1
            else [np.array(item, dtype=np.float64, copy=True) for item in instances]
        )
        self._authored_scene_asset_indices = self._scene_asset_indices(active_count, len(asset_refs))
        initial_pose = None
        if active_count == 1:
            body_matrix = self._body_from_visual(visual)
            initial_pose = self._pose_from_body(body_matrix)

        return self._start_authored_worker_after_discovery(initial_pose, cook_only=False)

    def cook_colliders(self) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before cooking physics colliders.")
            self.cooking_finished.emit(False, self._status_text)
            return False

        if not self._usd_sources:
            self._set_status("Physics cook skipped: this asset has no USD source with authored colliders.")
            self.cooking_finished.emit(False, self._status_text)
            return False

        if self._process is not None and self._worker_ready:
            text = "Physics colliders already cooked and ready."
            self._set_status(text)
            self.cooking_progress.emit(100, text)
            self.cooking_finished.emit(True, text)
            return True

        if self._process is not None:
            self._pending_play = False
            self._set_status("Physics collider cook already in progress.")
            return True

        if self._discovery_process is not None:
            self._pending_play = False
            self._set_status("Physics collider discovery already in progress.")
            return True

        self._sim_time = 0.0
        self._pending_play = False
        self._pending_step_after_start = False
        self._cooking_only = False
        self._startup_progress_active = True
        self._physics_mode = PHYSICS_MODE_AUTHORED
        self.cooking_progress.emit(0, "Cooking physics colliders after asset load...")
        self._set_status("Cooking authored physics colliders after asset load...")

        visual = np.array(self._current_visual_transform, dtype=np.float64, copy=True)
        transforms = (
            [np.array(item, dtype=np.float64, copy=True) for item in self._asset_source_transforms]
            if len(self._usd_sources) > 1
            else [np.array(visual, dtype=np.float64, copy=True)]
        )
        self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
        self._active_instance_count = len(transforms)
        self._last_start_instance_count = len(transforms)
        self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in transforms]
        self._authored_scene_instance_transforms = (
            [np.eye(4, dtype=np.float64)]
            if len(transforms) == 1
            else [np.array(item, dtype=np.float64, copy=True) for item in transforms]
        )
        self._authored_scene_asset_indices = self._default_asset_indices(len(transforms))
        self._last_start_play = False
        self._pending_instance_reset = len(transforms) > 1
        initial_pose = None
        if len(transforms) == 1:
            body_matrix = self._body_from_visual(visual)
            initial_pose = self._pose_from_body(body_matrix)
        return self._start_authored_worker_after_discovery(initial_pose, cook_only=False)

    def set_playing(self, playing: bool) -> None:
        if playing and self._discovery_process is not None:
            self._pending_play = True
            self._set_status("Physics collider discovery is in progress; playback will start when ready.")
            return

        if playing and self._process is None:
            self.restart(play=True)
            return

        if playing and not self._worker_ready:
            self._pending_play = True
            self._set_status("Physics worker starting...")
            return

        if playing:
            status = "Physics playing."
            if not self._timer.isActive():
                self._timer.start(max(1, int(self._dt * 1000)))
            if not self._running:
                self._running = True
                self.running_changed.emit(True)
            self._set_status(status)
            return

        self._pending_play = False
        if self._timer.isActive():
            self._timer.stop()
        if self._running:
            self._running = False
            self.running_changed.emit(False)
        self._set_status("Physics paused.")

    def step_once(self) -> None:
        if self._process is None:
            if self.restart(play=False):
                self._pending_step_after_start = True
            return
        self.set_playing(False)
        self._send_step()

    def drop_asset(self, count: int = 1) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before dropping physics.")
            self.running_changed.emit(False)
            return False
        selected_count = max(1, len(self._active_asset_refs()))
        drop_count = max(self._sanitize_drop_count(count), selected_count)
        self._authored_contact_instances_required = drop_count > 1
        self._authored_scene_asset_indices = self._drop_asset_indices(drop_count)
        visuals = self._drop_visual_transforms(drop_count)
        force_rebuild = (
            drop_count != self._scene_instance_count
            and not self._can_reuse_live_scene(self._active_asset_refs(), drop_count)
        )
        if not self.restart(
            visual_transform=visuals[0],
            play=True,
            force_rebuild=force_rebuild,
            instance_transforms=visuals,
        ):
            return False
        label = "asset" if drop_count == 1 else "assets"
        self._set_status(f"Dropped {drop_count} {label} from above the base scene.")
        return True

    def set_base_scene(self, scene_id: str) -> None:
        scene = str(scene_id or "plane").lower()
        if scene not in BASE_SCENES:
            scene = "plane"
        if scene == self._base_scene:
            return

        self._base_scene = scene
        if self._bounds is None:
            self._set_status(f"Physics base set to {scene}.")
            return

        active = self._process is not None
        was_running = self._running or self._pending_play
        visual = np.array(self._current_visual_transform, dtype=np.float64, copy=True)
        if active:
            instances = self._last_start_instance_transforms if self._scene_instance_count > 1 else None
            self.restart(visual_transform=visual, play=was_running, force_rebuild=True, instance_transforms=instances)
        else:
            self._set_status(f"Physics base set to {scene}.")

    def set_ccd_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._ccd_enabled:
            return

        self._ccd_enabled = enabled
        if self._process is not None:
            self._live_settings_dirty = True
        mode = "enabled" if enabled else "disabled"
        if self._bounds is None:
            self._set_status(f"CCD {mode}. Load an asset to apply it.")
            return

        if self._process is not None:
            self._send({"cmd": "set_ccd", "enabled": enabled})
            if self._worker_ready:
                self._set_status(f"CCD {mode}; restart the PhysX engine to re-author the scene setting.")
            else:
                self._set_status(
                    f"CCD {mode}; the current collider cook will finish, then restart the PhysX engine to apply it."
                )
        else:
            self._set_status(f"CCD {mode}. It will apply when physics starts.")

    def set_steps_per_second(self, steps_per_second: int) -> None:
        value = self._sanitize_steps_per_second(steps_per_second)
        if value == self._steps_per_second:
            return
        self._steps_per_second = value
        self._dt = 1.0 / float(value)
        if self._process is not None:
            self._live_settings_dirty = True
            self._set_status(f"PhysX steps set to {value}/s; restart the PhysX engine to re-author the scene.")
        else:
            self._set_status(f"PhysX steps set to {value}/s.")

    def set_substeps(self, substeps: int) -> None:
        value = self._sanitize_substeps(substeps)
        if value == self._substeps:
            return
        self._substeps = value
        self._set_status(f"PhysX substeps set to {value}x.")

    def set_device_mode(self, mode: str) -> None:
        value = self._sanitize_device_mode(mode)
        if value == self._physx_device_mode:
            return
        self._physx_device_mode = value
        if self._process is not None:
            self._live_settings_dirty = True
            self._set_status(f"PhysX device mode set to {value.upper()}; restart the PhysX engine to apply it.")
        else:
            self._set_status(f"PhysX device mode set to {value.upper()}.")

    def reset_physx_settings(self) -> None:
        had_live_scene = self._process is not None
        self.set_ccd_enabled(False)
        self.set_steps_per_second(DEFAULT_PHYSX_STEPS_PER_SECOND)
        self.set_substeps(DEFAULT_SUBSTEPS)
        self.set_device_mode(DEFAULT_PHYSX_DEVICE_MODE)
        if had_live_scene:
            self._set_status("PhysX settings reset to defaults; restart the PhysX engine to apply scene/device changes.")
        else:
            self._set_status("PhysX settings reset to defaults.")

    def restart_engine(
        self,
        visual_transform: Optional[np.ndarray] = None,
        play: bool = True,
        instance_transforms: Optional[list[np.ndarray]] = None,
    ) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before restarting the PhysX engine.")
            self.running_changed.emit(False)
            return False
        preserved_instances = instance_transforms
        if preserved_instances is None and self._scene_instance_count > 1:
            active_existing = max(1, min(self._active_instance_count, len(self._last_start_instance_transforms)))
            preserved_instances = [
                np.array(item, dtype=np.float64, copy=True)
                for item in self._last_start_instance_transforms[:active_existing]
            ]
        self._set_status("Restarting PhysX engine...")
        self._release_scene()
        return self.restart(
            visual_transform=visual_transform,
            play=play,
            force_rebuild=True,
            instance_transforms=preserved_instances,
        )

    def set_visual_transform(self, matrix: np.ndarray, zero_velocity: bool = True) -> None:
        visual = np.array(matrix, dtype=np.float64, copy=True)
        if not self._matrix_is_valid(visual):
            self._set_status("Physics transform ignored: invalid visual transform.")
            return
        self._current_visual_transform = visual
        if self._worker_ready:
            pose = self._pose_from_visual(visual)
            self._send({"cmd": "set_pose", "pose": pose.tolist(), "zero_velocity": bool(zero_velocity)})
        self.pose_changed.emit(np.array(visual, dtype=np.float64, copy=True))

    def _set_instance_transforms(self, transforms: list[np.ndarray], zero_velocity: bool = True) -> None:
        instances = self._normalize_instance_transforms(transforms, self._current_visual_transform)
        self._current_visual_transform = np.array(instances[0], dtype=np.float64, copy=True)
        payload = []
        bodies = []
        for index, visual in enumerate(instances):
            if not self._matrix_is_valid(visual):
                continue
            path = self._authored_asset_path(index)
            pose = self._pose_from_visual(visual)
            payload.append({"path": path, "pose": pose.astype(float).tolist()})
            bodies.append({"path": path, "matrix": np.array(visual, dtype=np.float64, copy=True)})
        if self._worker_ready and payload:
            self._send({"cmd": "set_poses", "poses": payload, "zero_velocity": bool(zero_velocity)})
        if bodies:
            self.pose_changed.emit(
                {
                    "root": np.array(instances[0], dtype=np.float64, copy=True),
                    "bodies": bodies,
                }
            )

    def set_grab_force_amount(self, amount: float) -> None:
        self._grab_force_amount = self._sanitize_grab_force_amount(amount)
        if self._pending_magnet is not None:
            self._pending_magnet["force_amount"] = self._grab_force_amount
            if self._worker_ready:
                self._send(self._pending_magnet)

    def set_drop_options(self, spacing: float, randomness: float) -> None:
        self._drop_spacing_amount = self._sanitize_drop_spacing(spacing)
        self._drop_randomness_amount = self._sanitize_drop_randomness(randomness)

    def begin_magnet(
        self,
        anchor_local_visual: np.ndarray,
        target_visual: np.ndarray,
        target_velocity_visual: Optional[np.ndarray] = None,
        body_path: Optional[str] = None,
    ) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before grabbing physics.")
            return False

        if not self._vector_is_valid(anchor_local_visual) or not self._vector_is_valid(target_visual):
            self._set_status("Physics grab skipped: invalid grab target.")
            return False

        message = self._magnet_message(anchor_local_visual, target_visual, target_velocity_visual, body_path)
        self._pending_magnet = message

        if self._process is None:
            if not self.restart(visual_transform=self._current_visual_transform, play=True):
                return False
            self._pending_magnet = message
            self._set_status("Starting physics grab...")
            return True

        if not self._worker_ready:
            self._pending_play = True
            self._set_status("Starting physics grab...")
            return True

        self._send(message)
        self.set_playing(True)
        self._set_status("Physics grab active.")
        return True

    def update_magnet(self, target_visual: np.ndarray, target_velocity_visual: Optional[np.ndarray] = None) -> None:
        if self._pending_magnet is None:
            return
        if not self._vector_is_valid(target_visual):
            self._set_status("Physics grab stopped: invalid grab target.")
            self.end_magnet(np.zeros(3, dtype=np.float64))
            return
        if target_velocity_visual is not None and not self._vector_is_valid(target_velocity_visual, MAX_SIM_POSITION):
            target_velocity_visual = np.zeros(3, dtype=np.float64)
        self._pending_magnet["target"] = self._point_visual_to_physics(target_visual).astype(float).tolist()
        self._pending_magnet["target_velocity"] = self._vector_visual_to_physics(
            np.zeros(3, dtype=np.float64) if target_velocity_visual is None else target_velocity_visual
        ).astype(float).tolist()
        if self._worker_ready:
            self._send(self._pending_magnet)

    def end_magnet(self, throw_velocity_visual: Optional[np.ndarray] = None) -> None:
        self._pending_magnet = None
        if throw_velocity_visual is not None and not self._vector_is_valid(throw_velocity_visual, MAX_SIM_POSITION):
            throw_velocity_visual = np.zeros(3, dtype=np.float64)
        velocity = self._vector_visual_to_physics(
            np.zeros(3, dtype=np.float64) if throw_velocity_visual is None else throw_velocity_visual
        )
        if self._worker_ready:
            self._send(
                {
                    "cmd": "release_magnet",
                    "velocity": velocity.astype(float).tolist(),
                    "angular_velocity": [0.0, 0.0, 0.0],
                }
            )
            self.set_playing(True)
        self._set_status("Physics grab released.")

    def shutdown(self) -> None:
        self._release_scene()

    def _start_authored_worker_after_discovery(
        self,
        initial_pose: Optional[np.ndarray],
        cook_only: bool = False,
    ) -> bool:
        asset_refs = self._active_asset_refs()
        if not asset_refs:
            self._set_status("Physics not started: no USD source is loaded.")
            self.cooking_finished.emit(False, self._status_text)
            return False

        cached_discoveries = [self._cached_discovery(asset_ref) for asset_ref in asset_refs]
        if all(discovery is not None for discovery in cached_discoveries):
            return self._start_worker_with_discoveries(
                [discovery for discovery in cached_discoveries if discovery is not None],
                initial_pose,
                cook_only,
            )

        self._pending_discovery_start = PendingPhysicsStart(
            asset_refs=list(asset_refs),
            initial_pose=None if initial_pose is None else np.array(initial_pose, dtype=np.float32, copy=True),
            cook_only=bool(cook_only),
        )

        if self._discovery_process is not None:
            self.cooking_progress.emit(8, "Discovering authored collider metadata...")
            self._set_status("Physics collider discovery already in progress.")
            return True

        missing_count = sum(1 for discovery in cached_discoveries if discovery is None)
        self.cooking_progress.emit(5, "Discovering authored collider metadata...")
        if len(asset_refs) > 1:
            self._set_status(
                f"Discovering authored collider metadata for {missing_count} of {len(asset_refs)} selected assets..."
            )
        else:
            self._set_status("Discovering authored collider metadata in background...")
        process = QProcess(self)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.readyReadStandardOutput.connect(self._on_discovery_stdout)
        process.readyReadStandardError.connect(self._on_discovery_stderr)
        process.finished.connect(self._on_discovery_finished)
        self._discovery_process = process
        self._discovery_buffer = ""
        self._discovery_error = ""
        args = ["-u", "-m", "core.physics_collider_discovery"]
        if len(asset_refs) > 1:
            args.extend(["--multi", *asset_refs])
        else:
            args.extend([asset_refs[0], AUTHORED_ASSET_PATH])
        process.start(
            sys.executable,
            args,
        )
        if not process.waitForStarted(1000):
            error = process.errorString()
            self._discovery_process = None
            self._pending_discovery_start = None
            process.deleteLater()
            discoveries = [self._empty_discovery() for _asset_ref in asset_refs]
            self._set_status(f"Collider discovery process could not start ({error}); OVPhysX will inspect USD only.")
            return self._start_worker_with_discoveries(discoveries, initial_pose, cook_only)
        return True

    def _start_worker_with_discovery(
        self,
        discovery: AuthoredColliderDiscovery,
        initial_pose: Optional[np.ndarray],
        cook_only: bool,
    ) -> bool:
        return self._start_worker_with_discoveries([discovery], initial_pose, cook_only)

    def _start_worker_with_discoveries(
        self,
        discoveries: list[AuthoredColliderDiscovery],
        initial_pose: Optional[np.ndarray],
        cook_only: bool,
    ) -> bool:
        try:
            scene_path = self._write_authored_scene(discoveries)
        except Exception as exc:
            self._set_status(f"Could not prepare authored physics scene: {exc}")
            self.cooking_finished.emit(False, self._status_text)
            return False
        return self._start_worker(scene_path, self._current_body_patterns, initial_pose, cook_only=cook_only)

    def _on_discovery_stdout(self) -> None:
        if self._discovery_process is None:
            return
        self._discovery_buffer += bytes(self._discovery_process.readAllStandardOutput()).decode("utf-8", "replace")

    def _on_discovery_stderr(self) -> None:
        if self._discovery_process is None:
            return
        text = bytes(self._discovery_process.readAllStandardError()).decode("utf-8", "replace").strip()
        if text:
            self._discovery_error = text

    def _on_discovery_finished(self, exit_code: int, _exit_status) -> None:
        process = self._discovery_process
        self._discovery_process = None
        if process is not None:
            process.deleteLater()

        pending = self._pending_discovery_start
        self._pending_discovery_start = None
        if pending is None:
            return
        if pending.asset_refs != self._active_asset_refs():
            return

        discoveries = self._discoveries_from_stdout(self._discovery_buffer, pending.asset_refs)
        if exit_code != 0 or discoveries is None:
            detail = self._discovery_error.splitlines()[-1] if self._discovery_error else "no discovery payload returned"
            self._set_status(f"Collider metadata discovery skipped ({detail}); OVPhysX will inspect USD only.")
            discoveries = [self._empty_discovery() for _asset_ref in pending.asset_refs]
        else:
            for asset_ref, discovery in zip(pending.asset_refs, discoveries):
                self._store_discovery(asset_ref, discovery)

        self.cooking_progress.emit(18, "Starting OVPhysX with discovered colliders...")
        self._start_worker_with_discoveries(discoveries, pending.initial_pose, pending.cook_only)

    def _start_worker(
        self,
        scene_path: Path,
        body_patterns: list[str],
        initial_pose: Optional[np.ndarray],
        cook_only: bool = False,
    ) -> bool:
        self._set_status("Starting OVPhysX worker...")
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("SIMREADY_OVPHYSX_DEVICE", self._physx_device_mode)
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_worker_stdout)
        self._process.readyReadStandardError.connect(self._on_worker_stderr)
        self._process.finished.connect(self._on_worker_finished)
        self._intentional_stop = False
        self._stdout_buffer = ""
        self._worker_ready = False
        self._step_in_flight = False
        self._cooking_only = bool(cook_only)

        self._process.start(sys.executable, ["-u", "-m", "core.physics_worker"])
        if not self._process.waitForStarted(5000):
            error = self._process.errorString() if self._process else "unknown error"
            was_cooking = self._cooking_only
            was_progress = self._startup_progress_active
            self._release_scene()
            self._set_status(f"Could not start OVPhysX worker: {error}")
            self.running_changed.emit(False)
            if was_cooking or was_progress:
                self.cooking_finished.emit(False, self._status_text)
            return False

        self._send(
            {
                "cmd": "start",
                "scene": str(scene_path),
                "body_patterns": body_patterns,
                "body_paths": self._current_body_paths,
                "articulation_paths": self._current_articulation_paths,
                "instance_paths": self._instance_root_paths(self._last_start_instance_count),
                "instance_reference_poses": self._instance_reference_poses(),
                "clone_source_path": self._runtime_clone_source_path,
                "clone_target_paths": self._runtime_clone_target_paths,
                "clone_parent_poses": [
                    np.array(pose, dtype=np.float32, copy=True).reshape(7).astype(float).tolist()
                    for pose in self._runtime_clone_parent_poses
                ],
                "clone_groups": self._runtime_clone_groups_payload(),
                "initial_pose": None if initial_pose is None else initial_pose.astype(float).tolist(),
                "contact_offset": self._contact_offset(),
                "cook_only": bool(cook_only),
                "ccd_enabled": bool(self._ccd_enabled),
                "steps_per_second": int(self._steps_per_second),
                "device_mode": self._physx_device_mode,
            }
        )
        self._cooked_asset_refs = tuple(self._active_asset_refs())
        self._cooked_base_scene = self._base_scene
        self._cooked_ccd_enabled = bool(self._ccd_enabled)
        self._cooked_steps_per_second = int(self._steps_per_second)
        self._cooked_physx_device_mode = self._physx_device_mode
        self._cooked_instance_capacity = max(1, int(self._last_start_instance_count))
        self._cooked_contact_instances_required = bool(self._authored_contact_instances_required)
        self._live_settings_dirty = False
        return True

    def _send_step(self) -> None:
        if not self._worker_ready or self._step_in_flight:
            return
        self._step_in_flight = True
        self._send({"cmd": "step", "dt": self._dt, "time": self._sim_time, "substeps": self._substeps})
        self._sim_time += self._dt

    def _send(self, message: dict) -> None:
        process = self._process
        if process is None or process.state() != QProcess.Running:
            return
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        process.write(payload)
        process.waitForBytesWritten(100)

    def _on_worker_stdout(self) -> None:
        if self._process is None:
            return
        self._stdout_buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._handle_worker_message(message)

    def _on_worker_stderr(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", "replace").strip()
        if text and "Traceback" in text:
            self._set_status(text.splitlines()[-1])

    def _handle_worker_message(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "progress":
            value = int(message.get("value", 0) or 0)
            text = str(message.get("message", "Cooking physics colliders..."))
            self._set_status(text)
            if self._startup_progress_active:
                self.cooking_progress.emit(max(0, min(99, value)), text)
            return

        if kind == "cooked":
            body_count = int(message.get("body_count", 1) or 1)
            shape_count = int(message.get("shape_count", 0) or 0)
            pattern = str(message.get("body_pattern", "authored USD bodies"))
            body_suffix = "body" if body_count == 1 else "bodies"
            if shape_count > 0:
                shape_suffix = "shape" if shape_count == 1 else "shapes"
                shape_text = f"{shape_count} collider {shape_suffix}"
            else:
                shape_text = "zero usable collider shapes"
            ccd_text = ", CCD on" if bool(message.get("ccd_enabled", False)) else ""
            device = str(message.get("device", self._physx_device_mode) or self._physx_device_mode).upper()
            device_text = f", {device} PhysX"
            authored_text = (
                f"{self._authored_collider_count} authored collider prims found, "
                if self._authored_collider_count > 0
                else ""
            )
            text = (
                f"Physics colliders cooked ({authored_text}{body_count} {body_suffix}, "
                f"{shape_text}, {pattern}{ccd_text}{device_text})."
            )
            warning = str(message.get("cook_warning", "") or "")
            device_warning = str(message.get("device_warning", "") or "")
            if device_warning:
                warning = f"{warning} {device_warning}".strip()
            if warning:
                text = f"{text} {warning}"
            self.cooking_progress.emit(100, text)
            self._set_status(text)
            was_pending_play = self._pending_play
            self._release_scene()
            cook_ok = shape_count > 0
            self.cooking_finished.emit(cook_ok, text)
            if was_pending_play and cook_ok:
                self.restart(play=True)
            return

        if kind == "started":
            self._worker_ready = True
            self._scene_instance_count = self._last_start_instance_count
            body_count = int(message.get("body_count", 1) or 1)
            if self._physics_mode == PHYSICS_MODE_AUTHORED:
                pattern = str(message.get("body_pattern", "authored USD bodies"))
                suffix = "body" if body_count == 1 else "bodies"
                shape_count = int(message.get("shape_count", 0) or 0)
                warning = str(message.get("cook_warning", "") or "")
                if shape_count <= 0:
                    authored_text = (
                        f"Discovery found {self._authored_collider_count} authored collider prims, but "
                        if self._authored_collider_count > 0
                        else ""
                    )
                    self._set_status(
                        f"{authored_text}OVPhysX exposed zero usable collision shapes for {pattern}; "
                        "physics was not started, so the asset will not fall through the ground."
                    )
                    if warning:
                        self._set_status(f"{self._status_text} {warning}")
                    if self._timer.isActive():
                        self._timer.stop()
                    if self._running:
                        self._running = False
                        self.running_changed.emit(False)
                    self._pending_play = False
                    self._pending_step_after_start = False
                    if self._startup_progress_active:
                        self.cooking_progress.emit(100, self._status_text)
                        self.cooking_finished.emit(False, self._status_text)
                    self._release_scene()
                    return
                shape_text = (
                    f", {shape_count} collider shapes"
                    if shape_count > 0
                    else ", authored collider tensors pending"
                )
                ccd_text = ", CCD on" if bool(message.get("ccd_enabled", False)) else ""
                device = str(message.get("device", self._physx_device_mode) or self._physx_device_mode).upper()
                device_text = f", {device} PhysX"
                prefix = "Physics reset with authored colliders"
                if not self._pending_play and not self._pending_step_after_start:
                    prefix = "Physics colliders cooked and ready"
                instance_text = ""
                if self._active_instance_count > 1:
                    instance_text = f", {self._active_instance_count} active asset instances"
                    if self._scene_instance_count > self._active_instance_count:
                        instance_text += f" from a {self._scene_instance_count}-instance cache"
                text = (
                    f"{prefix} ({self._authored_collider_count} authored prims, "
                    f"{body_count} {suffix}{shape_text}, "
                    f"{pattern}{instance_text}, {self._steps_per_second}/s, "
                    f"{self._substeps}x substeps{ccd_text}{device_text})."
                )
                device_warning = str(message.get("device_warning", "") or "")
                if device_warning:
                    warning = f"{warning} {device_warning}".strip()
                if warning:
                    text = f"{text} {warning}"
                self._set_status(text)
            else:
                text = "Physics reset with proxy collider."
                if bool(message.get("ccd_enabled", False)):
                    text = "Physics reset with proxy collider (CCD on)."
                self._set_status(text)
            if self._startup_progress_active:
                self.cooking_progress.emit(100, self._status_text)
                self.cooking_finished.emit(True, self._status_text)
                self._startup_progress_active = False
            if self._last_start_instance_count <= 1:
                self.set_visual_transform(self._last_start_visual_transform, zero_velocity=True)
            elif self._pending_instance_reset:
                self._set_instance_transforms(self._last_start_instance_transforms, zero_velocity=True)
                self._pending_instance_reset = False
            if self._pending_magnet is not None:
                self._send(self._pending_magnet)
            if self._pending_play:
                self.set_playing(True)
            if self._pending_step_after_start:
                self._pending_step_after_start = False
                self._send_step()
            return

        if kind == "ccd":
            enabled = bool(message.get("enabled", False))
            mode = "enabled" if enabled else "disabled"
            detail = str(message.get("message", "") or "")
            if detail:
                self._set_status(detail)
            else:
                self._set_status(f"CCD {mode}; current physics scene was not re-cooked.")
            return

        if kind == "pose":
            pose = np.array(message.get("pose", []), dtype=np.float64)
            if pose.size >= 7:
                if not self._pose_is_valid(pose[:7]):
                    self._handle_unstable_pose("Physics returned an invalid root pose; physics was stopped.")
                    return
                body_matrix = self._matrix_from_pose(pose[:7])
                visual_matrix = self._visual_from_body(body_matrix)
                if not self._matrix_is_valid(visual_matrix):
                    self._handle_unstable_pose("Physics returned an invalid asset transform; physics was stopped.")
                    return
                self._current_visual_transform = visual_matrix
                self._step_in_flight = False
                body_entries = []
                for item in message.get("bodies", []) or []:
                    if not isinstance(item, dict):
                        continue
                    body_pose = np.array(item.get("pose", []), dtype=np.float64)
                    if body_pose.size < 7:
                        continue
                    if not self._pose_is_valid(body_pose[:7]):
                        continue
                    path = str(item.get("path", "") or "")
                    if not path:
                        continue
                    body_visual = self._visual_from_body(self._matrix_from_pose(body_pose[:7]))
                    if not self._matrix_is_valid(body_visual):
                        continue
                    body_entries.append(
                        {
                            "path": path,
                            "matrix": body_visual,
                        }
                    )
                if body_entries:
                    self.pose_changed.emit(
                        {
                            "root": np.array(visual_matrix, dtype=np.float64, copy=True),
                            "bodies": body_entries,
                        }
                    )
                else:
                    self.pose_changed.emit(np.array(visual_matrix, dtype=np.float64, copy=True))
            return

        if kind == "error":
            self._step_in_flight = False
            was_cooking = self._cooking_only
            was_progress = self._startup_progress_active
            self._pending_play = False
            if self._timer.isActive():
                self._timer.stop()
            if self._running:
                self._running = False
                self.running_changed.emit(False)
            detail = message.get("message", "unknown error")
            if self._physics_mode == PHYSICS_MODE_AUTHORED:
                self._set_status(f"Authored collider cook failed; physics not started. {detail}")
            else:
                self._set_status(f"OVPhysX failed: {detail}")
            self._release_scene()
            if was_cooking or was_progress:
                self.cooking_finished.emit(False, self._status_text)
            return

        if kind == "stopped":
            self._worker_ready = False

    def _on_worker_finished(self, _exit_code: int, _exit_status) -> None:
        was_intentional = self._intentional_stop
        was_progress = self._startup_progress_active
        self._process = None
        self._worker_ready = False
        self._step_in_flight = False
        self._pending_instance_reset = False
        self._clear_live_cooked_scene_state()
        self._live_settings_dirty = False
        if self._timer.isActive():
            self._timer.stop()
        if self._running:
            self._running = False
            self.running_changed.emit(False)
        if not was_intentional:
            self._set_status("OVPhysX worker stopped.")
            if self._cooking_only or was_progress:
                self.cooking_finished.emit(False, self._status_text)
        self._cooking_only = False
        self._startup_progress_active = False

    def _release_scene(self) -> None:
        self._release_discovery_process()
        if self._timer.isActive():
            self._timer.stop()
        self._pending_magnet = None
        self._pending_play = False
        self._pending_step_after_start = False
        self._step_in_flight = False
        self._cooking_only = False
        self._startup_progress_active = False
        self._scene_instance_count = 1
        self._active_instance_count = 1
        self._clear_live_cooked_scene_state()
        self._live_settings_dirty = False
        self._clear_runtime_clone_state()
        if self._running:
            self._running = False
            self.running_changed.emit(False)

        process = self._process
        self._process = None
        self._worker_ready = False
        if process is None:
            return

        self._intentional_stop = True
        try:
            if process.state() == QProcess.Running:
                payload = (json.dumps({"cmd": "shutdown"}) + "\n").encode("utf-8")
                process.write(payload)
                process.waitForBytesWritten(100)
                process.closeWriteChannel()
                if not process.waitForFinished(3000):
                    process.terminate()
                    if not process.waitForFinished(1500):
                        process.kill()
                        process.waitForFinished(1500)
        finally:
            process.deleteLater()

    def _release_discovery_process(self) -> None:
        process = self._discovery_process
        self._discovery_process = None
        self._discovery_buffer = ""
        self._discovery_error = ""
        self._pending_discovery_start = None
        if process is None:
            return
        try:
            if process.state() == QProcess.Running:
                process.terminate()
                if not process.waitForFinished(1000):
                    process.kill()
                    process.waitForFinished(1000)
        finally:
            process.deleteLater()

    def _body_from_visual(self, visual_matrix: np.ndarray) -> np.ndarray:
        visual = np.array(visual_matrix, dtype=np.float64, copy=True)
        if self._physics_mode == PHYSICS_MODE_AUTHORED:
            return visual
        body_z = np.eye(4, dtype=np.float64)
        body_z[:3, :3] = visual[:3, :3]
        body_z[3, :3] = self._center @ visual[:3, :3] + visual[3, :3]
        return self._z_to_y_matrix(body_z)

    def _visual_from_body(self, body_matrix: np.ndarray) -> np.ndarray:
        if self._physics_mode == PHYSICS_MODE_AUTHORED:
            return np.array(body_matrix, dtype=np.float64, copy=True)
        body = self._y_to_z_matrix(body_matrix)
        visual = np.eye(4, dtype=np.float64)
        visual[:3, :3] = body[:3, :3]
        visual[3, :3] = body[3, :3] - self._center @ body[:3, :3]
        return visual

    def _pose_from_visual(self, visual_matrix: np.ndarray) -> np.ndarray:
        body = self._body_from_visual(visual_matrix)
        return self._pose_from_body(body)

    def _pose_from_body(self, body_matrix: np.ndarray) -> np.ndarray:
        body = np.asarray(body_matrix, dtype=np.float64).reshape(4, 4)
        quat = self._quat_xyzw_from_row_rotation(body[:3, :3])
        pose = np.zeros(7, dtype=np.float32)
        pose[:3] = body[3, :3].astype(np.float32)
        pose[3:] = quat.astype(np.float32)
        return pose

    def _magnet_message(
        self,
        anchor_local_visual: np.ndarray,
        target_visual: np.ndarray,
        target_velocity_visual: Optional[np.ndarray],
        body_path: Optional[str] = None,
    ) -> dict:
        target_velocity = (
            np.zeros(3, dtype=np.float64)
            if target_velocity_visual is None
            else np.asarray(target_velocity_visual, dtype=np.float64)
        )
        message = {
            "cmd": "set_magnet",
            "anchor": self._vector_visual_to_physics(anchor_local_visual).astype(float).tolist(),
            "target": self._point_visual_to_physics(target_visual).astype(float).tolist(),
            "target_velocity": self._vector_visual_to_physics(target_velocity).astype(float).tolist(),
            "estimated_mass": float(self._estimated_mass_kg),
            "natural_frequency": 7.0,
            "damping_ratio": 0.7,
            "max_acceleration": 70.0,
            "max_angular_acceleration": 10.0,
            "force_amount": self._grab_force_amount,
        }
        if body_path:
            message["body_path"] = str(body_path)
        return message

    def _write_authored_scene(
        self,
        discovery: Optional[AuthoredColliderDiscovery | list[AuthoredColliderDiscovery]] = None,
    ) -> Path:
        asset_refs = self._active_asset_refs()
        if not asset_refs:
            raise RuntimeError("No USD source is configured for authored collider physics")

        temp_dir = Path(tempfile.gettempdir()) / "simready_browser_physx"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._scene_path = temp_dir / "authored_asset_scene.usda"

        base_scene = self._base_scene_usda_z_up()
        if discovery is None:
            discoveries = [
                self._cached_discovery(asset_ref) or self._authored_collider_discovery(asset_ref)
                for asset_ref in asset_refs
            ]
        elif isinstance(discovery, AuthoredColliderDiscovery):
            discoveries = [discovery]
        else:
            discoveries = list(discovery or [])

        instance_transforms = self._normalize_instance_transforms(
            self._authored_scene_instance_transforms,
            np.eye(4, dtype=np.float64),
        )
        instance_count = len(instance_transforms)
        self._clear_runtime_clone_state()
        asset_indices = self._scene_asset_indices(instance_count, len(asset_refs))
        self._authored_scene_asset_indices = list(asset_indices)

        source_instances: dict[int, list[int]] = {}
        for instance_index, source_index in enumerate(asset_indices):
            source_instances.setdefault(int(source_index), []).append(instance_index)

        scene_entries: list[tuple[int, int, AuthoredColliderDiscovery]] = []
        use_authored_contact_instances = self._authored_contact_instances_required and instance_count > 1
        for source_index, instance_indices in source_instances.items():
            if source_index < 0 or source_index >= len(asset_refs):
                continue
            source_discovery = (
                discoveries[source_index]
                if source_index < len(discoveries)
                else self._empty_discovery()
            )
            if use_authored_contact_instances:
                for instance_index in instance_indices:
                    scene_entries.append((instance_index, source_index, source_discovery))
                continue

            source_instance_index = instance_indices[0]
            scene_entries.append((source_instance_index, source_index, source_discovery))
            clone_targets = [self._authored_asset_path(index) for index in instance_indices[1:]]
            if clone_targets:
                parent_poses = [
                    self._pose_from_visual(instance_transforms[index])
                    for index in instance_indices[1:]
                ]
                self._add_runtime_clone_group(
                    self._authored_asset_path(source_instance_index),
                    clone_targets,
                    parent_poses,
                )

        scene_entries.sort(key=lambda item: item[0])
        body_discoveries = [
            discoveries[source_index] if source_index < len(discoveries) else self._empty_discovery()
            for source_index in asset_indices
        ]

        self._scene_instance_count = instance_count
        self._current_body_patterns = []
        self._current_body_paths = []
        self._current_articulation_paths = []
        for index, item_discovery in enumerate(body_discoveries):
            if self._needs_dynamic_body_apis(item_discovery):
                body_paths = self._dynamic_body_paths(item_discovery, index)
                if item_discovery.body_paths:
                    self._extend_unique(
                        self._current_body_patterns,
                        self._map_authored_paths_for_index(
                            item_discovery.body_patterns or list(AUTHORED_BODY_PATTERNS),
                            index,
                        ),
                    )
                else:
                    self._extend_unique(self._current_body_patterns, self._dynamic_root_body_patterns(index))
                self._extend_unique(self._current_body_paths, body_paths)
            else:
                self._extend_unique(
                    self._current_body_patterns,
                    self._map_authored_paths_for_index(
                        item_discovery.body_patterns or list(AUTHORED_BODY_PATTERNS),
                        index,
                    ),
                )
                self._extend_unique(
                    self._current_body_paths,
                    self._map_authored_paths_for_index(item_discovery.body_paths, index),
                )
            self._extend_unique(
                self._current_articulation_paths,
                self._map_authored_paths_for_index(item_discovery.articulation_paths, index),
            )
        if not self._current_body_paths and instance_count > 1:
            self._current_body_paths = [self._authored_asset_path(index) for index in range(instance_count)]
        source_collider_count = sum(int(item[2].collider_count) for item in scene_entries)
        self._authored_collider_count = source_collider_count
        self._authored_override_count = 0
        scene_ccd = "1" if self._ccd_enabled else "0"
        asset_blocks = "\n\n".join(
            self._authored_asset_usda_block(
                instance_index,
                asset_refs[source_index],
                discovery_item.collision_overrides if self._needs_dynamic_body_apis(discovery_item) else "",
                instance_transforms[instance_index],
                discovery_item,
                self._estimated_mass_for_source(source_index),
            )
            for instance_index, source_index, discovery_item in scene_entries
        )
        if self._authored_collider_count > 0:
            if instance_count > len(scene_entries):
                self._set_status(
                    f"Found {self._authored_collider_count} authored collider prims across "
                    f"{len(scene_entries)} source assets; cloning to {instance_count} physics instances..."
                )
            elif use_authored_contact_instances and instance_count > 1:
                self._set_status(
                    f"Found {self._authored_collider_count} authored collider prims; "
                    f"cooking {instance_count} contact-enabled physics instances..."
                )
            else:
                self._set_status(
                    f"Found {self._authored_collider_count} authored collider prims; starting OVPhysX cook..."
                )
        else:
            self._set_status(
                "No authored collider prims were found by discovery; OVPhysX will inspect the loaded stage."
            )

        text = f"""#usda 1.0
(
    defaultPrim = "World"
    doc = "SimReady Browser OVPhysX authored-collider scene"
    framesPerSecond = {USD_STAGE_FRAMES_PER_SECOND}
    timeCodesPerSecond = {USD_STAGE_FRAMES_PER_SECOND}
    metersPerUnit = 1
    upAxis = "Z"
    kilogramsPerMass = 1
)

def Xform "World" (
    kind = "component"
)
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI", "MaterialBindingAPI"]
    )
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
        uniform token physxScene:solverType = "TGS"
        uniform uint physxScene:timeStepsPerSecond = {int(self._steps_per_second)}
        bool physxScene:enableStabilization = 1
        bool physxScene:enableExternalForcesEveryIteration = 1
        bool physxScene:solveArticulationContactLast = 1
        uniform uint physxScene:minPositionIterationCount = 8
        uniform uint physxScene:minVelocityIterationCount = 2
        bool physxScene:enableCCD = {scene_ccd}
        rel material:binding:physics = </World/PhysicsMaterials/DynamicMaterial> (
            bindMaterialAs = "weakerThanDescendants"
        )
    }}

{self._physics_materials_usda()}

{asset_blocks}

{base_scene}
}}
"""
        self._scene_path.write_text(text, encoding="utf-8")
        return self._scene_path

    def _write_proxy_scene(self, body_matrix: np.ndarray) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "simready_browser_physx"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._scene_path = temp_dir / "asset_proxy.usda"

        size_z = np.maximum(self._size, np.array([0.05, 0.05, 0.05], dtype=np.float64))
        size_y = np.array([size_z[0], size_z[2], size_z[1]], dtype=np.float64)
        body_pos = body_matrix[3, :3]
        body_quat = self._quat_xyzw_from_row_rotation(body_matrix[:3, :3])
        base_scene = self._base_scene_usda()
        proxy_mass = self._fmt(self._estimated_mass_kg)

        text = f"""#usda 1.0
(
    defaultPrim = "World"
    doc = "SimReady Browser OVPhysX proxy scene"
    framesPerSecond = {USD_STAGE_FRAMES_PER_SECOND}
    timeCodesPerSecond = {USD_STAGE_FRAMES_PER_SECOND}
    metersPerUnit = 1
    upAxis = "Y"
    kilogramsPerMass = 1
)

def Xform "World" (
    kind = "component"
)
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI", "MaterialBindingAPI"]
    )
    {{
        vector3f physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
        uniform token physxScene:solverType = "TGS"
        uniform uint physxScene:timeStepsPerSecond = {int(self._steps_per_second)}
        bool physxScene:enableStabilization = 1
        bool physxScene:enableExternalForcesEveryIteration = 1
        bool physxScene:solveArticulationContactLast = 1
        uniform uint physxScene:minPositionIterationCount = 8
        uniform uint physxScene:minVelocityIterationCount = 2
        bool physxScene:enableCCD = {"1" if self._ccd_enabled else "0"}
        rel material:binding:physics = </World/PhysicsMaterials/DynamicMaterial> (
            bindMaterialAs = "weakerThanDescendants"
        )
    }}

{self._physics_materials_usda()}

    def Xform "AssetProxy" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "MaterialBindingAPI"]
    )
    {{
        double3 xformOp:translate = ({self._fmt(body_pos[0])}, {self._fmt(body_pos[1])}, {self._fmt(body_pos[2])})
        quatd xformOp:orient = ({self._fmt(body_quat[3])}, {self._fmt(body_quat[0])}, {self._fmt(body_quat[1])}, {self._fmt(body_quat[2])})
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = {proxy_mass}
        vector3f physics:velocity = (0, 0, 0)
        vector3f physics:angularVelocity = (0, 0, 0)
        bool physxRigidBody:enableCCD = {"1" if self._ccd_enabled else "0"}
        rel material:binding:physics = </World/PhysicsMaterials/DynamicMaterial> (
            bindMaterialAs = "weakerThanDescendants"
        )

        def Cube "CubeGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            uniform token physics:approximation = "{BOX_COLLIDER_APPROXIMATION}"
            float physxCollision:contactOffset = {self._fmt(ISAAC_CUBOID_CONTACT_OFFSET)}
            float physxCollision:restOffset = {self._fmt(ISAAC_CUBOID_REST_OFFSET)}
            float physxCollision:torsionalPatchRadius = {self._fmt(ISAAC_CUBOID_TORSIONAL_PATCH_RADIUS)}
            float physxCollision:minTorsionalPatchRadius = {self._fmt(ISAAC_CUBOID_MIN_TORSIONAL_PATCH_RADIUS)}
            double3 xformOp:scale = ({self._fmt(size_y[0])}, {self._fmt(size_y[1])}, {self._fmt(size_y[2])})
            uniform token[] xformOpOrder = ["xformOp:scale"]
            rel material:binding:physics = </World/PhysicsMaterials/DynamicMaterial>
        }}
    }}

{base_scene}
}}
"""
        self._scene_path.write_text(text, encoding="utf-8")
        return self._scene_path

    def _authored_asset_usda_block(
        self,
        index: int,
        asset_ref: str,
        collision_overrides: str,
        transform: np.ndarray,
        discovery: Optional[AuthoredColliderDiscovery] = None,
        mass_kg: Optional[float] = None,
    ) -> str:
        name = self._authored_asset_name(index)
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        quat = self._quat_xyzw_from_row_rotation(matrix[:3, :3])
        schema_lines = [f"        prepend references = @{asset_ref}@"]
        dynamic_root = self._uses_dynamic_root(discovery)
        if dynamic_root:
            schema_lines.append(
                '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", '
                '"PhysxRigidBodyAPI", "MaterialBindingAPI"]'
            )
        xform_lines = [
            f"        double3 xformOp:translate = ({self._fmt(matrix[3, 0])}, {self._fmt(matrix[3, 1])}, {self._fmt(matrix[3, 2])})",
            f"        quatd xformOp:orient = ({self._fmt(quat[3])}, {self._fmt(quat[0])}, {self._fmt(quat[1])}, {self._fmt(quat[2])})",
            "        double3 xformOp:scale = (1, 1, 1)",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
        ]
        if dynamic_root:
            mass = self._fmt(mass_kg if mass_kg is not None else self._estimated_mass_kg)
            xform_lines.extend(
                [
                    f"        float physics:mass = {mass}",
                    "        vector3f physics:velocity = (0, 0, 0)",
                    "        vector3f physics:angularVelocity = (0, 0, 0)",
                    f"        bool physxRigidBody:enableCCD = {'1' if self._ccd_enabled else '0'}",
                    "        rel material:binding:physics = </World/PhysicsMaterials/DynamicMaterial> (",
                    '            bindMaterialAs = "weakerThanDescendants"',
                    "        )",
                ]
            )
        body_overrides = ""
        if self._needs_dynamic_body_apis(discovery) and not dynamic_root:
            body_overrides = self._dynamic_body_api_override_blocks(discovery.body_paths)
        ccd_overrides = ""
        if discovery is not None and not dynamic_root and not self._needs_dynamic_body_apis(discovery):
            ccd_overrides = self._ccd_body_api_override_blocks(self._ccd_body_paths(discovery))
        overrides = "\n".join(
            block for block in (body_overrides, ccd_overrides, collision_overrides) if str(block or "").strip()
        )
        overrides = f"\n{overrides}" if overrides else ""
        return f"""    def Xform "{name}" (
{chr(10).join(schema_lines)}
    )
    {{
{chr(10).join(xform_lines)}
{overrides}
    }}"""

    def _drop_visual_transforms(self, count: int) -> list[np.ndarray]:
        drop_count = self._sanitize_drop_count(count)
        asset_indices = self._scene_asset_indices(drop_count, len(self._active_asset_refs()))
        first = self._drop_visual_transform(asset_indices[0] if asset_indices else 0)
        if drop_count <= 1:
            return [first]

        rng = random.Random()
        transforms: list[np.ndarray] = []
        source_bounds = [self._drop_source_bounds(index) for index in asset_indices]
        sizes = [np.asarray(item["size"], dtype=np.float64).reshape(3) for item in source_bounds]
        max_size = np.maximum(np.max(np.stack(sizes, axis=0), axis=0), np.array([0.05, 0.05, 0.05]))
        spacing = self._drop_spacing_amount
        randomness = self._drop_randomness_amount
        clearance = max(0.005, min(0.22, float(np.max(max_size)) * 0.035 * max(0.15, spacing)))
        horizontal_diagonal = max(float(np.linalg.norm(max_size[:2])), 0.1)
        column_spacing = max(0.02, horizontal_diagonal * (0.08 + spacing * 0.95) + clearance)
        vertical_gap = max(float(max_size[2]) * (0.55 + spacing * 0.22) + clearance, 0.08)
        column_count = 1 if drop_count <= 6 else min(drop_count, max(2, math.ceil(math.sqrt(drop_count))))
        column_offsets = self._drop_column_offsets(column_count, column_spacing)
        rng.shuffle(column_offsets)
        base_center = np.asarray(self._center, dtype=np.float64).reshape(3)
        placed_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(drop_count):
            source_index = asset_indices[index] if index < len(asset_indices) else 0
            item_bounds = source_bounds[index] if index < len(source_bounds) else self._drop_source_bounds(source_index)
            size = np.asarray(item_bounds["size"], dtype=np.float64).reshape(3)
            item_clearance = max(clearance, min(0.2, float(np.max(size)) * 0.05))
            column = index % column_count
            layer = index // column_count
            visual = first
            bounds = self._drop_aabb(visual, source_index)
            for attempt in range(10):
                jitter_radius = horizontal_diagonal * 0.65 * randomness * rng.uniform(0.0, 1.25)
                jitter_angle = rng.uniform(0.0, math.tau)
                offset = column_offsets[column] + np.array(
                    [math.cos(jitter_angle) * jitter_radius, math.sin(jitter_angle) * jitter_radius],
                    dtype=np.float64,
                )
                yaw = rng.uniform(-math.pi, math.pi) * min(1.0, 0.35 + randomness * 0.35)
                rotation = first[:3, :3] @ self._z_up_yaw_rotation(yaw)
                target_center = np.array(base_center, dtype=np.float64, copy=True)
                target_center[0] += offset[0]
                target_center[1] += offset[1]
                target_center[2] = self._drop_center_z_for_source(source_index) + vertical_gap * layer
                if randomness > 0.0:
                    target_center[2] += rng.uniform(0.0, vertical_gap * 0.45 * randomness)
                visual = self._visual_transform_for_source_center(source_index, rotation, target_center)
                bounds = self._drop_aabb(visual, source_index)
                if bounds[0][2] < GROUND_TOP_Z + item_clearance:
                    target_center[2] += GROUND_TOP_Z + item_clearance - bounds[0][2]
                    visual = self._visual_transform_for_source_center(source_index, rotation, target_center)
                    bounds = self._drop_aabb(visual, source_index)
                if not any(self._aabb_intersects(bounds, previous, item_clearance) for previous in placed_bounds):
                    break

            guard = 0
            while any(self._aabb_intersects(bounds, previous, item_clearance) for previous in placed_bounds) and guard < drop_count + 4:
                target_center[2] += vertical_gap
                visual = self._visual_transform_for_source_center(source_index, rotation, target_center)
                bounds = self._drop_aabb(visual, source_index)
                guard += 1
            placed_bounds.append(bounds)
            transforms.append(visual)
        return transforms

    @staticmethod
    def _drop_column_offsets(count: int, spacing: float) -> list[np.ndarray]:
        total = max(1, int(count))
        if total <= 1:
            return [np.zeros(2, dtype=np.float64)]
        cols = max(1, math.ceil(math.sqrt(total)))
        rows = max(1, math.ceil(total / cols))
        offsets = []
        for index in range(total):
            row = index // cols
            col = index % cols
            offsets.append(
                np.array(
                    [
                        (float(col) - (float(cols) - 1.0) * 0.5) * float(spacing),
                        (float(row) - (float(rows) - 1.0) * 0.5) * float(spacing),
                    ],
                    dtype=np.float64,
                )
            )
        offsets.sort(key=lambda item: float(np.linalg.norm(item)))
        return offsets

    def _visual_transform_for_center(self, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
        visual = np.eye(4, dtype=np.float64)
        visual[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        visual[3, :3] = np.asarray(center, dtype=np.float64).reshape(3) - self._center @ visual[:3, :3]
        return visual

    def _visual_transform_for_source_center(
        self,
        source_index: int,
        rotation: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        source_bounds = self._drop_source_bounds(source_index)
        source_center = np.asarray(source_bounds["center"], dtype=np.float64).reshape(3)
        visual = np.eye(4, dtype=np.float64)
        visual[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        visual[3, :3] = np.asarray(center, dtype=np.float64).reshape(3) - source_center @ visual[:3, :3]
        return visual

    def _drop_aabb(self, visual: np.ndarray, source_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(visual, dtype=np.float64).reshape(4, 4)
        source_bounds = self._drop_source_bounds(source_index)
        source_center = np.asarray(source_bounds["center"], dtype=np.float64).reshape(3)
        source_size = np.asarray(source_bounds["size"], dtype=np.float64).reshape(3)
        half = np.maximum(source_size, np.array([0.05, 0.05, 0.05])) * 0.5
        center = source_center @ matrix[:3, :3] + matrix[3, :3]
        extent = half @ np.abs(matrix[:3, :3])
        return center - extent, center + extent

    @staticmethod
    def _aabb_intersects(
        a: tuple[np.ndarray, np.ndarray],
        b: tuple[np.ndarray, np.ndarray],
        clearance: float = 0.0,
    ) -> bool:
        a_min, a_max = a
        b_min, b_max = b
        pad = max(0.0, float(clearance))
        return bool(np.all(a_min < b_max + pad) and np.all(a_max > b_min - pad))

    def _drop_visual_transform(self, source_index: int = 0) -> np.ndarray:
        visual = np.array(self._current_visual_transform, dtype=np.float64, copy=True).reshape(4, 4)
        if not self._matrix_is_valid(visual):
            visual = np.eye(4, dtype=np.float64)

        source_bounds = self._drop_source_bounds(source_index)
        source_center = np.asarray(source_bounds["center"], dtype=np.float64).reshape(3)
        source_size = np.asarray(source_bounds["size"], dtype=np.float64).reshape(3)
        half_z = max(float(source_size[2]) * 0.5, 0.05)
        extent = max(float(source_bounds.get("extent", 0.0) or 0.0), float(np.linalg.norm(source_size)) * 0.5, half_z)

        drop_height = max(
            float(source_size[2]) * DROP_HEIGHT_SIZE_SCALE,
            extent * DROP_HEIGHT_EXTENT_SCALE,
            DROP_HEIGHT_MIN,
        )
        drop_height = min(drop_height, DROP_HEIGHT_MAX)

        target_center = np.array(source_center, dtype=np.float64, copy=True)
        target_center[2] = max(target_center[2], GROUND_TOP_Z + half_z + drop_height)
        visual[3, :3] = target_center - source_center @ visual[:3, :3]
        return visual

    def _drop_center_z_for_source(self, source_index: int) -> float:
        source_bounds = self._drop_source_bounds(source_index)
        size = np.asarray(source_bounds["size"], dtype=np.float64).reshape(3)
        half_z = max(float(size[2]) * 0.5, 0.05)
        extent = max(float(source_bounds.get("extent", 0.0) or 0.0), float(np.linalg.norm(size)) * 0.5, half_z)
        drop_height = max(float(size[2]) * DROP_HEIGHT_SIZE_SCALE, extent * DROP_HEIGHT_EXTENT_SCALE, DROP_HEIGHT_MIN)
        return GROUND_TOP_Z + half_z + min(drop_height, DROP_HEIGHT_MAX)

    def _drop_source_bounds(self, source_index: int) -> dict:
        try:
            index = max(0, int(source_index))
        except Exception:
            index = 0
        if self._asset_source_bounds:
            index = min(index, len(self._asset_source_bounds) - 1)
            return self._asset_source_bounds[index]
        return self._bounds or {"center": self._center.tolist(), "size": self._size.tolist(), "extent": 1.0}

    @classmethod
    def _normalize_instance_transforms(
        cls,
        transforms: Optional[list[np.ndarray]],
        fallback: np.ndarray,
    ) -> list[np.ndarray]:
        fallback_matrix = np.array(fallback, dtype=np.float64, copy=True).reshape(4, 4)
        try:
            raw_items = list(transforms or [])
        except TypeError:
            raw_items = []

        normalized: list[np.ndarray] = []
        for item in raw_items[:MAX_DROP_ASSETS]:
            try:
                matrix = np.array(item, dtype=np.float64, copy=True).reshape(4, 4)
            except Exception:
                matrix = np.array(fallback_matrix, dtype=np.float64, copy=True)
            if not cls._matrix_is_valid(matrix):
                matrix = np.array(fallback_matrix, dtype=np.float64, copy=True)
            normalized.append(matrix)
        return normalized or [fallback_matrix]

    @staticmethod
    def _sanitize_drop_count(count: int) -> int:
        try:
            value = int(count)
        except Exception:
            value = 1
        return max(MIN_DROP_ASSETS, min(MAX_DROP_ASSETS, value))

    def _drop_asset_indices(self, count: int) -> list[int]:
        source_count = max(1, len(self._active_asset_refs()))
        total = max(source_count, self._sanitize_drop_count(count))
        indices: list[int] = []
        base = total // source_count
        remainder = total % source_count
        for source_index in range(source_count):
            copies = base + (1 if source_index < remainder else 0)
            indices.extend([source_index] * copies)
        ordered: list[int] = []
        for offset in range(max(indices.count(index) for index in range(source_count))):
            for source_index in range(source_count):
                item_index = sum(indices.count(index) for index in range(source_index)) + offset
                if item_index < len(indices) and indices[item_index] == source_index:
                    ordered.append(source_index)
        return ordered[:total] or [0]

    def _default_asset_indices(self, count: int) -> list[int]:
        source_count = max(1, len(self._active_asset_refs()))
        total = max(1, int(count))
        if total == source_count:
            return list(range(source_count))
        return [index % source_count for index in range(total)]

    def _scene_asset_indices(self, count: int, source_count: int) -> list[int]:
        total = max(1, int(count))
        sources = max(1, int(source_count))
        current = []
        for item in self._authored_scene_asset_indices[:total]:
            try:
                value = int(item)
            except Exception:
                value = 0
            current.append(max(0, min(sources - 1, value)))
        if len(current) == total:
            return current
        if total == sources:
            return list(range(sources))
        return [index % sources for index in range(total)]

    @staticmethod
    def _authored_asset_name(index: int) -> str:
        return AUTHORED_ASSET_NAME if index <= 0 else f"Instance_{index + 1:02d}"

    @classmethod
    def _authored_asset_path(cls, index: int) -> str:
        return f"/World/{cls._authored_asset_name(index)}"

    @classmethod
    def _dynamic_root_body_patterns(cls, index: int) -> list[str]:
        root = cls._authored_asset_path(index)
        return [root, f"{root}/*", f"{root}/Geometry/*", f"{root}/Geometry/*/*"]

    @staticmethod
    def _needs_dynamic_body_apis(discovery: Optional[AuthoredColliderDiscovery]) -> bool:
        if discovery is None:
            return False
        return (
            int(discovery.collider_count) > 0
            and not discovery.rigid_body_paths
            and not discovery.articulation_paths
        )

    @classmethod
    def _uses_dynamic_root(cls, discovery: Optional[AuthoredColliderDiscovery]) -> bool:
        return cls._needs_dynamic_body_apis(discovery) and not list(getattr(discovery, "body_paths", []) or [])

    @classmethod
    def _dynamic_body_paths(cls, discovery: Optional[AuthoredColliderDiscovery], index: int) -> list[str]:
        if not cls._needs_dynamic_body_apis(discovery):
            return []
        paths = list(getattr(discovery, "body_paths", []) or [])
        if not paths:
            paths = [AUTHORED_ASSET_PATH]
        return cls._map_authored_paths_for_index(paths, index)

    @classmethod
    def _instance_root_paths(cls, count: int) -> list[str]:
        return [cls._authored_asset_path(index) for index in range(max(1, int(count)))]

    def _instance_reference_poses(self) -> list[dict]:
        transforms = self._normalize_instance_transforms(
            self._authored_scene_instance_transforms,
            np.eye(4, dtype=np.float64),
        )
        poses = []
        for index, matrix in enumerate(transforms):
            poses.append(
                {
                    "path": self._authored_asset_path(index),
                    "pose": self._pose_from_visual(matrix).astype(float).tolist(),
                }
            )
        return poses

    @classmethod
    def _expand_authored_paths(cls, paths: list[str], count: int) -> list[str]:
        expanded: list[str] = []
        for index in range(max(1, int(count))):
            cls._extend_unique(expanded, cls._map_authored_paths_for_index(paths, index))
        return expanded

    @classmethod
    def _map_authored_paths_for_index(cls, paths: list[str], index: int) -> list[str]:
        mapped_paths: list[str] = []
        root = cls._authored_asset_path(index)
        for path in paths or []:
            text = str(path or "").strip()
            if not text:
                continue
            if text == AUTHORED_ASSET_PATH:
                mapped = root
            elif text.startswith(f"{AUTHORED_ASSET_PATH}/"):
                mapped = root + text[len(AUTHORED_ASSET_PATH) :]
            else:
                mapped = text
            if mapped not in mapped_paths:
                mapped_paths.append(mapped)
        return mapped_paths

    @staticmethod
    def _extend_unique(target: list[str], values: list[str]) -> None:
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in target:
                target.append(text)

    @staticmethod
    def _z_up_yaw_rotation(angle: float) -> np.ndarray:
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        return np.array(
            [
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _usd_asset_reference(source: str) -> str:
        text = str(source or "").strip().replace("\\", "/")
        if text.startswith(f"s3://{S3_BUCKET}/"):
            key = text[len(f"s3://{S3_BUCKET}/") :]
            return f"{S3_HTTP_ROOT}/{urllib.parse.quote(key, safe='/')}"
        if "://" not in text and text:
            try:
                text = Path(text).expanduser().resolve().as_posix()
            except Exception:
                pass
        return text.replace("@", "%40")

    def _authored_collider_discovery(self, asset_ref: str) -> AuthoredColliderDiscovery:
        """Discover SimReady authored colliders and cook overrides without making new colliders."""
        payload_discovery = self._payload_collider_discovery(asset_ref)
        if payload_discovery.collider_count > 0:
            return payload_discovery

        if self._should_use_slow_usd_discovery(asset_ref):
            stage_discovery = self._stage_collider_discovery(asset_ref)
            if stage_discovery is not None and stage_discovery.collider_count > 0:
                return stage_discovery
            if stage_discovery is not None and stage_discovery.body_paths:
                return stage_discovery
        return payload_discovery

    def _active_asset_refs(self) -> list[str]:
        refs = [self._usd_asset_reference(source) for source in self._usd_sources if str(source or "").strip()]
        if not refs and self._usd_source:
            refs = [self._usd_asset_reference(self._usd_source)]
        return refs

    def _can_reuse_live_scene(self, asset_refs: list[str], active_count: int) -> bool:
        if self._process is None or not self._worker_ready:
            return False
        try:
            count = max(1, int(active_count))
        except Exception:
            count = 1
        return (
            tuple(asset_refs or []) == self._cooked_asset_refs
            and self._base_scene == self._cooked_base_scene
            and bool(self._ccd_enabled) == bool(self._cooked_ccd_enabled)
            and int(self._steps_per_second) == int(self._cooked_steps_per_second)
            and self._physx_device_mode == self._cooked_physx_device_mode
            and bool(self._authored_contact_instances_required) == bool(self._cooked_contact_instances_required)
            and self._cooked_instance_capacity >= count
        )

    def _pad_instance_transforms(self, transforms: list[np.ndarray], capacity: int) -> list[np.ndarray]:
        instances = self._normalize_instance_transforms(transforms, self._current_visual_transform)
        try:
            target = max(len(instances), int(capacity))
        except Exception:
            target = len(instances)
        if target <= len(instances):
            return [np.array(item, dtype=np.float64, copy=True) for item in instances]

        padded = [np.array(item, dtype=np.float64, copy=True) for item in instances]
        for index in range(len(padded), target):
            padded.append(self._hidden_instance_transform(index))
        return padded

    @staticmethod
    def _hidden_instance_transform(index: int = 0) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[3, 0] = 8000.0
        matrix[3, 1] = float(max(0, int(index))) * 8.0
        matrix[3, 2] = -1000.0
        return matrix

    def _clear_runtime_clone_state(self) -> None:
        self._runtime_clone_source_path = ""
        self._runtime_clone_target_paths = []
        self._runtime_clone_parent_poses = []
        self._runtime_clone_groups = []

    def _add_runtime_clone_group(
        self,
        source_path: str,
        target_paths: list[str],
        parent_poses: list[np.ndarray],
    ) -> None:
        source = str(source_path or "").strip()
        targets = [str(path or "").strip() for path in target_paths if str(path or "").strip()]
        if not source or not targets:
            return
        poses = [np.array(pose, dtype=np.float32, copy=True).reshape(7) for pose in parent_poses[: len(targets)]]
        if len(poses) != len(targets):
            poses = []
        group = {"source": source, "targets": targets, "parent_poses": poses}
        self._runtime_clone_groups.append(group)
        if not self._runtime_clone_source_path:
            self._runtime_clone_source_path = source
            self._runtime_clone_target_paths = list(targets)
            self._runtime_clone_parent_poses = [np.array(pose, dtype=np.float32, copy=True) for pose in poses]

    def _runtime_clone_groups_payload(self) -> list[dict]:
        payload: list[dict] = []
        for group in self._runtime_clone_groups:
            source = str(group.get("source", "") or "").strip()
            targets = [str(path or "").strip() for path in group.get("targets", []) if str(path or "").strip()]
            poses = [
                np.array(pose, dtype=np.float32, copy=True).reshape(7).astype(float).tolist()
                for pose in group.get("parent_poses", [])
            ]
            if source and targets:
                payload.append({"source": source, "targets": targets, "parent_poses": poses})
        return payload

    def _clear_live_cooked_scene_state(self) -> None:
        self._cooked_asset_refs = ()
        self._cooked_base_scene = ""
        self._cooked_ccd_enabled = False
        self._cooked_steps_per_second = 0
        self._cooked_physx_device_mode = ""
        self._cooked_instance_capacity = 0
        self._cooked_contact_instances_required = False

    @staticmethod
    def _configured_asset_sources(bounds: dict, usd_source: Optional[str]) -> list[str]:
        if usd_source:
            return [str(usd_source)]
        try:
            raw_sources = list(bounds.get("_asset_sources", []) if isinstance(bounds, dict) else [])
        except TypeError:
            raw_sources = []
        sources: list[str] = []
        for item in raw_sources:
            text = str(item or "").strip()
            if text and text not in sources:
                sources.append(text)
        return sources

    @classmethod
    def _configured_asset_transforms(cls, bounds: dict, source_count: int) -> list[np.ndarray]:
        if source_count <= 0:
            return [np.eye(4, dtype=np.float64)]
        raw_transforms = []
        if isinstance(bounds, dict):
            try:
                raw_transforms = list(bounds.get("_asset_layout_transforms", []) or [])
            except TypeError:
                raw_transforms = []
        transforms: list[np.ndarray] = []
        for item in raw_transforms[:source_count]:
            try:
                matrix = np.array(item, dtype=np.float64, copy=True).reshape(4, 4)
            except Exception:
                matrix = np.eye(4, dtype=np.float64)
            if not cls._matrix_is_valid(matrix):
                matrix = np.eye(4, dtype=np.float64)
            transforms.append(matrix)
        while len(transforms) < source_count:
            transforms.append(np.eye(4, dtype=np.float64))
        return transforms

    @classmethod
    def _configured_asset_bounds(cls, bounds: dict, source_count: int) -> list[dict]:
        if source_count <= 0:
            return []
        raw_bounds = []
        if isinstance(bounds, dict):
            try:
                raw_bounds = list(bounds.get("_asset_bounds", []) or [])
            except TypeError:
                raw_bounds = []
        fallback = cls._normalize_bounds(bounds if isinstance(bounds, dict) else {})
        normalized: list[dict] = []
        for item in raw_bounds[:source_count]:
            try:
                normalized.append(cls._normalize_bounds(item if isinstance(item, dict) else {}))
            except Exception:
                normalized.append(dict(fallback))
        while len(normalized) < source_count:
            normalized.append(dict(fallback))
        return normalized

    @staticmethod
    def _empty_discovery() -> AuthoredColliderDiscovery:
        return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], [], 0, 0)

    @classmethod
    def _cached_discovery(cls, asset_ref: str) -> Optional[AuthoredColliderDiscovery]:
        key = str(asset_ref or "")
        if not key:
            return None
        cached = cls._DISCOVERY_CACHE.get(key)
        if cached is not None:
            return cached
        cached = cls._load_cached_discovery_from_disk(key)
        if cached is not None:
            cls._remember_discovery(key, cached)
        return cached

    @classmethod
    def _store_discovery(cls, asset_ref: str, discovery: AuthoredColliderDiscovery) -> None:
        key = str(asset_ref or "")
        if not key:
            return
        cls._remember_discovery(key, discovery)
        cls._write_cached_discovery_to_disk(key, discovery)

    @classmethod
    def _remember_discovery(cls, asset_ref: str, discovery: AuthoredColliderDiscovery) -> None:
        key = str(asset_ref or "")
        if not key:
            return
        cls._DISCOVERY_CACHE[key] = discovery
        if key in cls._DISCOVERY_CACHE_ORDER:
            cls._DISCOVERY_CACHE_ORDER.remove(key)
        cls._DISCOVERY_CACHE_ORDER.append(key)
        while len(cls._DISCOVERY_CACHE_ORDER) > cls._DISCOVERY_CACHE_LIMIT:
            old_key = cls._DISCOVERY_CACHE_ORDER.pop(0)
            cls._DISCOVERY_CACHE.pop(old_key, None)

    @classmethod
    def _load_cached_discovery_from_disk(cls, asset_ref: str) -> Optional[AuthoredColliderDiscovery]:
        try:
            path = cls._discovery_cache_path(asset_ref)
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("version", 0) or 0) != DISCOVERY_CACHE_VERSION:
                return None
            data = payload.get("discovery")
            if not isinstance(data, dict):
                return None
            return cls._discovery_from_payload(data)
        except Exception:
            return None

    @classmethod
    def _write_cached_discovery_to_disk(cls, asset_ref: str, discovery: AuthoredColliderDiscovery) -> None:
        try:
            path = cls._discovery_cache_path(asset_ref)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": DISCOVERY_CACHE_VERSION,
                "asset_ref": asset_ref,
                "discovery": cls._discovery_to_payload(discovery),
            }
            path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _discovery_cache_path(asset_ref: str) -> Path:
        digest = hashlib.sha1(str(asset_ref or "").encode("utf-8", "replace")).hexdigest()
        return Path(__file__).resolve().parents[1] / "cache" / "physics_discovery" / f"{digest}.json"

    @staticmethod
    def _discovery_from_stdout(text: str) -> Optional[AuthoredColliderDiscovery]:
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
        return PhysicsController._discovery_from_payload(payload)

    @staticmethod
    def _discoveries_from_stdout(text: str, asset_refs: list[str]) -> Optional[list[AuthoredColliderDiscovery]]:
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
        if "discoveries" not in payload:
            discovery = PhysicsController._discovery_from_payload(payload)
            return [discovery] if len(asset_refs) == 1 else None
        raw_items = payload.get("discoveries")
        if not isinstance(raw_items, list) or len(raw_items) != len(asset_refs):
            return None
        discoveries: list[AuthoredColliderDiscovery] = []
        for item in raw_items:
            if not isinstance(item, dict):
                return None
            discoveries.append(PhysicsController._discovery_from_payload(item))
        return discoveries

    @staticmethod
    def _discovery_from_payload(payload: dict) -> AuthoredColliderDiscovery:
        try:
            collider_count = int(payload.get("collider_count", 0) or 0)
        except Exception:
            collider_count = 0
        return AuthoredColliderDiscovery(
            PhysicsController._collision_overrides_from_payload(payload),
            [str(path) for path in payload.get("body_patterns", []) if str(path or "").strip()]
            or list(AUTHORED_BODY_PATTERNS),
            [str(path) for path in payload.get("body_paths", []) if str(path or "").strip()],
            [str(path) for path in payload.get("rigid_body_paths", []) if str(path or "").strip()],
            [str(path) for path in payload.get("articulation_paths", []) if str(path or "").strip()],
            collider_count,
            0,
        )

    @classmethod
    def _collision_overrides_from_payload(cls, payload: dict) -> str:
        explicit = str(payload.get("collision_overrides", "") or "")
        if explicit.strip():
            return explicit

        collider_paths: list[str] = []
        for item in payload.get("colliders", []) or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "") or "").strip()
            if not path.startswith(f"{AUTHORED_ASSET_PATH}/"):
                continue

            prim_type = str(item.get("prim_type", "") or "")
            if prim_type and prim_type != "Mesh":
                continue

            schemas = {str(schema).lower() for schema in item.get("schemas", []) or []}
            approximation = str(item.get("approximation", "") or "").strip().lower()
            collider_type = str(item.get("type", "") or "").strip().lower()
            dynamic_ready = {
                "sdf",
                "convexhull",
                "convexdecomposition",
            }
            if approximation in dynamic_ready or collider_type in dynamic_ready:
                continue
            if "physxsdfmeshcollisionapi" in schemas:
                continue
            if path not in collider_paths:
                collider_paths.append(path)
        return cls._collision_override_blocks(collider_paths)

    @staticmethod
    def _discovery_to_payload(discovery: AuthoredColliderDiscovery) -> dict:
        return {
            "collision_overrides": str(discovery.collision_overrides or ""),
            "body_patterns": list(discovery.body_patterns or []),
            "body_paths": list(discovery.body_paths or []),
            "rigid_body_paths": list(discovery.rigid_body_paths or []),
            "articulation_paths": list(discovery.articulation_paths or []),
            "collider_count": int(discovery.collider_count),
            "override_count": 0,
        }

    @classmethod
    def _collision_override_blocks(cls, collider_paths: list[str]) -> str:
        tree: dict[str, dict] = {}
        for path in collider_paths or []:
            text = str(path or "").strip()
            if not text.startswith(f"{AUTHORED_ASSET_PATH}/"):
                continue
            parts = [part for part in text[len(f"{AUTHORED_ASSET_PATH}/") :].split("/") if part]
            if not parts:
                continue
            node = tree
            for part in parts:
                node = node.setdefault(cls._usd_name(part), {})
            node["__collider__"] = {}
        return "\n".join(cls._render_collision_override_tree(tree, 2))

    @classmethod
    def _render_collision_override_tree(cls, tree: dict, indent_level: int) -> list[str]:
        lines: list[str] = []
        indent = " " * (indent_level * 4)
        child_indent = " " * ((indent_level + 1) * 4)
        for name in sorted(key for key in tree.keys() if key != "__collider__"):
            child = tree.get(name, {})
            is_collider = "__collider__" in child
            if is_collider:
                lines.append(f'{indent}over "{name}" (')
                lines.append(
                    f'{child_indent}prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", '
                    '"PhysxConvexDecompositionCollisionAPI"]'
                )
                lines.append(f"{indent})")
            else:
                lines.append(f'{indent}over "{name}"')
            lines.append(f"{indent}{{")
            if is_collider:
                lines.append(
                    f'{child_indent}uniform token physics:approximation = "{DYNAMIC_COLLIDER_APPROXIMATION}"'
                )
                lines.append(f"{child_indent}int physxConvexDecompositionCollision:maxConvexHulls = 32")
                lines.append(f"{child_indent}int physxConvexDecompositionCollision:hullVertexLimit = 64")
            lines.extend(cls._render_collision_override_tree(child, indent_level + 1))
            lines.append(f"{indent}}}")
        return lines

    def _stage_collider_discovery(self, asset_ref: str) -> Optional[AuthoredColliderDiscovery]:
        helper_pythons = self._usd_discovery_python_candidates()
        if not helper_pythons:
            return None

        for helper_python in helper_pythons:
            try:
                result = subprocess.run(
                    [
                        helper_python,
                        "-u",
                        "-m",
                        "core.usd_collision_discovery",
                        asset_ref,
                        AUTHORED_ASSET_PATH,
                    ],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    capture_output=True,
                    text=True,
                    timeout=90.0,
                    check=False,
                )
            except Exception:
                continue

            if result.returncode != 0:
                continue

            payload = None
            for line in reversed((result.stdout or "").splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if isinstance(payload, dict):
                return self._discovery_from_stage_payload(payload)

        return None

    def _discovery_from_stage_payload(self, payload: dict) -> AuthoredColliderDiscovery:
        if not isinstance(payload, dict):
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], [], 0, 0)

        body_patterns = [str(path) for path in payload.get("body_patterns", []) if str(path or "").strip()]
        body_paths = [str(path) for path in payload.get("body_paths", []) if str(path or "").strip()]
        rigid_body_paths = [str(path) for path in payload.get("rigid_body_paths", []) if str(path or "").strip()]
        collider_count = int(payload.get("collider_count", 0) or 0)
        if collider_count <= 0:
            return AuthoredColliderDiscovery("", body_patterns or list(AUTHORED_BODY_PATTERNS), [], [], [], 0, 0)
        return AuthoredColliderDiscovery(
            self._collision_overrides_from_payload(payload),
            body_patterns or list(AUTHORED_BODY_PATTERNS),
            body_paths,
            rigid_body_paths,
            [str(path) for path in payload.get("articulation_paths", []) if str(path or "").strip()],
            collider_count,
            0,
        )

    @staticmethod
    def _should_use_slow_usd_discovery(asset_ref: str) -> bool:
        if os.environ.get(ENABLE_SLOW_USD_COLLIDER_DISCOVERY_ENV) == "1":
            return True
        text = str(asset_ref or "")
        if text.startswith("http://") or text.startswith("https://") or text.startswith("s3://"):
            return False
        return bool(text and "://" not in text)

    @staticmethod
    def _usd_discovery_python() -> Optional[str]:
        candidates = PhysicsController._usd_discovery_python_candidates()
        return candidates[0] if candidates else None

    @staticmethod
    def _usd_discovery_python_candidates() -> list[str]:
        root = Path(__file__).resolve().parents[1]
        candidates = [
            os.environ.get(USD_DISCOVERY_PYTHON_ENV, ""),
            str(root / ".usd_discovery_venv" / "Scripts" / "python.exe"),
            str(root / ".usd_discovery_venv" / "bin" / "python"),
            sys.executable,
        ]
        result: list[str] = []
        seen: set[str] = set()
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
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(str(resolved))
        return result

    def _payload_collider_discovery(self, asset_ref: str) -> AuthoredColliderDiscovery:
        base_text = self._read_simready_payload_text(asset_ref, "base.usda")
        instances_text = self._read_simready_payload_text(asset_ref, "instances.usda")
        if not base_text or not instances_text:
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], [], 0, 0)

        collision_instances = self._collision_instances(instances_text)
        if not collision_instances:
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], [], 0, 0)

        references = self._simready_instance_references(base_text)
        collision_refs = [(path, name) for path, name in references if name in collision_instances]
        if not collision_refs:
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], [], 0, 0)

        body_patterns = self._authored_body_patterns_from_refs(collision_refs)
        body_paths = self._authored_body_paths_from_refs(collision_refs)
        collision_paths = self._authored_collider_paths_from_refs(collision_refs, collision_instances)
        return AuthoredColliderDiscovery(
            self._collision_override_blocks(collision_paths),
            body_patterns,
            body_paths,
            [],
            [],
            len(collision_refs),
            0,
        )

    def _authored_body_patterns_from_refs(self, refs: list[tuple[tuple[str, ...], str]]) -> list[str]:
        patterns: list[str] = []

        def add(path: str) -> None:
            if path and path not in patterns and len(patterns) < MAX_DISCOVERED_BODY_PATTERNS:
                patterns.append(path)

        for rel_path, instance_name in refs:
            clean_parts = [self._usd_name(part) for part in rel_path if part]
            instance_name = self._usd_name(instance_name)
            if not clean_parts:
                continue
            if clean_parts[0] == "Geometry" and len(clean_parts) > 1:
                object_path = f"{AUTHORED_ASSET_PATH}/Geometry/{clean_parts[1]}"
                if len(clean_parts) >= 3:
                    mesh_path = f"{object_path}/{clean_parts[-1]}"
                elif instance_name:
                    mesh_path = f"{object_path}/{instance_name}"
                else:
                    mesh_path = object_path
            else:
                full_parts = [AUTHORED_ASSET_PATH, *clean_parts]
                object_path = "/".join(full_parts[:2])
                mesh_path = f"{object_path}/{instance_name}" if len(clean_parts) == 1 and instance_name else "/".join(full_parts)
            add(object_path)
            add(f"{object_path}/*")
            add(mesh_path)
            add(f"{mesh_path}/*")

        for fallback in AUTHORED_BODY_PATTERNS:
            add(fallback)
        return patterns or list(AUTHORED_BODY_PATTERNS)

    def _authored_body_paths_from_refs(self, refs: list[tuple[tuple[str, ...], str]]) -> list[str]:
        paths: list[str] = []

        def add(path: str) -> None:
            if path and path not in paths and len(paths) < MAX_DISCOVERED_BODY_PATTERNS:
                paths.append(path)

        for rel_path, _instance_name in refs:
            clean_parts = [self._usd_name(part) for part in rel_path if part]
            if not clean_parts:
                continue
            if clean_parts[0] == "Geometry" and len(clean_parts) > 1:
                add(f"{AUTHORED_ASSET_PATH}/Geometry/{clean_parts[1]}")
            else:
                full_parts = [AUTHORED_ASSET_PATH, *clean_parts]
                add("/".join(full_parts[:2]))
        return paths

    def _authored_collider_paths_from_refs(
        self,
        refs: list[tuple[tuple[str, ...], str]],
        collision_instances: dict[str, str],
    ) -> list[str]:
        paths: list[str] = []

        def add(path: str) -> None:
            if path and path not in paths and len(paths) < MAX_DISCOVERED_BODY_PATTERNS:
                paths.append(path)

        for rel_path, instance_name in refs:
            approximation = str(collision_instances.get(instance_name, "") or "").strip().lower()
            if approximation in {"sdf", "convexdecomposition", "convexhull"}:
                continue

            clean_parts = [self._usd_name(part) for part in rel_path if part]
            instance_name = self._usd_name(instance_name)
            if not clean_parts:
                continue
            if clean_parts[0] == "Geometry" and len(clean_parts) > 1:
                object_path = f"{AUTHORED_ASSET_PATH}/Geometry/{clean_parts[1]}"
                if len(clean_parts) >= 3:
                    mesh_path = f"{object_path}/{clean_parts[-1]}"
                elif instance_name:
                    mesh_path = f"{object_path}/{instance_name}"
                else:
                    mesh_path = object_path
            else:
                full_parts = [AUTHORED_ASSET_PATH, *clean_parts]
                object_path = "/".join(full_parts[:2])
                mesh_path = f"{object_path}/{instance_name}" if len(clean_parts) == 1 and instance_name else "/".join(full_parts)
            add(mesh_path)
        return paths

    @staticmethod
    def _collision_instances(instances_text: str) -> dict[str, str]:
        instances: dict[str, str] = {}
        pattern = re.compile(
            r'\b(?:def|over|class)\s+(?:\w+\s+)?"([^"]+)"\s*(?P<meta>\([^{}]*?\))?\s*\{',
            re.MULTILINE | re.DOTALL,
        )
        for match in pattern.finditer(instances_text):
            name = match.group(1)
            meta = match.group("meta") or ""
            body_start = match.end() - 1
            body_end = PhysicsController._matching_brace(instances_text, body_start)
            if body_end <= body_start:
                continue
            body = instances_text[body_start + 1 : body_end]
            if "PhysicsCollisionAPI" not in f"{meta}\n{body}":
                continue
            approximation_match = re.search(r'physics:approximation\s*=\s*"([^"]+)"', body)
            approximation = approximation_match.group(1).strip() if approximation_match else ""
            if not approximation and "PhysxSDFMeshCollisionAPI" in f"{meta}\n{body}":
                approximation = "sdf"
            elif not approximation and "PhysxConvexDecompositionCollisionAPI" in f"{meta}\n{body}":
                approximation = "convexDecomposition"
            elif not approximation and "PhysxConvexHullCollisionAPI" in f"{meta}\n{body}":
                approximation = "convexHull"
            instances[name] = approximation
        return instances

    @staticmethod
    def _simready_instance_references(base_text: str) -> list[tuple[tuple[str, ...], str]]:
        refs: list[tuple[tuple[str, ...], str]] = []
        stack: list[tuple[int, str]] = []
        prim_pattern = re.compile(r'\b(?:def|over|class)\s+\w+\s+"([^"]+)"')
        ref_pattern = re.compile(r'references\s*=\s*(?:\[[^\]]*)?@[^@\r\n]*instances\.usda@</Instances/([^>]+)>')

        for raw_line in base_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))

            if stripped.startswith("}"):
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                continue

            prim_match = prim_pattern.search(stripped)
            if prim_match:
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                stack.append((indent, prim_match.group(1)))

            ref_match = ref_pattern.search(stripped)
            if not ref_match:
                continue

            names = tuple(name for _level, name in stack)
            if "Geometry" in names:
                names = names[names.index("Geometry") :]
            instance_name = ref_match.group(1).strip()
            item = (names, instance_name)
            if names and item not in refs:
                refs.append(item)
        return refs

    @staticmethod
    def _matching_brace(text: str, open_index: int) -> int:
        depth = 0
        for index in range(open_index, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @classmethod
    def _read_simready_payload_text(cls, asset_ref: str, name: str) -> str:
        if not asset_ref:
            return ""

        cache_key = (str(asset_ref), str(name))
        cached = cls._PAYLOAD_TEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        text = ""
        url = ""
        if asset_ref.startswith("http://") or asset_ref.startswith("https://"):
            base = asset_ref.rsplit("/", 1)[0]
            url = f"{base}/payloads/{name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SimReadyBrowser/1.0"})
                with urllib.request.urlopen(req, timeout=1.5) as response:
                    text = response.read().decode("utf-8", "replace")
            except Exception:
                text = ""
            cls._store_payload_text(cache_key, text)
            return text

        if "://" in asset_ref:
            return ""

        try:
            path = Path(asset_ref)
            payload = path.parent / "payloads" / name
            if payload.exists():
                text = payload.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        cls._store_payload_text(cache_key, text)
        return text

    @classmethod
    def _store_payload_text(cls, cache_key: tuple[str, str], text: str) -> None:
        cls._PAYLOAD_TEXT_CACHE[cache_key] = text
        if cache_key in cls._PAYLOAD_TEXT_CACHE_ORDER:
            cls._PAYLOAD_TEXT_CACHE_ORDER.remove(cache_key)
        cls._PAYLOAD_TEXT_CACHE_ORDER.append(cache_key)
        while len(cls._PAYLOAD_TEXT_CACHE_ORDER) > cls._PAYLOAD_TEXT_CACHE_LIMIT:
            old_key = cls._PAYLOAD_TEXT_CACHE_ORDER.pop(0)
            cls._PAYLOAD_TEXT_CACHE.pop(old_key, None)

    @staticmethod
    def _usd_name(value: str) -> str:
        return str(value or "").replace("\\", "_").replace('"', "_")

    def _base_scene_usda_z_up(self) -> str:
        if self._base_scene == "none":
            return ""
        parts = [self._ground_usda_z_up()]
        if self._base_scene == "ramp":
            parts.append(self._ramp_usda_z_up())
        elif self._base_scene == "obstacles":
            parts.extend(
                [
                    self._box_usda_z_up("ObstacleA", (-2.4, -1.4, 0.35), (1.2, 1.2, 0.7)),
                    self._box_usda_z_up("ObstacleB", (1.8, 1.2, 0.6), (1.0, 1.6, 1.2)),
                    self._box_usda_z_up("ObstacleC", (0.0, -2.8, 0.25), (2.0, 0.6, 0.5)),
                ]
            )
        return "\n".join(parts)

    def _physics_materials_usda(self) -> str:
        return f"""    def Scope "PhysicsMaterials"
    {{
        def Material "DynamicMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {self._fmt(ISAAC_DYNAMIC_STATIC_FRICTION)}
            float physics:dynamicFriction = {self._fmt(ISAAC_DYNAMIC_FRICTION)}
            float physics:restitution = {self._fmt(ISAAC_DYNAMIC_RESTITUTION)}
        }}

        def Material "GroundMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {self._fmt(ISAAC_GROUND_STATIC_FRICTION)}
            float physics:dynamicFriction = {self._fmt(ISAAC_GROUND_DYNAMIC_FRICTION)}
            float physics:restitution = {self._fmt(ISAAC_GROUND_RESTITUTION)}
        }}
    }}"""

    def _dynamic_body_api_override_blocks(self, body_paths: list[str]) -> str:
        tree: dict[str, dict] = {}
        for path in body_paths or []:
            text = str(path or "").strip()
            if not text.startswith(f"{AUTHORED_ASSET_PATH}/"):
                continue
            parts = [part for part in text[len(f"{AUTHORED_ASSET_PATH}/") :].split("/") if part]
            if not parts:
                continue
            node = tree
            for part in parts:
                node = node.setdefault(self._usd_name(part), {})
            node["__rigid_body__"] = {}
        return "\n".join(self._render_dynamic_body_api_tree(tree, 2))

    def _ccd_body_api_override_blocks(self, body_paths: list[str]) -> str:
        tree: dict[str, dict] = {}
        for path in body_paths or []:
            text = str(path or "").strip()
            if not text.startswith(f"{AUTHORED_ASSET_PATH}/"):
                continue
            parts = [part for part in text[len(f"{AUTHORED_ASSET_PATH}/") :].split("/") if part]
            if not parts:
                continue
            node = tree
            for part in parts:
                node = node.setdefault(self._usd_name(part), {})
            node["__ccd_body__"] = {}
        return "\n".join(self._render_ccd_body_api_tree(tree, 2))

    @staticmethod
    def _ccd_body_paths(discovery: Optional[AuthoredColliderDiscovery]) -> list[str]:
        paths: list[str] = []
        if discovery is None:
            return paths
        PhysicsController._extend_unique(paths, list(getattr(discovery, "rigid_body_paths", []) or []))
        PhysicsController._extend_unique(paths, list(getattr(discovery, "body_paths", []) or []))
        return paths

    def _render_ccd_body_api_tree(self, tree: dict, indent_level: int) -> list[str]:
        lines: list[str] = []
        indent = " " * (indent_level * 4)
        child_indent = " " * ((indent_level + 1) * 4)
        for name in sorted(key for key in tree.keys() if key != "__ccd_body__"):
            child = tree.get(name, {})
            is_rigid_body = "__ccd_body__" in child
            if is_rigid_body:
                lines.append(f'{indent}over "{name}" (')
                lines.append(f'{child_indent}prepend apiSchemas = ["PhysxRigidBodyAPI"]')
                lines.append(f"{indent})")
            else:
                lines.append(f'{indent}over "{name}"')
            lines.append(f"{indent}{{")
            if is_rigid_body:
                lines.append(f"{child_indent}bool physxRigidBody:enableCCD = {'1' if self._ccd_enabled else '0'}")
            lines.extend(self._render_ccd_body_api_tree(child, indent_level + 1))
            lines.append(f"{indent}}}")
        return lines

    def _render_dynamic_body_api_tree(self, tree: dict, indent_level: int) -> list[str]:
        lines: list[str] = []
        indent = " " * (indent_level * 4)
        child_indent = " " * ((indent_level + 1) * 4)
        for name in sorted(key for key in tree.keys() if key != "__rigid_body__"):
            child = tree.get(name, {})
            is_rigid_body = "__rigid_body__" in child
            if is_rigid_body:
                lines.append(f'{indent}over "{name}" (')
                lines.append(
                    f'{child_indent}prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", '
                    '"MaterialBindingAPI"]'
                )
                lines.append(f"{indent})")
            else:
                lines.append(f'{indent}over "{name}"')
            lines.append(f"{indent}{{")
            if is_rigid_body:
                lines.append(f"{child_indent}vector3f physics:velocity = (0, 0, 0)")
                lines.append(f"{child_indent}vector3f physics:angularVelocity = (0, 0, 0)")
                lines.append(f"{child_indent}bool physxRigidBody:enableCCD = {'1' if self._ccd_enabled else '0'}")
                lines.append(f"{child_indent}rel material:binding:physics = </World/PhysicsMaterials/DynamicMaterial> (")
                lines.append(f'{child_indent}    bindMaterialAs = "weakerThanDescendants"')
                lines.append(f"{child_indent})")
            lines.extend(self._render_dynamic_body_api_tree(child, indent_level + 1))
            lines.append(f"{indent}}}")
        return lines

    def _contact_offset(self) -> float:
        min_size = float(np.min(np.maximum(self._size, 0.001)))
        return max(0.003, min(0.1, min_size * 0.025))

    def _estimated_mass_for_source(self, source_index: int) -> float:
        try:
            bounds = self._drop_source_bounds(source_index)
            size = np.asarray(bounds.get("size", self._size), dtype=np.float64).reshape(3)
        except Exception:
            size = np.asarray(self._size, dtype=np.float64).reshape(3)
        return self._estimate_asset_mass(size)

    @staticmethod
    def _estimate_asset_mass(size: np.ndarray) -> float:
        try:
            dims = np.asarray(size, dtype=np.float64).reshape(3)
        except Exception:
            dims = np.ones(3, dtype=np.float64)
        dims = np.maximum(np.abs(dims), 0.02)
        volume = float(np.prod(dims))
        if not math.isfinite(volume) or volume <= 0.0:
            volume = 1.0
        mass = volume * ESTIMATED_ASSET_DENSITY_KG_M3
        return max(MIN_ESTIMATED_MASS_KG, min(mass, MAX_ESTIMATED_MASS_KG))

    @staticmethod
    def _sanitize_grab_force_amount(value: float) -> float:
        try:
            amount = float(value)
        except Exception:
            amount = DEFAULT_GRAB_FORCE_AMOUNT
        if not math.isfinite(amount):
            amount = DEFAULT_GRAB_FORCE_AMOUNT
        return max(MIN_GRAB_FORCE_AMOUNT, min(amount, MAX_GRAB_FORCE_AMOUNT))

    @staticmethod
    def _sanitize_drop_spacing(value: float) -> float:
        try:
            amount = float(value)
        except Exception:
            amount = DEFAULT_DROP_SPACING
        if not math.isfinite(amount):
            amount = DEFAULT_DROP_SPACING
        return max(MIN_DROP_SPACING, min(amount, MAX_DROP_SPACING))

    @staticmethod
    def _sanitize_drop_randomness(value: float) -> float:
        try:
            amount = float(value)
        except Exception:
            amount = DEFAULT_DROP_RANDOMNESS
        if not math.isfinite(amount):
            amount = DEFAULT_DROP_RANDOMNESS
        return max(MIN_DROP_RANDOMNESS, min(amount, MAX_DROP_RANDOMNESS))

    @staticmethod
    def _sanitize_steps_per_second(value: int) -> int:
        try:
            steps = int(value)
        except Exception:
            steps = DEFAULT_PHYSX_STEPS_PER_SECOND
        return max(MIN_PHYSX_STEPS_PER_SECOND, min(steps, MAX_PHYSX_STEPS_PER_SECOND))

    @staticmethod
    def _sanitize_substeps(value: int) -> int:
        try:
            substeps = int(value)
        except Exception:
            substeps = DEFAULT_SUBSTEPS
        return max(MIN_PHYSX_SUBSTEPS, min(substeps, MAX_PHYSX_SUBSTEPS))

    @staticmethod
    def _sanitize_device_mode(value: str) -> str:
        mode = str(value or "").strip().lower()
        if mode.startswith("cuda"):
            mode = "gpu"
        if mode not in PHYSX_DEVICE_MODES:
            mode = DEFAULT_PHYSX_DEVICE_MODE
        return mode

    def _ground_usda_z_up(self) -> str:
        ground_z = -GROUND_THICKNESS * 0.5
        return f"""    def Xform "Ground"
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "GroundGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            uniform token physics:approximation = "{BOX_COLLIDER_APPROXIMATION}"
            float physxCollision:contactOffset = {self._fmt(ISAAC_CUBOID_CONTACT_OFFSET)}
            float physxCollision:restOffset = {self._fmt(ISAAC_CUBOID_REST_OFFSET)}
            float physxCollision:torsionalPatchRadius = {self._fmt(ISAAC_CUBOID_TORSIONAL_PATCH_RADIUS)}
            float physxCollision:minTorsionalPatchRadius = {self._fmt(ISAAC_CUBOID_MIN_TORSIONAL_PATCH_RADIUS)}
            double3 xformOp:translate = (0, 0, {self._fmt(ground_z)})
            double3 xformOp:scale = ({self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_THICKNESS)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial>
        }}
    }}"""

    def _ramp_usda_z_up(self) -> str:
        return f"""    def Xform "Ramp"
    {{
        def Mesh "RampGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 3, 2, 1, 0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2, 4, 5, 6, 7]
            point3f[] points = [(-1.7, -1.1, 0), (1.7, -1.1, 0), (1.7, 1.1, 0), (-1.7, 1.1, 0), (-1.7, -1.1, 0.12), (1.7, -1.1, 1.25), (1.7, 1.1, 1.25), (-1.7, 1.1, 0.12)]
            uniform token subdivisionScheme = "none"
            uniform token physics:approximation = "convexDecomposition"
            int physxConvexDecompositionCollision:maxConvexHulls = 1
            int physxConvexDecompositionCollision:hullVertexLimit = 64
            color3f[] primvars:displayColor = [(0.22, 0.24, 0.25)]
            uniform token primvars:displayColor:interpolation = "constant"
            rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial>
        }}
    }}"""

    def _box_usda_z_up(self, name: str, translate: tuple[float, float, float], scale: tuple[float, float, float]) -> str:
        tx, ty, tz = translate
        sx, sy, sz = scale
        return f"""    def Xform "{name}"
    {{
        double3 xformOp:translate = ({self._fmt(tx)}, {self._fmt(ty)}, {self._fmt(tz)})
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]

        def Cube "Geom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            uniform token physics:approximation = "{BOX_COLLIDER_APPROXIMATION}"
            float physxCollision:contactOffset = {self._fmt(ISAAC_CUBOID_CONTACT_OFFSET)}
            float physxCollision:restOffset = {self._fmt(ISAAC_CUBOID_REST_OFFSET)}
            float physxCollision:torsionalPatchRadius = {self._fmt(ISAAC_CUBOID_TORSIONAL_PATCH_RADIUS)}
            float physxCollision:minTorsionalPatchRadius = {self._fmt(ISAAC_CUBOID_MIN_TORSIONAL_PATCH_RADIUS)}
            double3 xformOp:scale = ({self._fmt(sx)}, {self._fmt(sy)}, {self._fmt(sz)})
            uniform token[] xformOpOrder = ["xformOp:scale"]
            rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial>
        }}
    }}"""

    def _base_scene_usda(self) -> str:
        if self._base_scene == "none":
            return ""
        parts = [self._ground_usda()]
        if self._base_scene == "ramp":
            parts.append(self._ramp_usda())
        elif self._base_scene == "obstacles":
            parts.extend(
                [
                    self._box_usda("ObstacleA", (-2.4, 0.35, 1.4), (1.2, 0.7, 1.2)),
                    self._box_usda("ObstacleB", (1.8, 0.6, -1.2), (1.0, 1.2, 1.6)),
                    self._box_usda("ObstacleC", (0.0, 0.25, 2.8), (2.0, 0.5, 0.6)),
                ]
            )
        return "\n".join(parts)

    def _ground_usda(self) -> str:
        ground_y = -GROUND_THICKNESS * 0.5
        return f"""    def Xform "Ground"
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "GroundGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            uniform token physics:approximation = "{BOX_COLLIDER_APPROXIMATION}"
            float physxCollision:contactOffset = {self._fmt(ISAAC_CUBOID_CONTACT_OFFSET)}
            float physxCollision:restOffset = {self._fmt(ISAAC_CUBOID_REST_OFFSET)}
            float physxCollision:torsionalPatchRadius = {self._fmt(ISAAC_CUBOID_TORSIONAL_PATCH_RADIUS)}
            float physxCollision:minTorsionalPatchRadius = {self._fmt(ISAAC_CUBOID_MIN_TORSIONAL_PATCH_RADIUS)}
            double3 xformOp:translate = (0, {self._fmt(ground_y)}, 0)
            double3 xformOp:scale = ({self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_THICKNESS)}, {self._fmt(GROUND_HALF_SIZE * 2.0)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial>
        }}
    }}"""

    def _ramp_usda(self) -> str:
        return f"""    def Xform "Ramp"
    {{
        def Mesh "RampGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 2, 3, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7, 4, 7, 6, 5]
            point3f[] points = [(-1.7, 0, -1.1), (1.7, 0, -1.1), (1.7, 0, 1.1), (-1.7, 0, 1.1), (-1.7, 0.12, -1.1), (1.7, 1.25, -1.1), (1.7, 1.25, 1.1), (-1.7, 0.12, 1.1)]
            uniform token subdivisionScheme = "none"
            uniform token physics:approximation = "convexDecomposition"
            int physxConvexDecompositionCollision:maxConvexHulls = 1
            int physxConvexDecompositionCollision:hullVertexLimit = 64
            color3f[] primvars:displayColor = [(0.22, 0.24, 0.25)]
            uniform token primvars:displayColor:interpolation = "constant"
            rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial>
        }}
    }}"""

    def _box_usda(self, name: str, translate: tuple[float, float, float], scale: tuple[float, float, float]) -> str:
        tx, ty, tz = translate
        sx, sy, sz = scale
        return f"""    def Xform "{name}"
    {{
        double3 xformOp:translate = ({self._fmt(tx)}, {self._fmt(ty)}, {self._fmt(tz)})
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]

        def Cube "Geom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            uniform token physics:approximation = "{BOX_COLLIDER_APPROXIMATION}"
            float physxCollision:contactOffset = {self._fmt(ISAAC_CUBOID_CONTACT_OFFSET)}
            float physxCollision:restOffset = {self._fmt(ISAAC_CUBOID_REST_OFFSET)}
            float physxCollision:torsionalPatchRadius = {self._fmt(ISAAC_CUBOID_TORSIONAL_PATCH_RADIUS)}
            float physxCollision:minTorsionalPatchRadius = {self._fmt(ISAAC_CUBOID_MIN_TORSIONAL_PATCH_RADIUS)}
            double3 xformOp:scale = ({self._fmt(sx)}, {self._fmt(sy)}, {self._fmt(sz)})
            uniform token[] xformOpOrder = ["xformOp:scale"]
            rel material:binding:physics = </World/PhysicsMaterials/GroundMaterial>
        }}
    }}"""

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.status_changed.emit(text)

    def _handle_unstable_pose(self, text: str) -> None:
        was_progress = self._startup_progress_active
        was_cooking = self._cooking_only
        self._set_status(text)
        self._release_scene()
        if was_cooking or was_progress:
            self.cooking_finished.emit(False, text)

    @classmethod
    def _normalize_bounds(cls, bounds: dict) -> dict:
        center = np.array(bounds.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        if center.size != 3 or not np.all(np.isfinite(center)):
            center = np.zeros(3, dtype=np.float64)

        extent = float(bounds.get("extent", 1.0))
        if not math.isfinite(extent) or extent <= 0:
            extent = 1.0

        size_value = bounds.get("size")
        try:
            size = np.array(size_value, dtype=np.float64)
        except Exception:
            size = np.array([], dtype=np.float64)
        if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0):
            side = max((2.0 * extent) / math.sqrt(3.0), 0.25)
            size = np.array([side, side, side], dtype=np.float64)

        size = np.maximum(np.abs(size), np.array([0.05, 0.05, 0.05], dtype=np.float64))
        return {"center": center.tolist(), "extent": extent, "size": size.tolist()}

    @classmethod
    def _matrix_from_pose(cls, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = cls._row_rotation_from_quat_xyzw(pose[3:7])
        matrix[3, :3] = pose[:3]
        return matrix

    @staticmethod
    def _pose_is_valid(pose) -> bool:
        try:
            arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        except Exception:
            return False
        if arr.size < 7 or not np.all(np.isfinite(arr[:7])):
            return False
        if float(np.linalg.norm(arr[:3])) > MAX_SIM_POSITION:
            return False
        quat_norm = float(np.linalg.norm(arr[3:7]))
        return math.isfinite(quat_norm) and quat_norm > 1.0e-8

    @staticmethod
    def _matrix_is_valid(matrix) -> bool:
        try:
            arr = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        except Exception:
            return False
        return np.all(np.isfinite(arr)) and float(np.linalg.norm(arr[3, :3])) <= MAX_SIM_POSITION

    @staticmethod
    def _vector_is_valid(vector, limit: float = MAX_SIM_POSITION) -> bool:
        try:
            arr = np.asarray(vector, dtype=np.float64).reshape(3)
        except Exception:
            return False
        return np.all(np.isfinite(arr)) and float(np.linalg.norm(arr)) <= float(limit)

    @staticmethod
    def _row_rotation_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in quat[:4]]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        col = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )
        return col.T

    @classmethod
    def _quat_xyzw_from_row_rotation(cls, row_rotation: np.ndarray) -> np.ndarray:
        m = np.asarray(row_rotation, dtype=np.float64).T
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([x, y, z, w], dtype=np.float64)
        quat /= max(float(np.linalg.norm(quat)), 1e-9)
        return quat

    @staticmethod
    def _z_to_y_matrix(matrix: np.ndarray) -> np.ndarray:
        return Y_TO_Z_MATRIX @ np.asarray(matrix, dtype=np.float64).reshape(4, 4) @ Z_TO_Y_MATRIX

    @staticmethod
    def _y_to_z_matrix(matrix: np.ndarray) -> np.ndarray:
        return Z_TO_Y_MATRIX @ np.asarray(matrix, dtype=np.float64).reshape(4, 4) @ Y_TO_Z_MATRIX

    @staticmethod
    def _vector_z_to_y(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float64).reshape(3) @ Z_TO_Y_ROTATION

    @staticmethod
    def _point_z_to_y(point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=np.float64).reshape(3) @ Z_TO_Y_ROTATION

    def _vector_visual_to_physics(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64).reshape(3)
        if self._physics_mode == PHYSICS_MODE_AUTHORED:
            return arr
        return arr @ Z_TO_Y_ROTATION

    def _point_visual_to_physics(self, point: np.ndarray) -> np.ndarray:
        arr = np.asarray(point, dtype=np.float64).reshape(3)
        if self._physics_mode == PHYSICS_MODE_AUTHORED:
            return arr
        return arr @ Z_TO_Y_ROTATION

    @staticmethod
    def _fmt(value: float) -> str:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        return f"{value:.9g}"
