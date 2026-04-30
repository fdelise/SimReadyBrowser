"""Qt-side OVPhysX controller.

OVPhysX 0.3.7 currently crashes if a PhysX instance is created after Qt has
initialized in this app, so simulation runs in a small worker subprocess. The
Qt process owns UI state and converts worker poses back into OVRTX transforms.
"""

from __future__ import annotations

import json
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
from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal


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
DEFAULT_DT = 1.0 / 60.0
DEFAULT_SUBSTEPS = 4
BASE_SCENES = {"plane", "ramp", "obstacles"}
ESTIMATED_ASSET_DENSITY_KG_M3 = 260.0
MIN_ESTIMATED_MASS_KG = 0.05
MAX_ESTIMATED_MASS_KG = 500.0
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
    articulation_paths: list[str]
    collider_count: int
    override_count: int


class PhysicsController(QObject):
    """Owns a worker process and streams visual transforms into the viewport."""

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
        self._pending_play = False
        self._pending_step_after_start = False
        self._step_in_flight = False
        self._cooking_only = False
        self._startup_progress_active = False

        self._bounds: Optional[dict] = None
        self._usd_source: Optional[str] = None
        self._center = np.zeros(3, dtype=np.float64)
        self._size = np.ones(3, dtype=np.float64)
        self._estimated_mass_kg = 10.0
        self._sim_time = 0.0
        self._dt = DEFAULT_DT
        self._substeps = DEFAULT_SUBSTEPS
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
        self._ccd_enabled = False
        self._scene_instance_count = 1
        self._last_start_instance_count = 1
        self._last_start_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._pending_instance_reset = False

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_scene(self) -> bool:
        return self._process is not None and self._worker_ready

    @property
    def current_visual_transform(self) -> np.ndarray:
        return np.array(self._current_visual_transform, dtype=np.float64, copy=True)

    def configure_asset(self, bounds: dict, usd_source: Optional[str] = None) -> None:
        self.shutdown()
        self._pending_magnet = None
        self._bounds = self._normalize_bounds(bounds)
        self._usd_source = str(usd_source) if usd_source else None
        self._center = np.array(self._bounds["center"], dtype=np.float64)
        self._size = np.array(self._bounds["size"], dtype=np.float64)
        self._estimated_mass_kg = self._estimate_asset_mass(self._size)
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._physics_mode = PHYSICS_MODE_AUTHORED if self._usd_source else PHYSICS_MODE_PROXY
        self._current_body_patterns = list(AUTHORED_BODY_PATTERNS)
        self._current_body_paths = []
        self._current_articulation_paths = []
        self._authored_collider_count = 0
        self._authored_override_count = 0
        self._scene_instance_count = 1
        self._last_start_instance_count = 1
        self._last_start_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._pending_instance_reset = False
        if self._usd_source:
            self._set_status("Physics ready. Colliders will cook when asset loading finishes.")
        else:
            self._set_status("Physics unavailable: no USD source to inspect authored colliders.")

    def clear_asset(self) -> None:
        self.shutdown()
        self._bounds = None
        self._usd_source = None
        self._pending_magnet = None
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._estimated_mass_kg = 10.0
        self._physics_mode = PHYSICS_MODE_PROXY
        self._current_body_patterns = list(AUTHORED_BODY_PATTERNS)
        self._current_body_paths = []
        self._current_articulation_paths = []
        self._authored_collider_count = 0
        self._authored_override_count = 0
        self._scene_instance_count = 1
        self._last_start_instance_count = 1
        self._last_start_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._pending_instance_reset = False
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

        if not self._usd_source:
            self._set_status("Physics not started: this asset has no USD source with authored colliders.")
            self.running_changed.emit(False)
            return False

        visual = (
            np.array(visual_transform, dtype=np.float64, copy=True)
            if visual_transform is not None
            else np.array(self._current_visual_transform, dtype=np.float64, copy=True)
        )
        if instance_transforms is None and self._scene_instance_count > 1:
            instance_transforms = self._last_start_instance_transforms
            if instance_transforms:
                visual = np.array(instance_transforms[0], dtype=np.float64, copy=True)
        instances = self._normalize_instance_transforms(instance_transforms, visual)
        instance_count = len(instances)
        force_rebuild = bool(force_rebuild or instance_count != self._scene_instance_count)

        if self._process is not None and self._worker_ready and not force_rebuild:
            self._sim_time = 0.0
            self._pending_step_after_start = False
            self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
            self._last_start_instance_count = instance_count
            self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in instances]
            self._last_start_play = bool(play)
            if instance_count > 1:
                self._set_instance_transforms(instances, zero_velocity=True)
                self._set_status(f"Physics reset {instance_count} asset instances using cooked colliders.")
            else:
                self.set_visual_transform(visual, zero_velocity=True)
                self._set_status("Physics reset using cooked colliders.")
            self.set_playing(bool(play))
            return True

        if self._process is not None and not self._worker_ready and not force_rebuild:
            self._pending_play = bool(play)
            self._pending_step_after_start = False
            self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
            self._last_start_instance_count = instance_count
            self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in instances]
            self._last_start_play = bool(play)
            self._pending_instance_reset = instance_count > 1
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
        self._last_start_instance_count = instance_count
        self._last_start_instance_transforms = [np.array(item, dtype=np.float64, copy=True) for item in instances]
        self._last_start_play = bool(play)
        self._pending_instance_reset = instance_count > 1
        self._authored_scene_instance_transforms = (
            [np.eye(4, dtype=np.float64)]
            if instance_count == 1
            else [np.array(item, dtype=np.float64, copy=True) for item in instances]
        )
        initial_pose = None
        if instance_count == 1:
            body_matrix = self._body_from_visual(visual)
            initial_pose = self._pose_from_body(body_matrix)

        scene_path = self._write_authored_scene()
        return self._start_worker(scene_path, self._current_body_patterns, initial_pose)

    def cook_colliders(self) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before cooking physics colliders.")
            self.cooking_finished.emit(False, self._status_text)
            return False

        if not self._usd_source:
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

        self._sim_time = 0.0
        self._pending_play = False
        self._pending_step_after_start = False
        self._cooking_only = False
        self._startup_progress_active = True
        self._physics_mode = PHYSICS_MODE_AUTHORED
        self.cooking_progress.emit(0, "Cooking physics colliders after asset load...")
        self._set_status("Cooking authored physics colliders after asset load...")

        visual = np.array(self._current_visual_transform, dtype=np.float64, copy=True)
        self._last_start_visual_transform = np.array(visual, dtype=np.float64, copy=True)
        self._last_start_instance_count = 1
        self._last_start_instance_transforms = [np.array(visual, dtype=np.float64, copy=True)]
        self._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        self._last_start_play = False
        body_matrix = self._body_from_visual(visual)
        initial_pose = self._pose_from_body(body_matrix)
        scene_path = self._write_authored_scene()
        return self._start_worker(scene_path, self._current_body_patterns, initial_pose, cook_only=False)

    def set_playing(self, playing: bool) -> None:
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
        drop_count = self._sanitize_drop_count(count)
        visuals = self._drop_visual_transforms(drop_count)
        force_rebuild = drop_count != self._scene_instance_count
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
        mode = "enabled" if enabled else "disabled"
        if self._bounds is None:
            self._set_status(f"CCD {mode}. Load an asset to apply it.")
            return

        if self._process is not None:
            self._send({"cmd": "set_ccd", "enabled": enabled})
            if self._worker_ready:
                self._set_status(f"CCD {mode}; current cooked collider cache was left intact.")
            else:
                self._set_status(
                    f"CCD {mode}; current collider cook will finish without restarting, then the setting applies next start."
                )
        else:
            self._set_status(f"CCD {mode}. It will apply when physics starts.")

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
                "initial_pose": None if initial_pose is None else initial_pose.astype(float).tolist(),
                "contact_offset": self._contact_offset(),
                "cook_only": bool(cook_only),
                "ccd_enabled": bool(self._ccd_enabled),
            }
        )
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
            authored_text = (
                f"{self._authored_collider_count} authored collider prims traversed, "
                if self._authored_collider_count > 0
                else ""
            )
            text = (
                f"Physics colliders cooked ({authored_text}{body_count} {body_suffix}, "
                f"{shape_text}, {pattern}{ccd_text})."
            )
            warning = str(message.get("cook_warning", "") or "")
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
                        f"Traversal found {self._authored_collider_count} authored collider prims, but "
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
                prefix = "Physics reset with authored colliders"
                if not self._pending_play and not self._pending_step_after_start:
                    prefix = "Physics colliders cooked and ready"
                instance_text = ""
                if self._scene_instance_count > 1:
                    instance_text = f", {self._scene_instance_count} asset instances"
                text = (
                    f"{prefix} ({self._authored_collider_count} authored prims, "
                    f"{body_count} {suffix}{shape_text}, "
                    f"{pattern}{instance_text}, {self._substeps}x substeps{ccd_text})."
                )
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
        if self._timer.isActive():
            self._timer.stop()
        self._pending_magnet = None
        self._pending_play = False
        self._pending_step_after_start = False
        self._step_in_flight = False
        self._cooking_only = False
        self._startup_progress_active = False
        self._scene_instance_count = 1
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

    def _write_authored_scene(self) -> Path:
        if not self._usd_source:
            raise RuntimeError("No USD source is configured for authored collider physics")

        temp_dir = Path(tempfile.gettempdir()) / "simready_browser_physx"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._scene_path = temp_dir / "authored_asset_scene.usda"

        asset_ref = self._usd_asset_reference(self._usd_source)
        base_scene = self._base_scene_usda_z_up()
        discovery = self._authored_collider_discovery(asset_ref)
        instance_transforms = self._normalize_instance_transforms(
            self._authored_scene_instance_transforms,
            np.eye(4, dtype=np.float64),
        )
        instance_count = len(instance_transforms)
        self._scene_instance_count = instance_count
        self._current_body_patterns = self._expand_authored_paths(
            discovery.body_patterns or list(AUTHORED_BODY_PATTERNS),
            instance_count,
        )
        self._current_body_paths = self._expand_authored_paths(discovery.body_paths, instance_count)
        if not self._current_body_paths and instance_count > 1:
            self._current_body_paths = [self._authored_asset_path(index) for index in range(instance_count)]
        self._current_articulation_paths = self._expand_authored_paths(discovery.articulation_paths, instance_count)
        self._authored_collider_count = int(discovery.collider_count)
        self._authored_override_count = int(discovery.override_count)
        collision_overrides = discovery.collision_overrides
        scene_ccd = "1" if self._ccd_enabled else "0"
        asset_blocks = "\n\n".join(
            self._authored_asset_usda_block(index, asset_ref, collision_overrides, instance_transforms[index])
            for index in range(instance_count)
        )
        if self._authored_collider_count > 0:
            self._set_status(
                f"Traversed {self._authored_collider_count} authored collider prims; starting OVPhysX cook..."
            )
        else:
            self._set_status(
                "No authored collider prims were found in the SimReady payload traversal; OVPhysX will inspect the stage."
            )
        proxy_mass = self._fmt(self._estimated_mass_kg)

        text = f"""#usda 1.0
(
    defaultPrim = "World"
    doc = "SimReady Browser OVPhysX authored-collider scene"
    metersPerUnit = 1
    upAxis = "Z"
    kilogramsPerMass = 1
)

def Xform "World" (
    kind = "component"
)
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI"]
    )
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
        uniform token physxScene:solverType = "TGS"
        bool physxScene:enableStabilization = 1
        bool physxScene:enableExternalForcesEveryIteration = 1
        bool physxScene:solveArticulationContactLast = 1
        uniform uint physxScene:minPositionIterationCount = 8
        uniform uint physxScene:minVelocityIterationCount = 2
        bool physxScene:enableCCD = {scene_ccd}
    }}

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
    metersPerUnit = 1
    upAxis = "Y"
    kilogramsPerMass = 1
)

def Xform "World" (
    kind = "component"
)
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI"]
    )
    {{
        vector3f physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
        uniform token physxScene:solverType = "TGS"
        bool physxScene:enableStabilization = 1
        bool physxScene:enableExternalForcesEveryIteration = 1
        bool physxScene:solveArticulationContactLast = 1
        uniform uint physxScene:minPositionIterationCount = 8
        uniform uint physxScene:minVelocityIterationCount = 2
        bool physxScene:enableCCD = {"1" if self._ccd_enabled else "0"}
    }}

    def Xform "AssetProxy" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
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

        def Cube "CubeGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:scale = ({self._fmt(size_y[0])}, {self._fmt(size_y[1])}, {self._fmt(size_y[2])})
            uniform token[] xformOpOrder = ["xformOp:scale"]
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
    ) -> str:
        name = self._authored_asset_name(index)
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        quat = self._quat_xyzw_from_row_rotation(matrix[:3, :3])
        xform_lines = [
            f"        double3 xformOp:translate = ({self._fmt(matrix[3, 0])}, {self._fmt(matrix[3, 1])}, {self._fmt(matrix[3, 2])})",
            f"        quatd xformOp:orient = ({self._fmt(quat[3])}, {self._fmt(quat[0])}, {self._fmt(quat[1])}, {self._fmt(quat[2])})",
            "        double3 xformOp:scale = (1, 1, 1)",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
        ]
        overrides = f"\n{collision_overrides}" if collision_overrides else ""
        return f"""    def Xform "{name}" (
        prepend references = @{asset_ref}@
    )
    {{
{chr(10).join(xform_lines)}
{overrides}
    }}"""

    def _drop_visual_transforms(self, count: int) -> list[np.ndarray]:
        drop_count = self._sanitize_drop_count(count)
        first = self._drop_visual_transform()
        if drop_count <= 1:
            return [first]

        rng = random.Random()
        transforms: list[np.ndarray] = []
        size = np.maximum(np.asarray(self._size, dtype=np.float64).reshape(3), np.array([0.05, 0.05, 0.05]))
        clearance = max(0.03, min(0.25, float(np.max(size)) * 0.08))
        horizontal_diagonal = max(float(np.linalg.norm(size[:2])), 0.1)
        column_spacing = horizontal_diagonal + clearance * 2.0
        vertical_gap = max(float(size[2]) + clearance, 0.12)
        column_count = 1 if drop_count <= 6 else min(drop_count, max(2, math.ceil(math.sqrt(drop_count))))
        column_offsets = self._drop_column_offsets(column_count, column_spacing)
        base_center = self._center @ first[:3, :3] + first[3, :3]
        placed_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(drop_count):
            column = index % column_count
            layer = index // column_count
            jitter_radius = clearance * rng.uniform(0.0, 0.25)
            jitter_angle = rng.uniform(0.0, math.tau)
            offset = column_offsets[column] + np.array(
                [math.cos(jitter_angle) * jitter_radius, math.sin(jitter_angle) * jitter_radius],
                dtype=np.float64,
            )
            yaw = rng.uniform(-0.45, 0.45)
            rotation = first[:3, :3] @ self._z_up_yaw_rotation(yaw)
            target_center = np.array(base_center, dtype=np.float64, copy=True)
            target_center[0] += offset[0]
            target_center[1] += offset[1]
            target_center[2] += vertical_gap * layer
            visual = self._visual_transform_for_center(rotation, target_center)
            bounds = self._drop_aabb(visual)
            if bounds[0][2] < GROUND_TOP_Z + clearance:
                target_center[2] += GROUND_TOP_Z + clearance - bounds[0][2]
                visual = self._visual_transform_for_center(rotation, target_center)
                bounds = self._drop_aabb(visual)

            guard = 0
            while any(self._aabb_intersects(bounds, previous, clearance) for previous in placed_bounds) and guard < drop_count + 4:
                target_center[2] += vertical_gap
                visual = self._visual_transform_for_center(rotation, target_center)
                bounds = self._drop_aabb(visual)
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

    def _drop_aabb(self, visual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(visual, dtype=np.float64).reshape(4, 4)
        half = np.maximum(np.asarray(self._size, dtype=np.float64).reshape(3), np.array([0.05, 0.05, 0.05])) * 0.5
        center = self._center @ matrix[:3, :3] + matrix[3, :3]
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

    def _drop_visual_transform(self) -> np.ndarray:
        visual = np.array(self._current_visual_transform, dtype=np.float64, copy=True).reshape(4, 4)
        if not self._matrix_is_valid(visual):
            visual = np.eye(4, dtype=np.float64)

        half_z = max(float(self._size[2]) * 0.5, 0.05)
        extent = max(float(np.linalg.norm(self._size)) * 0.5, half_z)
        if self._bounds:
            try:
                extent = max(extent, float(self._bounds.get("extent", extent)))
            except Exception:
                pass

        drop_height = max(
            float(self._size[2]) * DROP_HEIGHT_SIZE_SCALE,
            extent * DROP_HEIGHT_EXTENT_SCALE,
            DROP_HEIGHT_MIN,
        )
        drop_height = min(drop_height, DROP_HEIGHT_MAX)

        target_center = np.array(self._center, dtype=np.float64, copy=True)
        target_center[2] = max(target_center[2], GROUND_TOP_Z + half_z + drop_height)
        visual[3, :3] = target_center - self._center @ visual[:3, :3]
        return visual

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

    @staticmethod
    def _authored_asset_name(index: int) -> str:
        return AUTHORED_ASSET_NAME if index <= 0 else f"{AUTHORED_ASSET_NAME}_{index + 1:02d}"

    @classmethod
    def _authored_asset_path(cls, index: int) -> str:
        return f"/World/{cls._authored_asset_name(index)}"

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
                if mapped not in expanded:
                    expanded.append(mapped)
        return expanded

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
        stage_discovery = self._stage_collider_discovery(asset_ref)
        if stage_discovery is not None:
            return stage_discovery
        return self._payload_collider_discovery(asset_ref)

    def _stage_collider_discovery(self, asset_ref: str) -> Optional[AuthoredColliderDiscovery]:
        helper_python = self._usd_discovery_python()
        if not helper_python:
            return None

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
            return None

        if result.returncode != 0:
            return None

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
        if not isinstance(payload, dict):
            return None

        override_paths = [str(path) for path in payload.get("override_paths", [])]
        body_patterns = [str(path) for path in payload.get("body_patterns", []) if str(path or "").strip()]
        collider_count = int(payload.get("collider_count", 0) or 0)
        override_count = int(payload.get("override_count", 0) or 0)
        if collider_count <= 0:
            return AuthoredColliderDiscovery("", body_patterns or list(AUTHORED_BODY_PATTERNS), [], [], 0, 0)
        return AuthoredColliderDiscovery(
            self._format_absolute_collision_overrides(override_paths),
            body_patterns or list(AUTHORED_BODY_PATTERNS),
            [str(path) for path in payload.get("body_paths", []) if str(path or "").strip()],
            [str(path) for path in payload.get("articulation_paths", []) if str(path or "").strip()],
            collider_count,
            override_count,
        )

    @staticmethod
    def _usd_discovery_python() -> Optional[str]:
        root = Path(__file__).resolve().parents[1]
        candidates = [
            os.environ.get(USD_DISCOVERY_PYTHON_ENV, ""),
            str(root / ".usd_discovery_venv" / "Scripts" / "python.exe"),
            str(root / ".usd_discovery_venv" / "bin" / "python"),
        ]
        current = Path(sys.executable).resolve()
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
            if resolved == current and not os.environ.get(USD_DISCOVERY_PYTHON_ENV):
                continue
            return str(resolved)
        return None

    def _payload_collider_discovery(self, asset_ref: str) -> AuthoredColliderDiscovery:
        base_text = self._read_simready_payload_text(asset_ref, "base.usda")
        instances_text = self._read_simready_payload_text(asset_ref, "instances.usda")
        if not base_text or not instances_text:
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], 0, 0)

        collision_instances = self._collision_instances(instances_text)
        if not collision_instances:
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], 0, 0)

        references = self._simready_instance_references(base_text)
        collision_refs = [(path, name) for path, name in references if name in collision_instances]
        if not collision_refs:
            return AuthoredColliderDiscovery("", list(AUTHORED_BODY_PATTERNS), [], [], 0, 0)

        override_refs = [(path, name) for path, name in collision_refs if collision_instances.get(name) == "sdf"]
        collision_overrides = self._format_collision_overrides(override_refs)
        body_patterns = self._authored_body_patterns_from_refs(collision_refs)
        body_paths = self._authored_body_paths_from_refs(collision_refs)
        return AuthoredColliderDiscovery(
            collision_overrides,
            body_patterns,
            body_paths,
            [],
            len(collision_refs),
            len(override_refs),
        )

    def _authored_collision_overrides(self, asset_ref: str) -> str:
        return self._authored_collider_discovery(asset_ref).collision_overrides

    def _format_absolute_collision_overrides(self, paths: list[str]) -> str:
        root: dict = {}
        for path in paths:
            text = str(path or "").strip()
            if not text.startswith(f"{AUTHORED_ASSET_PATH}/"):
                continue
            rel_parts = [self._usd_name(part) for part in text[len(AUTHORED_ASSET_PATH) + 1 :].split("/") if part]
            if not rel_parts:
                continue
            node = root
            for part in rel_parts:
                node = node.setdefault(part, {})
            node["__collider__"] = True

        if not root:
            return ""

        lines: list[str] = []

        def emit_node(name: str, node: dict, indent: int) -> None:
            pad = " " * indent
            is_collider = bool(node.get("__collider__"))
            children = [(child, child_node) for child, child_node in node.items() if child != "__collider__"]
            if is_collider:
                lines.extend(
                    [
                        f'{pad}over "{name}" (',
                        ' ' * (indent + 4)
                        + 'prepend apiSchemas = ["PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]',
                        f"{pad})",
                        f"{pad}{{",
                        ' ' * (indent + 4) + 'uniform token physics:approximation = "convexDecomposition"',
                        ' ' * (indent + 4) + "int physxConvexDecompositionCollision:maxConvexHulls = 32",
                        ' ' * (indent + 4) + "int physxConvexDecompositionCollision:hullVertexLimit = 64",
                    ]
                )
            else:
                lines.extend([f'{pad}over "{name}"', f"{pad}{{"])
            for child, child_node in children:
                emit_node(child, child_node, indent + 4)
            lines.append(f"{pad}}}")

        for child, child_node in root.items():
            emit_node(child, child_node, 8)
        return "\n".join(lines)

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
            full_parts = [AUTHORED_ASSET_PATH, *clean_parts]
            if clean_parts[0] == "Geometry" and len(clean_parts) > 1:
                object_path = "/".join(full_parts[:4])
                if len(clean_parts) == 2 and instance_name:
                    mesh_path = f"{object_path}/{instance_name}"
                else:
                    mesh_path = "/".join(full_parts)
            else:
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
            full_parts = [AUTHORED_ASSET_PATH, *clean_parts]
            if clean_parts[0] == "Geometry" and len(clean_parts) > 1:
                add("/".join(full_parts[:4]))
            else:
                add("/".join(full_parts[:2]))
        return paths

    def _format_collision_overrides(self, refs: list[tuple[tuple[str, ...], str]]) -> str:
        overrides: list[tuple[str, str, str]] = []
        for rel_path, instance_name in refs:
            clean_parts = [self._usd_name(part) for part in rel_path if part]
            instance_name = self._usd_name(instance_name)
            if len(clean_parts) >= 3 and clean_parts[0] == "Geometry":
                object_name = clean_parts[1]
                mesh_name = clean_parts[-1]
            elif len(clean_parts) == 2 and clean_parts[0] == "Geometry":
                object_name = clean_parts[1]
                mesh_name = instance_name
            elif len(clean_parts) >= 2:
                object_name = clean_parts[0]
                mesh_name = clean_parts[-1]
            elif len(clean_parts) == 1:
                object_name = clean_parts[0]
                mesh_name = instance_name
            else:
                continue
            item = (object_name, mesh_name, instance_name)
            if item not in overrides:
                overrides.append(item)

        if not overrides:
            return ""

        lines = ['        over "Geometry"', "        {"]
        for object_name, mesh_name, instance_name in overrides:
            lines.extend(
                [
                    f'            over "{self._usd_name(object_name)}"',
                    "            {",
                    f'                over "{self._usd_name(mesh_name)}" (',
                    '                    prepend apiSchemas = ["PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]',
                    "                )",
                    "                {",
                    '                    uniform token physics:approximation = "convexDecomposition"',
                    "                    int physxConvexDecompositionCollision:maxConvexHulls = 32",
                    "                    int physxConvexDecompositionCollision:hullVertexLimit = 64",
                ]
            )
            if instance_name:
                lines.extend(
                    [
                        "",
                        f'                    over "{self._usd_name(instance_name)}" (',
                        '                        prepend apiSchemas = ["PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]',
                        "                    )",
                        "                    {",
                        '                        uniform token physics:approximation = "convexDecomposition"',
                        "                        int physxConvexDecompositionCollision:maxConvexHulls = 32",
                        "                        int physxConvexDecompositionCollision:hullVertexLimit = 64",
                        "                    }",
                    ]
                )
            lines.extend(["                }", "            }"])
        lines.append("        }")
        return "\n".join(lines)

    @staticmethod
    def _sdf_collision_instances(instances_text: str) -> set[str]:
        return {name for name, approximation in PhysicsController._collision_instances(instances_text).items() if approximation == "sdf"}

    @staticmethod
    def _collision_instances(instances_text: str) -> dict[str, str]:
        instances: dict[str, str] = {}
        pattern = re.compile(
            r'\bover\s+"([^"]+)"\s*(?P<meta>\([^{}]*?\))?\s*\{',
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
            instances[name] = approximation
        return instances

    @staticmethod
    def _simready_instance_references(base_text: str) -> list[tuple[tuple[str, ...], str]]:
        refs: list[tuple[tuple[str, ...], str]] = []
        stack: list[tuple[int, str]] = []
        prim_pattern = re.compile(r'\b(?:def|over|class)\s+\w+\s+"([^"]+)"')
        ref_pattern = re.compile(r'references\s*=\s*@instances\.usda@</Instances/([^>]+)>')

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

    @staticmethod
    def _read_simready_payload_text(asset_ref: str, name: str) -> str:
        if not asset_ref:
            return ""

        url = ""
        if asset_ref.startswith("http://") or asset_ref.startswith("https://"):
            base = asset_ref.rsplit("/", 1)[0]
            url = f"{base}/payloads/{name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SimReadyBrowser/1.0"})
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    return response.read().decode("utf-8", "replace")
            except Exception:
                return ""

        if "://" in asset_ref:
            return ""

        try:
            path = Path(asset_ref)
            payload = path.parent / "payloads" / name
            if payload.exists():
                return payload.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        return ""

    @staticmethod
    def _usd_name(value: str) -> str:
        return str(value or "").replace("\\", "_").replace('"', "_")

    def _base_scene_usda_z_up(self) -> str:
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

    def _contact_offset(self) -> float:
        min_size = float(np.min(np.maximum(self._size, 0.001)))
        return max(0.003, min(0.04, min_size * 0.025))

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

    def _ground_usda_z_up(self) -> str:
        ground_z = -GROUND_THICKNESS * 0.5
        return f"""    def Xform "Ground"
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "GroundGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:translate = (0, 0, {self._fmt(ground_z)})
            double3 xformOp:scale = ({self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_THICKNESS)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}"""

    def _ramp_usda_z_up(self) -> str:
        return f"""    def Xform "Ramp"
    {{
        def Mesh "RampGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]
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
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:scale = ({self._fmt(sx)}, {self._fmt(sy)}, {self._fmt(sz)})
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}"""

    def _base_scene_usda(self) -> str:
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
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:translate = (0, {self._fmt(ground_y)}, 0)
            double3 xformOp:scale = ({self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_THICKNESS)}, {self._fmt(GROUND_HALF_SIZE * 2.0)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}"""

    def _ramp_usda(self) -> str:
        return f"""    def Xform "Ramp"
    {{
        def Mesh "RampGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]
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
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:scale = ({self._fmt(sx)}, {self._fmt(sy)}, {self._fmt(sz)})
            uniform token[] xformOpOrder = ["xformOp:scale"]
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
