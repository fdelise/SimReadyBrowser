"""Optional OVPhysX simulation bridge for the SimReady viewport.

OVPhysX simulates a lightweight box proxy for the loaded asset, then emits a
USD row-vector transform that OVRTX can apply to the visual asset root.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal


PhysX = None  # type: ignore
TensorType = None  # type: ignore
OVPHYSX_AVAILABLE: Optional[bool] = None
OVPHYSX_IMPORT_ERROR: Optional[BaseException] = None

PROXY_PATH = "/World/AssetProxy"
GROUND_HALF_SIZE = 20.0
GROUND_THICKNESS = 0.05
GROUND_TOP_Z = 0.0
DEFAULT_DT = 1.0 / 60.0


def _ensure_ovphysx_available() -> bool:
    global PhysX, TensorType, OVPHYSX_AVAILABLE, OVPHYSX_IMPORT_ERROR

    if OVPHYSX_AVAILABLE is not None:
        return OVPHYSX_AVAILABLE

    try:
        from ovphysx import PhysX as PhysXClass  # type: ignore

        try:
            from ovphysx.types import TensorType as TensorTypeClass  # type: ignore
        except Exception:
            from ovphysx import TensorType as TensorTypeClass  # type: ignore

        PhysX = PhysXClass
        TensorType = TensorTypeClass
        OVPHYSX_AVAILABLE = True
    except Exception as exc:
        OVPHYSX_IMPORT_ERROR = exc
        OVPHYSX_AVAILABLE = False

    return OVPHYSX_AVAILABLE


class PhysicsController(QObject):
    """Owns a small OVPhysX scene and streams visual transforms."""

    pose_changed = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)

        self._physx = None
        self._usd_handle = None
        self._pose_binding = None
        self._velocity_binding = None
        self._pose_buffer: Optional[np.ndarray] = None
        self._velocity_buffer: Optional[np.ndarray] = None

        self._bounds: Optional[dict] = None
        self._center = np.zeros(3, dtype=np.float64)
        self._size = np.ones(3, dtype=np.float64)
        self._sim_time = 0.0
        self._dt = DEFAULT_DT
        self._running = False
        self._status_text = "Load an asset, then use Play or Restart physics."
        self._scene_path: Optional[Path] = None
        self._current_visual_transform = np.eye(4, dtype=np.float64)

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_scene(self) -> bool:
        return self._physx is not None

    @property
    def current_visual_transform(self) -> np.ndarray:
        return np.array(self._current_visual_transform, dtype=np.float64, copy=True)

    def configure_asset(self, bounds: dict) -> None:
        self.shutdown()
        self._bounds = self._normalize_bounds(bounds)
        self._center = np.array(self._bounds["center"], dtype=np.float64)
        self._size = np.array(self._bounds["size"], dtype=np.float64)
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._set_status("Physics ready. Play drops the asset proxy onto the ground.")

    def clear_asset(self) -> None:
        self.shutdown()
        self._bounds = None
        self._current_visual_transform = np.eye(4, dtype=np.float64)
        self._set_status("Load an asset, then use Play or Restart physics.")

    def restart(self, visual_transform: Optional[np.ndarray] = None, play: bool = True) -> bool:
        if self._bounds is None:
            self._set_status("Load an asset before starting physics.")
            self.running_changed.emit(False)
            return False

        if not _ensure_ovphysx_available():
            detail = f": {OVPHYSX_IMPORT_ERROR}" if OVPHYSX_IMPORT_ERROR else ""
            self._set_status(f"OVPhysX is not installed{detail}. Run launch.bat to install it.")
            self.running_changed.emit(False)
            return False

        self._release_scene()
        self._sim_time = 0.0

        visual = (
            np.array(visual_transform, dtype=np.float64, copy=True)
            if visual_transform is not None
            else self._default_drop_visual_transform()
        )
        body_matrix = self._body_from_visual(visual)
        scene_path = self._write_proxy_scene(body_matrix)

        try:
            self._set_status("Starting OVPhysX proxy simulation...")
            self._physx = PhysX(device="cpu")
            result = self._physx.add_usd(str(scene_path))
            self._usd_handle = result[0] if isinstance(result, tuple) else result
            self._physx.wait_all()
            self._pose_binding = self._physx.create_tensor_binding(
                prim_paths=[PROXY_PATH],
                tensor_type=TensorType.RIGID_BODY_POSE,
            )
            self._pose_buffer = np.zeros(self._pose_binding.shape, dtype=np.float32)
            try:
                self._velocity_binding = self._physx.create_tensor_binding(
                    prim_paths=[PROXY_PATH],
                    tensor_type=TensorType.RIGID_BODY_VELOCITY,
                )
                self._velocity_buffer = np.zeros(self._velocity_binding.shape, dtype=np.float32)
            except Exception:
                self._velocity_binding = None
                self._velocity_buffer = None
            self._read_emit_pose()
            self.set_playing(play)
            if not play:
                self._set_status("Physics reset and paused.")
            return True
        except Exception as exc:
            self._release_scene()
            self._set_status(f"OVPhysX startup failed: {exc}")
            self.running_changed.emit(False)
            return False

    def set_playing(self, playing: bool) -> None:
        if playing and self._physx is None:
            self.restart(play=True)
            return

        if playing:
            if not self._timer.isActive():
                self._timer.start(int(self._dt * 1000))
            self._running = True
            self.running_changed.emit(True)
            self._set_status("Physics playing.")
            return

        if self._timer.isActive():
            self._timer.stop()
        if self._running:
            self._running = False
            self.running_changed.emit(False)
        self._set_status("Physics paused.")

    def step_once(self) -> None:
        if self._physx is None:
            if not self.restart(play=False):
                return
        self.set_playing(False)
        self._step()

    def set_visual_transform(self, matrix: np.ndarray, zero_velocity: bool = True) -> None:
        visual = np.array(matrix, dtype=np.float64, copy=True)
        self._current_visual_transform = visual
        if self._physx is not None and self._pose_binding is not None and self._pose_buffer is not None:
            pose = self._pose_from_visual(visual)
            self._pose_buffer[0, :] = pose
            self._pose_binding.write(self._pose_buffer)
            if zero_velocity and self._velocity_binding is not None and self._velocity_buffer is not None:
                self._velocity_buffer.fill(0.0)
                self._velocity_binding.write(self._velocity_buffer)
        self.pose_changed.emit(np.array(visual, dtype=np.float64, copy=True))

    def shutdown(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._running:
            self._running = False
            self.running_changed.emit(False)
        self._release_scene()

    def _step(self) -> None:
        if self._physx is None or self._pose_binding is None or self._pose_buffer is None:
            return

        try:
            step_sync = getattr(self._physx, "step_sync", None)
            if callable(step_sync):
                try:
                    step_sync(self._dt, self._sim_time)
                except TypeError:
                    step_sync(self._dt)
            else:
                self._physx.step(self._dt, self._sim_time)
                self._physx.wait_all()
            self._sim_time += self._dt
            self._read_emit_pose()
        except Exception as exc:
            self.set_playing(False)
            self._set_status(f"Physics step failed: {exc}")

    def _read_emit_pose(self) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return

        self._pose_binding.read(self._pose_buffer)
        body_matrix = self._matrix_from_pose(self._pose_buffer[0])
        visual_matrix = self._visual_from_body(body_matrix)
        self._current_visual_transform = visual_matrix
        self.pose_changed.emit(np.array(visual_matrix, dtype=np.float64, copy=True))

    def _release_scene(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

        for binding_name in ("_pose_binding", "_velocity_binding"):
            binding = getattr(self, binding_name)
            if binding is not None:
                try:
                    binding.destroy()
                except Exception:
                    pass
                setattr(self, binding_name, None)

        self._pose_buffer = None
        self._velocity_buffer = None

        if self._physx is not None:
            try:
                if self._usd_handle is not None:
                    self._physx.remove_usd(self._usd_handle)
                    self._physx.wait_all()
            except Exception:
                pass
            try:
                self._physx.release()
            except Exception:
                pass

        self._physx = None
        self._usd_handle = None
        if self._running:
            self._running = False
            self.running_changed.emit(False)

    def _default_drop_visual_transform(self) -> np.ndarray:
        body = np.eye(4, dtype=np.float64)
        half_z = max(float(self._size[2]) * 0.5, 0.05)
        drop_height = max(float(self._size[2]) * 1.5, 0.75)
        body[3, :3] = self._center
        body[3, 2] = max(body[3, 2], GROUND_TOP_Z + half_z + drop_height)
        return self._visual_from_body(body)

    def _body_from_visual(self, visual_matrix: np.ndarray) -> np.ndarray:
        visual = np.array(visual_matrix, dtype=np.float64, copy=True)
        body = np.eye(4, dtype=np.float64)
        body[:3, :3] = visual[:3, :3]
        body[3, :3] = self._center @ visual[:3, :3] + visual[3, :3]
        return body

    def _visual_from_body(self, body_matrix: np.ndarray) -> np.ndarray:
        body = np.array(body_matrix, dtype=np.float64, copy=True)
        visual = np.eye(4, dtype=np.float64)
        visual[:3, :3] = body[:3, :3]
        visual[3, :3] = body[3, :3] - self._center @ body[:3, :3]
        return visual

    def _pose_from_visual(self, visual_matrix: np.ndarray) -> np.ndarray:
        body = self._body_from_visual(visual_matrix)
        quat = self._quat_xyzw_from_row_rotation(body[:3, :3])
        pose = np.zeros(7, dtype=np.float32)
        pose[:3] = body[3, :3].astype(np.float32)
        pose[3:] = quat.astype(np.float32)
        return pose

    def _write_proxy_scene(self, body_matrix: np.ndarray) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "simready_browser_physx"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._scene_path = temp_dir / "asset_proxy.usda"

        size = np.maximum(self._size, np.array([0.05, 0.05, 0.05], dtype=np.float64))
        collider_scale = size
        ground_z = GROUND_TOP_Z - GROUND_THICKNESS * 0.5
        body_transform = self._usd_matrix(body_matrix)

        text = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "physicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "AssetProxy" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 10
        matrix4d xformOp:transform = {body_transform}
        uniform token[] xformOpOrder = ["xformOp:transform"]

        def Cube "Collider" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            uniform token purpose = "guide"
            double size = 1
            double3 xformOp:scale = ({self._fmt(collider_scale[0])}, {self._fmt(collider_scale[1])}, {self._fmt(collider_scale[2])})
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}

    def Cube "Ground" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token purpose = "guide"
        double size = 1
        double3 xformOp:translate = (0, 0, {self._fmt(ground_z)})
        double3 xformOp:scale = ({self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_HALF_SIZE * 2.0)}, {self._fmt(GROUND_THICKNESS)})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
"""
        self._scene_path.write_text(text, encoding="utf-8")
        return self._scene_path

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.status_changed.emit(text)

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

    @classmethod
    def _usd_matrix(cls, matrix: np.ndarray) -> str:
        m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        rows = []
        for row in m:
            rows.append("(" + ", ".join(cls._fmt(v) for v in row) + ")")
        return "( " + ", ".join(rows) + " )"

    @staticmethod
    def _fmt(value: float) -> str:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        return f"{value:.9g}"
