"""Qt-side OVPhysX controller.

OVPhysX 0.3.7 currently crashes if a PhysX instance is created after Qt has
initialized in this app, so simulation runs in a small worker subprocess. The
Qt process owns UI state and converts worker poses back into OVRTX transforms.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal


PROXY_PATH = "/World/AssetProxy"
GROUND_HALF_SIZE = 20.0
GROUND_THICKNESS = 0.05
GROUND_TOP_Z = 0.0
DEFAULT_DT = 1.0 / 60.0

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


class PhysicsController(QObject):
    """Owns a worker process and streams visual transforms into the viewport."""

    pose_changed = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

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
        return self._process is not None and self._worker_ready

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

        self._release_scene()
        self._sim_time = 0.0
        self._pending_play = bool(play)
        self._pending_step_after_start = False

        visual = (
            np.array(visual_transform, dtype=np.float64, copy=True)
            if visual_transform is not None
            else self._default_drop_visual_transform()
        )
        body_matrix = self._body_from_visual(visual)
        scene_path = self._write_proxy_scene(body_matrix)
        return self._start_worker(scene_path)

    def set_playing(self, playing: bool) -> None:
        if playing and self._process is None:
            self.restart(play=True)
            return

        if playing and not self._worker_ready:
            self._pending_play = True
            self._set_status("Physics worker starting...")
            return

        if playing:
            if not self._timer.isActive():
                self._timer.start(max(1, int(self._dt * 1000)))
            if not self._running:
                self._running = True
                self.running_changed.emit(True)
            self._set_status("Physics playing.")
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

    def set_visual_transform(self, matrix: np.ndarray, zero_velocity: bool = True) -> None:
        visual = np.array(matrix, dtype=np.float64, copy=True)
        self._current_visual_transform = visual
        if self._worker_ready:
            pose = self._pose_from_visual(visual)
            self._send({"cmd": "set_pose", "pose": pose.tolist(), "zero_velocity": bool(zero_velocity)})
        self.pose_changed.emit(np.array(visual, dtype=np.float64, copy=True))

    def shutdown(self) -> None:
        self._release_scene()

    def _start_worker(self, scene_path: Path) -> bool:
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

        self._process.start(sys.executable, ["-u", "-m", "core.physics_worker"])
        if not self._process.waitForStarted(5000):
            error = self._process.errorString() if self._process else "unknown error"
            self._release_scene()
            self._set_status(f"Could not start OVPhysX worker: {error}")
            self.running_changed.emit(False)
            return False

        self._send({"cmd": "start", "scene": str(scene_path)})
        return True

    def _send_step(self) -> None:
        if not self._worker_ready or self._step_in_flight:
            return
        self._step_in_flight = True
        self._send({"cmd": "step", "dt": self._dt, "time": self._sim_time})
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
        if kind == "started":
            self._worker_ready = True
            self._set_status("Physics reset and ready.")
            if self._pending_play:
                self.set_playing(True)
            if self._pending_step_after_start:
                self._pending_step_after_start = False
                self._send_step()
            return

        if kind == "pose":
            pose = np.array(message.get("pose", []), dtype=np.float64)
            if pose.size >= 7:
                body_matrix = self._matrix_from_pose(pose[:7])
                visual_matrix = self._visual_from_body(body_matrix)
                self._current_visual_transform = visual_matrix
                self._step_in_flight = False
                self.pose_changed.emit(np.array(visual_matrix, dtype=np.float64, copy=True))
            return

        if kind == "error":
            self._step_in_flight = False
            self._pending_play = False
            if self._timer.isActive():
                self._timer.stop()
            if self._running:
                self._running = False
                self.running_changed.emit(False)
            self._set_status(f"OVPhysX failed: {message.get('message', 'unknown error')}")
            return

        if kind == "stopped":
            self._worker_ready = False

    def _on_worker_finished(self, _exit_code: int, _exit_status) -> None:
        was_intentional = self._intentional_stop
        self._process = None
        self._worker_ready = False
        self._step_in_flight = False
        if self._timer.isActive():
            self._timer.stop()
        if self._running:
            self._running = False
            self.running_changed.emit(False)
        if not was_intentional:
            self._set_status("OVPhysX worker stopped.")

    def _release_scene(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._pending_play = False
        self._pending_step_after_start = False
        self._step_in_flight = False
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

    def _default_drop_visual_transform(self) -> np.ndarray:
        body_z = np.eye(4, dtype=np.float64)
        half_z = max(float(self._size[2]) * 0.5, 0.05)
        drop_height = max(float(self._size[2]) * 1.5, 0.75)
        body_z[3, :3] = self._center
        body_z[3, 2] = max(body_z[3, 2], GROUND_TOP_Z + half_z + drop_height)
        return self._visual_from_body(self._z_to_y_matrix(body_z))

    def _body_from_visual(self, visual_matrix: np.ndarray) -> np.ndarray:
        visual = np.array(visual_matrix, dtype=np.float64, copy=True)
        body_z = np.eye(4, dtype=np.float64)
        body_z[:3, :3] = visual[:3, :3]
        body_z[3, :3] = self._center @ visual[:3, :3] + visual[3, :3]
        return self._z_to_y_matrix(body_z)

    def _visual_from_body(self, body_matrix: np.ndarray) -> np.ndarray:
        body = self._y_to_z_matrix(body_matrix)
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

        size_z = np.maximum(self._size, np.array([0.05, 0.05, 0.05], dtype=np.float64))
        size_y = np.array([size_z[0], size_z[2], size_z[1]], dtype=np.float64)
        ground_y = -GROUND_THICKNESS * 0.5
        body_pos = body_matrix[3, :3]
        body_quat = self._quat_xyzw_from_row_rotation(body_matrix[:3, :3])

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
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "AssetProxy" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        double3 xformOp:translate = ({self._fmt(body_pos[0])}, {self._fmt(body_pos[1])}, {self._fmt(body_pos[2])})
        quatd xformOp:orient = ({self._fmt(body_quat[3])}, {self._fmt(body_quat[0])}, {self._fmt(body_quat[1])}, {self._fmt(body_quat[2])})
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        float physics:mass = 10
        vector3f physics:velocity = (0, 0, 0)
        vector3f physics:angularVelocity = (0, 0, 0)

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

    def Xform "Ground" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
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

    @staticmethod
    def _z_to_y_matrix(matrix: np.ndarray) -> np.ndarray:
        return Y_TO_Z_MATRIX @ np.asarray(matrix, dtype=np.float64).reshape(4, 4) @ Z_TO_Y_MATRIX

    @staticmethod
    def _y_to_z_matrix(matrix: np.ndarray) -> np.ndarray:
        return Z_TO_Y_MATRIX @ np.asarray(matrix, dtype=np.float64).reshape(4, 4) @ Y_TO_Z_MATRIX

    @staticmethod
    def _fmt(value: float) -> str:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        return f"{value:.9g}"
