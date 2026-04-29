"""Subprocess worker for OVPhysX simulation.

The parent process talks to this module with newline-delimited JSON over
stdin/stdout. Keeping OVPhysX outside Qt avoids an access violation observed
when constructing PhysX after QApplication/QCoreApplication exists.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import numpy as np


PROXY_PATH = "/World/AssetProxy"


class PhysicsWorker:
    def __init__(self):
        self._physx = None
        self._usd_handle = None
        self._pose_binding = None
        self._velocity_binding = None
        self._wrench_binding = None
        self._pose_buffer: Optional[np.ndarray] = None
        self._velocity_buffer: Optional[np.ndarray] = None
        self._wrench_buffer: Optional[np.ndarray] = None
        self._magnet: Optional[dict] = None
        self._body_patterns = [PROXY_PATH]
        self._active_body_pattern = PROXY_PATH

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                cmd = message.get("cmd")
                if cmd == "start":
                    self.start(
                        message["scene"],
                        message.get("body_patterns", message.get("body_pattern", PROXY_PATH)),
                        message.get("initial_pose"),
                    )
                elif cmd == "step":
                    self.step(float(message.get("dt", 1.0 / 60.0)), float(message.get("time", 0.0)))
                elif cmd == "set_pose":
                    self.set_pose(message.get("pose", []), bool(message.get("zero_velocity", True)))
                elif cmd == "set_magnet":
                    self.set_magnet(message)
                elif cmd == "release_magnet":
                    self.release_magnet(message.get("velocity", []), message.get("angular_velocity", []))
                elif cmd == "shutdown":
                    self.shutdown()
                    self._emit({"type": "stopped"})
                    return
            except Exception as exc:
                self._emit({"type": "error", "message": str(exc)})

    def start(self, scene_path: str, body_patterns=None, initial_pose=None) -> None:
        self.shutdown()
        self._body_patterns = self._normalize_patterns(body_patterns)
        self._active_body_pattern = self._body_patterns[0]

        import ovphysx  # noqa: F401
        from ovphysx import PhysX

        try:
            from ovphysx.types import TensorType
        except Exception:
            from ovphysx import TensorType

        try:
            self._physx = PhysX(device="cpu")
        except TypeError:
            self._physx = PhysX()

        result = self._physx.add_usd(scene_path)
        self._usd_handle = result[0] if isinstance(result, tuple) else result
        self._physx.wait_all()

        self._pose_binding = self._create_tensor_binding(TensorType.RIGID_BODY_POSE)
        self._pose_buffer = np.zeros(self._pose_binding.shape, dtype=np.float32)
        try:
            self._pose_binding.read(self._pose_buffer)
        except Exception:
            pass

        try:
            self._velocity_binding = self._create_tensor_binding(TensorType.RIGID_BODY_VELOCITY)
            self._velocity_buffer = np.zeros(self._velocity_binding.shape, dtype=np.float32)
        except Exception:
            self._velocity_binding = None
            self._velocity_buffer = None

        try:
            self._wrench_binding = self._create_tensor_binding(TensorType.RIGID_BODY_WRENCH)
            self._wrench_buffer = np.zeros(self._wrench_binding.shape, dtype=np.float32)
        except Exception:
            self._wrench_binding = None
            self._wrench_buffer = None

        if initial_pose is not None:
            self._write_pose(initial_pose, zero_velocity=True, emit=False)

        self._emit(
            {
                "type": "started",
                "body_pattern": self._active_body_pattern,
                "body_count": self._binding_count(self._pose_binding),
            }
        )
        self._emit_pose()

    def step(self, dt: float, sim_time: float) -> None:
        if self._physx is None:
            raise RuntimeError("Physics scene is not started")
        self._apply_magnet(max(float(dt), 1.0e-5))
        step_sync = getattr(self._physx, "step_sync", None)
        if callable(step_sync):
            try:
                step_sync(dt, sim_time)
            except TypeError:
                step_sync(dt)
        else:
            self._physx.step(dt, sim_time)
            self._physx.wait_all()
        self._emit_pose()

    def set_pose(self, pose, zero_velocity: bool) -> None:
        self._write_pose(pose, zero_velocity=zero_velocity, emit=True)

    def _write_pose(self, pose, zero_velocity: bool, emit: bool) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return
        arr = np.array(pose, dtype=np.float32).reshape(7)
        self._pose_buffer[0, :] = arr
        self._pose_binding.write(self._pose_buffer)
        if zero_velocity and self._velocity_binding is not None and self._velocity_buffer is not None:
            self._velocity_buffer.fill(0.0)
            self._velocity_binding.write(self._velocity_buffer)
        if emit:
            self._emit_pose()

    def set_magnet(self, message: dict) -> None:
        target = np.array(message.get("target", []), dtype=np.float32)
        anchor = np.array(message.get("anchor", []), dtype=np.float32)
        target_velocity = np.array(message.get("target_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        if target.size != 3 or anchor.size != 3:
            self._magnet = None
            self._clear_wrench()
            return
        if target_velocity.size != 3:
            target_velocity = np.zeros(3, dtype=np.float32)

        self._magnet = {
            "target": target.reshape(3),
            "anchor": anchor.reshape(3),
            "target_velocity": target_velocity.reshape(3),
            "stiffness": float(message.get("stiffness", 520.0)),
            "damping": float(message.get("damping", 62.0)),
            "max_force": float(message.get("max_force", 3200.0)),
        }

    def release_magnet(self, velocity, angular_velocity=None) -> None:
        self._magnet = None
        self._clear_wrench()

        if self._velocity_binding is None or self._velocity_buffer is None:
            return

        linear = np.array(velocity, dtype=np.float32)
        angular = np.array(angular_velocity if angular_velocity is not None else [], dtype=np.float32)
        if linear.size != 3:
            linear = np.zeros(3, dtype=np.float32)
        if angular.size != 3:
            angular = np.zeros(3, dtype=np.float32)

        self._velocity_buffer.fill(0.0)
        if self._velocity_buffer.shape[-1] >= 3:
            self._velocity_buffer[0, 0:3] = self._clamp_vector(linear.reshape(3), 18.0)
        if self._velocity_buffer.shape[-1] >= 6:
            self._velocity_buffer[0, 3:6] = self._clamp_vector(angular.reshape(3), 18.0)
        self._velocity_binding.write(self._velocity_buffer)

    def shutdown(self) -> None:
        self._magnet = None
        for name in ("_pose_binding", "_velocity_binding", "_wrench_binding"):
            binding = getattr(self, name)
            if binding is not None:
                try:
                    binding.destroy()
                except Exception:
                    pass
                setattr(self, name, None)

        self._pose_buffer = None
        self._velocity_buffer = None
        self._wrench_buffer = None

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

    def _create_tensor_binding(self, tensor_type):
        last_exc: Optional[Exception] = None
        for pattern in self._body_patterns:
            attempts = [
                lambda p=pattern: self._physx.create_tensor_binding(pattern=p, tensor_type=tensor_type),
            ]
            if "*" not in pattern and "?" not in pattern:
                attempts.extend(
                    [
                        lambda p=pattern: self._physx.create_tensor_binding(prim_paths=[p], tensor_type=tensor_type),
                        lambda p=pattern: self._physx.create_tensor_binding(p, tensor_type=tensor_type),
                    ]
                )

            for attempt in attempts:
                try:
                    binding = attempt()
                    if self._binding_count(binding) <= 0:
                        raise RuntimeError(f"No rigid bodies matched {pattern}")
                    self._active_body_pattern = pattern
                    return binding
                except Exception as exc:
                    last_exc = exc
        joined = ", ".join(self._body_patterns)
        raise RuntimeError(f"Could not bind tensor for {joined}: {last_exc}")

    @staticmethod
    def _normalize_patterns(patterns) -> list[str]:
        if isinstance(patterns, str):
            raw = [patterns]
        else:
            try:
                raw = list(patterns or [])
            except TypeError:
                raw = []
        normalized = []
        for item in raw:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized or [PROXY_PATH]

    @staticmethod
    def _binding_count(binding) -> int:
        count = getattr(binding, "count", None)
        if count is not None:
            try:
                return int(count)
            except Exception:
                pass
        shape = getattr(binding, "shape", None)
        if shape is not None:
            try:
                return int(shape[0])
            except Exception:
                pass
        return 1

    def _emit_pose(self) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return
        self._pose_binding.read(self._pose_buffer)
        self._emit({"type": "pose", "pose": self._pose_buffer[0].astype(float).tolist()})

    def _apply_magnet(self, dt: float) -> None:
        if self._magnet is None or self._pose_binding is None or self._pose_buffer is None:
            return

        self._pose_binding.read(self._pose_buffer)
        pose = np.array(self._pose_buffer[0, :], dtype=np.float32)
        position = pose[:3]
        rotation = self._row_rotation_from_quat_xyzw(pose[3:7])

        anchor_local = self._magnet["anchor"]
        anchor_world = position + anchor_local @ rotation
        target = self._magnet["target"]
        target_velocity = self._magnet["target_velocity"]

        linear_velocity = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        if self._velocity_binding is not None and self._velocity_buffer is not None:
            self._velocity_binding.read(self._velocity_buffer)
            if self._velocity_buffer.shape[-1] >= 3:
                linear_velocity = np.array(self._velocity_buffer[0, 0:3], dtype=np.float32)
            if self._velocity_buffer.shape[-1] >= 6:
                angular_velocity = np.array(self._velocity_buffer[0, 3:6], dtype=np.float32)

        lever = anchor_world - position
        point_velocity = linear_velocity + np.cross(angular_velocity, lever)
        offset = target - anchor_world
        force = (
            offset * float(self._magnet["stiffness"])
            + (target_velocity - point_velocity) * float(self._magnet["damping"])
        )
        force = self._clamp_vector(force, float(self._magnet["max_force"]))

        if self._wrench_binding is not None and self._wrench_buffer is not None:
            self._wrench_buffer.fill(0.0)
            self._wrench_buffer[0, 0:3] = force
            self._wrench_buffer[0, 6:9] = anchor_world
            self._wrench_binding.write(self._wrench_buffer)
            return

        if self._velocity_binding is not None and self._velocity_buffer is not None:
            desired_linear = linear_velocity + force * (dt / 10.0)
            desired_angular = angular_velocity + np.cross(lever, force) * (dt / 4.0)
            self._velocity_buffer.fill(0.0)
            self._velocity_buffer[0, 0:3] = self._clamp_vector(desired_linear, 12.0)
            if self._velocity_buffer.shape[-1] >= 6:
                self._velocity_buffer[0, 3:6] = self._clamp_vector(desired_angular, 12.0)
            self._velocity_binding.write(self._velocity_buffer)

    def _clear_wrench(self) -> None:
        if self._wrench_binding is None or self._wrench_buffer is None:
            return
        self._wrench_buffer.fill(0.0)
        try:
            self._wrench_binding.write(self._wrench_buffer)
        except Exception:
            pass

    @staticmethod
    def _row_rotation_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in quat[:4]]
        norm = float(np.linalg.norm([x, y, z, w]))
        if norm < 1.0e-9:
            return np.eye(3, dtype=np.float32)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        col = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )
        return col.T

    @staticmethod
    def _clamp_vector(vector: np.ndarray, limit: float) -> np.ndarray:
        arr = np.array(vector, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(arr))
        if norm > max(float(limit), 1.0e-6):
            arr *= float(limit) / norm
        return arr

    @staticmethod
    def _emit(message: dict) -> None:
        sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def main() -> int:
    worker = PhysicsWorker()
    try:
        worker.run()
    finally:
        worker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
