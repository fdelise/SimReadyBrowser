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
        self._pose_buffer: Optional[np.ndarray] = None
        self._velocity_buffer: Optional[np.ndarray] = None

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                cmd = message.get("cmd")
                if cmd == "start":
                    self.start(message["scene"])
                elif cmd == "step":
                    self.step(float(message.get("dt", 1.0 / 60.0)), float(message.get("time", 0.0)))
                elif cmd == "set_pose":
                    self.set_pose(message.get("pose", []), bool(message.get("zero_velocity", True)))
                elif cmd == "shutdown":
                    self.shutdown()
                    self._emit({"type": "stopped"})
                    return
            except Exception as exc:
                self._emit({"type": "error", "message": str(exc)})

    def start(self, scene_path: str) -> None:
        self.shutdown()

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
            self._velocity_binding = self._create_tensor_binding(TensorType.RIGID_BODY_VELOCITY)
            self._velocity_buffer = np.zeros(self._velocity_binding.shape, dtype=np.float32)
        except Exception:
            self._velocity_binding = None
            self._velocity_buffer = None

        self._emit({"type": "started"})
        self._emit_pose()

    def step(self, dt: float, sim_time: float) -> None:
        if self._physx is None:
            raise RuntimeError("Physics scene is not started")
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
        if self._pose_binding is None or self._pose_buffer is None:
            return
        arr = np.array(pose, dtype=np.float32).reshape(7)
        self._pose_buffer[0, :] = arr
        self._pose_binding.write(self._pose_buffer)
        if zero_velocity and self._velocity_binding is not None and self._velocity_buffer is not None:
            self._velocity_buffer.fill(0.0)
            self._velocity_binding.write(self._velocity_buffer)
        self._emit_pose()

    def shutdown(self) -> None:
        for name in ("_pose_binding", "_velocity_binding"):
            binding = getattr(self, name)
            if binding is not None:
                try:
                    binding.destroy()
                except Exception:
                    pass
                setattr(self, name, None)

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

    def _create_tensor_binding(self, tensor_type):
        attempts = (
            lambda: self._physx.create_tensor_binding(prim_paths=[PROXY_PATH], tensor_type=tensor_type),
            lambda: self._physx.create_tensor_binding(pattern=PROXY_PATH, tensor_type=tensor_type),
            lambda: self._physx.create_tensor_binding(PROXY_PATH, tensor_type=tensor_type),
        )
        last_exc: Optional[Exception] = None
        for attempt in attempts:
            try:
                binding = attempt()
                if getattr(binding, "count", 1) == 0:
                    raise RuntimeError(f"No rigid bodies matched {PROXY_PATH}")
                return binding
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Could not bind tensor for {PROXY_PATH}: {last_exc}")

    def _emit_pose(self) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return
        self._pose_binding.read(self._pose_buffer)
        self._emit({"type": "pose", "pose": self._pose_buffer[0].astype(float).tolist()})

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
