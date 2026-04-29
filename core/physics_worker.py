"""Subprocess worker for OVPhysX simulation.

The parent process talks to this module with newline-delimited JSON over
stdin/stdout. Keeping OVPhysX outside Qt avoids an access violation observed
when constructing PhysX after QApplication/QCoreApplication exists.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

import numpy as np


PROXY_PATH = "/World/AssetProxy"
DEFAULT_GRAB_MASS_KG = 10.0
MIN_GRAB_MASS_KG = 0.05
MAX_GRAB_CONTROL_MASS_KG = 500.0
GRAB_NATURAL_FREQUENCY = 7.0
GRAB_DAMPING_RATIO = 0.7
GRAB_MAX_ACCELERATION = 70.0
GRAB_MAX_ANGULAR_ACCELERATION = 10.0
DEFAULT_GRAB_FORCE_AMOUNT = 2.0
MIN_GRAB_FORCE_AMOUNT = 0.25
MAX_GRAB_FORCE_AMOUNT = 5.0


class PhysicsWorker:
    def __init__(self):
        self._physx = None
        self._usd_handle = None
        self._pose_binding = None
        self._velocity_binding = None
        self._wrench_binding = None
        self._mass_binding = None
        self._pose_buffer: Optional[np.ndarray] = None
        self._reference_pose_buffer: Optional[np.ndarray] = None
        self._velocity_buffer: Optional[np.ndarray] = None
        self._wrench_buffer: Optional[np.ndarray] = None
        self._mass_buffer: Optional[np.ndarray] = None
        self._magnet: Optional[dict] = None
        self._body_patterns = [PROXY_PATH]
        self._active_body_pattern = PROXY_PATH
        self._active_body_count = 0
        self._effective_mass_kg = DEFAULT_GRAB_MASS_KG
        self._contact_offset = 0.01
        self._shape_count = 0
        self._cook_warning = ""

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
                        message.get("contact_offset", 0.01),
                        bool(message.get("cook_only", False)),
                    )
                elif cmd == "step":
                    self.step(
                        float(message.get("dt", 1.0 / 60.0)),
                        float(message.get("time", 0.0)),
                        int(message.get("substeps", 1) or 1),
                    )
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

    def start(
        self,
        scene_path: str,
        body_patterns=None,
        initial_pose=None,
        contact_offset=0.01,
        cook_only: bool = False,
    ) -> None:
        self.shutdown()
        self._shape_count = 0
        self._cook_warning = ""
        self._emit_progress(4, "Preparing OVPhysX collider cook...")
        self._body_patterns = self._normalize_patterns(body_patterns)
        self._active_body_pattern = self._body_patterns[0]
        try:
            self._contact_offset = max(0.001, min(0.08, float(contact_offset)))
        except Exception:
            self._contact_offset = 0.01

        self._emit_progress(10, "Importing OVPhysX runtime...")
        import ovphysx  # noqa: F401
        from ovphysx import PhysX

        try:
            from ovphysx.types import TensorType
        except Exception:
            from ovphysx import TensorType

        self._emit_progress(18, "Creating PhysX scene...")
        device = os.environ.get("SIMREADY_OVPHYSX_DEVICE", "cuda")
        try:
            self._physx = PhysX(device=device)
        except Exception:
            try:
                self._physx = PhysX(device="cpu")
            except TypeError:
                self._physx = PhysX()

        self._emit_progress(30, "Loading USD and authored physics payloads...")
        result = self._physx.add_usd(scene_path)
        self._usd_handle = result[0] if isinstance(result, tuple) else result
        self._physx.wait_all()

        self._emit_progress(58, "Discovering authored rigid bodies...")
        self._pose_binding = self._create_best_tensor_binding(TensorType.RIGID_BODY_POSE)
        self._active_body_count = self._binding_count(self._pose_binding)
        self._pose_buffer = np.zeros(self._pose_binding.shape, dtype=np.float32)
        try:
            self._pose_binding.read(self._pose_buffer)
        except Exception:
            pass
        self._reference_pose_buffer = np.array(self._pose_buffer, dtype=np.float32, copy=True)

        self._emit_progress(70, "Binding velocity and force tensors...")
        try:
            self._velocity_binding = self._create_tensor_binding(
                TensorType.RIGID_BODY_VELOCITY,
                patterns=[self._active_body_pattern],
            )
            self._velocity_buffer = np.zeros(self._velocity_binding.shape, dtype=np.float32)
        except Exception:
            self._velocity_binding = None
            self._velocity_buffer = None

        try:
            self._wrench_binding = self._create_tensor_binding(
                TensorType.RIGID_BODY_WRENCH,
                patterns=[self._active_body_pattern],
            )
            self._wrench_buffer = np.zeros(self._wrench_binding.shape, dtype=np.float32)
        except Exception:
            self._wrench_binding = None
            self._wrench_buffer = None

        self._effective_mass_kg = DEFAULT_GRAB_MASS_KG
        try:
            self._mass_binding = self._create_tensor_binding(
                TensorType.RIGID_BODY_MASS,
                patterns=[self._active_body_pattern],
                update_active=False,
            )
            self._mass_buffer = np.zeros(self._mass_binding.shape, dtype=np.float32)
            self._effective_mass_kg = self._read_active_mass_kg(DEFAULT_GRAB_MASS_KG)
        except Exception:
            self._mass_binding = None
            self._mass_buffer = None

        self._emit_progress(82, "Cooking collider shapes...")
        self._shape_count = self._tune_shape_properties(TensorType)
        if self._shape_count <= 0:
            self._shape_count = self._force_collider_cook(TensorType)
        if self._shape_count <= 0:
            self._cook_warning = (
                "OVPhysX finished the authored collider cook, but did not expose collider shape tensors yet. "
                "Physics will not play until the authored collider binding is fixed."
            )
            self._emit_progress(92, "Finalizing authored collider cook...")

        self._emit_progress(94, "Finalizing cooked collider cache...")
        if cook_only:
            self._emit(
                {
                    "type": "cooked",
                    "body_pattern": self._active_body_pattern,
                    "body_count": self._active_body_count,
                    "shape_count": self._shape_count,
                    "cook_warning": self._cook_warning,
                }
            )
            return

        if initial_pose is not None:
            self._write_pose(initial_pose, zero_velocity=True, emit=False)

        self._emit(
            {
                "type": "started",
                "body_pattern": self._active_body_pattern,
                "body_count": self._active_body_count,
                "shape_count": self._shape_count,
                "cook_warning": self._cook_warning,
            }
        )
        self._emit_pose()

    def step(self, dt: float, sim_time: float, substeps: int = 1) -> None:
        if self._physx is None:
            raise RuntimeError("Physics scene is not started")
        dt = max(float(dt), 1.0e-5)
        substeps = max(1, min(16, int(substeps)))
        sub_dt = dt / float(substeps)

        self._apply_magnet(dt)
        step_n_sync = getattr(self._physx, "step_n_sync", None)
        if callable(step_n_sync) and substeps > 1:
            step_n_sync(substeps, sub_dt, sim_time)
            self._emit_pose()
            return

        step_sync = getattr(self._physx, "step_sync", None)
        if callable(step_sync):
            for index in range(substeps):
                try:
                    step_sync(sub_dt, sim_time + index * sub_dt)
                except TypeError:
                    step_sync(sub_dt)
        else:
            for index in range(substeps):
                self._physx.step(sub_dt, sim_time + index * sub_dt)
                self._physx.wait_all()
        self._emit_pose()

    def set_pose(self, pose, zero_velocity: bool) -> None:
        self._write_pose(pose, zero_velocity=zero_velocity, emit=True)

    def _write_pose(self, pose, zero_velocity: bool, emit: bool) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return
        root_pose = np.array(pose, dtype=np.float32).reshape(7)
        root_matrix = self._matrix_from_pose(root_pose)
        reference = self._reference_pose_buffer
        if reference is not None and reference.shape[0] == self._pose_buffer.shape[0]:
            for index in range(self._pose_buffer.shape[0]):
                body_matrix = self._matrix_from_pose(reference[index, :]) @ root_matrix
                self._pose_buffer[index, :] = self._pose_from_matrix(body_matrix)
        else:
            self._pose_buffer[0, :] = root_pose
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

        estimated_mass = self._sanitize_mass(message.get("estimated_mass", self._effective_mass_kg), self._effective_mass_kg)
        mass = self._read_active_mass_kg(estimated_mass)
        self._effective_mass_kg = mass
        stiffness, damping, max_force, speed_limit, max_angular_accel = self._magnet_response(message, mass)
        force_amount = self._sanitize_force_amount(message.get("force_amount", DEFAULT_GRAB_FORCE_AMOUNT))

        self._magnet = {
            "target": target.reshape(3),
            "anchor": self._root_anchor_to_active_body_local(anchor.reshape(3)),
            "target_velocity": self._clamp_vector(target_velocity.reshape(3), speed_limit),
            "mass": mass,
            "force_amount": force_amount,
            "stiffness": stiffness,
            "damping": damping,
            "max_force": max_force,
            "max_angular_accel": max_angular_accel,
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

        speed_limit = self._release_speed_limit(self._effective_mass_kg)
        self._velocity_buffer.fill(0.0)
        if self._velocity_buffer.shape[-1] >= 3:
            self._velocity_buffer[0, 0:3] = self._clamp_vector(linear.reshape(3), speed_limit)
        if self._velocity_buffer.shape[-1] >= 6:
            self._velocity_buffer[0, 3:6] = self._clamp_vector(angular.reshape(3), min(12.0, speed_limit))
        self._velocity_binding.write(self._velocity_buffer)

    def shutdown(self) -> None:
        self._magnet = None
        for name in ("_pose_binding", "_velocity_binding", "_wrench_binding", "_mass_binding"):
            binding = getattr(self, name)
            if binding is not None:
                try:
                    binding.destroy()
                except Exception:
                    pass
                setattr(self, name, None)

        self._pose_buffer = None
        self._reference_pose_buffer = None
        self._velocity_buffer = None
        self._wrench_buffer = None
        self._mass_buffer = None
        self._effective_mass_kg = DEFAULT_GRAB_MASS_KG

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

    def _create_best_tensor_binding(self, tensor_type):
        last_exc: Optional[Exception] = None
        candidates = []
        for pattern in self._body_patterns:
            for attempt in self._binding_attempts(pattern, tensor_type):
                binding = None
                try:
                    binding = attempt()
                    count = self._binding_count(binding)
                    if count <= 0:
                        raise RuntimeError(f"No rigid bodies matched {pattern}")
                    candidates.append((count, self._pattern_score(pattern), pattern, binding))
                    break
                except Exception as exc:
                    last_exc = exc
                    if binding is not None:
                        try:
                            binding.destroy()
                        except Exception:
                            pass

        if not candidates:
            joined = ", ".join(self._body_patterns)
            raise RuntimeError(f"Could not bind tensor for {joined}: {last_exc}")

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _count, _score, pattern, binding = candidates[0]
        for _other_count, _other_score, _other_pattern, other_binding in candidates[1:]:
            try:
                other_binding.destroy()
            except Exception:
                pass
        self._active_body_pattern = pattern
        return binding

    def _create_tensor_binding(self, tensor_type, patterns=None, update_active: bool = True):
        last_exc: Optional[Exception] = None
        pattern_list = self._normalize_patterns(patterns or self._body_patterns)
        for pattern in pattern_list:
            for attempt in self._binding_attempts(pattern, tensor_type):
                binding = None
                try:
                    binding = attempt()
                    if self._binding_count(binding) <= 0:
                        raise RuntimeError(f"No rigid bodies matched {pattern}")
                    if update_active:
                        self._active_body_pattern = pattern
                    return binding
                except Exception as exc:
                    last_exc = exc
                    if binding is not None:
                        try:
                            binding.destroy()
                        except Exception:
                            pass
        joined = ", ".join(pattern_list)
        raise RuntimeError(f"Could not bind tensor for {joined}: {last_exc}")

    def _binding_attempts(self, pattern: str, tensor_type):
        attempts = [lambda p=pattern: self._physx.create_tensor_binding(pattern=p, tensor_type=tensor_type)]
        if not self._has_wildcards(pattern):
            attempts.extend(
                [
                    lambda p=pattern: self._physx.create_tensor_binding(prim_paths=[p], tensor_type=tensor_type),
                    lambda p=pattern: self._physx.create_tensor_binding([p], tensor_type=tensor_type),
                ]
            )
        return attempts

    @staticmethod
    def _has_wildcards(pattern: str) -> bool:
        return any(char in str(pattern) for char in "*?[")

    @staticmethod
    def _pattern_score(pattern: str) -> int:
        score = len(pattern.replace("*", "").replace("?", ""))
        if "RootNode" in pattern:
            score += 1000
        if "Geometry" in pattern:
            score += 100
        return score

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
        pose = np.array(self._pose_buffer[0], dtype=np.float32, copy=True)
        if self._reference_pose_buffer is not None and self._reference_pose_buffer.shape[0] > 0:
            try:
                ref_matrix = self._matrix_from_pose(self._reference_pose_buffer[0])
                body_matrix = self._matrix_from_pose(pose)
                root_matrix = np.linalg.inv(ref_matrix) @ body_matrix
                pose = self._pose_from_matrix(root_matrix)
            except Exception:
                pass
        self._emit({"type": "pose", "pose": pose.astype(float).tolist()})

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
        mass = max(float(self._magnet.get("mass", self._effective_mass_kg)), MIN_GRAB_MASS_KG)
        force = self._clamp_grab_force(force, mass, lever)
        torque = self._clamp_grab_torque(np.cross(lever, force), mass, lever)

        if self._wrench_binding is not None and self._wrench_buffer is not None:
            self._wrench_buffer.fill(0.0)
            if self._wrench_buffer.shape[-1] >= 3:
                self._wrench_buffer[0, 0:3] = force
            if self._wrench_buffer.shape[-1] >= 6:
                self._wrench_buffer[0, 3:6] = torque
            if self._wrench_buffer.shape[-1] >= 9:
                self._wrench_buffer[0, 6:9] = anchor_world
            self._wrench_binding.write(self._wrench_buffer)
            return

        if self._velocity_binding is not None and self._velocity_buffer is not None:
            desired_linear = linear_velocity + (force / mass) * dt
            lever_length = max(float(np.linalg.norm(lever)), 0.05)
            approximate_inertia = max(mass * lever_length * lever_length, 0.001)
            desired_angular = angular_velocity + (torque / approximate_inertia) * dt
            self._velocity_buffer.fill(0.0)
            self._velocity_buffer[0, 0:3] = self._clamp_vector(desired_linear, self._release_speed_limit(mass))
            if self._velocity_buffer.shape[-1] >= 6:
                self._velocity_buffer[0, 3:6] = self._clamp_vector(desired_angular, 10.0)
            self._velocity_binding.write(self._velocity_buffer)

    def _root_anchor_to_active_body_local(self, anchor_root: np.ndarray) -> np.ndarray:
        anchor = np.asarray(anchor_root, dtype=np.float32).reshape(3)
        if self._reference_pose_buffer is None or self._reference_pose_buffer.shape[0] <= 0:
            return anchor

        try:
            reference_matrix = self._matrix_from_pose(np.asarray(self._reference_pose_buffer[0], dtype=np.float32))
            reference_rotation = reference_matrix[:3, :3]
            reference_position = reference_matrix[3, :3]
            return ((anchor.astype(np.float64) - reference_position) @ reference_rotation.T).astype(np.float32)
        except Exception:
            return anchor

    def _read_active_mass_kg(self, fallback: float) -> float:
        if self._mass_binding is None or self._mass_buffer is None:
            return self._sanitize_mass(fallback, DEFAULT_GRAB_MASS_KG)

        try:
            self._mass_binding.read(self._mass_buffer)
            values = np.asarray(self._mass_buffer, dtype=np.float64).reshape(-1)
        except Exception:
            return self._sanitize_mass(fallback, DEFAULT_GRAB_MASS_KG)

        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size <= 0:
            return self._sanitize_mass(fallback, DEFAULT_GRAB_MASS_KG)

        # Treat multi-body matches as one grabbed assembly for gain scaling.
        return self._sanitize_mass(float(np.sum(finite)), fallback)

    def _magnet_response(self, message: dict, mass: float) -> tuple[float, float, float, float, float]:
        control_mass = max(MIN_GRAB_MASS_KG, min(float(mass), MAX_GRAB_CONTROL_MASS_KG))
        force_amount = self._sanitize_force_amount(message.get("force_amount", DEFAULT_GRAB_FORCE_AMOUNT))
        force_scale = math.sqrt(force_amount)
        frequency = self._finite_float(message.get("natural_frequency", GRAB_NATURAL_FREQUENCY), GRAB_NATURAL_FREQUENCY)
        frequency *= force_scale
        frequency = max(1.8, min(10.0, frequency))
        damping_ratio = self._finite_float(message.get("damping_ratio", GRAB_DAMPING_RATIO), GRAB_DAMPING_RATIO)
        damping_ratio = max(0.25, min(1.1, damping_ratio))
        max_accel = self._finite_float(message.get("max_acceleration", GRAB_MAX_ACCELERATION), GRAB_MAX_ACCELERATION)
        max_accel = max(14.0, min(180.0, max_accel * force_amount))
        max_angular_accel = self._finite_float(
            message.get("max_angular_acceleration", GRAB_MAX_ANGULAR_ACCELERATION),
            GRAB_MAX_ANGULAR_ACCELERATION,
        )
        max_angular_accel = max(4.0, min(35.0, max_angular_accel))

        stiffness = self._finite_float(message.get("stiffness", control_mass * frequency * frequency), 0.0)
        damping = self._finite_float(message.get("damping", 2.0 * damping_ratio * control_mass * frequency), 0.0)
        max_force = self._finite_float(message.get("max_force", control_mass * max_accel), control_mass * max_accel)

        stiffness = max(0.1, min(stiffness, control_mass * 160.0))
        damping = max(0.1, min(damping, control_mass * 40.0))
        max_force = max(control_mass * 14.0, min(max_force, control_mass * 220.0, 12000.0))
        return stiffness, damping, max_force, self._release_speed_limit(control_mass), max_angular_accel

    def _clamp_grab_force(self, force: np.ndarray, mass: float, lever: np.ndarray) -> np.ndarray:
        force_limit = float(self._magnet.get("max_force", mass * GRAB_MAX_ACCELERATION)) if self._magnet else mass * GRAB_MAX_ACCELERATION
        angular_limit = float(self._magnet.get("max_angular_accel", GRAB_MAX_ANGULAR_ACCELERATION)) if self._magnet else GRAB_MAX_ANGULAR_ACCELERATION
        force_amount = float(self._magnet.get("force_amount", DEFAULT_GRAB_FORCE_AMOUNT)) if self._magnet else DEFAULT_GRAB_FORCE_AMOUNT
        force_scale = math.sqrt(max(MIN_GRAB_FORCE_AMOUNT, min(force_amount, MAX_GRAB_FORCE_AMOUNT)))
        lever_length = max(float(np.linalg.norm(lever)), 0.05)

        # A corner grab creates torque. Capping force by angular acceleration keeps
        # small/light objects from spinning violently while still allowing lift.
        torque_limited_force = max(mass * 60.0 * force_scale, mass * lever_length * angular_limit * 2.0 * force_scale)
        limit = min(force_limit, torque_limited_force)
        return self._clamp_vector(force, max(limit, mass * 12.0, 0.25))

    def _clamp_grab_torque(self, torque: np.ndarray, mass: float, lever: np.ndarray) -> np.ndarray:
        angular_limit = float(self._magnet.get("max_angular_accel", GRAB_MAX_ANGULAR_ACCELERATION)) if self._magnet else GRAB_MAX_ANGULAR_ACCELERATION
        force_amount = float(self._magnet.get("force_amount", DEFAULT_GRAB_FORCE_AMOUNT)) if self._magnet else DEFAULT_GRAB_FORCE_AMOUNT
        force_scale = math.sqrt(max(MIN_GRAB_FORCE_AMOUNT, min(force_amount, MAX_GRAB_FORCE_AMOUNT)))
        lever_length = max(float(np.linalg.norm(lever)), 0.05)
        approximate_inertia = max(mass * lever_length * lever_length, 0.001)
        torque_limit = max(approximate_inertia * angular_limit * force_scale, mass * 0.04, 0.02)
        return self._clamp_vector(torque, torque_limit)

    @staticmethod
    def _release_speed_limit(mass: float) -> float:
        mass = max(MIN_GRAB_MASS_KG, min(float(mass), MAX_GRAB_CONTROL_MASS_KG))
        return max(3.5, min(18.0, 5.0 + 2.2 * math.sqrt(mass)))

    @classmethod
    def _sanitize_mass(cls, value, fallback: float) -> float:
        mass = cls._finite_float(value, fallback)
        if mass <= 0.0:
            mass = cls._finite_float(fallback, DEFAULT_GRAB_MASS_KG)
        return max(MIN_GRAB_MASS_KG, min(mass, MAX_GRAB_CONTROL_MASS_KG))

    @classmethod
    def _sanitize_force_amount(cls, value) -> float:
        amount = cls._finite_float(value, DEFAULT_GRAB_FORCE_AMOUNT)
        return max(MIN_GRAB_FORCE_AMOUNT, min(amount, MAX_GRAB_FORCE_AMOUNT))

    @staticmethod
    def _finite_float(value, fallback: float) -> float:
        try:
            result = float(value)
        except Exception:
            result = float(fallback)
        if not math.isfinite(result):
            result = float(fallback)
        return result

    def _clear_wrench(self) -> None:
        if self._wrench_binding is None or self._wrench_buffer is None:
            return
        self._wrench_buffer.fill(0.0)
        try:
            self._wrench_binding.write(self._wrench_buffer)
        except Exception:
            pass

    def _tune_shape_properties(self, tensor_types) -> int:
        shape_count = 0
        contact_binding = None
        rest_binding = None
        material_binding = None
        shape_patterns = self._shape_binding_patterns()
        try:
            contact_binding = self._create_tensor_binding(
                tensor_types.RIGID_BODY_CONTACT_OFFSET,
                patterns=shape_patterns,
                update_active=False,
            )
            contact_buffer = np.zeros(contact_binding.shape, dtype=np.float32)
            contact_binding.read(contact_buffer)
            shape_count = self._shape_entry_count(contact_binding)
            contact_buffer = np.maximum(contact_buffer, np.float32(self._contact_offset))
            contact_binding.write(contact_buffer)

            try:
                rest_binding = self._create_tensor_binding(
                    tensor_types.RIGID_BODY_REST_OFFSET,
                    patterns=shape_patterns,
                    update_active=False,
                )
                rest_buffer = np.zeros(rest_binding.shape, dtype=np.float32)
                rest_binding.read(rest_buffer)
                rest_buffer = np.minimum(rest_buffer, contact_buffer - np.float32(0.0005))
                rest_buffer = np.minimum(rest_buffer, np.float32(0.0))
                rest_binding.write(rest_buffer)
            except Exception:
                pass

            try:
                material_binding = self._create_tensor_binding(
                    tensor_types.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
                    patterns=shape_patterns,
                    update_active=False,
                )
                material_buffer = np.zeros(material_binding.shape, dtype=np.float32)
                material_binding.read(material_buffer)
                if material_buffer.ndim == 3 and material_buffer.shape[-1] >= 3:
                    material_buffer[:, :, 2] = np.minimum(np.maximum(material_buffer[:, :, 2], 0.0), 0.05)
                    material_binding.write(material_buffer)
            except Exception:
                pass
        except Exception:
            shape_count = 0
        finally:
            for binding in (contact_binding, rest_binding, material_binding):
                if binding is not None:
                    try:
                        binding.destroy()
                    except Exception:
                        pass
        return int(shape_count)

    def _shape_binding_patterns(self) -> list[str]:
        patterns: list[str] = []

        def add(pattern: str) -> None:
            text = str(pattern or "").strip()
            if text and text not in patterns:
                patterns.append(text)

        add(self._active_body_pattern)
        for pattern in [self._active_body_pattern, *self._body_patterns]:
            add(pattern)
            if not self._has_wildcards(pattern):
                add(f"{pattern}/*")
                add(f"{pattern}/*/*")
                add(f"{pattern}/*/*/*")
        return patterns or [self._active_body_pattern]

    def _force_collider_cook(self, tensor_types) -> int:
        if self._physx is None:
            return 0

        for attempt in range(3):
            self._emit_progress(84 + attempt * 3, "Forcing OVPhysX collider cook pass...")
            try:
                step_n_sync = getattr(self._physx, "step_n_sync", None)
                if callable(step_n_sync):
                    step_n_sync(1, 1.0e-5, attempt * 1.0e-5)
                else:
                    step_sync = getattr(self._physx, "step_sync", None)
                    if callable(step_sync):
                        try:
                            step_sync(1.0e-5, attempt * 1.0e-5)
                        except TypeError:
                            step_sync(1.0e-5)
                    else:
                        self._physx.step(1.0e-5, attempt * 1.0e-5)
                wait_all = getattr(self._physx, "wait_all", None)
                if callable(wait_all):
                    wait_all()
            except Exception:
                pass

            shape_count = self._tune_shape_properties(tensor_types)
            if shape_count > 0:
                return int(shape_count)

        return 0

    @staticmethod
    def _shape_entry_count(binding) -> int:
        shape = getattr(binding, "shape", None)
        if shape is None:
            return 0
        try:
            dims = tuple(int(value) for value in shape)
        except Exception:
            return 0
        if len(dims) < 2:
            return 0
        return max(0, dims[0]) * max(0, dims[1])

    @classmethod
    def _matrix_from_pose(cls, pose: np.ndarray) -> np.ndarray:
        arr = np.asarray(pose, dtype=np.float64).reshape(7)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = cls._row_rotation_from_quat_xyzw(arr[3:7])
        matrix[3, :3] = arr[:3]
        return matrix

    @classmethod
    def _pose_from_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        pose = np.zeros(7, dtype=np.float32)
        pose[:3] = mat[3, :3].astype(np.float32)
        pose[3:] = cls._quat_xyzw_from_row_rotation(mat[:3, :3]).astype(np.float32)
        return pose

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

    def _emit_progress(self, value: int, message: str) -> None:
        self._emit({"type": "progress", "value": int(value), "message": str(message)})


def main() -> int:
    worker = PhysicsWorker()
    try:
        worker.run()
    finally:
        worker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
