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
MAX_GRAB_FORCE_AMOUNT = 100.0
MAX_SIM_POSITION = 10000.0


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
        self._body_paths: list[str] = []
        self._articulation_paths: list[str] = []
        self._instance_paths: list[str] = []
        self._instance_reference_poses: dict[str, np.ndarray] = {}
        self._bound_body_paths: list[str] = []
        self._body_index_by_path: dict[str, int] = {}
        self._active_body_pattern = PROXY_PATH
        self._active_body_index = 0
        self._active_body_count = 0
        self._effective_mass_kg = DEFAULT_GRAB_MASS_KG
        self._contact_offset = 0.01
        self._shape_count = 0
        self._cook_warning = ""
        self._ccd_enabled = False
        self._unstable = False

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
                        scene_path=message["scene"],
                        body_patterns=message.get("body_patterns", message.get("body_pattern", PROXY_PATH)),
                        initial_pose=message.get("initial_pose"),
                        contact_offset=message.get("contact_offset", 0.01),
                        cook_only=bool(message.get("cook_only", False)),
                        body_paths=message.get("body_paths", []),
                        articulation_paths=message.get("articulation_paths", []),
                        instance_paths=message.get("instance_paths", []),
                        instance_reference_poses=message.get("instance_reference_poses", []),
                        clone_source_path=message.get("clone_source_path", ""),
                        clone_target_paths=message.get("clone_target_paths", []),
                        clone_parent_poses=message.get("clone_parent_poses", []),
                        clone_groups=message.get("clone_groups", []),
                        ccd_enabled=bool(message.get("ccd_enabled", False)),
                    )
                elif cmd == "step":
                    self.step(
                        float(message.get("dt", 1.0 / 60.0)),
                        float(message.get("time", 0.0)),
                        int(message.get("substeps", 1) or 1),
                    )
                elif cmd == "set_pose":
                    self.set_pose(message.get("pose", []), bool(message.get("zero_velocity", True)))
                elif cmd == "set_poses":
                    self.set_poses(message.get("poses", []), bool(message.get("zero_velocity", True)))
                elif cmd == "set_magnet":
                    self.set_magnet(message)
                elif cmd == "release_magnet":
                    self.release_magnet(message.get("velocity", []), message.get("angular_velocity", []))
                elif cmd == "set_ccd":
                    self.set_ccd_enabled(bool(message.get("enabled", False)))
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
        body_paths=None,
        articulation_paths=None,
        instance_paths=None,
        instance_reference_poses=None,
        clone_source_path: str = "",
        clone_target_paths=None,
        clone_parent_poses=None,
        clone_groups=None,
        ccd_enabled: bool = False,
    ) -> None:
        self.shutdown()
        self._shape_count = 0
        self._cook_warning = ""
        self._ccd_enabled = bool(ccd_enabled)
        self._unstable = False
        self._emit_progress(4, "Preparing OVPhysX collider cook...")
        self._body_patterns = self._normalize_patterns(body_patterns)
        self._body_paths = self._normalize_optional_paths(body_paths)
        self._articulation_paths = self._normalize_optional_paths(articulation_paths)
        self._instance_paths = self._normalize_optional_paths(instance_paths)
        self._instance_reference_poses = self._normalize_pose_map(instance_reference_poses)
        self._bound_body_paths = []
        self._body_index_by_path = {}
        self._active_body_index = 0
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
        device = os.environ.get("SIMREADY_OVPHYSX_DEVICE", "cpu").strip().lower() or "cpu"
        if device.startswith("cuda"):
            device = "gpu"
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
        self._clone_runtime_instances(clone_groups, clone_source_path, clone_target_paths, clone_parent_poses)

        self._emit_progress(58, "Discovering authored rigid bodies...")
        self._pose_binding = self._create_pose_binding(TensorType.RIGID_BODY_POSE)
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
                prim_paths=self._bound_body_paths,
            )
            self._velocity_buffer = np.zeros(self._velocity_binding.shape, dtype=np.float32)
        except Exception:
            self._velocity_binding = None
            self._velocity_buffer = None

        try:
            self._wrench_binding = self._create_tensor_binding(
                TensorType.RIGID_BODY_WRENCH,
                patterns=[self._active_body_pattern],
                prim_paths=self._bound_body_paths,
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
                prim_paths=self._bound_body_paths,
                update_active=False,
            )
            self._mass_buffer = np.zeros(self._mass_binding.shape, dtype=np.float32)
            self._effective_mass_kg = self._read_body_mass_kg(self._active_body_index, DEFAULT_GRAB_MASS_KG)
        except Exception:
            self._mass_binding = None
            self._mass_buffer = None

        self._emit_progress(82, "Binding collider shapes from cooked sources...")
        self._shape_count = self._tune_shape_properties(TensorType)
        if self._shape_count <= 0:
            self._shape_count = self._force_collider_cook(TensorType)
        if self._shape_count <= 0:
            self._cook_warning = (
                "OVPhysX finished the authored collider cook, but did not expose collider shape tensors yet. "
                "Physics will not play until the authored collider binding is fixed."
            )
            self._emit_progress(92, "Finalizing authored collider cook...")

        self._emit_progress(94, "Finalizing cloned physics instance cache...")
        if cook_only:
            self._emit(
                {
                    "type": "cooked",
                    "body_pattern": self._active_body_pattern,
                    "body_count": self._active_body_count,
                    "body_paths": self._bound_body_paths,
                    "shape_count": self._shape_count,
                    "cook_warning": self._cook_warning,
                    "ccd_enabled": self._ccd_enabled,
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
                "body_paths": self._bound_body_paths,
                "shape_count": self._shape_count,
                "cook_warning": self._cook_warning,
                "ccd_enabled": self._ccd_enabled,
            }
        )
        self._emit_pose()

    def step(self, dt: float, sim_time: float, substeps: int = 1) -> None:
        if self._physx is None:
            raise RuntimeError("Physics scene is not started")
        if self._unstable:
            return
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

    def set_poses(self, poses, zero_velocity: bool) -> None:
        self._write_poses(poses, zero_velocity=zero_velocity, emit=True)

    def _write_pose(self, pose, zero_velocity: bool, emit: bool) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return
        root_pose = np.array(pose, dtype=np.float32).reshape(7)
        if not self._pose_is_valid(root_pose):
            self._fail_unstable("Physics received an invalid reset pose; physics was stopped.")
            return
        root_matrix = self._matrix_from_pose(root_pose)
        reference = self._reference_pose_buffer
        if reference is not None and reference.shape[0] == self._pose_buffer.shape[0]:
            wrote_any = False
            for index in range(self._pose_buffer.shape[0]):
                if not self._pose_is_valid(reference[index, :]):
                    self._pose_buffer[index, :] = root_pose
                    continue
                body_matrix = self._matrix_from_pose(reference[index, :]) @ root_matrix
                self._pose_buffer[index, :] = self._pose_from_matrix(body_matrix)
                wrote_any = True
            if not wrote_any:
                self._pose_buffer[0, :] = root_pose
        else:
            self._pose_buffer[0, :] = root_pose
        self._pose_binding.write(self._pose_buffer)
        if zero_velocity and self._velocity_binding is not None and self._velocity_buffer is not None:
            self._velocity_buffer.fill(0.0)
            self._velocity_binding.write(self._velocity_buffer)
        if emit:
            self._emit_pose()

    def _write_poses(self, poses, zero_velocity: bool, emit: bool) -> None:
        if self._pose_binding is None or self._pose_buffer is None:
            return

        target_poses = self._normalize_pose_map(poses)
        if not target_poses:
            self._fail_unstable("Physics received no valid reset poses; physics was stopped.")
            return

        reference = self._reference_pose_buffer
        if reference is None or reference.shape[0] != self._pose_buffer.shape[0]:
            first_pose = next(iter(target_poses.values()))
            self._write_pose(first_pose, zero_velocity=zero_velocity, emit=emit)
            return

        for index in range(self._pose_buffer.shape[0]):
            body_path = self._bound_body_paths[index] if index < len(self._bound_body_paths) else ""
            target_path = self._instance_path_for_body(body_path, target_poses)
            if not target_path:
                if len(target_poses) == 1:
                    target_path = next(iter(target_poses.keys()))
                else:
                    continue

            target_pose = target_poses[target_path]
            if not self._pose_is_valid(target_pose):
                continue

            if not self._pose_is_valid(reference[index, :]):
                self._pose_buffer[index, :] = target_pose
                continue

            reference_root = self._instance_reference_poses.get(target_path)
            if reference_root is None or not self._pose_is_valid(reference_root):
                if body_path == target_path:
                    self._pose_buffer[index, :] = target_pose
                    continue
                reference_root = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            try:
                reference_body_matrix = self._matrix_from_pose(reference[index, :])
                reference_root_matrix = self._matrix_from_pose(reference_root)
                target_root_matrix = self._matrix_from_pose(target_pose)
                local_matrix = reference_body_matrix @ np.linalg.inv(reference_root_matrix)
                self._pose_buffer[index, :] = self._pose_from_matrix(local_matrix @ target_root_matrix)
            except Exception:
                self._pose_buffer[index, :] = target_pose

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
        if (
            not np.all(np.isfinite(target))
            or not np.all(np.isfinite(anchor))
            or not np.all(np.isfinite(target_velocity))
            or float(np.linalg.norm(target)) > MAX_SIM_POSITION
        ):
            self._magnet = None
            self._clear_wrench()
            self._emit({"type": "error", "message": "Physics grab target became invalid; physics was stopped."})
            self._unstable = True
            return

        body_path = str(message.get("body_path", "") or "")
        body_index = self._body_index_from_path(body_path)
        estimated_mass = self._sanitize_mass(message.get("estimated_mass", self._effective_mass_kg), self._effective_mass_kg)
        mass = self._read_body_mass_kg(body_index, estimated_mass)
        self._effective_mass_kg = mass
        stiffness, damping, max_force, speed_limit, max_angular_accel = self._magnet_response(message, mass)
        force_amount = self._sanitize_force_amount(message.get("force_amount", DEFAULT_GRAB_FORCE_AMOUNT))
        anchor_local = anchor.reshape(3)
        if not body_path:
            anchor_local = self._root_anchor_to_active_body_local(anchor_local)

        self._magnet = {
            "target": target.reshape(3),
            "anchor": anchor_local,
            "target_velocity": self._clamp_vector(target_velocity.reshape(3), speed_limit),
            "body_path": body_path,
            "body_index": body_index,
            "mass": mass,
            "force_amount": force_amount,
            "stiffness": stiffness,
            "damping": damping,
            "max_force": max_force,
            "max_angular_accel": max_angular_accel,
        }

    def release_magnet(self, velocity, angular_velocity=None) -> None:
        body_index = self._clamp_body_index(
            int(self._magnet.get("body_index", self._active_body_index)) if self._magnet else self._active_body_index
        )
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
        if not np.all(np.isfinite(linear)):
            linear = np.zeros(3, dtype=np.float32)
        if not np.all(np.isfinite(angular)):
            angular = np.zeros(3, dtype=np.float32)

        speed_limit = self._release_speed_limit(self._effective_mass_kg)
        self._velocity_binding.read(self._velocity_buffer)
        body_index = self._buffer_body_index(self._velocity_buffer, body_index)
        if self._velocity_buffer.shape[-1] >= 3:
            self._velocity_buffer[body_index, 0:3] = self._clamp_vector(linear.reshape(3), speed_limit)
        if self._velocity_buffer.shape[-1] >= 6:
            self._velocity_buffer[body_index, 3:6] = self._clamp_vector(angular.reshape(3), min(12.0, speed_limit))
        self._velocity_binding.write(self._velocity_buffer)

    def set_ccd_enabled(self, enabled: bool) -> None:
        self._ccd_enabled = bool(enabled)
        mode = "enabled" if self._ccd_enabled else "disabled"
        if self._physx is None:
            self._emit(
                {
                    "type": "ccd",
                    "enabled": self._ccd_enabled,
                    "applied": False,
                    "message": f"CCD {mode}. It will apply when the next physics scene starts.",
                }
            )
            return

        self._emit(
            {
                "type": "ccd",
                "enabled": self._ccd_enabled,
                "applied": False,
                "message": (
                    f"CCD {mode}; current cooked collider cache was left intact. "
                    "OVPhysX does not expose a live CCD tensor, so this setting applies on the next physics scene start."
                ),
            }
        )

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
        self._body_paths = []
        self._articulation_paths = []
        self._instance_paths = []
        self._instance_reference_poses = {}
        self._bound_body_paths = []
        self._body_index_by_path = {}
        self._active_body_index = 0
        self._effective_mass_kg = DEFAULT_GRAB_MASS_KG
        self._unstable = False

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

    def _create_pose_binding(self, tensor_type):
        if self._body_paths:
            binding = None
            try:
                binding = self._physx.create_tensor_binding(prim_paths=self._body_paths, tensor_type=tensor_type)
                count = self._binding_count(binding)
                if count <= 0:
                    raise RuntimeError("No exact rigid bodies matched discovered body paths")
                self._bound_body_paths = self._body_paths[:count]
                self._body_index_by_path = {path: index for index, path in enumerate(self._bound_body_paths)}
                self._active_body_pattern = self._bound_body_paths[0]
                self._active_body_index = 0
                return binding
            except Exception:
                if binding is not None:
                    try:
                        binding.destroy()
                    except Exception:
                        pass

        binding = self._create_best_tensor_binding(tensor_type)
        count = self._binding_count(binding)
        if count == 1 and not self._has_wildcards(self._active_body_pattern):
            self._bound_body_paths = [self._active_body_pattern]
            self._body_index_by_path = {self._active_body_pattern: 0}
        else:
            self._bound_body_paths = []
            self._body_index_by_path = {}
        self._active_body_index = 0
        return binding

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

    def _create_tensor_binding(self, tensor_type, patterns=None, update_active: bool = True, prim_paths=None):
        last_exc: Optional[Exception] = None
        exact_paths = self._normalize_optional_paths(prim_paths)
        if exact_paths:
            binding = None
            try:
                binding = self._physx.create_tensor_binding(prim_paths=exact_paths, tensor_type=tensor_type)
                if self._binding_count(binding) <= 0:
                    raise RuntimeError(f"No rigid bodies matched explicit body paths")
                return binding
            except Exception as exc:
                last_exc = exc
                if binding is not None:
                    try:
                        binding.destroy()
                    except Exception:
                        pass

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
    def _normalize_optional_paths(paths) -> list[str]:
        try:
            raw = list(paths or [])
        except TypeError:
            raw = []
        normalized = []
        for item in raw:
            text = str(item or "").strip()
            if text and "*" not in text and "?" not in text and "[" not in text and text not in normalized:
                normalized.append(text)
        return normalized

    @classmethod
    def _normalize_pose_map(cls, items) -> dict[str, np.ndarray]:
        try:
            raw_items = list(items or [])
        except TypeError:
            raw_items = []
        poses: dict[str, np.ndarray] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "") or "").strip()
            if not path:
                continue
            if not path.startswith("/"):
                path = "/" + path
            try:
                pose = np.array(item.get("pose", []), dtype=np.float32).reshape(7)
            except Exception:
                continue
            if not cls._pose_is_valid(pose):
                continue
            poses[path] = pose
        return poses

    def _clone_runtime_instances(self, clone_groups, source_path: str = "", target_paths=None, parent_poses=None) -> None:
        if self._physx is None:
            return

        groups = self._normalize_clone_groups(clone_groups)
        if not groups:
            source = str(source_path or "").strip()
            if source and not source.startswith("/"):
                source = "/" + source
            targets = [path for path in self._normalize_optional_paths(target_paths) if path and path != source]
            if source and targets:
                groups = [{"source": source, "targets": targets, "parent_poses": list(parent_poses or [])}]
        if not groups:
            return

        clone_count = sum(len(group["targets"]) for group in groups)
        self._emit_progress(44, f"Instancing {clone_count + len(groups)} physics copies from cooked sources...")
        for group in groups:
            source = group["source"]
            targets = group["targets"]
            parent_transforms = self._clone_parent_transforms(group.get("parent_poses", []), len(targets))
            try:
                if parent_transforms:
                    self._physx.clone(source, targets, parent_transforms=parent_transforms)
                else:
                    self._physx.clone(source, targets)
            except TypeError:
                self._physx.clone(source, targets)
        self._physx.wait_all()

    def _normalize_clone_groups(self, groups) -> list[dict]:
        try:
            raw_items = list(groups or [])
        except TypeError:
            raw_items = []
        normalized: list[dict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "") or "").strip()
            if source and not source.startswith("/"):
                source = "/" + source
            targets = [path for path in self._normalize_optional_paths(item.get("targets", [])) if path and path != source]
            if not source or not targets:
                continue
            normalized.append(
                {
                    "source": source,
                    "targets": targets,
                    "parent_poses": list(item.get("parent_poses", []) or []),
                }
            )
        return normalized

    @staticmethod
    def _clone_parent_transforms(poses, count: int) -> list[tuple[float, float, float, float, float, float, float]]:
        try:
            raw_items = list(poses or [])
        except TypeError:
            raw_items = []
        expected = max(0, int(count))
        transforms: list[tuple[float, float, float, float, float, float, float]] = []
        for item in raw_items[:expected]:
            try:
                pose = np.array(item, dtype=np.float32).reshape(7)
            except Exception:
                continue
            if not PhysicsWorker._pose_is_valid(pose):
                continue
            transforms.append(tuple(float(value) for value in pose))
        return transforms if len(transforms) == expected else []

    def _instance_path_for_body(self, body_path: str, target_poses: dict[str, np.ndarray]) -> str:
        text = str(body_path or "").strip()
        if not text:
            return ""
        if not text.startswith("/"):
            text = "/" + text

        candidates = self._instance_paths or list(target_poses.keys())
        best = ""
        for root in candidates:
            root_text = str(root or "").strip()
            if not root_text:
                continue
            if not root_text.startswith("/"):
                root_text = "/" + root_text
            if root_text not in target_poses:
                continue
            if text == root_text or text.startswith(f"{root_text}/"):
                if len(root_text) > len(best):
                    best = root_text
        return best

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
        try:
            self._pose_binding.read(self._pose_buffer)
        except Exception as exc:
            self._fail_unstable(f"Physics pose read failed; physics was stopped. {exc}")
            return
        valid_indices = self._valid_pose_indices(self._pose_buffer)
        if not valid_indices:
            self._fail_unstable("Physics simulation produced no valid rigid body poses; physics was stopped.")
            return
        root_index = valid_indices[0]
        pose = np.array(self._pose_buffer[root_index], dtype=np.float32, copy=True)
        body_path = self._bound_body_paths[root_index] if root_index < len(self._bound_body_paths) else ""
        instance_path = self._instance_path_for_body(body_path, self._instance_reference_poses)
        if (
            instance_path
            and self._reference_pose_buffer is not None
            and self._reference_pose_buffer.shape[0] > root_index
            and self._pose_is_valid(self._reference_pose_buffer[root_index])
            and self._pose_is_valid(self._instance_reference_poses.get(instance_path))
        ):
            try:
                reference_body_matrix = self._matrix_from_pose(self._reference_pose_buffer[root_index])
                reference_root_matrix = self._matrix_from_pose(self._instance_reference_poses[instance_path])
                body_matrix = self._matrix_from_pose(pose)
                local_matrix = reference_body_matrix @ np.linalg.inv(reference_root_matrix)
                root_matrix = np.linalg.inv(local_matrix) @ body_matrix
                pose = self._pose_from_matrix(root_matrix)
            except Exception:
                pass
        elif self._reference_pose_buffer is not None and self._reference_pose_buffer.shape[0] > root_index:
            reference_pose = self._reference_pose_buffer[root_index]
            if self._pose_is_valid(reference_pose):
                try:
                    ref_matrix = self._matrix_from_pose(reference_pose)
                    body_matrix = self._matrix_from_pose(pose)
                    root_matrix = np.linalg.inv(ref_matrix) @ body_matrix
                    pose = self._pose_from_matrix(root_matrix)
                except Exception:
                    pass

        message = {"type": "pose", "pose": pose.astype(float).tolist()}
        bodies = []
        if self._bound_body_paths:
            for index in valid_indices:
                if index >= len(self._bound_body_paths):
                    continue
                path = self._bound_body_paths[index]
                if not path or self._has_wildcards(path):
                    continue
                bodies.append(
                    {
                        "path": path,
                        "pose": np.array(self._pose_buffer[index], dtype=np.float32, copy=True).astype(float).tolist(),
                    }
                )
        if bodies:
            message["bodies"] = bodies
        self._emit(message)

    def _apply_magnet(self, dt: float) -> None:
        if self._magnet is None or self._pose_binding is None or self._pose_buffer is None:
            return

        try:
            self._pose_binding.read(self._pose_buffer)
        except Exception as exc:
            self._fail_unstable(f"Physics pose read failed during grab; physics was stopped. {exc}")
            return
        body_index = self._clamp_body_index(int(self._magnet.get("body_index", self._active_body_index)))
        pose = np.array(self._pose_buffer[body_index, :], dtype=np.float32)
        if not self._pose_is_valid(pose):
            self._fail_unstable("Physics simulation produced an invalid grabbed-body pose; physics was stopped.")
            return
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
            velocity_index = self._buffer_body_index(self._velocity_buffer, body_index)
            if self._velocity_buffer.shape[-1] >= 3:
                linear_velocity = np.array(self._velocity_buffer[velocity_index, 0:3], dtype=np.float32)
            if self._velocity_buffer.shape[-1] >= 6:
                angular_velocity = np.array(self._velocity_buffer[velocity_index, 3:6], dtype=np.float32)
        if not np.all(np.isfinite(linear_velocity)):
            linear_velocity = np.zeros(3, dtype=np.float32)
        if not np.all(np.isfinite(angular_velocity)):
            angular_velocity = np.zeros(3, dtype=np.float32)

        lever = anchor_world - position
        point_velocity = linear_velocity + np.cross(angular_velocity, lever)
        offset = target - anchor_world
        force = (
            offset * float(self._magnet["stiffness"])
            + (target_velocity - point_velocity) * float(self._magnet["damping"])
        )
        if not np.all(np.isfinite(force)):
            self._fail_unstable("Physics grab force became invalid; physics was stopped.")
            return
        mass = max(float(self._magnet.get("mass", self._effective_mass_kg)), MIN_GRAB_MASS_KG)
        force = self._clamp_grab_force(force, mass, lever)
        torque = self._clamp_grab_torque(np.cross(lever, force), mass, lever)
        if not np.all(np.isfinite(force)) or not np.all(np.isfinite(torque)):
            self._fail_unstable("Physics grab force became invalid; physics was stopped.")
            return

        if self._wrench_binding is not None and self._wrench_buffer is not None:
            wrench_index = self._buffer_body_index(self._wrench_buffer, body_index)
            self._wrench_buffer.fill(0.0)
            if self._wrench_buffer.shape[-1] >= 3:
                self._wrench_buffer[wrench_index, 0:3] = force
            if self._wrench_buffer.shape[-1] >= 6:
                self._wrench_buffer[wrench_index, 3:6] = torque
            if self._wrench_buffer.shape[-1] >= 9:
                self._wrench_buffer[wrench_index, 6:9] = anchor_world
            self._wrench_binding.write(self._wrench_buffer)
            return

        if self._velocity_binding is not None and self._velocity_buffer is not None:
            velocity_index = self._buffer_body_index(self._velocity_buffer, body_index)
            desired_linear = linear_velocity + (force / mass) * dt
            lever_length = max(float(np.linalg.norm(lever)), 0.05)
            approximate_inertia = max(mass * lever_length * lever_length, 0.001)
            desired_angular = angular_velocity + (torque / approximate_inertia) * dt
            self._velocity_buffer.fill(0.0)
            self._velocity_buffer[velocity_index, 0:3] = self._clamp_vector(desired_linear, self._release_speed_limit(mass))
            if self._velocity_buffer.shape[-1] >= 6:
                self._velocity_buffer[velocity_index, 3:6] = self._clamp_vector(desired_angular, 10.0)
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

    def _read_body_mass_kg(self, body_index: int, fallback: float) -> float:
        if self._mass_binding is None or self._mass_buffer is None:
            return self._sanitize_mass(fallback, DEFAULT_GRAB_MASS_KG)

        try:
            self._mass_binding.read(self._mass_buffer)
            mass_array = np.asarray(self._mass_buffer, dtype=np.float64)
        except Exception:
            return self._sanitize_mass(fallback, DEFAULT_GRAB_MASS_KG)

        if mass_array.ndim <= 0:
            values = mass_array.reshape(-1)
        else:
            index = self._buffer_body_index(mass_array, body_index)
            values = mass_array[index].reshape(-1)
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size <= 0:
            return self._sanitize_mass(fallback, DEFAULT_GRAB_MASS_KG)

        return self._sanitize_mass(float(np.sum(finite)), fallback)

    def _body_index_from_path(self, body_path: str) -> int:
        if body_path:
            direct = self._body_index_by_path.get(body_path)
            if direct is not None:
                return self._clamp_body_index(direct)
            for known_path, index in self._body_index_by_path.items():
                if known_path.endswith(body_path) or body_path.endswith(known_path):
                    return self._clamp_body_index(index)
        return self._clamp_body_index(self._active_body_index)

    def _clamp_body_index(self, index: int) -> int:
        count = 1
        if self._pose_buffer is not None and getattr(self._pose_buffer, "ndim", 0) > 0:
            count = max(1, int(self._pose_buffer.shape[0]))
        return max(0, min(int(index), count - 1))

    @staticmethod
    def _buffer_body_index(buffer: np.ndarray, index: int) -> int:
        try:
            count = int(buffer.shape[0])
        except Exception:
            count = 1
        return max(0, min(int(index), max(1, count) - 1))

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

    @classmethod
    def _valid_pose_indices(cls, poses) -> list[int]:
        try:
            arr = np.asarray(poses)
            count = int(arr.shape[0])
        except Exception:
            return []
        valid = []
        for index in range(count):
            if cls._pose_is_valid(arr[index]):
                valid.append(index)
        return valid

    def _fail_unstable(self, message: str) -> None:
        if self._unstable:
            return
        self._unstable = True
        self._magnet = None
        self._clear_wrench()
        self._emit({"type": "error", "message": message})

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
        shape_prim_paths = self._bound_body_paths if self._bound_body_paths else None
        try:
            contact_binding = self._create_tensor_binding(
                tensor_types.RIGID_BODY_CONTACT_OFFSET,
                patterns=shape_patterns,
                prim_paths=shape_prim_paths,
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
                    prim_paths=shape_prim_paths,
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
                    prim_paths=shape_prim_paths,
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
