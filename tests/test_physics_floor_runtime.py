from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np


# App repository root.
ROOT = Path(__file__).resolve().parents[1]
REQUIRE_RUNTIME_ENV = "SIMREADY_REQUIRE_OVPHYSX"
START_TIMEOUT_SECONDS = 45.0
STEP_TIMEOUT_SECONDS = 10.0
START_HEIGHT_METERS = 1.25
FALL_THROUGH_LIMIT_Z = -0.25
SIM_FRAME_COUNT = 240
S3_COFFEE_CUP_SOURCE = (
    "s3://omniverse-content-production/"
    "Assets/Isaac/6.0/Isaac/SimReady/Food/Beverage/Coffee_Cup_A01/"
    "sm_food_beverage_coffeeCup_a01_01.usd"
)
COFFEE_CUP_BOUNDS = {
    "center": [0.0, 0.0, 0.06],
    "size": [0.18, 0.18, 0.16],
    "extent": 0.16,
}

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_QT_APP = None


def _qt_app():
    global _QT_APP
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local runtime install
        raise unittest.SkipTest(f"PyQt5 is required for physics scene authoring: {exc}") from exc

    app = QApplication.instance()
    if app is None:
        _QT_APP = QApplication([])
        app = _QT_APP
    return app


def _require_runtime() -> bool:
    value = os.environ.get(REQUIRE_RUNTIME_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _WorkerSession:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "core.physics_worker"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout: queue.Queue[str] = queue.Queue()
        self.stderr_lines: list[str] = []
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def send(self, message: dict) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("Physics worker stdin is closed")
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def read_until(self, wanted: set[str], timeout: float) -> dict:
        deadline = time.monotonic() + float(timeout)
        last_non_json = ""
        while time.monotonic() < deadline:
            try:
                line = self.stdout.get(timeout=0.05)
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                last_non_json = line
                continue

            if message.get("type") in wanted:
                return message

        stderr = self.stderr_text()
        return_code = self.proc.poll()
        raise TimeoutError(
            f"Timed out waiting for {sorted(wanted)} from physics worker "
            f"(returncode={return_code}, last_stdout={last_non_json!r}, stderr={stderr!r})"
        )

    def stderr_text(self) -> str:
        return "\n".join(self.stderr_lines[-40:])

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.send({"cmd": "shutdown"})
                    self.proc.wait(timeout=5.0)
                except Exception:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5.0)
        finally:
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            for thread in self._threads:
                thread.join(timeout=0.5)

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.stdout.put(line.rstrip("\r\n"))

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\r\n"))


class PhysicsFloorRuntimeTest(unittest.TestCase):
    """Runtime floor checks through the app's USD authoring and OVPhysX worker path."""

    def test_s3_coffee_cup_never_passes_through_box_floor(self):
        controller, start_message = self._build_start_message()
        self.addCleanup(controller.shutdown)

        session = _WorkerSession()
        self.addCleanup(session.close)
        session.send(start_message)

        started = self._read_start(session)
        self.assertGreater(
            int(started.get("shape_count", 0) or 0),
            0,
            f"OVPhysX started without cooked collider shapes: {started}",
        )

        initial_message = self._read_pose_message(session, timeout=STEP_TIMEOUT_SECONDS)
        initial_pose = self._pose_from_message(initial_message)
        min_root_z = float(initial_pose[2])
        min_body_z = self._min_body_z(initial_message)

        for frame in range(SIM_FRAME_COUNT):
            session.send(
                {
                    "cmd": "step",
                    "dt": controller._dt,
                    "time": frame * controller._dt,
                    "substeps": controller._substeps,
                    }
            )
            message = self._read_pose_message(session, timeout=STEP_TIMEOUT_SECONDS)
            pose = self._pose_from_message(message)
            root_z = float(pose[2])
            min_root_z = min(min_root_z, root_z)
            body_z = self._min_body_z(message)
            min_body_z = min(min_body_z, body_z)
            self.assertGreaterEqual(
                root_z,
                FALL_THROUGH_LIMIT_Z,
                f"S3 coffee cup root crossed below the floor escape plane at frame {frame}: "
                f"root_z={root_z:.4f}, pose={pose}",
            )
            self.assertGreaterEqual(
                body_z,
                FALL_THROUGH_LIMIT_Z,
                f"S3 coffee cup body crossed below the floor escape plane at frame {frame}: "
                f"body_z={body_z:.4f}, message={message}",
            )

        self.assertGreaterEqual(min_root_z, FALL_THROUGH_LIMIT_Z)
        self.assertGreaterEqual(min_body_z, FALL_THROUGH_LIMIT_Z)

    def test_authored_scene_floor_is_a_box_collider(self):
        controller, start_message = self._build_start_message()
        self.addCleanup(controller.shutdown)
        scene_text = Path(start_message["scene"]).read_text(encoding="utf-8")

        self.assertIn('def Cube "GroundGeom"', scene_text)
        self.assertIn('uniform token physics:approximation = "boundingCube"', scene_text)
        self.assertNotIn('def Mesh "GroundGeom"', scene_text)

    def _build_start_message(self):
        _qt_app()
        from core.physics_controller import PHYSICS_MODE_AUTHORED, PhysicsController

        controller = PhysicsController()
        controller.configure_asset(COFFEE_CUP_BOUNDS, usd_source=S3_COFFEE_CUP_SOURCE)
        controller.set_base_scene("plane")
        controller.set_steps_per_second(60)
        controller.set_substeps(4)
        controller._physics_mode = PHYSICS_MODE_AUTHORED
        controller._authored_scene_instance_transforms = [np.eye(4, dtype=np.float64)]
        controller._last_start_instance_count = 1
        controller._active_instance_count = 1
        controller._last_start_instance_transforms = [np.eye(4, dtype=np.float64)]

        asset_ref = controller._active_asset_refs()[0]
        discovery = controller._authored_collider_discovery(asset_ref)
        self.assertGreater(
            discovery.collider_count,
            0,
            f"S3 coffee cup should expose authored colliders through the normal SimReady payload path: {asset_ref}",
        )
        scene_path = controller._write_authored_scene(discovery)

        visual = np.eye(4, dtype=np.float64)
        visual[3, 2] = START_HEIGHT_METERS
        initial_pose = controller._pose_from_body(controller._body_from_visual(visual))

        return controller, {
            "cmd": "start",
            "scene": str(scene_path),
            "body_patterns": controller._current_body_patterns,
            "body_paths": controller._current_body_paths,
            "articulation_paths": controller._current_articulation_paths,
            "instance_paths": controller._instance_root_paths(controller._last_start_instance_count),
            "instance_reference_poses": controller._instance_reference_poses(),
            "clone_source_path": controller._runtime_clone_source_path,
            "clone_target_paths": controller._runtime_clone_target_paths,
            "clone_parent_poses": [
                np.array(pose, dtype=np.float32, copy=True).reshape(7).astype(float).tolist()
                for pose in controller._runtime_clone_parent_poses
            ],
            "clone_groups": controller._runtime_clone_groups_payload(),
            "initial_pose": initial_pose.astype(float).tolist(),
            "contact_offset": controller._contact_offset(),
            "cook_only": False,
            "ccd_enabled": controller._ccd_enabled,
            "steps_per_second": controller._steps_per_second,
            "device_mode": controller._physx_device_mode,
        }

    def _read_start(self, session: _WorkerSession) -> dict:
        try:
            message = session.read_until({"started", "error"}, START_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            self._unavailable_or_fail(str(exc), session)

        if message.get("type") == "started":
            return message

        error_text = str(message.get("message", ""))
        if self._is_runtime_unavailable(error_text, session):
            self._unavailable_or_fail(error_text, session)
        self.fail(f"Physics worker failed before simulation start: {error_text}")

    def _read_pose_message(self, session: _WorkerSession, timeout: float) -> dict:
        try:
            message = session.read_until({"pose", "error"}, timeout)
        except TimeoutError as exc:
            self.fail(str(exc))

        if message.get("type") == "error":
            self.fail(f"Physics worker failed during simulation: {message.get('message', '')}")

        self._pose_from_message(message)
        return message

    def _pose_from_message(self, message: dict) -> list[float]:
        pose = message.get("pose", [])
        self.assertEqual(len(pose), 7, f"pose message should contain xyz + quaternion: {message}")
        self.assertTrue(np.all(np.isfinite(np.asarray(pose, dtype=np.float64))), f"pose contains non-finite values: {pose}")
        return [float(value) for value in pose]

    def _min_body_z(self, message: dict) -> float:
        values: list[float] = []
        for item in message.get("bodies", []) or []:
            if not isinstance(item, dict):
                continue
            pose = item.get("pose", [])
            if len(pose) < 3:
                continue
            try:
                z = float(pose[2])
            except Exception:
                continue
            if np.isfinite(z):
                values.append(z)
        self.assertTrue(values, f"pose message should include body poses for the S3 coffee cup: {message}")
        return min(values)

    def _unavailable_or_fail(self, reason: str, session: _WorkerSession) -> None:
        detail = f"{reason}\nstderr:\n{session.stderr_text()}"
        if _require_runtime():
            self.fail(detail)
        raise unittest.SkipTest(
            f"OVPhysX runtime is unavailable in this environment. "
            f"Set {REQUIRE_RUNTIME_ENV}=1 to make this a hard failure.\n{detail}"
        )

    def _is_runtime_unavailable(self, error_text: str, session: _WorkerSession) -> bool:
        text = f"{error_text}\n{session.stderr_text()}".lower()
        return_code = session.proc.poll()
        return (
            "no module named" in text and "ovphysx" in text
            or "dll load failed" in text
            or "access violation" in text
            or return_code in {-1073741819, 3221225477}
        )


if __name__ == "__main__":
    unittest.main()
