from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.physics_worker import PhysicsWorker


AUTHORED_ASSET_PATH = "/World/Asset"
AUTHORED_BODY_PATTERNS = [AUTHORED_ASSET_PATH]


def _collecting_worker() -> tuple[PhysicsWorker, list[dict]]:
    worker = PhysicsWorker()
    messages: list[dict] = []
    worker._emit = lambda message: messages.append(dict(message))  # type: ignore[method-assign]
    return worker, messages


def _write_local_authored_asset(temp_dir: Path) -> Path:
    asset = temp_dir / "authored_collider_asset.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "RootNode"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RootNode"
{
    def Cube "AuthoredCollider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        double size = 1
    }
}
""",
        encoding="utf-8",
    )
    return asset


def _asset_reference(source: str | Path) -> str:
    text = str(source).replace("\\", "/")
    if "://" not in text:
        text = Path(text).resolve().as_posix()
    return text.replace("@", "%40")


def _write_authored_scene(
    temp_dir: Path,
    asset_path: str | Path,
    asset_extra: str = "",
    root_body: bool = True,
) -> Path:
    scene = temp_dir / "authored_scene.usda"
    asset_ref = _asset_reference(asset_path)
    root_api = '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]' if root_body else ""
    root_attrs = (
        """        bool physics:rigidBodyEnabled = 1
        bool physics:kinematicEnabled = 0
        bool physics:startsAsleep = 0
        float physics:mass = 10
        vector3f physics:velocity = (0, 0, 0)
        vector3f physics:angularVelocity = (0, 0, 0)
"""
        if root_body
        else ""
    )
    scene.write_text(
        f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    kilogramsPerMass = 1
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "Asset" (
        prepend references = @{asset_ref}@
{root_api}
    )
    {{
{root_attrs}

{asset_extra}
    }}

    def Xform "Ground"
    {{
        def Cube "GroundGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:translate = (0, 0, -0.25)
            double3 xformOp:scale = (40, 40, 0.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}
}}
""",
        encoding="utf-8",
    )
    return scene


def _write_ramp_wedge_scene(temp_dir: Path, asset_path: str | Path) -> Path:
    scene = temp_dir / "ramp_wedge_scene.usda"
    asset_ref = _asset_reference(asset_path)
    scene.write_text(
        f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    kilogramsPerMass = 1
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "Asset" (
        prepend references = @{asset_ref}@
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:rigidBodyEnabled = 1
        bool physics:kinematicEnabled = 0
        bool physics:startsAsleep = 0
        float physics:mass = 10
        vector3f physics:velocity = (0, 0, 0)
        vector3f physics:angularVelocity = (0, 0, 0)
    }}

    def Xform "Ground"
    {{
        def Cube "GroundGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:translate = (0, 0, -0.25)
            double3 xformOp:scale = (40, 40, 0.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}

    def Xform "Ramp"
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
        }}
    }}
}}
""",
        encoding="utf-8",
    )
    return scene


def _last_message(messages: list[dict], kind: str) -> dict:
    for message in reversed(messages):
        if message.get("type") == kind:
            return message
    raise AssertionError(f"missing {kind} message in {messages}")


def _test_referenced_collider_drop(temp_dir: Path) -> None:
    asset = _write_local_authored_asset(temp_dir)
    scene = _write_authored_scene(temp_dir, asset)

    worker, messages = _collecting_worker()
    try:
        initial_pose = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        worker.start(str(scene), AUTHORED_BODY_PATTERNS, initial_pose, contact_offset=0.02, cook_only=False)
        started = _last_message(messages, "started")
        assert started["body_pattern"] == AUTHORED_ASSET_PATH
        assert int(started["shape_count"]) > 0, started

        for index in range(180):
            worker.step(1.0 / 60.0, index / 60.0, substeps=4)
        pose = np.array(_last_message(messages, "pose")["pose"], dtype=np.float32)
        assert pose[2] > 0.35, f"authored collider fell through floor: final z={pose[2]:.4f}"
    finally:
        worker.shutdown()


def _test_corner_grab_lifts_and_tumbles(temp_dir: Path) -> None:
    asset = _write_local_authored_asset(temp_dir)
    scene = _write_authored_scene(temp_dir, asset)

    worker, messages = _collecting_worker()
    try:
        initial_pose = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        worker.start(str(scene), AUTHORED_BODY_PATTERNS, initial_pose, contact_offset=0.02, cook_only=False)
        started = _last_message(messages, "started")
        assert int(started["shape_count"]) > 0, started

        worker.set_magnet(
            {
                "target": [0.5, 0.5, 3.2],
                "anchor": [0.5, 0.5, 0.5],
                "target_velocity": [0.0, 0.0, 0.0],
                "estimated_mass": 10.0,
                "natural_frequency": 7.0,
                "damping_ratio": 0.7,
                "max_acceleration": 70.0,
                "max_angular_acceleration": 10.0,
            }
        )
        for index in range(160):
            worker.step(1.0 / 60.0, index / 60.0, substeps=4)

        pose = np.array(_last_message(messages, "pose")["pose"], dtype=np.float32)
        tilt = float(np.linalg.norm(pose[3:6]))
        assert pose[2] > 1.25, f"corner grab did not lift the body: final z={pose[2]:.4f}"
        assert tilt > 0.01, f"corner grab did not apply visible torque: tilt={tilt:.4f}"
    finally:
        worker.shutdown()


def _test_ramp_wedge_scene_drop(temp_dir: Path) -> None:
    asset = _write_local_authored_asset(temp_dir)
    worker = None
    try:
        scene = _write_ramp_wedge_scene(temp_dir, asset)
        text = scene.read_text(encoding="utf-8")
        assert "PhysicsCollisionAPI" in text, "ramp scene did not author a collision API"
        assert "PhysxConvexDecompositionCollisionAPI" in text, "ramp scene did not author a convex ramp collider"

        worker, messages = _collecting_worker()
        initial_pose = np.array([0.0, 0.0, 2.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        worker.start(
            str(scene),
            AUTHORED_BODY_PATTERNS,
            initial_pose,
            contact_offset=0.02,
            cook_only=False,
        )
        started = _last_message(messages, "started")
        assert int(started["shape_count"]) > 0, started

        for index in range(180):
            worker.step(1.0 / 60.0, index / 60.0, substeps=4)
        pose = np.array(_last_message(messages, "pose")["pose"], dtype=np.float32)
        assert pose[2] > 0.65, f"ramp collider did not catch the asset above the floor: final z={pose[2]:.4f}"
    finally:
        if worker is not None:
            worker.shutdown()


def _test_cached_simready_cooks() -> None:
    asset = Path("cache") / "inspect_asset.usd"
    if not asset.exists():
        print("cached SimReady asset not present; skipped")
        return

    with tempfile.TemporaryDirectory(prefix="simready_cached_cook_") as temp:
        scene = _write_authored_scene(Path(temp), asset.resolve())

        worker, messages = _collecting_worker()
        try:
            worker.start(
                str(scene),
                [
                    "/World/Asset/Geometry/*",
                    "/World/Asset/Geometry/*/*",
                    "/World/Asset/Geometry/sm_tape_clear_a02_obj_00",
                    *AUTHORED_BODY_PATTERNS,
                ],
                None,
                contact_offset=0.02,
                cook_only=True,
            )
            cooked = _last_message(messages, "cooked")
            assert cooked["body_pattern"] in {AUTHORED_ASSET_PATH, "/World/Asset/Geometry/sm_tape_clear_a02_obj_00"}
            assert int(cooked["shape_count"]) > 0, cooked
        finally:
            worker.shutdown()


def _test_s3_simready_cooks() -> None:
    asset = (
        "s3://omniverse-content-production/Assets/Isaac/6.0/Isaac/SimReady/"
        "Industrial/Hardware/Tapes/Clear_Tape/sm_tape_clear_a02_01.usd"
    )
    with tempfile.TemporaryDirectory(prefix="simready_s3_cook_") as temp:
        scene = _write_authored_scene(Path(temp), asset)

        worker, messages = _collecting_worker()
        try:
            worker.start(
                str(scene),
                [
                    "/World/Asset/Geometry/*",
                    "/World/Asset/Geometry/*/*",
                    "/World/Asset/Geometry/sm_tape_clear_a02_obj_00",
                    *AUTHORED_BODY_PATTERNS,
                ],
                None,
                contact_offset=0.02,
                cook_only=True,
            )
            cooked = _last_message(messages, "cooked")
            assert cooked["body_pattern"] in {AUTHORED_ASSET_PATH, "/World/Asset/Geometry/sm_tape_clear_a02_obj_00"}
            assert int(cooked["shape_count"]) > 0, cooked
        finally:
            worker.shutdown()


def _test_https_simready_cooks(use_overrides: bool = True, drop: bool = False) -> None:
    asset = (
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Clear_Tape/"
        "sm_tape_clear_a02_01.usd"
    )
    with tempfile.TemporaryDirectory(prefix="simready_https_cook_") as temp:
        tape_collision_overrides = """        over "Geometry"
    {
        over "sm_tape_clear_a02_obj_00"
        {
            over "sm_tape_clear_a02_mesh_00" (
                prepend apiSchemas = ["PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]
            )
            {
                uniform token physics:approximation = "convexDecomposition"
                int physxConvexDecompositionCollision:maxConvexHulls = 32
                int physxConvexDecompositionCollision:hullVertexLimit = 64

                over "sm_tape_clear_a02_mesh_00" (
                    prepend apiSchemas = ["PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]
                )
                {
                    uniform token physics:approximation = "convexDecomposition"
                    int physxConvexDecompositionCollision:maxConvexHulls = 32
                    int physxConvexDecompositionCollision:hullVertexLimit = 64
                }
            }
        }
    }""" if use_overrides else ""
        scene = _write_authored_scene(Path(temp), asset, tape_collision_overrides, root_body=False)

        worker, messages = _collecting_worker()
        try:
            initial_pose = (
                np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                if drop
                else None
            )
            worker.start(
                str(scene),
                [
                    "/World/Asset/Geometry/*",
                    "/World/Asset/Geometry/*/*",
                    "/World/Asset/Geometry/sm_tape_clear_a02_obj_00",
                    *AUTHORED_BODY_PATTERNS,
                ],
                initial_pose,
                contact_offset=0.02,
                cook_only=not drop,
            )
            message = _last_message(messages, "started" if drop else "cooked")
            assert message["body_pattern"] in {AUTHORED_ASSET_PATH, "/World/Asset/Geometry/sm_tape_clear_a02_obj_00", "/World/Asset/Geometry/*"}
            assert int(message["shape_count"]) > 0, message
            if drop:
                for index in range(180):
                    worker.step(1.0 / 60.0, index / 60.0, substeps=4)
                pose = np.array(_last_message(messages, "pose")["pose"], dtype=np.float32)
                assert pose[2] > -0.05, f"SimReady asset fell through floor: final z={pose[2]:.4f}"
        finally:
            worker.shutdown()


def main() -> int:
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()
        if mode == "drop":
            with tempfile.TemporaryDirectory(prefix="simready_physx_authored_") as temp:
                _test_referenced_collider_drop(Path(temp))
            print("authored referenced-collider drop test passed")
            return 0
        if mode == "grab":
            with tempfile.TemporaryDirectory(prefix="simready_physx_grab_") as temp:
                _test_corner_grab_lifts_and_tumbles(Path(temp))
            print("authored corner-grab force test passed")
            return 0
        if mode == "ramp":
            with tempfile.TemporaryDirectory(prefix="simready_physx_ramp_") as temp:
                _test_ramp_wedge_scene_drop(Path(temp))
            print("ramp convex-wedge scene drop test passed")
            return 0
        if mode == "cached":
            _test_cached_simready_cooks()
            print("cached SimReady authored-collider cook test passed")
            return 0
        if mode == "s3":
            _test_s3_simready_cooks()
            print("S3 SimReady authored-collider cook test passed")
            return 0
        if mode == "https":
            _test_https_simready_cooks()
            print("HTTPS SimReady authored-collider cook test passed")
            return 0
        if mode == "https-nooverrides":
            _test_https_simready_cooks(use_overrides=False)
            print("HTTPS SimReady authored-collider cook test passed without overrides")
            return 0
        if mode == "https-drop":
            _test_https_simready_cooks(drop=True)
            print("HTTPS SimReady authored-collider drop test passed")
            return 0
        raise SystemExit(f"unknown mode: {mode}")

    with tempfile.TemporaryDirectory(prefix="simready_physx_authored_") as temp:
        _test_referenced_collider_drop(Path(temp))
    _test_cached_simready_cooks()
    print("authored collider cook and drop tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
