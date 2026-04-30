from __future__ import annotations

import json
import subprocess
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


def _write_multi_drop_scene(temp_dir: Path, asset_path: str | Path, count: int = 4) -> Path:
    scene = temp_dir / "multi_drop_scene.usda"
    asset_ref = _asset_reference(asset_path)
    asset_blocks = []
    for index in range(max(1, int(count))):
        name = "Asset" if index == 0 else f"Asset_{index + 1:02d}"
        x = (index % 2) * 0.25
        y = ((index // 2) % 2) * 0.25
        z = 1.4 + index * 0.55
        asset_blocks.append(
            f"""    def Xform "{name}" (
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
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""
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

{chr(10).join(asset_blocks)}

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


def _write_jointed_multibody_scene(temp_dir: Path) -> Path:
    scene = temp_dir / "jointed_multibody_scene.usda"
    scene.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    kilogramsPerMass = 1
)

def Xform "World"
{
    def PhysicsScene "PhysicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }

    def Xform "Asset"
    {
        def Xform "Base" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            bool physics:rigidBodyEnabled = 1
            bool physics:kinematicEnabled = 1
            bool physics:startsAsleep = 0
            float physics:mass = 100
            double3 xformOp:translate = (0, 0, 0.7)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Cube "BaseCollider" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {
                float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
                double size = 1
                double3 xformOp:scale = (0.12, 1.0, 1.2)
                uniform token[] xformOpOrder = ["xformOp:scale"]
            }
        }

        def Xform "Door" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            bool physics:rigidBodyEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:startsAsleep = 0
            float physics:mass = 8
            vector3f physics:velocity = (0, 0, 0)
            vector3f physics:angularVelocity = (0, 0, 0)
            double3 xformOp:translate = (0.5, 0, 0.7)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Cube "DoorCollider" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {
                float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
                double size = 1
                double3 xformOp:translate = (0.32, 0, 0)
                double3 xformOp:scale = (0.64, 0.08, 1.0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
        }

        def PhysicsRevoluteJoint "DoorHinge"
        {
            rel physics:body0 = </World/Asset/Base>
            rel physics:body1 = </World/Asset/Door>
            point3f physics:localPos0 = (0.5, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            token physics:axis = "Z"
            float physics:lowerLimit = -120
            float physics:upperLimit = 120
        }
    }

    def Xform "Ground"
    {
        def Cube "GroundGeom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double size = 1
            double3 xformOp:translate = (0, 0, -0.25)
            double3 xformOp:scale = (40, 40, 0.5)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
}
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


def _test_multi_referenced_collider_drop(temp_dir: Path) -> None:
    asset = _write_local_authored_asset(temp_dir)
    count = 4
    scene = _write_multi_drop_scene(temp_dir, asset, count=count)
    body_paths = ["/World/Asset" if index == 0 else f"/World/Asset_{index + 1:02d}" for index in range(count)]

    worker, messages = _collecting_worker()
    try:
        worker.start(
            str(scene),
            body_paths,
            None,
            contact_offset=0.02,
            cook_only=False,
            body_paths=body_paths,
        )
        started = _last_message(messages, "started")
        assert int(started["body_count"]) >= count, started
        assert int(started["shape_count"]) > 0, started

        reset_poses = []
        for index, body_path in enumerate(body_paths):
            reset_poses.append(
                {
                    "path": body_path,
                    "pose": [0.15 * (index % 2), 0.15 * (index // 2), 1.8 + index * 0.45, 0.0, 0.0, 0.0, 1.0],
                }
            )
        worker.set_poses(reset_poses, zero_velocity=True)
        reset_message = _last_message(messages, "pose")
        for index, body_path in enumerate(body_paths):
            pose = _body_pose(reset_message, body_path)
            assert pose[2] > 1.5 + index * 0.35, f"{body_path} did not reset without recook: z={pose[2]:.4f}"

        for index in range(180):
            worker.step(1.0 / 60.0, index / 60.0, substeps=4)
        pose_message = _last_message(messages, "pose")
        for body_path in body_paths:
            pose = _body_pose(pose_message, body_path)
            assert pose[2] > 0.35, f"{body_path} fell through floor: final z={pose[2]:.4f}"
    finally:
        worker.shutdown()


def _test_scene_level_ccd_cooks(temp_dir: Path) -> None:
    asset = _write_local_authored_asset(temp_dir)
    scene = _write_authored_scene(temp_dir, asset)
    text = scene.read_text(encoding="utf-8")
    text = text.replace(
        '    def PhysicsScene "PhysicsScene"\n    {',
        '    def PhysicsScene "PhysicsScene" (\n'
        '        prepend apiSchemas = ["PhysxSceneAPI"]\n'
        '    )\n'
        '    {',
    ).replace(
        "        float physics:gravityMagnitude = 9.81",
        "        float physics:gravityMagnitude = 9.81\n"
        "        bool physxScene:enableCCD = 1",
    )
    scene.write_text(text, encoding="utf-8")

    worker, messages = _collecting_worker()
    try:
        initial_pose = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        worker.start(str(scene), AUTHORED_BODY_PATTERNS, initial_pose, contact_offset=0.02, cook_only=True)
        cooked = _last_message(messages, "cooked")
        assert int(cooked["shape_count"]) > 0, cooked
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


def _body_pose(message: dict, path: str) -> np.ndarray:
    for item in message.get("bodies", []) or []:
        if item.get("path") == path:
            return np.array(item["pose"], dtype=np.float32)
    raise AssertionError(f"missing body pose for {path}: {message}")


def _test_jointed_multibody_grab(temp_dir: Path) -> None:
    scene = _write_jointed_multibody_scene(temp_dir)
    worker, messages = _collecting_worker()
    try:
        body_paths = ["/World/Asset/Base", "/World/Asset/Door"]
        worker.start(
            str(scene),
            ["/World/Asset/*", AUTHORED_ASSET_PATH],
            None,
            contact_offset=0.02,
            cook_only=False,
            body_paths=body_paths,
        )
        started = _last_message(messages, "started")
        assert int(started["body_count"]) >= 2, started
        assert started.get("body_paths") == body_paths, started

        start_pose = _last_message(messages, "pose")
        start_base = _body_pose(start_pose, "/World/Asset/Base")
        start_door = _body_pose(start_pose, "/World/Asset/Door")

        worker.set_magnet(
            {
                "body_path": "/World/Asset/Door",
                "target": [1.1, 0.65, 1.1],
                "anchor": [0.55, 0.0, 0.0],
                "target_velocity": [0.0, 0.0, 0.0],
                "estimated_mass": 8.0,
                "natural_frequency": 7.0,
                "damping_ratio": 0.7,
                "max_acceleration": 90.0,
                "max_angular_acceleration": 14.0,
                "force_amount": 3.0,
            }
        )
        for index in range(180):
            worker.step(1.0 / 60.0, index / 60.0, substeps=4)

        end_pose = _last_message(messages, "pose")
        end_base = _body_pose(end_pose, "/World/Asset/Base")
        end_door = _body_pose(end_pose, "/World/Asset/Door")
        base_delta = float(np.linalg.norm(end_base[:3] - start_base[:3]))
        door_delta = float(np.linalg.norm(end_door[:3] - start_door[:3]))
        door_rotation_delta = float(np.linalg.norm(end_door[3:7] - start_door[3:7]))
        assert base_delta < 0.05, f"kinematic base moved unexpectedly: delta={base_delta:.4f}"
        assert door_delta > 0.03 or door_rotation_delta > 0.03, (
            f"door body did not respond to targeted multibody grab: "
            f"translation={door_delta:.4f}, rotation={door_rotation_delta:.4f}"
        )
    finally:
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


def _test_selected_multi_simready_cooks() -> None:
    from PyQt5.QtCore import QCoreApplication

    from core.physics_controller import PhysicsController

    QCoreApplication.instance() or QCoreApplication([])
    assets = [
        (
            "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
            "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Clear_Tape/"
            "sm_tape_clear_a02_01.usd"
        ),
        (
            "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
            "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Tape_Dispenser_002/"
            "sm_tape_dispenser_a02_01.usd"
        ),
    ]
    transforms = []
    for x in (0.0, 0.8):
        matrix = np.eye(4, dtype=np.float64)
        matrix[3, 0] = x
        transforms.append(matrix.tolist())

    controller = PhysicsController()
    bounds = {
        "center": [0.4, 0.0, 0.15],
        "size": [1.2, 0.5, 0.4],
        "extent": 0.75,
        "_multi_asset": True,
        "_asset_count": len(assets),
        "_asset_sources": assets,
        "_asset_layout_transforms": transforms,
    }
    controller.configure_asset(bounds, usd_source=None)
    discoveries = [
        controller._authored_collider_discovery(controller._usd_asset_reference(asset))
        for asset in assets
    ]
    scene = controller._write_authored_scene(discoveries)
    scene_text = scene.read_text(encoding="utf-8")
    assert "</World/Instance_02/Geometry/sm_support_a02_obj_00>" in scene_text
    assert "</World/Instance_02/Geometry/sm_pakingtape_a02_obj_00>" in scene_text

    start_message = {
        "cmd": "start",
        "scene": str(scene),
        "body_patterns": controller._current_body_patterns,
        "body_paths": controller._current_body_paths,
        "articulation_paths": controller._current_articulation_paths,
        "instance_paths": controller._instance_root_paths(len(assets)),
        "instance_reference_poses": controller._instance_reference_poses(),
        "initial_pose": None,
        "contact_offset": 0.02,
        "cook_only": True,
        "ccd_enabled": False,
    }
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "core.physics_worker"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(
        json.dumps(start_message, separators=(",", ":")) + "\n" + json.dumps({"cmd": "shutdown"}) + "\n",
        timeout=240,
    )
    messages = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    cooked_messages = [message for message in messages if message.get("type") == "cooked"]
    assert cooked_messages, f"missing cooked message\nstdout={stdout}\nstderr={stderr}"
    cooked = cooked_messages[-1]
    assert int(cooked.get("body_count", 0)) >= 3, cooked
    assert int(cooked.get("shape_count", 0)) > 0, cooked
    assert "body relationship /World/Asset/Geometry" not in stderr


def _test_controller_instanced_drop_uses_runtime_clones(temp_dir: Path) -> None:
    from PyQt5.QtCore import QCoreApplication

    from core.physics_controller import AuthoredColliderDiscovery, PhysicsController

    QCoreApplication.instance() or QCoreApplication([])
    asset = _write_local_authored_asset(temp_dir)
    controller = PhysicsController()
    controller.configure_asset(
        {"center": [0, 0, 0], "size": [1, 1, 1], "extent": 1},
        usd_source=str(asset),
    )
    transforms = []
    for index in range(4):
        matrix = np.eye(4, dtype=np.float64)
        matrix[3, 0] = float(index) * 1.5
        matrix[3, 2] = 2.0 + float(index) * 0.05
        transforms.append(matrix)
    controller._authored_scene_instance_transforms = transforms
    controller._last_start_instance_count = len(transforms)
    controller._active_instance_count = len(transforms)

    collision_overrides = """
        over "AuthoredCollider" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            bool physics:rigidBodyEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:startsAsleep = 0
            float physics:mass = 10
            vector3f physics:velocity = (0, 0, 0)
            vector3f physics:angularVelocity = (0, 0, 0)
        }
"""
    discovery = AuthoredColliderDiscovery(
        collision_overrides,
        ["/World/Asset/AuthoredCollider"],
        ["/World/Asset/AuthoredCollider"],
        [],
        1,
        1,
    )
    scene = controller._write_authored_scene(discovery)
    scene_text = scene.read_text(encoding="utf-8")
    assert scene_text.count("prepend references") == 1, "drop scene should reference the source asset once"
    assert controller._authored_collider_count == 1, "collider count should report source colliders, not cloned instances"
    assert controller._runtime_clone_source_path == "/World/Asset"
    assert controller._runtime_clone_target_paths == ["/World/Instance_02", "/World/Instance_03", "/World/Instance_04"]

    start_message = {
        "cmd": "start",
        "scene": str(scene),
        "body_patterns": controller._current_body_patterns,
        "body_paths": controller._current_body_paths,
        "articulation_paths": controller._current_articulation_paths,
        "instance_paths": controller._instance_root_paths(len(transforms)),
        "instance_reference_poses": controller._instance_reference_poses(),
        "clone_source_path": controller._runtime_clone_source_path,
        "clone_target_paths": controller._runtime_clone_target_paths,
        "clone_parent_poses": [pose.astype(float).tolist() for pose in controller._runtime_clone_parent_poses],
        "initial_pose": None,
        "contact_offset": 0.02,
        "cook_only": True,
        "ccd_enabled": False,
    }
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "core.physics_worker"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(
        json.dumps(start_message, separators=(",", ":")) + "\n" + json.dumps({"cmd": "shutdown"}) + "\n",
        timeout=120,
    )
    messages = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    cooked_messages = [message for message in messages if message.get("type") == "cooked"]
    assert cooked_messages, f"missing cooked message\nstdout={stdout}\nstderr={stderr}"
    cooked = cooked_messages[-1]
    assert int(cooked.get("body_count", 0)) >= 4, cooked
    assert int(cooked.get("shape_count", 0)) > 0, cooked
    assert "Failed to queue clone" not in stdout + stderr


def _test_https_simready_clone_drop() -> None:
    from PyQt5.QtCore import QCoreApplication

    from core.physics_controller import PhysicsController

    QCoreApplication.instance() or QCoreApplication([])
    asset = (
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Clear_Tape/"
        "sm_tape_clear_a02_01.usd"
    )
    controller = PhysicsController()
    controller.configure_asset(
        {"center": [0, 0, 0.1], "size": [0.35, 0.2, 0.15], "extent": 0.35},
        usd_source=asset,
    )
    transforms = []
    for index in range(4):
        matrix = np.eye(4, dtype=np.float64)
        matrix[3, 0] = float(index % 2) * 0.45
        matrix[3, 1] = float(index // 2) * 0.45
        matrix[3, 2] = 1.2 + float(index) * 0.1
        transforms.append(matrix)
    controller._authored_scene_instance_transforms = transforms
    controller._last_start_instance_count = len(transforms)
    controller._active_instance_count = len(transforms)
    discovery = controller._authored_collider_discovery(controller._usd_asset_reference(asset))
    scene = controller._write_authored_scene(discovery)
    scene_text = scene.read_text(encoding="utf-8")
    assert scene_text.count("prepend references") == 1, "SimReady clone-drop scene should reference the source once"
    assert controller._authored_collider_count == discovery.collider_count
    assert len(controller._runtime_clone_target_paths) == len(transforms) - 1

    start_message = {
        "cmd": "start",
        "scene": str(scene),
        "body_patterns": controller._current_body_patterns,
        "body_paths": controller._current_body_paths,
        "articulation_paths": controller._current_articulation_paths,
        "instance_paths": controller._instance_root_paths(len(transforms)),
        "instance_reference_poses": controller._instance_reference_poses(),
        "clone_source_path": controller._runtime_clone_source_path,
        "clone_target_paths": controller._runtime_clone_target_paths,
        "clone_parent_poses": [pose.astype(float).tolist() for pose in controller._runtime_clone_parent_poses],
        "initial_pose": None,
        "contact_offset": 0.02,
        "cook_only": True,
        "ccd_enabled": False,
    }
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "core.physics_worker"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(
        json.dumps(start_message, separators=(",", ":")) + "\n" + json.dumps({"cmd": "shutdown"}) + "\n",
        timeout=240,
    )
    messages = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    cooked_messages = [message for message in messages if message.get("type") == "cooked"]
    assert cooked_messages, f"missing cooked message\nstdout={stdout}\nstderr={stderr}"
    cooked = cooked_messages[-1]
    assert int(cooked.get("body_count", 0)) >= len(transforms), cooked
    assert int(cooked.get("shape_count", 0)) > 0, cooked
    assert "Failed to queue clone" not in stdout + stderr


def _test_selected_assets_split_drop_uses_clone_groups(temp_dir: Path) -> None:
    from PyQt5.QtCore import QCoreApplication

    from core.physics_controller import AuthoredColliderDiscovery, PhysicsController

    QCoreApplication.instance() or QCoreApplication([])
    asset_a = _write_local_authored_asset(temp_dir)
    asset_b = temp_dir / "authored_collider_asset_b.usda"
    asset_b.write_text(asset_a.read_text(encoding="utf-8"), encoding="utf-8")

    controller = PhysicsController()
    bounds = {
        "center": [0.5, 0.5, 0.5],
        "size": [1, 1, 1],
        "extent": 1,
        "_asset_sources": [str(asset_a), str(asset_b)],
        "_asset_bounds": [
            {"center": [0, 0, 0.5], "size": [1.0, 1.0, 1.0], "extent": 0.9},
            {"center": [0, 0, 0.2], "size": [0.5, 0.4, 0.4], "extent": 0.45},
        ],
        "_asset_layout_transforms": [np.eye(4).tolist(), np.eye(4).tolist()],
    }
    controller.configure_asset(bounds, usd_source=None)
    controller.set_drop_options(0.55, 1.6)
    controller._authored_scene_asset_indices = controller._drop_asset_indices(5)
    transforms = controller._drop_visual_transforms(5)
    for index, transform in enumerate(transforms):
        source_index = controller._authored_scene_asset_indices[index]
        assert controller._drop_aabb(transform, source_index)[0][2] > 0.01
    controller._authored_scene_instance_transforms = transforms
    controller._last_start_instance_count = len(transforms)
    controller._active_instance_count = len(transforms)

    discovery = AuthoredColliderDiscovery(
        """
        over "AuthoredCollider" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            bool physics:rigidBodyEnabled = 1
            bool physics:kinematicEnabled = 0
            bool physics:startsAsleep = 0
            float physics:mass = 10
            vector3f physics:velocity = (0, 0, 0)
            vector3f physics:angularVelocity = (0, 0, 0)
        }
""",
        ["/World/Asset/AuthoredCollider"],
        ["/World/Asset/AuthoredCollider"],
        [],
        1,
        1,
    )
    scene = controller._write_authored_scene([discovery, discovery])
    scene_text = scene.read_text(encoding="utf-8")
    assert scene_text.count("prepend references") == 2, "selected drop should reference each selected source once"
    assert controller._authored_collider_count == 2, "collider count should report selected source colliders, not drop copies"
    assert controller._authored_scene_asset_indices == [0, 1, 0, 1, 0]
    assert len(controller._runtime_clone_groups) == 2
    assert controller._runtime_clone_groups[0]["source"] == "/World/Asset"
    assert controller._runtime_clone_groups[0]["targets"] == ["/World/Instance_03", "/World/Instance_05"]
    assert controller._runtime_clone_groups[1]["source"] == "/World/Instance_02"
    assert controller._runtime_clone_groups[1]["targets"] == ["/World/Instance_04"]

    start_message = {
        "cmd": "start",
        "scene": str(scene),
        "body_patterns": controller._current_body_patterns,
        "body_paths": controller._current_body_paths,
        "articulation_paths": controller._current_articulation_paths,
        "instance_paths": controller._instance_root_paths(len(transforms)),
        "instance_reference_poses": controller._instance_reference_poses(),
        "clone_groups": controller._runtime_clone_groups_payload(),
        "initial_pose": None,
        "contact_offset": 0.02,
        "cook_only": True,
        "ccd_enabled": False,
    }
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "core.physics_worker"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(
        json.dumps(start_message, separators=(",", ":")) + "\n" + json.dumps({"cmd": "shutdown"}) + "\n",
        timeout=120,
    )
    messages = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    cooked_messages = [message for message in messages if message.get("type") == "cooked"]
    assert cooked_messages, f"missing cooked message\nstdout={stdout}\nstderr={stderr}"
    cooked = cooked_messages[-1]
    assert int(cooked.get("body_count", 0)) >= len(transforms), cooked
    assert int(cooked.get("shape_count", 0)) > 0, cooked
    assert "Failed to queue clone" not in stdout + stderr


def main() -> int:
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()
        if mode == "drop":
            with tempfile.TemporaryDirectory(prefix="simready_physx_authored_") as temp:
                _test_referenced_collider_drop(Path(temp))
            print("authored referenced-collider drop test passed")
            return 0
        if mode == "multi":
            with tempfile.TemporaryDirectory(prefix="simready_physx_multi_") as temp:
                _test_multi_referenced_collider_drop(Path(temp))
            print("authored multi-instance drop test passed")
            return 0
        if mode == "grab":
            with tempfile.TemporaryDirectory(prefix="simready_physx_grab_") as temp:
                _test_corner_grab_lifts_and_tumbles(Path(temp))
            print("authored corner-grab force test passed")
            return 0
        if mode == "ccd":
            with tempfile.TemporaryDirectory(prefix="simready_physx_ccd_") as temp:
                _test_scene_level_ccd_cooks(Path(temp))
            print("scene-level CCD authored-collider cook test passed")
            return 0
        if mode == "ramp":
            with tempfile.TemporaryDirectory(prefix="simready_physx_ramp_") as temp:
                _test_ramp_wedge_scene_drop(Path(temp))
            print("ramp convex-wedge scene drop test passed")
            return 0
        if mode == "multibody":
            with tempfile.TemporaryDirectory(prefix="simready_physx_multibody_") as temp:
                _test_jointed_multibody_grab(Path(temp))
            print("jointed multibody targeted-grab test passed")
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
        if mode == "selected-multi":
            _test_selected_multi_simready_cooks()
            print("selected multi-asset authored-collider cook test passed")
            return 0
        if mode == "clone-drop":
            with tempfile.TemporaryDirectory(prefix="simready_physx_clone_drop_") as temp:
                _test_controller_instanced_drop_uses_runtime_clones(Path(temp))
            print("clone-backed multi-drop cook test passed")
            return 0
        if mode == "https-clone-drop":
            _test_https_simready_clone_drop()
            print("HTTPS SimReady clone-backed multi-drop cook test passed")
            return 0
        if mode == "selected-clone-drop":
            with tempfile.TemporaryDirectory(prefix="simready_physx_selected_clone_drop_") as temp:
                _test_selected_assets_split_drop_uses_clone_groups(Path(temp))
            print("selected-asset split clone-drop cook test passed")
            return 0
        raise SystemExit(f"unknown mode: {mode}")

    with tempfile.TemporaryDirectory(prefix="simready_physx_authored_") as temp:
        _test_referenced_collider_drop(Path(temp))
    _test_cached_simready_cooks()
    print("authored collider cook and drop tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
