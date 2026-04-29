from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.physics_controller import PhysicsController


CLEAR_TAPE_ASSET = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Clear_Tape/"
    "sm_tape_clear_a02_01.usd"
)

DISPENSER_ASSET = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Tape_Dispenser_002/"
    "sm_tape_dispenser_a02_01.usd"
)


def _check(asset: str, expected_pattern: str, expected_override: str) -> None:
    controller = PhysicsController()
    discovery = controller._authored_collider_discovery(asset)
    print(asset.rsplit("/", 1)[-1])
    print(f"collider_refs={discovery.collider_count}")
    print(f"convex_override_refs={discovery.override_count}")
    for pattern in discovery.body_patterns[:12]:
        print(f"pattern={pattern}")

    assert discovery.collider_count > 0, "expected authored collision references"
    assert discovery.override_count > 0, "expected SDF collision override references"
    assert any(expected_pattern in pattern for pattern in discovery.body_patterns), "expected discovered object body path"
    assert "convexDecomposition" in discovery.collision_overrides
    assert expected_override in discovery.collision_overrides, "expected collider override path"


def main() -> int:
    _check(CLEAR_TAPE_ASSET, "sm_tape_clear_a02_obj_00", "sm_tape_clear_a02_mesh_00")
    _check(DISPENSER_ASSET, "sm_support_a02_obj_00", "sm_support_a02_mesh_00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
