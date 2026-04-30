"""Async-friendly collider discovery entry point for PhysicsController.

This module is launched in a separate Python process so slow OpenUSD traversal
or SimReady payload reads never block the Qt UI thread.
"""

from __future__ import annotations

import json
import sys

from PyQt5.QtCore import QCoreApplication

from core.physics_controller import PhysicsController


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m core.physics_collider_discovery <asset-ref> [root-path]\n"
            "   or: python -m core.physics_collider_discovery --multi <asset-ref>..."
        )

    # PhysicsController owns the current discovery code. Construct it in this
    # subprocess so its synchronous USD work is isolated from the UI process.
    _app = QCoreApplication.instance() or QCoreApplication([])
    controller = PhysicsController()
    if argv[1] == "--multi":
        asset_refs = [item for item in argv[2:] if str(item or "").strip()]
    else:
        asset_refs = [argv[1]]

    discoveries = [controller._authored_collider_discovery(asset_ref) for asset_ref in asset_refs]
    controller.shutdown()

    def payload(discovery):
        return {
            "collision_overrides": discovery.collision_overrides,
            "body_patterns": discovery.body_patterns,
            "body_paths": discovery.body_paths,
            "articulation_paths": discovery.articulation_paths,
            "collider_count": discovery.collider_count,
            "override_count": discovery.override_count,
        }

    if argv[1] == "--multi":
        print(
            json.dumps(
                {
                    "discoveries": [payload(discovery) for discovery in discoveries],
                },
                separators=(",", ":"),
            )
        )
        return 0

    discovery = discoveries[0]
    print(
        json.dumps(
            payload(discovery),
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
