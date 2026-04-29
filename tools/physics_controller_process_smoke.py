from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QTimer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.physics_controller import PhysicsController


def main() -> int:
    app = QCoreApplication.instance() or QCoreApplication([])
    controller = PhysicsController()
    result = {"ok": False, "message": "timed out"}

    def finish(ok: bool, message: str) -> None:
        result["ok"] = bool(ok)
        result["message"] = str(message)
        app.quit()

    default_asset = (
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/SimReady/Industrial/Hardware/Tapes/Clear_Tape/"
        "sm_tape_clear_a02_01.usd"
    )
    asset = sys.argv[1] if len(sys.argv) > 1 else default_asset
    controller.configure_asset(
        {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "extent": 0.866},
        usd_source=asset,
    )
    controller.cooking_finished.connect(finish)
    controller.status_changed.connect(lambda text: print(f"status: {text}"))
    QTimer.singleShot(120000, lambda: finish(False, "timed out waiting for cook"))

    if not controller.cook_colliders():
        print(controller.status_text)
        controller.shutdown()
        return 1

    app.exec_()
    controller.shutdown()
    print(result["message"])
    return 0 if result["ok"] and "0 usable" not in result["message"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
