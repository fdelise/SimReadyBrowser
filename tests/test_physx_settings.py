import unittest

import numpy as np
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

from core.physics_controller import (
    AuthoredColliderDiscovery,
    DEFAULT_PHYSX_DEVICE_MODE,
    DEFAULT_PHYSX_STEPS_PER_SECOND,
    DEFAULT_SUBSTEPS,
    PhysicsController,
)
from ui.controls_panel import (
    PHYSX_DEFAULT_CCD,
    PHYSX_DEFAULT_DEVICE,
    PHYSX_DEFAULT_STEPS_PER_SECOND,
    PHYSX_DEFAULT_SUBSTEPS,
    ControlsPanel,
)


_QT_APP = None


def _ensure_qapplication():
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        _QT_APP = QApplication([])
        app = _QT_APP
    return app


class PhysXSettingsTest(unittest.TestCase):
    def test_controls_panel_exposes_all_hdri_dome_environments(self):
        _ensure_qapplication()
        panel = ControlsPanel()
        self.addCleanup(panel.deleteLater)

        values = [
            panel._dome_environment.itemData(index)
            for index in range(panel._dome_environment.count())
        ]

        self.assertEqual(values, ["flat", "blurry_studio", "automotive_show", "outdoor_day"])

    def test_controls_panel_uses_fast_session_defaults_and_clears_old_persistence(self):
        _ensure_qapplication()
        settings = QSettings("NVIDIA Corporation", "NVIDIA SimReady Browser")
        self.addCleanup(lambda: (settings.remove("physics"), settings.sync()))
        settings.setValue("physics/settings_version", 1)
        settings.setValue("physics/steps_per_second", 15)
        settings.setValue("physics/substeps", 16)
        settings.setValue("physics/device_mode", "cpu")
        settings.setValue("physics/ccd_enabled", True)
        settings.sync()

        panel = ControlsPanel()
        self.addCleanup(panel.deleteLater)

        values = panel.physics_settings()
        self.assertEqual(values["steps_per_second"], PHYSX_DEFAULT_STEPS_PER_SECOND)
        self.assertEqual(values["substeps"], PHYSX_DEFAULT_SUBSTEPS)
        self.assertEqual(values["device_mode"], PHYSX_DEFAULT_DEVICE)
        self.assertEqual(values["ccd_enabled"], PHYSX_DEFAULT_CCD)

        settings.sync()
        self.assertFalse(settings.contains("physics/steps_per_second"))
        self.assertFalse(settings.contains("physics/substeps"))
        self.assertFalse(settings.contains("physics/device_mode"))
        self.assertFalse(settings.contains("physics/ccd_enabled"))

        panel._physics_steps.setValue(120)
        panel._physics_substeps.setValue(8)
        panel._physics_device.setCurrentIndex(panel._physics_device.findData("cpu"))
        panel._physics_ccd.setChecked(True)
        settings.sync()

        self.assertFalse(settings.contains("physics/steps_per_second"))
        self.assertFalse(settings.contains("physics/substeps"))
        self.assertFalse(settings.contains("physics/device_mode"))
        self.assertFalse(settings.contains("physics/ccd_enabled"))

    def test_steps_substeps_and_device_sanitize_and_reset(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)

        self.assertFalse(controller._ccd_enabled)
        self.assertEqual(controller._substeps, DEFAULT_SUBSTEPS)
        controller.set_steps_per_second(120)
        controller.set_substeps(8)
        controller.set_device_mode("cuda:0")
        controller.set_ccd_enabled(True)

        self.assertEqual(controller._steps_per_second, 120)
        self.assertAlmostEqual(controller._dt, 1.0 / 120.0)
        self.assertEqual(controller._substeps, 8)
        self.assertEqual(controller._physx_device_mode, "gpu")
        self.assertTrue(controller._ccd_enabled)

        controller.set_steps_per_second(1)
        controller.set_substeps(99)
        controller.set_device_mode("bogus")

        self.assertEqual(controller._steps_per_second, 15)
        self.assertEqual(controller._substeps, 16)
        self.assertEqual(controller._physx_device_mode, DEFAULT_PHYSX_DEVICE_MODE)

        controller.reset_physx_settings()

        self.assertEqual(controller._steps_per_second, DEFAULT_PHYSX_STEPS_PER_SECOND)
        self.assertAlmostEqual(controller._dt, 1.0 / float(DEFAULT_PHYSX_STEPS_PER_SECOND))
        self.assertEqual(controller._substeps, DEFAULT_SUBSTEPS)
        self.assertEqual(controller._physx_device_mode, DEFAULT_PHYSX_DEVICE_MODE)
        self.assertFalse(controller._ccd_enabled)

    def test_proxy_scene_authors_configured_steps_and_ccd(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)
        controller._size = np.array([0.2, 0.2, 0.2], dtype=np.float64)
        controller._estimated_mass_kg = 1.0
        controller.set_steps_per_second(120)

        path = controller._write_proxy_scene(np.eye(4, dtype=np.float64))
        text = path.read_text(encoding="utf-8")

        self.assertIn("framesPerSecond = 60", text)
        self.assertIn("timeCodesPerSecond = 60", text)
        self.assertIn("uniform uint physxScene:timeStepsPerSecond = 120", text)
        self.assertIn("bool physxScene:enableCCD = 0", text)

        controller.set_ccd_enabled(True)
        path = controller._write_proxy_scene(np.eye(4, dtype=np.float64))
        text = path.read_text(encoding="utf-8")
        self.assertIn("bool physxScene:enableCCD = 1", text)
        self.assertIn("bool physxRigidBody:enableCCD = 1", text)

    def test_authored_rigid_bodies_get_body_level_ccd_override(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)
        controller.set_ccd_enabled(True)
        discovery = AuthoredColliderDiscovery(
            collision_overrides="",
            body_patterns=["/World/Asset/Body"],
            body_paths=["/World/Asset/Body"],
            rigid_body_paths=["/World/Asset/Body"],
            articulation_paths=[],
            collider_count=1,
            override_count=0,
        )

        block = controller._authored_asset_usda_block(
            0,
            "s3://bucket/example.usd",
            "",
            np.eye(4, dtype=np.float64),
            discovery,
        )

        self.assertIn('prepend apiSchemas = ["PhysxRigidBodyAPI"]', block)
        self.assertIn("bool physxRigidBody:enableCCD = 1", block)

    def test_articulated_body_paths_get_global_ccd_override(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)
        controller.set_ccd_enabled(True)
        discovery = AuthoredColliderDiscovery(
            collision_overrides="",
            body_patterns=["/World/Asset/Robot/Link"],
            body_paths=["/World/Asset/Robot/Link"],
            rigid_body_paths=[],
            articulation_paths=["/World/Asset/Robot"],
            collider_count=1,
            override_count=0,
        )

        block = controller._authored_asset_usda_block(
            0,
            "s3://bucket/robot.usd",
            "",
            np.eye(4, dtype=np.float64),
            discovery,
        )

        self.assertIn('over "Robot"', block)
        self.assertIn('over "Link" (', block)
        self.assertIn("bool physxRigidBody:enableCCD = 1", block)

    def test_drop_does_not_clear_enabled_ccd(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)
        controller.configure_asset(
            {"center": [0.0, 0.0, 0.1], "size": [0.2, 0.2, 0.2], "extent": 0.2},
            usd_source="s3://bucket/example.usd",
        )
        controller.set_ccd_enabled(True)
        seen_ccd_values = []

        def fake_restart(*_args, **_kwargs):
            seen_ccd_values.append(controller._ccd_enabled)
            return True

        controller.restart = fake_restart

        self.assertTrue(controller.drop_asset(3))
        self.assertTrue(controller._ccd_enabled)
        self.assertEqual(seen_ccd_values, [True])


if __name__ == "__main__":
    unittest.main()
