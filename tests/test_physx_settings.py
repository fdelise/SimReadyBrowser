import unittest

import numpy as np

from core.physics_controller import (
    AuthoredColliderDiscovery,
    DEFAULT_PHYSX_DEVICE_MODE,
    DEFAULT_PHYSX_STEPS_PER_SECOND,
    DEFAULT_SUBSTEPS,
    PhysicsController,
)


class PhysXSettingsTest(unittest.TestCase):
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

    def test_authored_rigid_bodies_get_body_level_ccd_override(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)
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
        self.assertIn("bool physxRigidBody:enableCCD = 0", block)

    def test_articulated_body_paths_get_global_ccd_override(self):
        controller = PhysicsController()
        self.addCleanup(controller.shutdown)
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
        self.assertIn("bool physxRigidBody:enableCCD = 0", block)


if __name__ == "__main__":
    unittest.main()
