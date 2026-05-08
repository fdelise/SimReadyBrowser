import unittest

from core.scene_explorer_model import build_scene_tree, render_path_from_physics_path


class SceneExplorerModelTests(unittest.TestCase):
    def test_maps_physics_body_paths_to_render_tree(self):
        payload = build_scene_tree(
            [{"name": "Handtruck", "source": "s3://bucket/handtruck.usd"}],
            [
                {
                    "body_paths": [
                        "/World/Asset/Geometry/Frame",
                        "/World/Asset/Geometry/Wheels/Left",
                    ],
                    "rigid_body_paths": ["/World/Asset/Geometry/Frame"],
                    "articulation_paths": [],
                    "collider_count": 2,
                }
            ],
        )

        root = payload["roots"][0]
        self.assertEqual(root["path"], "/SimReadyAsset")
        self.assertEqual(root["properties"]["collider_count"], 2)
        geometry = root["children"][0]
        self.assertEqual(geometry["path"], "/SimReadyAsset/Geometry")
        child_paths = {child["path"] for child in geometry["children"]}
        self.assertIn("/SimReadyAsset/Geometry/Frame", child_paths)
        self.assertIn("/SimReadyAsset/Geometry/Wheels", child_paths)

    def test_maps_multi_asset_instance_to_render_root(self):
        self.assertEqual(
            render_path_from_physics_path("/World/Instance_02/Geometry/Cup", asset_index=1),
            "/SimReadyAsset_02/Geometry/Cup",
        )


if __name__ == "__main__":
    unittest.main()
