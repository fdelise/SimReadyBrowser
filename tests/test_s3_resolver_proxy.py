import unittest

from core.ovrtx_renderer import _rewrite_usd_asset_references_for_s3


class S3ResolverProxyTests(unittest.TestCase):
    def test_case_corrects_coffee_cup_texture_refs_to_actual_s3_objects(self):
        layer = """
        asset inputs:diffuse_texture = @../Textures/t_coffeeCup_brown_a01_alb.png@
        asset inputs:normalmap_texture = @../Textures/t_coffeeCup_brown_a01_nor.png@
        prepend references = @./geometries.usd@
        """
        asset_root = "Assets/Isaac/6.0/Isaac/SimReady/Food/Beverage/Coffee_Cup_A01/"
        object_keys = {
            asset_root + "payloads/materials.usda",
            asset_root + "payloads/geometries.usd",
            asset_root + "textures/t_coffeeCup_brown_a01_alb.png",
            asset_root + "textures/t_coffeeCup_brown_a01_nor.png",
        }

        rewritten, patch_count, refs = _rewrite_usd_asset_references_for_s3(
            layer,
            layer_key=asset_root + "payloads/materials.usda",
            asset_root_key=asset_root,
            object_keys=object_keys,
        )

        self.assertEqual(patch_count, 2)
        self.assertEqual(
            refs,
            {
                asset_root + "textures/t_coffeeCup_brown_a01_alb.png",
                asset_root + "textures/t_coffeeCup_brown_a01_nor.png",
            },
        )
        self.assertIn("@../textures/t_coffeeCup_brown_a01_alb.png@", rewritten)
        self.assertIn("@../textures/t_coffeeCup_brown_a01_nor.png@", rewritten)
        self.assertIn("@./geometries.usd@", rewritten)

    def test_leaves_mdl_source_asset_for_renderer_resolution(self):
        layer = 'asset info:mdl:sourceAsset = @OmniPBR.mdl@'
        asset_root = "Assets/Isaac/6.0/Isaac/SimReady/Food/Beverage/Coffee_Cup_A01/"
        object_keys = {
            asset_root + "payloads/materials.usda",
            asset_root + "source_assets/OmniPBR.mdl",
        }

        rewritten, patch_count, refs = _rewrite_usd_asset_references_for_s3(
            layer,
            layer_key=asset_root + "payloads/materials.usda",
            asset_root_key=asset_root,
            object_keys=object_keys,
        )

        self.assertEqual(patch_count, 0)
        self.assertEqual(refs, set())
        self.assertIn("@OmniPBR.mdl@", rewritten)


if __name__ == "__main__":
    unittest.main()
