import unittest
import tempfile
from pathlib import Path

from core.s3_client import (
    AssetInfo,
    S3_BASE_URL,
    S3_BUCKET,
    S3Client,
    S3Location,
    _category_from_asset_key,
    parse_s3_location,
)


class S3CategoryTests(unittest.TestCase):
    def test_hides_asset_leaf_folder_for_browser_category(self):
        key = (
            "Assets/Isaac/6.0/Isaac/SimReady/"
            "Food/Beverage/food_beverage_coffeeCup_a01/"
            "sm_food_beverage_coffeeCup_a01_01.usd"
        )
        self.assertEqual(_category_from_asset_key(key), "Food / Beverage")

    def test_keeps_parent_folder_for_nested_categories(self):
        key = (
            "Assets/Isaac/6.0/Isaac/SimReady/"
            "Food/Prepared/Bread/food_prepared_bagel_b01/"
            "sm_food_prepared_bagel_b01_01.usd"
        )
        self.assertEqual(_category_from_asset_key(key), "Food / Prepared / Bread")

    def test_manifest_entries_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = S3Client(cache_dir=Path(temp_dir))
            assets = client._parse_manifest(
                [
                    {
                        "name": "Coffee Cup",
                        "path": (
                            "Assets/Isaac/6.0/Isaac/SimReady/"
                            "Food/Beverage/food_beverage_coffeeCup_a01/"
                            "sm_food_beverage_coffeeCup_a01_01.usd"
                        ),
                        "category": "Food / Beverage / food_beverage_coffeeCup_a01",
                    }
                ]
            )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].category, "Food / Beverage")

    def test_parse_custom_s3_location(self):
        location = parse_s3_location("s3://example-bucket/Robots/SimReady")
        self.assertEqual(location.bucket, "example-bucket")
        self.assertEqual(location.prefix, "Robots/SimReady/")
        self.assertEqual(location.root_uri, "s3://example-bucket/Robots/SimReady/")

    def test_parse_default_bucket_uses_known_regional_endpoint(self):
        location = parse_s3_location(f"s3://{S3_BUCKET}/Assets/Isaac/6.0/Isaac/SimReady/")
        self.assertEqual(location.base_url, S3_BASE_URL)

    def test_parse_virtual_host_https_location(self):
        location = parse_s3_location("https://example-bucket.s3.us-west-2.amazonaws.com/Robots/SimReady/")
        self.assertEqual(location.bucket, "example-bucket")
        self.assertEqual(location.prefix, "Robots/SimReady/")
        self.assertEqual(location.base_url, "https://example-bucket.s3.us-west-2.amazonaws.com")

    def test_manifest_assets_keep_custom_source_metadata(self):
        location = S3Location(
            bucket="example-bucket",
            prefix="Robots/SimReady/",
            base_url="https://example-bucket.s3.amazonaws.com",
            name="Robot Shelf",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = S3Client(cache_dir=Path(temp_dir))
            assets = client._parse_manifest(
                [{"path": "Arms/arm_asset/sm_arm.usd", "thumbnail": "Arms/arm_asset/thumbnail.png"}],
                location,
            )
        self.assertEqual(len(assets), 1)
        asset = assets[0]
        self.assertEqual(asset.bucket, "example-bucket")
        self.assertEqual(asset.source_uri, "s3://example-bucket/Robots/SimReady/")
        self.assertEqual(asset.usd_key, "Robots/SimReady/Arms/arm_asset/sm_arm.usd")
        self.assertEqual(asset.category, "Arms")
        self.assertEqual(
            asset.usd_url,
            "https://example-bucket.s3.amazonaws.com/Robots/SimReady/Arms/arm_asset/sm_arm.usd",
        )

    def test_thumbnail_result_sets_local_cache_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = S3Client(cache_dir=Path(temp_dir))
            asset = AssetInfo(
                name="Coffee Cup",
                usd_key="Assets/cup/sm_cup.usd",
                thumbnail_key="Assets/cup/thumb.png",
            )
            request_key = "thumbnail-request"
            client._thumbnail_requests.add(request_key)
            thumb_path = Path(temp_dir) / "thumb.png"

            client._on_thumbnail_done(asset, {"path": thumb_path}, request_key)

            self.assertEqual(asset.local_thumbnail, thumb_path)
            self.assertNotIn(request_key, client._thumbnail_requests)


if __name__ == "__main__":
    unittest.main()
