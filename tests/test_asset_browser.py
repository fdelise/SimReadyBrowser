import tempfile
import unittest
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from core.s3_client import AssetInfo, S3Client
from ui.asset_browser import AssetBrowserPanel


_QT_APP = None


def _ensure_qapplication():
    global _QT_APP
    app = QApplication.instance()
    if app is None:
        _QT_APP = QApplication([])
        app = _QT_APP
    return app


class AssetBrowserTests(unittest.TestCase):
    def test_s3_locations_use_compact_folder_dropdown_without_source_tree(self):
        _ensure_qapplication()
        with tempfile.TemporaryDirectory() as temp_dir:
            client = S3Client(cache_dir=Path(temp_dir))
            panel = AssetBrowserPanel(client)
            self.addCleanup(panel.deleteLater)

            self.assertFalse(hasattr(panel, "_source_tree"))

            panel._on_assets_loaded(
                [
                    AssetInfo(
                        name="Coffee Cup",
                        usd_key="Food/Beverage/cup/sm_cup.usd",
                        category="Food / Beverage",
                        bucket="bucket-a",
                        base_url="https://bucket-a.s3.amazonaws.com",
                        source_prefix="Assets/",
                        source_uri="s3://bucket-a/Assets/",
                        source_name="Bucket A",
                    ),
                    AssetInfo(
                        name="Handtruck",
                        usd_key="Robots/handtruck/sm_handtruck.usd",
                        category="Robots",
                        bucket="bucket-b",
                        base_url="https://bucket-b.s3.amazonaws.com",
                        source_prefix="Assets/",
                        source_uri="s3://bucket-b/Assets/",
                        source_name="Bucket B",
                    ),
                ]
            )

            folders = [panel._cat_combo.itemText(index) for index in range(panel._cat_combo.count())]
            self.assertEqual(folders, ["All", "Food / Beverage", "Robots"])

            panel._cat_combo.setCurrentText("Robots")
            panel._filter_assets()
            self.assertEqual([asset.name for asset in panel._visible_assets], ["Handtruck"])


if __name__ == "__main__":
    unittest.main()
