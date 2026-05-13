import unittest
from unittest import mock

from core import ovrtx_renderer
from core.ovrtx_renderer import DOME_LIGHT_PATH, OVRTXRenderer, _ensure_dome_environment_texture


class DomeEnvironmentTests(unittest.TestCase):
    def test_flat_dome_environment_is_png_texture(self):
        path = _ensure_dome_environment_texture("flat")
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix.lower(), ".png")
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_blurry_studio_environment_is_hdr_texture(self):
        path = _ensure_dome_environment_texture("blurry_studio")
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix.lower(), ".hdr")
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"#?RADIANCE\n"))
        self.assertIn(b"-Y 512 +X 1024", data[:128])
        self.assertIn("softbox", path.name)

    def test_additional_hdri_environments_are_hdr_textures(self):
        for mode, filename_hint in (
            ("automotive_show", "automotive_show"),
            ("outdoor_day", "outdoor_day"),
        ):
            with self.subTest(mode=mode):
                path = _ensure_dome_environment_texture(mode)
                self.assertTrue(path.exists())
                self.assertEqual(path.suffix.lower(), ".hdr")
                data = path.read_bytes()
                self.assertTrue(data.startswith(b"#?RADIANCE\n"))
                self.assertIn(b"-Y 512 +X 1024", data[:128])
                self.assertIn(filename_hint, path.name)

    def test_dome_update_only_writes_existing_intensity_attribute(self):
        calls = []

        class FakePrimMode:
            MUST_EXIST = object()

        class FakeRenderer:
            def write_attribute(self, **kwargs):
                calls.append(kwargs)

            def add_usd_layer(self, *_args, **_kwargs):
                raise AssertionError("dome updates must not add a root layer after stage load")

        class FakeSignal:
            def emit(self, message):
                raise AssertionError(message)

        dummy = OVRTXRenderer.__new__(OVRTXRenderer)
        dummy._renderer = FakeRenderer()
        dummy._stage_loaded = True
        dummy._dome_intensity = 0.75
        dummy._dome_environment = "blurry_studio"
        dummy._dome_texture_warning_shown = False
        dummy.status_changed = FakeSignal()

        with mock.patch.object(ovrtx_renderer, "PrimMode", FakePrimMode):
            OVRTXRenderer._apply_dome_light(dummy)

        self.assertEqual([item["attribute_name"] for item in calls], ["inputs:intensity"])
        self.assertEqual(calls[0]["prim_paths"], [DOME_LIGHT_PATH])

    def test_review_layer_contains_selected_hdr_texture(self):
        class FakeSignal:
            def emit(self, message):
                raise AssertionError(message)

        dummy = OVRTXRenderer.__new__(OVRTXRenderer)
        dummy._width = 320
        dummy._height = 240
        dummy._dome_texture_warning_shown = False
        dummy.status_changed = FakeSignal()

        for mode in ("blurry_studio", "automotive_show", "outdoor_day"):
            with self.subTest(mode=mode):
                dummy._dome_environment = mode
                layer = OVRTXRenderer._review_layer(dummy)

                self.assertIn("asset inputs:texture:file = @", layer)
                self.assertIn(".hdr@", layer)


if __name__ == "__main__":
    unittest.main()
