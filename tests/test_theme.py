import unittest

from styles import nvidia_theme


class ThemeTests(unittest.TestCase):
    def test_dropdown_text_uses_white_on_dark_dropdowns(self):
        stylesheet = nvidia_theme.get_stylesheet()

        self.assertIn("QComboBox {", stylesheet)
        self.assertIn(f"background-color: {nvidia_theme.COLOR_BG_WIDGET};", stylesheet)
        self.assertIn(f"color: {nvidia_theme.COLOR_TEXT_PRIMARY};", stylesheet)
        self.assertIn("QComboBox QAbstractItemView::item", stylesheet)
        self.assertIn(f"selection-color: {nvidia_theme.COLOR_TEXT_PRIMARY};", stylesheet)


if __name__ == "__main__":
    unittest.main()
