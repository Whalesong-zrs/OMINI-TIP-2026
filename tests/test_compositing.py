"""Numerical compositing checks; no Blender import or render is required."""
from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from omini_camera_render.compositing import composite_over_background


@unittest.skipIf(np is None, "NumPy is required for numerical compositing checks")
class CompositingTests(unittest.TestCase):
    def test_transparent_foreground_preserves_background(self):
        foreground = np.zeros((4, 4, 4), dtype=np.uint8)
        foreground[..., :3] = (231, 99, 17)
        background = np.full((4, 4, 3), 83, dtype=np.uint8)
        result, shadow = composite_over_background(foreground, foreground, background, 0.5, 0.02)
        np.testing.assert_array_equal(result, background)
        self.assertFalse(shadow.any())

    def test_opaque_foreground_preserves_foreground_color(self):
        foreground = np.empty((4, 4, 4), dtype=np.uint8)
        foreground[:] = (200, 100, 50, 255)
        background = np.full((4, 4, 3), 83, dtype=np.uint8)
        result, shadow = composite_over_background(foreground, foreground, background, 1.0, 0.0)
        np.testing.assert_array_equal(result, foreground[..., :3])
        self.assertFalse(shadow.any())

    def test_partial_alpha_is_applied_exactly_once(self):
        foreground = np.empty((3, 3, 4), dtype=np.uint8)
        foreground[:] = (200, 100, 50, 128)
        background = np.empty((3, 3, 3), dtype=np.uint8)
        background[:] = (40, 60, 80)
        result, shadow = composite_over_background(foreground, foreground, background, 1.0, 0.0)
        expected = np.empty_like(background)
        expected[:] = (120, 80, 65)
        np.testing.assert_array_equal(result, expected)
        self.assertFalse(shadow.any())

    def test_shadow_darkens_background_not_target_or_its_edge(self):
        foreground = np.zeros((7, 7, 4), dtype=np.uint8)
        foreground[3, 3] = (200, 100, 50, 255)
        combined = foreground.copy()
        combined[..., 3] = 128
        combined[3, 3, 3] = 255
        background = np.full((7, 7, 3), 100, dtype=np.uint8)
        result, shadow = composite_over_background(foreground, combined, background, 0.5, 0.0)
        np.testing.assert_array_equal(result[3, 3], (200, 100, 50))
        np.testing.assert_array_equal(result[2, 3], (100, 100, 100))
        np.testing.assert_array_equal(result[0, 0], (75, 75, 75))
        self.assertEqual(int(shadow[3, 3]), 0)
        self.assertEqual(int(shadow[2, 3]), 0)
        self.assertEqual(int(shadow[0, 0]), 64)

    def test_shadow_floor_discards_low_alpha(self):
        foreground = np.zeros((3, 3, 4), dtype=np.uint8)
        combined = foreground.copy()
        combined[..., 3] = 3
        background = np.full((3, 3, 3), 100, dtype=np.uint8)
        result, shadow = composite_over_background(foreground, combined, background, 1.0, 0.02)
        np.testing.assert_array_equal(result, background)
        self.assertFalse(shadow.any())

    def test_invalid_shapes_dtype_and_shadow_options_are_rejected(self):
        foreground = np.zeros((3, 3, 4), dtype=np.uint8)
        background = np.zeros((3, 3, 3), dtype=np.uint8)
        for strength, floor in ((-0.1, 0.0), (1.1, 0.0), (0.5, 1.0), (float("nan"), 0.0)):
            with self.subTest(strength=strength, floor=floor):
                with self.assertRaises(ValueError):
                    composite_over_background(foreground, foreground, background, strength, floor)
        with self.assertRaises(ValueError):
            composite_over_background(foreground.astype(float), foreground, background, 0.5, 0.0)
        with self.assertRaises(ValueError):
            composite_over_background(foreground, foreground, background[:2], 0.5, 0.0)


if __name__ == "__main__":
    unittest.main()
