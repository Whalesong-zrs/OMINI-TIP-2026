"""Regression tests for projected front/back triangle coverage cancellation."""

import unittest

import numpy as np

from omini_camera_render.raster import fill_triangle_union


class TriangleCoverageTests(unittest.TestCase):
    def test_duplicate_triangle_preserves_interior(self):
        triangle = np.array([[[2, 2], [13, 2], [2, 13]]], dtype=np.int32)
        once = np.zeros((16, 16), dtype=np.uint8)
        twice = np.zeros_like(once)
        fill_triangle_union(once, triangle)
        fill_triangle_union(twice, np.repeat(triangle, 2, axis=0))
        np.testing.assert_array_equal(once, twice)
        self.assertEqual(int(twice[4, 4]), 255)

    def test_front_and_back_surfaces_form_solid_square(self):
        front = np.array(
            [[[2, 2], [13, 2], [13, 13]], [[2, 2], [13, 13], [2, 13]]],
            dtype=np.int32,
        )
        all_faces = np.concatenate([front, front[:, ::-1]], axis=0)
        result = np.zeros((16, 16), dtype=np.uint8)
        fill_triangle_union(result, all_faces)
        self.assertTrue(np.all(result[3:13, 3:13] == 255))
        self.assertTrue(np.all(result[:2] == 0))

    def test_empty_face_list_keeps_destination_unchanged(self):
        result = np.zeros((8, 8), dtype=np.uint8)
        fill_triangle_union(result, np.empty((0, 3, 2), dtype=np.int32))
        self.assertFalse(result.any())


if __name__ == "__main__":
    unittest.main()
