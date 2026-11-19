"""Blender-independent silhouette coverage rasterization."""

from __future__ import annotations

import cv2
import numpy as np


def fill_triangle_union(mask: np.ndarray, triangles: np.ndarray) -> None:
    """Paint an opaque union of integer-pixel triangles into a uint8 mask.

    One fillPoly call over all triangles applies an even-odd contour rule: front
    and back surfaces can cancel and leave a wireframe-like mask. A silhouette
    needs a coverage union, so each convex triangle is filled separately onto
    the same destination. No depth buffer is needed for opaque coverage.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Destination mask must be a two-dimensional uint8 array")
    triangles = np.asarray(triangles, dtype=np.int32)
    if triangles.size == 0:
        return
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("Triangles must have shape (N, 3, 2)")
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 255, lineType=cv2.LINE_8)
