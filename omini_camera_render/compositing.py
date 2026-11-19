"""Blender-independent straight-alpha and shadow compositing helpers.

Array channel order follows OpenCV: BGR for backgrounds and BGRA for PNGs.
PNG color channels are straight (unassociated) alpha, not premultiplied.
"""
from __future__ import annotations

import math

import numpy as np


def composite_over_background(object_bgra: np.ndarray, combined_bgra: np.ndarray,
                              background_bgr: np.ndarray, shadow_strength: float,
                              shadow_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply foreground alpha once; darken background outside the target only.

    The combined image supplies shadow-catcher alpha, not foreground RGB. The
    returned shadow mask includes the requested opacity multiplier. The catcher
    alpha is an artistic opacity, not a calibrated physical shadow measurement.
    """
    if object_bgra.shape != combined_bgra.shape or object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
        raise ValueError("Object and combined renders must be matching BGRA arrays")
    if background_bgr.shape != object_bgra.shape[:2] + (3,):
        raise ValueError("Background and foreground dimensions differ")
    if any(array.dtype != np.uint8 for array in (object_bgra, combined_bgra, background_bgr)):
        raise ValueError("Compositing expects uint8 PNG arrays")
    if not math.isfinite(shadow_strength) or not 0 <= shadow_strength <= 1:
        raise ValueError("shadow_strength must be finite and within [0, 1]")
    if not math.isfinite(shadow_floor) or not 0 <= shadow_floor < 1:
        raise ValueError("shadow_floor must be finite and within [0, 1)")
    alpha = object_bgra[..., 3].astype(np.float32) / 255
    combined_alpha = combined_bgra[..., 3].astype(np.float32) / 255
    # Dilate target support by one pixel so antialiased edges are not darkened
    # twice by the adjacent catcher. This equals a binary 3x3 square dilation.
    height, width = alpha.shape
    padded = np.pad(alpha > 0, 1, mode="constant", constant_values=False)
    object_region = np.zeros((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            object_region |= padded[dy:dy + height, dx:dx + width]
    shadow = np.clip((combined_alpha - shadow_floor) / (1 - shadow_floor), 0, 1) * shadow_strength
    shadow[object_region] = 0
    shadowed_background = background_bgr.astype(np.float32) * (1 - shadow[..., None])
    color = object_bgra[..., :3].astype(np.float32) * alpha[..., None] + shadowed_background * (1 - alpha[..., None])
    return np.clip(np.rint(color), 0, 255).astype(np.uint8), np.clip(np.rint(shadow * 255), 0, 255).astype(np.uint8)
