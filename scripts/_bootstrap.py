"""Locate this checkout and optional Blender-version-specific dependencies."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    dependency_dir = root / ".blender-deps" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir))
    sys.path.insert(0, str(root))
    return root
