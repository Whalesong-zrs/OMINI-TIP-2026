"""Blender launcher for the standalone renderer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import bootstrap

bootstrap()

from omini_camera_render.render import main

main()
