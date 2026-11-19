"""Run the standalone camera fitter with Blender --python."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import bootstrap

bootstrap()

from omini_camera_render.camera import main

if __name__ == "__main__":
    main()
