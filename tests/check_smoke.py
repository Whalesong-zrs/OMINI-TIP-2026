#!/usr/bin/env python3
"""Check existing fit/render smoke outputs without running Blender.

python tests/check_smoke.py --fit outputs/demo-fit --render outputs/demo-render
No original data are modified. The PNG size check uses the standard library.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def required_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty output: {path}")


def png_size(path: Path) -> tuple[int, int]:
    required_file(path)
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def within(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes output directory: {relative}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.8,
                        help="Synthetic fixture sanity threshold, not a benchmark result.")
    args = parser.parse_args()
    if not 0.0 <= args.min_iou <= 1.0:
        parser.error("--min-iou must lie in [0, 1]")
    fit = args.fit.resolve()
    render = args.render.resolve()
    required_file(fit / "camera.json")
    required_file(fit / "scene.blend")
    camera = json.loads((fit / "camera.json").read_text(encoding="utf-8"))
    if camera.get("schema_version") != 1 or not camera.get("model_mesh_names"):
        raise ValueError("Missing target/schema metadata in camera.json")
    iou = float(camera["score_exact"]["iou"])
    if not math.isfinite(iou) or iou < args.min_iou:
        raise ValueError(f"Synthetic silhouette-fit IoU {iou} is below {args.min_iou}")
    resolution = tuple(camera["resolution"])
    if len(resolution) != 2 or any(not isinstance(x, int) or x < 1 for x in resolution):
        raise ValueError("Invalid fit resolution")
    for name in ("target_mask.png", "aligned_mask.png", "mask_overlay.png"):
        if png_size(fit / name) != resolution:
            raise ValueError(f"Fit image size mismatch: {name}")
    for key in ("camera_matrix_world", "projection_matrix"):
        matrix = camera[key]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError(f"Invalid matrix shape: {key}")
        if any(not math.isfinite(float(value)) for row in matrix for value in row):
            raise ValueError(f"Non-finite matrix: {key}")
    required_file(render / "manifest.json")
    manifest = json.loads((render / "manifest.json").read_text(encoding="utf-8"))
    render_resolution = tuple(manifest["resolution"])
    if len(render_resolution) != 2 or any(not isinstance(x, int) or x < 1 for x in render_resolution):
        raise ValueError("Invalid render resolution")
    views = manifest["views"]
    if not views:
        raise ValueError("No rendered views")
    for view in views:
        files = view["files"]
        for key in ("rgba", "object_rgba", "mask", "depth", "normal", "normal_preview", "albedo", "camera"):
            required_file(within(render, files[key]))
        sizes = {png_size(within(render, files[key]))
                 for key in ("rgba", "object_rgba", "mask", "normal_preview", "albedo")}
        if sizes != {render_resolution}:
            raise ValueError(f"Render pass size mismatch: {view['view_id']}")
        if files.get("composite"):
            if png_size(within(render, files["composite"])) not in sizes:
                raise ValueError(f"Composite size mismatch: {view['view_id']}")
        for relative in files.values():
            if relative:
                required_file(within(render, relative))
    print(json.dumps({"status": "ok", "fit_resolution": list(resolution),
                      "fit_iou": iou, "rendered_views": len(views)}, indent=2))


if __name__ == "__main__":
    main()
