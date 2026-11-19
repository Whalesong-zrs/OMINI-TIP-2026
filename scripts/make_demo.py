#!/usr/bin/env python3
"""Create a tiny, synthetic alignment/rendering fixture inside Blender.

No paper assets, model downloads, segmentation network, or external API are used.
Run with Blender, not ordinary Python; see the repository README.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Blender's --python executes a file without adding its directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import bootstrap

bootstrap()

import bpy
from mathutils import Vector

from omini_camera_render.common import prepare_output_dir, write_json


def material(name: str, color: tuple[float, float, float]):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = 0.6
    return result


def add_box(name, location, dimensions, surface):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(surface)
    bevel = obj.modifiers.new("SmallEdgeBevel", "BEVEL")
    bevel.width = 0.025
    bevel.segments = 1
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    return obj


def save_pixels(path: Path, width: int, height: int, pixels, scene) -> None:
    image = bpy.data.images.new(path.stem, width, height, alpha=True, float_buffer=True)
    image.pixels.foreach_set(pixels)
    image.update()
    image.save_render(str(path), scene=scene)
    bpy.data.images.remove(image)


def create_fixture(output: Path, resolution: int, samples: int) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2

    # An asymmetric connected-looking assembly, deliberately not rotationally
    # symmetric. Mesh geometry is original and generated locally by this file.
    orange = material("DemoOrange", (0.65, 0.12, 0.025))
    blue = material("DemoBlue", (0.025, 0.17, 0.55))
    parts = [
        add_box("DemoBase", (0.0, 0.0, 0.28), (1.5, 0.8, 0.5), orange),
        add_box("DemoTower", (-0.46, 0.0, 0.95), (0.48, 0.65, 0.95), blue),
        add_box("DemoArm", (0.12, -0.08, 1.33), (1.10, 0.42, 0.25), orange),
    ]
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    target = bpy.context.object
    target.name = "DemoTarget"
    target["omini_target"] = True

    # glTF applies its axis conversion on export and Blender reverses it on
    # import. Selected-only export intentionally excludes the reference camera.
    bpy.ops.export_scene.gltf(
        filepath=str(output / "mesh.glb"), export_format="GLB", use_selection=True
    )

    corners = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
    lower = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    upper = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    center = (lower + upper) * 0.5
    azimuth = math.radians(45.0)
    elevation = math.radians(15.0)
    distance = 3.8
    camera_data = bpy.data.cameras.new("DemoReferenceCamera")
    camera_data.type = "PERSP"
    camera_data.lens = 50.0
    camera_data.sensor_width = 36.0
    camera_data.sensor_height = 24.0
    camera_data.sensor_fit = "AUTO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = center + Vector(
        (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
    )
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (center - camera.location).to_track_quat("-Z", "Y")
    bpy.context.view_layer.update()

    # A tiny, linear HDR environment with a broad bright lobe for contact
    # shadows. This is a synthetic fixture, not an estimated or measured HDR.
    ew, eh = 64, 32
    environment = bpy.data.images.new("DemoEnvironment", ew, eh, alpha=True, float_buffer=True)
    env_pixels = []
    for y in range(eh):
        latitude = math.pi * (y / (eh - 1) - 0.5)
        for x in range(ew):
            longitude = 2.0 * math.pi * x / ew
            lobe = max(0.0, math.cos(longitude - 2.3)) ** 10
            lobe *= max(0.0, math.cos(latitude - 0.7)) ** 16
            energy = 0.18 + 5.0 * lobe
            env_pixels.extend((energy, energy, energy, 1.0))
    environment.pixels.foreach_set(env_pixels)
    environment.update()
    environment.filepath_raw = str(output / "environment.exr")
    environment.file_format = "OPEN_EXR"
    environment.save()
    world = bpy.data.worlds.new("DemoWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    env_node = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env_node.image = environment
    background_node = world.node_tree.nodes.new("ShaderNodeBackground")
    world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(env_node.outputs["Color"], background_node.inputs["Color"])
    world.node_tree.links.new(background_node.outputs["Background"], world_output.inputs["Surface"])

    rgba_path = output / "foreground_rgba.png"
    scene.render.filepath = str(rgba_path)
    bpy.ops.render.render(write_still=True)
    foreground = bpy.data.images.load(str(rgba_path), check_existing=False)
    foreground.alpha_mode = "STRAIGHT"
    rgba = list(foreground.pixels[:])
    count = resolution * resolution
    binary_mask = []
    reference = []
    clean_background = []
    for i in range(count):
        row = i // resolution
        t = row / max(1, resolution - 1)
        clean = (0.07 + 0.025 * t, 0.11 + 0.025 * t, 0.15 + 0.025 * t)
        alpha = min(1.0, max(0.0, rgba[4 * i + 3]))
        selected = 1.0 if alpha >= 0.5 else 0.0
        binary_mask.extend((selected, selected, selected, 1.0))
        clean_background.extend((*clean, 1.0))
        reference.extend(tuple(rgba[4 * i + c] * alpha + clean[c] * (1.0 - alpha) for c in range(3)) + (1.0,))
    save_pixels(output / "mask.png", resolution, resolution, binary_mask, scene)
    save_pixels(output / "background.png", resolution, resolution, clean_background, scene)
    save_pixels(output / "reference.png", resolution, resolution, reference, scene)
    bpy.data.images.remove(foreground)

    # Record the known fixture camera only for debugging. The fit command must
    # derive its own camera from mesh.glb + mask.png, not consume this file.
    manifest = {
        "purpose": "Synthetic smoke-test fixture; not a paper experiment or accuracy benchmark.",
        "resolution": [resolution, resolution],
        "mesh": "mesh.glb",
        "mask": "mask.png",
        "image": "reference.png",
        "background": "background.png",
        "environment": "environment.exr",
        "known_camera_for_debug_only": {
            "azimuth_deg": 45.0,
            "elevation_deg": 15.0,
            "distance": distance,
            "focal_mm": 50.0,
            "target": list(center),
        },
        "mask_policy": "Original image coordinates, binary coverage at alpha >= 0.5; not a recentered cutout.",
        "blender_version": bpy.app.version_string,
    }
    write_json(output / "demo.json", manifest)
    return manifest


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args(argv)
    if args.resolution < 32 or args.samples < 1:
        parser.error("resolution must be >=32 and samples must be >=1")
    output = prepare_output_dir(args.output)
    result = create_fixture(output, args.resolution, args.samples)
    print("OMINI_DEMO_CREATED", json.dumps({"output": str(output), "resolution": result["resolution"]}))


if __name__ == "__main__":
    main()
