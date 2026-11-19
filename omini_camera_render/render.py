"""Render a calibrated single-object Blender scene and composite its background.

Extracted and simplified from OMINI's ``render_control_passes.py``. This module
does not estimate illumination, train textures, invoke a VLM, or generate motion.
Run it through Blender using ``scripts/render.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import cv2
import numpy as np
from mathutils import Matrix, Vector

from .common import prepare_output_dir, validate_file, write_json
from .compositing import composite_over_background
from .device import configure_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Calibrated .blend file.")
    parser.add_argument("--camera-json", type=Path, help="Defaults to camera.json next to the scene, when present.")
    parser.add_argument("--background", type=Path, help="Already inpainted/target-removed background; optional.")
    parser.add_argument("--environment", type=Path, help="Equirectangular HDR/EXR environment; optional.")
    parser.add_argument("--strength", type=float, default=1.0, help="Environment strength, when --environment is given.")
    parser.add_argument("--environment-rotation", type=float, default=0.0, help="Environment Z rotation in degrees.")
    parser.add_argument("--keep-scene-lights", action="store_true", help="Keep explicit lights when replacing the world with an HDR environment.")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory; existing files are never deleted.")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4, help="CPU render threads; does not change the global system configuration.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "optix", "metal", "hip"), default="auto")
    parser.add_argument("--width", type=int, help="Output width; height defaults to a proportional size.")
    parser.add_argument("--height", type=int, help="Output height; width defaults to a proportional size.")
    parser.add_argument("--resize-background", action="store_true", help="Explicitly permit background resampling to the output resolution.")
    parser.add_argument("--yaw", type=float, nargs="+", default=[0.0], help="Object yaw(s) around world +Z; camera remains fixed.")
    parser.add_argument("--frame", type=int, default=1, help="Single scene frame to evaluate before applying yaw transforms.")
    parser.add_argument("--shadow-catcher", action="store_true", help="Add a horizontal ground plane at the target's bottom.")
    parser.add_argument("--shadow-strength", type=float, default=0.35)
    parser.add_argument("--shadow-floor", type=float, default=0.02)
    parser.add_argument("--ground-size", type=float, default=6.0, help="Shadow plane side length as a multiple of the target's largest extent.")
    args = parser.parse_args(argv)
    for name in ("strength", "environment_rotation", "shadow_strength", "shadow_floor", "ground_size"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if any(not math.isfinite(yaw) for yaw in args.yaw):
        parser.error("--yaw values must be finite")
    if args.strength < 0 or args.ground_size <= 0:
        parser.error("--strength must be nonnegative and --ground-size must be positive")
    if not 0 <= args.shadow_strength <= 1 or not 0 <= args.shadow_floor < 1:
        parser.error("--shadow-strength must be in [0, 1] and --shadow-floor in [0, 1)")
    if args.samples < 1 or args.threads < 1:
        parser.error("--samples and --threads must be positive integers")
    if any(value is not None and value < 8 for value in (args.width, args.height)):
        parser.error("Output dimensions must be at least 8 pixels")
    return args


def target_meshes(scene, camera_metadata: dict) -> list:
    """Select the calibrated target, not every mesh in an arbitrary .blend."""
    marked = [obj for obj in scene.objects if obj.type == "MESH" and obj.get("omini_target", False)]
    if marked:
        return marked
    names = camera_metadata.get("model_mesh_names", [])
    if not isinstance(names, list) or not names:
        raise ValueError("No calibrated target found. Use a scene from fit_camera or mark target meshes with obj['omini_target'] = True.")
    missing = [name for name in names if name not in scene.objects or scene.objects[name].type != "MESH"]
    if missing:
        raise ValueError(f"Target meshes from camera.json are missing: {missing}")
    return [scene.objects[name] for name in names]


def parent_depth(obj) -> int:
    depth = 0
    while obj.parent is not None:
        depth += 1
        obj = obj.parent
    return depth


def bounds(objects: list) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not corners:
        raise ValueError("Target geometry is empty")
    minimum = Vector([min(point[i] for point in corners) for i in range(3)])
    maximum = Vector([max(point[i] for point in corners) for i in range(3)])
    if max(maximum - minimum) <= 0:
        raise ValueError("Target geometry has zero extent")
    return minimum, maximum


def output_size(scene, metadata: dict, width: int | None, height: int | None) -> tuple[int, int]:
    source = [int(scene.render.resolution_x), int(scene.render.resolution_y)]
    if "resolution" in metadata and list(metadata["resolution"]) != source:
        raise ValueError(f"camera.json resolution {metadata['resolution']} does not match .blend resolution {source}")
    if width is None and height is None:
        return source[0], source[1]
    if width is None:
        width = max(8, round(height * source[0] / source[1]))
    if height is None:
        height = max(8, round(width * source[1] / source[0]))
    if abs(width / height - source[0] / source[1]) > 1.0 / min(width, height):
        raise ValueError("Output dimensions must preserve the calibrated aspect ratio. Specify only --width or --height for proportional resizing.")
    return width, height


def configure_scene(scene, samples: int, width: int, height: int, threads: int) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    # Unlike the historical Medium Contrast setting, None is available in both
    # the Blender 3.6 Filmic and Blender 4.x AgX configurations.
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.use_compositing = True
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_normal = True
    layer.use_pass_diffuse_color = True


def set_environment(scene, path: Path | None, strength: float, rotation: float, keep_lights: bool) -> dict:
    if path is None:
        if scene.world is None:
            world = bpy.data.worlds.new("OMINI_NeutralWorld")
            world.use_nodes = True
            world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
            world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
            scene.world = world
            return {"source": "neutral_world_fallback", "strength": 1.0}
        return {"source": "existing_scene_world", "name": scene.world.name}
    world = bpy.data.worlds.new("OMINI_Environment")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation)
    texture = nodes.new("ShaderNodeTexEnvironment")
    texture.projection = "EQUIRECTANGULAR"
    texture.image = bpy.data.images.load(str(path), check_existing=True)
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputWorld")
    # World shader normals are the environment lookup direction.
    links.new(coordinates.outputs["Normal"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    disabled = []
    if not keep_lights:
        for obj in scene.objects:
            if obj.type == "LIGHT" and not obj.hide_render:
                obj.hide_render = True
                disabled.append(obj.name)
    return {"source": str(path), "strength": strength, "rotation_degrees": rotation, "disabled_scene_lights": disabled}


def add_shadow_catcher(scene, objects: list, size_multiplier: float):
    minimum, maximum = bounds(objects)
    extent = max(maximum - minimum)
    bpy.ops.mesh.primitive_plane_add(
        size=extent * size_multiplier,
        location=((minimum.x + maximum.x) / 2, (minimum.y + maximum.y) / 2, minimum.z),
    )
    plane = bpy.context.object
    plane.name = "OMINI_ShadowCatcher"
    plane.is_shadow_catcher = True
    material = bpy.data.materials.new("OMINI_ShadowCatcherMaterial")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
    material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1
    plane.data.materials.append(material)
    return plane


def add_file_output(tree, socket, directory: Path, name: str, file_format: str, color_mode: str, depth: str) -> None:
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.base_path = str(directory)
    node.file_slots[0].path = name + "_"
    node.format.file_format = file_format
    node.format.color_mode = color_mode
    node.format.color_depth = depth
    tree.links.new(socket, node.inputs[0])


def setup_passes(scene, directory: Path) -> None:
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layer = tree.nodes.new("CompositorNodeRLayers")
    layer.layer = scene.view_layers[0].name
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layer.outputs["Image"], composite.inputs["Image"])
    add_file_output(tree, layer.outputs["Depth"], directory, "depth", "OPEN_EXR", "RGB", "32")
    add_file_output(tree, layer.outputs["Normal"], directory, "normal", "OPEN_EXR", "RGB", "32")
    add_file_output(tree, layer.outputs["DiffCol"], directory, "albedo", "PNG", "RGB", "8")
    scale = tree.nodes.new("CompositorNodeMixRGB")
    scale.blend_type = "MULTIPLY"
    scale.use_clamp = False
    scale.inputs[0].default_value = 1
    scale.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    tree.links.new(layer.outputs["Normal"], scale.inputs[1])
    bias = tree.nodes.new("CompositorNodeMixRGB")
    bias.blend_type = "ADD"
    bias.use_clamp = False
    bias.inputs[0].default_value = 1
    bias.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    tree.links.new(scale.outputs[0], bias.inputs[1])
    add_file_output(tree, bias.outputs[0], directory, "normal_preview", "PNG", "RGB", "8")


def finalize_pass(directory: Path, name: str, suffix: str, destination_name: str | None = None) -> Path:
    candidates = list(directory.glob(name + "_*" + suffix))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {name} compositor pass, found {len(candidates)}")
    destination = directory / ((destination_name or name) + suffix)
    candidates[0].replace(destination)
    return destination


def read_image(path: Path, flags: int) -> np.ndarray:
    result = cv2.imread(str(path), flags)
    if result is None:
        raise ValueError(f"Cannot read image: {path}")
    return result


def save_image(path: Path, array: np.ndarray) -> None:
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Could not write image: {path}")


def matrix_rows(matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def render(args: argparse.Namespace) -> dict:
    scene_path = validate_file(args.scene, "Blender scene")
    if scene_path.suffix.lower() != ".blend":
        raise ValueError("--scene must be a .blend file")
    camera_path = args.camera_json
    if camera_path is None and (scene_path.parent / "camera.json").is_file():
        camera_path = scene_path.parent / "camera.json"
    metadata = {}
    if camera_path is not None:
        camera_path = validate_file(camera_path, "Camera metadata")
        metadata = json.loads(camera_path.read_text(encoding="utf-8"))
    background_path = validate_file(args.background, "Background") if args.background is not None else None
    environment_path = validate_file(args.environment, "Environment") if args.environment is not None else None
    if environment_path is not None and environment_path.suffix.lower() not in (".hdr", ".exr"):
        raise ValueError("--environment must be an HDR or EXR file")
    bpy.ops.wm.open_mainfile(filepath=str(scene_path), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("Scene has no active camera")
    scene.frame_set(args.frame)
    objects = target_meshes(scene, metadata)
    # Assign a parent's world transform before its children. Otherwise a target
    # with mesh-to-mesh parenting could receive the yaw twice through inheritance.
    objects.sort(key=parent_depth)
    names = {obj.name for obj in objects}
    hidden = []
    for obj in scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj.name not in names
            if obj.hide_render:
                hidden.append(obj.name)
    width, height = output_size(scene, metadata, args.width, args.height)
    background = None
    background_info = None
    if background_path is not None:
        background = read_image(background_path, cv2.IMREAD_COLOR)
        original_size = [background.shape[1], background.shape[0]]
        if original_size != [width, height]:
            if not args.resize_background:
                raise ValueError(f"Background is {original_size}, output is {[width, height]}; use --resize-background to explicitly allow resampling.")
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)
        background_info = {"path": str(background_path), "original_resolution": original_size,
                           "output_resolution": [width, height], "resampled": original_size != [width, height]}
    output = prepare_output_dir(args.output)
    configure_scene(scene, args.samples, width, height, args.threads)
    device = configure_device(scene, args.device)
    environment = set_environment(scene, environment_path, args.strength, args.environment_rotation, args.keep_scene_lights)
    minimum, maximum = bounds(objects)
    center = (minimum + maximum) / 2
    initial_matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
    catcher = add_shadow_catcher(scene, objects, args.ground_size) if args.shadow_catcher else None
    views = []
    for index, yaw in enumerate(args.yaw):
        directory = output / f"view_{index:03d}"
        directory.mkdir()
        transform = Matrix.Translation(center) @ Matrix.Rotation(math.radians(yaw), 4, "Z") @ Matrix.Translation(-center)
        for obj in objects:
            obj.matrix_world = transform @ initial_matrices[obj.name]
        bpy.context.view_layer.update()
        if catcher is not None:
            catcher.hide_render = True
        setup_passes(scene, directory)
        scene.render.filepath = str(directory / "rgba_object.png")
        bpy.ops.render.render(write_still=True)
        finalize_pass(directory, "depth", ".exr")
        finalize_pass(directory, "normal", ".exr")
        finalize_pass(directory, "normal_preview", ".png", "normal")
        finalize_pass(directory, "albedo", ".png")
        object_rgba = read_image(directory / "rgba_object.png", cv2.IMREAD_UNCHANGED)
        if object_rgba.ndim != 3 or object_rgba.shape[2] != 4:
            raise RuntimeError("Blender did not produce a four-channel PNG")
        save_image(directory / "mask.png", object_rgba[..., 3])
        if catcher is not None:
            catcher.hide_render = False
            scene.use_nodes = False  # do not overwrite object-only depth/normal passes
            scene.render.filepath = str(directory / "rgba.png")
            bpy.ops.render.render(write_still=True)
        else:
            # Preserve the exact PNG rather than an unnecessary color round-trip.
            (directory / "rgba.png").write_bytes((directory / "rgba_object.png").read_bytes())
        files = {
            "rgba": f"{directory.name}/rgba.png", "object_rgba": f"{directory.name}/rgba_object.png",
            "mask": f"{directory.name}/mask.png", "depth": f"{directory.name}/depth.exr",
            "normal": f"{directory.name}/normal.exr", "normal_preview": f"{directory.name}/normal.png",
            "albedo": f"{directory.name}/albedo.png", "camera": f"{directory.name}/camera.json",
        }
        if background is not None:
            combined_rgba = read_image(directory / "rgba.png", cv2.IMREAD_UNCHANGED)
            composite, shadow = composite_over_background(
                object_rgba, combined_rgba, background,
                args.shadow_strength if catcher is not None else 0, args.shadow_floor,
            )
            save_image(directory / "composite.png", composite)
            save_image(directory / "shadow_mask.png", shadow)
            files.update({"composite": f"{directory.name}/composite.png", "shadow_mask": f"{directory.name}/shadow_mask.png"})
        projection = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height,
                                               scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
        view = {"view_id": directory.name, "directory": directory.name, "yaw_degrees": yaw, "files": files}
        write_json(directory / "camera.json", {
            "schema_version": 1, "resolution": [width, height], "camera_matrix_world": matrix_rows(camera.matrix_world),
            "projection_matrix": matrix_rows(projection), "object_yaw_degrees": yaw,
            "model_rotation_center_world": list(center), "object_matrices_world": {obj.name: matrix_rows(obj.matrix_world) for obj in objects},
        })
        views.append(view)
        print(f"OMINI_RENDER_VIEW {index + 1}/{len(args.yaw)} {directory}", flush=True)
    manifest = {
        "schema_version": 1, "scene": str(scene_path), "camera_json": str(camera_path) if camera_path else None,
        "blender_version": bpy.app.version_string, "resolution": [width, height], "samples": args.samples,
        "device": device, "background": background_info, "environment": environment,
        "model_mesh_names": [obj.name for obj in objects], "hidden_non_target_meshes": hidden,
        "frame": args.frame, "rotation_convention": "object yaw about world +Z; calibrated camera remains fixed",
        "shadow_catcher": {"enabled": catcher is not None, "strength": args.shadow_strength,
                           "floor": args.shadow_floor, "ground_size_multiplier": args.ground_size},
        "mask_semantics": "object-only alpha coverage, uint8 [0,255]; excludes the shadow catcher",
        "passes_semantics": "depth and normal EXRs are object-only raw Cycles passes; normal.png is a mapped display preview",
        "compositing": "PNG straight-alpha foreground applied once; shadow catcher darkens the supplied background outside the object",
        "render_calls": len(views) * (2 if catcher is not None else 1), "views": views,
    }
    write_json(output / "manifest.json", manifest)
    print(f"OMINI_RENDER_COMPLETE {output / 'manifest.json'}", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    render(parse_args(argv))


if __name__ == "__main__":
    main()
