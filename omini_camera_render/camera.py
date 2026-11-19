"""Silhouette-based perspective camera alignment inside Blender.

Extracted from the OMINI camera fitter, with explicit portable inputs and no
training, segmentation-model, lighting-estimator, or API dependency. This aligns
a mesh silhouette; it does not recover verified physical camera ground truth.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import bpy
import cv2
import numpy as np
from mathutils import Quaternion, Vector
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize

from .common import prepare_output_dir, validate_file, write_json
from .raster import fill_triangle_union

@dataclass
class Pose:
    azimuth_deg: float
    elevation_deg: float
    distance: float
    focal_mm: float
    shift_x: float
    shift_y: float
    roll_deg: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.azimuth_deg,
                self.elevation_deg,
                self.distance,
                self.focal_mm,
                self.shift_x,
                self.shift_y,
                self.roll_deg,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "Pose":
        return cls(*(float(value) for value in values))


@dataclass
class Score:
    total: float
    iou: float
    chamfer: float
    bbox: float
    area: float


@dataclass
class Candidate:
    pose: Pose
    score: Score
    stage: str
    optimizer: dict[str, object] | None = None


@dataclass
class Geometry:
    vertices_world_h: np.ndarray
    faces: np.ndarray


class ScoreContext:
    def __init__(self, target_mask: np.ndarray, long_side: int, padding: float = 0.25, full_frame: bool = False):
        if target_mask.ndim != 2:
            raise ValueError("target_mask must be a 2D array")
        target_mask = np.where(target_mask, 255, 0).astype(np.uint8)
        if not target_mask.any():
            raise ValueError("target_mask must contain foreground pixels")
        self.full_height, self.full_width = target_mask.shape
        self.target_bbox_full = mask_bbox(target_mask)
        x0, y0, x1, y1 = self.target_bbox_full
        span = max(x1 - x0, y1 - y0)
        pad = max(4.0, span * padding)
        self.roi_x0 = max(0.0, x0 - pad)
        self.roi_y0 = max(0.0, y0 - pad)
        self.roi_x1 = min(float(self.full_width), x1 + pad)
        self.roi_y1 = min(float(self.full_height), y1 + pad)
        if full_frame:
            self.roi_x0, self.roi_y0 = 0.0, 0.0
            self.roi_x1, self.roi_y1 = float(self.full_width), float(self.full_height)
        roi_width = max(1.0, self.roi_x1 - self.roi_x0)
        roi_height = max(1.0, self.roi_y1 - self.roi_y0)
        scale = float(long_side) / max(roi_width, roi_height)
        self.width = max(1, int(round(roi_width * scale)))
        self.height = max(1, int(round(roi_height * scale)))

        crop = target_mask[
            int(math.floor(self.roi_y0)) : int(math.ceil(self.roi_y1)),
            int(math.floor(self.roi_x0)) : int(math.ceil(self.roi_x1)),
        ].astype(np.uint8)
        self.target = cv2.resize(
            crop,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        ) >= 128
        if not self.target.any():
            raise ValueError(
                "Foreground vanished at the scoring resolution; increase "
                "--search-long-side / --final-score-long-side for a thin mask"
            )
        self.target_bbox = mask_bbox(self.target)
        self.target_area = int(self.target.sum())
        self.target_edge = mask_edge(self.target)
        self.target_edge_distance = distance_transform_edt(~self.target_edge)
        target_width = self.target_bbox[2] - self.target_bbox[0]
        target_height = self.target_bbox[3] - self.target_bbox[1]
        self.target_diag = max(1.0, math.hypot(target_width, target_height))

    def full_to_score_pixels(self, pixels: np.ndarray) -> np.ndarray:
        result = np.empty_like(pixels)
        result[:, 0] = (
            (pixels[:, 0] - self.roi_x0)
            * (self.width - 1)
            / max(1.0, self.roi_x1 - self.roi_x0)
        )
        result[:, 1] = (
            (pixels[:, 1] - self.roi_y0)
            * (self.height - 1)
            / max(1.0, self.roi_y1 - self.roi_y0)
        )
        return result

    def rasterize(self, pixels_full: np.ndarray, valid_vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        score_pixels = self.full_to_score_pixels(pixels_full)
        triangles = np.rint(np.clip(score_pixels[faces], -1.0e7, 1.0e7)).astype(np.int32)
        valid_faces = valid_vertices[faces].all(axis=1)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        fill_triangle_union(mask, triangles[valid_faces])
        return mask >= 128

    def evaluate(self, rendered: np.ndarray, pose: Pose) -> Score:
        return score_masks(self.target, rendered, self, pose)


def import_model(path: Path) -> list[bpy.types.Object]:
    """Import one OBJ/GLTF/GLB asset, preserving its hierarchy and materials."""
    before = set(bpy.data.objects)
    if path.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError("Supported mesh formats are OBJ, GLB, and GLTF")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"No mesh objects found in {path}")
    bpy.context.view_layer.update()
    # Imported cameras/lights are not used. Preserve world transforms if any
    # mesh or empty happens to be parented beneath one of those objects.
    world_matrices = {obj: obj.matrix_world.copy() for obj in imported}
    for obj in imported:
        if obj.parent is not None and obj.parent.type in {"LIGHT", "CAMERA"}:
            obj.parent = None
            obj.matrix_world = world_matrices[obj]
    for obj in imported:
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in meshes:
        obj["omini_target"] = True
        obj.hide_render = False
        obj.hide_viewport = False
    if path.suffix.lower() == ".obj":
        untextured = [obj for obj in meshes
                      if not any(slot.material is not None for slot in obj.material_slots)]
        if untextured:
            attach_obj_pbr_material(path, untextured)
    bpy.context.view_layer.update()
    return meshes


def load_target_mask(path: Path) -> np.ndarray:
    """Read a grayscale mask or the alpha of an RGBA foreground cutout.

    The image must retain the original image's pixel coordinates, not a cropped,
    recentered, or independently resized object-upload image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode mask image: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[..., 3]
        rgb = image[..., :3]
        grayscale_rgb = (
            np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 0], rgb[..., 2])
        )
        opaque = image.dtype in (np.uint8, np.uint16) and np.all(
            alpha == np.iinfo(image.dtype).max
        )
        # A grayscale mask saved as an opaque RGBA PNG is still a binary mask;
        # a color RGBA foreground instead derives coverage from its alpha.
        image = rgb[..., 0] if opaque and grayscale_rgb else alpha
    elif image.ndim == 3:
        if image.shape[2] != 3 or not np.array_equal(image[..., 0], image[..., 1]) or not np.array_equal(image[..., 0], image[..., 2]):
            raise ValueError("RGB mask must be grayscale; use RGBA alpha for a color foreground")
        image = image[..., 0]
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("Mask must be a two-dimensional image at least 8 by 8 pixels")
    # Binary PNGs sometimes encode the foreground as 1 rather than 255/65535.
    if image.dtype in (np.uint8, np.uint16) and int(image.max()) <= 1:
        mask = image != 0
    elif image.dtype == np.uint8:
        mask = image >= 128
    elif image.dtype == np.uint16:
        mask = image >= 32768
    else:
        raise ValueError("Mask must use 8-bit or 16-bit integer pixels")
    if not mask.any() or mask.all():
        raise ValueError("Mask must contain both foreground and background pixels")
    return mask


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True,
                        help="Original-coordinate binary mask, or RGBA foreground with alpha")
    parser.add_argument("--image", type=Path, help="Optional source RGB image for the overlay")
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty directory; existing results are never overwritten")
    parser.add_argument("--proxy-faces", type=int, default=20000)
    parser.add_argument("--search-long-side", type=int, default=256)
    parser.add_argument("--final-score-long-side", type=int, default=512)
    parser.add_argument("--azimuth-step", type=float, default=15.0)
    parser.add_argument("--elevations", type=float, nargs="+", default=[-15.0, 0.0, 15.0, 30.0, 45.0])
    parser.add_argument("--focals", type=float, nargs="+", default=[35.0, 50.0, 70.0])
    parser.add_argument("--coarse-keep", type=int, default=12)
    parser.add_argument("--refine-seeds", type=int, default=3,
                        help="Use 0 to disable local refinement for a quick diagnostic")
    parser.add_argument("--refine-maxfev", type=int, default=260)
    parser.add_argument("--elevation-min", type=float, default=-30.0)
    parser.add_argument("--elevation-max", type=float, default=70.0)
    parser.add_argument("--roll-bound", type=float, default=10.0)
    args = parser.parse_args(argv)
    numeric = [args.azimuth_step, args.elevation_min, args.elevation_max, args.roll_bound,
               *args.elevations, *args.focals]
    if not all(math.isfinite(x) for x in numeric):
        parser.error("All numeric camera parameters must be finite")
    if not 0 < args.azimuth_step <= 360:
        parser.error("--azimuth-step must be in (0, 360]")
    if args.proxy_faces < 4 or min(args.search_long_side, args.final_score_long_side) < 32:
        parser.error("Use at least 4 proxy faces and 32-pixel scoring resolution")
    if args.coarse_keep < 1 or args.refine_seeds < 0 or args.refine_maxfev < 1:
        parser.error("Require positive coarse-keep/maxfev and nonnegative refine-seeds")
    if not -89 < args.elevation_min < args.elevation_max < 89:
        parser.error("Elevation bounds must be ordered strictly within (-89, 89)")
    if not all(args.elevation_min <= x <= args.elevation_max for x in args.elevations):
        parser.error("Coarse elevations must lie within the elevation bounds")
    if not all(24 <= x <= 120 for x in args.focals):
        parser.error("Coarse focal lengths must lie in [24, 120] mm")
    if not 0 <= args.roll_bound <= 180:
        parser.error("--roll-bound must lie in [0, 180]")
    return args


def attach_obj_pbr_material(path: Path, objects: Sequence[bpy.types.Object]) -> None:
    """Bind Rodin's sidecar PBR textures when an OBJ has no MTL file."""
    texture_paths = {
        "diffuse": path.parent / "texture_diffuse.png",
        "roughness": path.parent / "texture_roughness.png",
        "metallic": path.parent / "texture_metallic.png",
        "normal": path.parent / "texture_normal.png",
    }
    if not texture_paths["diffuse"].is_file():
        return

    material = bpy.data.materials.new(name=f"{path.stem}_RodinPBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError("Principled BSDF node is unavailable")

    def image_node(kind: str, non_color: bool = False):
        texture_path = texture_paths[kind]
        if not texture_path.is_file():
            return None
        image = bpy.data.images.load(str(texture_path), check_existing=True)
        if non_color:
            image.colorspace_settings.name = "Non-Color"
        node = nodes.new("ShaderNodeTexImage")
        node.name = f"Rodin_{kind}"
        node.label = texture_path.name
        node.image = image
        return node

    diffuse = image_node("diffuse")
    roughness = image_node("roughness", non_color=True)
    metallic = image_node("metallic", non_color=True)
    normal = image_node("normal", non_color=True)
    if diffuse is not None:
        links.new(diffuse.outputs["Color"], principled.inputs["Base Color"])
    if roughness is not None:
        links.new(roughness.outputs["Color"], principled.inputs["Roughness"])
    if metallic is not None:
        links.new(metallic.outputs["Color"], principled.inputs["Metallic"])
    if normal is not None:
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.65
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])

    for obj in objects:
        # Preserve an existing MTL/Principled material instead of replacing it.
        if not any(slot.material is not None for slot in obj.material_slots):
            obj.data.materials.clear()
            obj.data.materials.append(material)
    print(
        "OMINI_CAMERA_MATERIAL",
        f"model={path}",
        "source=rodin_obj_sidecars",
        flush=True,
    )


def model_bounds(objects: Iterable[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    minimum = np.asarray([min(point[i] for point in corners) for i in range(3)])
    maximum = np.asarray([max(point[i] for point in corners) for i in range(3)])
    return minimum, maximum


def duplicate_proxy(
    objects: Sequence[bpy.types.Object], target_faces: int
) -> list[bpy.types.Object]:
    total_faces = sum(len(obj.data.polygons) for obj in objects)
    ratio = min(1.0, float(target_faces) / max(1, total_faces))
    proxies: list[bpy.types.Object] = []
    for index, original in enumerate(objects):
        proxy = original.copy()
        proxy.data = original.data.copy()
        proxy.name = f"CameraFitProxy_{index:02d}"
        bpy.context.collection.objects.link(proxy)
        bpy.ops.object.select_all(action="DESELECT")
        proxy.select_set(True)
        bpy.context.view_layer.objects.active = proxy
        modifier = proxy.modifiers.new("CameraFitDecimate", "DECIMATE")
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        proxies.append(proxy)
    return proxies


def extract_geometry(objects: Sequence[bpy.types.Object]) -> Geometry:
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    offset = 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        local = np.asarray([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
        if len(local) == 0 or not np.isfinite(local).all():
            raise ValueError(f"Mesh has empty or non-finite vertices: {obj.name}")
        local_h = np.concatenate([local, np.ones((len(local), 1))], axis=1)
        world = np.asarray(obj.matrix_world, dtype=np.float64)
        vertices_world_h = local_h @ world.T
        faces = np.asarray(
            [triangle.vertices[:] for triangle in obj.data.loop_triangles],
            dtype=np.int32,
        )
        if len(faces) == 0:
            raise ValueError(f"Mesh contains no triangles: {obj.name}")
        vertices_parts.append(vertices_world_h)
        faces_parts.append(faces + offset)
        offset += len(local)
    return Geometry(
        vertices_world_h=np.concatenate(vertices_parts, axis=0),
        faces=np.concatenate(faces_parts, axis=0),
    )


def create_camera() -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("RecoveredCamera")
    camera_data.sensor_fit = "AUTO"
    camera_data.sensor_width = 36.0
    camera_data.sensor_height = 24.0
    camera = bpy.data.objects.new("RecoveredCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def set_camera_pose(camera: bpy.types.Object, target: Vector, pose: Pose) -> None:
    azimuth = math.radians(pose.azimuth_deg)
    elevation = math.radians(pose.elevation_deg)
    offset = Vector(
        (
            pose.distance * math.cos(elevation) * math.cos(azimuth),
            pose.distance * math.cos(elevation) * math.sin(azimuth),
            pose.distance * math.sin(elevation),
        )
    )
    camera.location = target + offset
    look = (target - camera.location).to_track_quat("-Z", "Y")
    roll = Quaternion((0.0, 0.0, 1.0), math.radians(pose.roll_deg))
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = look @ roll
    camera.data.type = "PERSP"
    camera.data.lens = pose.focal_mm
    camera.data.shift_x = pose.shift_x
    camera.data.shift_y = pose.shift_y
    bpy.context.view_layer.update()


def project_geometry(
    camera: bpy.types.Object,
    target: Vector,
    pose: Pose,
    geometry: Geometry,
    full_width: int,
    full_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_camera_pose(camera, target, pose)
    projection = np.asarray(
        camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(),
            x=full_width,
            y=full_height,
            scale_x=1.0,
            scale_y=1.0,
        ),
        dtype=np.float64,
    )
    view = np.asarray(camera.matrix_world.inverted(), dtype=np.float64)
    clip = geometry.vertices_world_h @ (projection @ view).T
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0.0)
    ndc = np.zeros((len(clip), 2), dtype=np.float64)
    ndc[valid] = clip[valid, :2] / clip[valid, 3:4]
    pixels = np.empty_like(ndc)
    pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (full_width - 1)
    pixels[:, 1] = (0.5 - ndc[:, 1] * 0.5) * (full_height - 1)
    return pixels, valid


def projected_bbox(pixels: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float]:
    points = pixels[valid]
    if len(points) == 0:
        raise RuntimeError("All proxy vertices are behind the camera")
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def fit_distance_and_shift(
    base_pose: Pose,
    camera: bpy.types.Object,
    target: Vector,
    geometry: Geometry,
    context: ScoreContext,
    distance_bounds: tuple[float, float],
) -> Pose:
    pose = Pose.from_array(base_pose.as_array())
    pose.shift_x = 0.0
    pose.shift_y = 0.0
    target_bbox = context.target_bbox_full
    target_width = max(1.0, target_bbox[2] - target_bbox[0])
    target_height = max(1.0, target_bbox[3] - target_bbox[1])

    for _ in range(3):
        pixels, valid = project_geometry(
            camera,
            target,
            pose,
            geometry,
            context.full_width,
            context.full_height,
        )
        bbox = projected_bbox(pixels, valid)
        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        ratio = math.sqrt((width / target_width) * (height / target_height))
        pose.distance = float(
            np.clip(pose.distance * ratio, distance_bounds[0], distance_bounds[1])
        )

    pixels, valid = project_geometry(
        camera,
        target,
        pose,
        geometry,
        context.full_width,
        context.full_height,
    )
    bbox = projected_bbox(pixels, valid)
    current_center = np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])
    target_center = np.asarray(
        [
            (target_bbox[0] + target_bbox[2]) * 0.5,
            (target_bbox[1] + target_bbox[3]) * 0.5,
        ]
    )

    epsilon = 0.05
    x_probe = Pose.from_array(pose.as_array())
    x_probe.shift_x = epsilon
    x_pixels, x_valid = project_geometry(
        camera,
        target,
        x_probe,
        geometry,
        context.full_width,
        context.full_height,
    )
    x_bbox = projected_bbox(x_pixels, x_valid)
    x_center = (x_bbox[0] + x_bbox[2]) * 0.5
    x_slope = (x_center - current_center[0]) / epsilon

    y_probe = Pose.from_array(pose.as_array())
    y_probe.shift_y = epsilon
    y_pixels, y_valid = project_geometry(
        camera,
        target,
        y_probe,
        geometry,
        context.full_width,
        context.full_height,
    )
    y_bbox = projected_bbox(y_pixels, y_valid)
    y_center = (y_bbox[1] + y_bbox[3]) * 0.5
    y_slope = (y_center - current_center[1]) / epsilon

    if abs(x_slope) > 1.0e-6:
        pose.shift_x = float(np.clip((target_center[0] - current_center[0]) / x_slope, -1.5, 1.5))
    if abs(y_slope) > 1.0e-6:
        pose.shift_y = float(np.clip((target_center[1] - current_center[1]) / y_slope, -1.5, 1.5))
    return pose


def render_proxy_mask(
    pose: Pose,
    camera: bpy.types.Object,
    target: Vector,
    geometry: Geometry,
    context: ScoreContext,
) -> np.ndarray:
    pixels, valid = project_geometry(
        camera,
        target,
        pose,
        geometry,
        context.full_width,
        context.full_height,
    )
    return context.rasterize(pixels, valid, geometry.faces)


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mask_edge(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return mask & ~(eroded.astype(bool))


def score_masks(
    target: np.ndarray,
    rendered: np.ndarray,
    context: ScoreContext,
    pose: Pose,
) -> Score:
    rendered_area = int(rendered.sum())
    if rendered_area == 0:
        return Score(total=10.0, iou=0.0, chamfer=4.0, bbox=4.0, area=4.0)
    intersection = int(np.logical_and(target, rendered).sum())
    union = int(np.logical_or(target, rendered).sum())
    iou = intersection / max(1, union)

    rendered_edge = mask_edge(rendered)
    if rendered_edge.any():
        rendered_distance = distance_transform_edt(~rendered_edge)
        chamfer = (
            float(context.target_edge_distance[rendered_edge].mean())
            + float(rendered_distance[context.target_edge].mean())
        ) / (2.0 * context.target_diag)
    else:
        chamfer = 4.0

    target_bbox = context.target_bbox
    rendered_bbox = mask_bbox(rendered)
    bbox_scale = np.asarray(
        [context.width, context.height, context.width, context.height],
        dtype=np.float64,
    )
    bbox_loss = float(
        np.mean(
            np.abs(np.asarray(rendered_bbox) - np.asarray(target_bbox))
            / np.maximum(1.0, bbox_scale)
        )
    )
    area_loss = abs(math.log((rendered_area + 1.0) / (context.target_area + 1.0)))
    roll_prior = 0.002 * (pose.roll_deg / 10.0) ** 2
    focal_prior = 0.002 * math.log(max(1.0, pose.focal_mm) / 50.0) ** 2
    total = (
        0.55 * (1.0 - iou)
        + 0.30 * chamfer
        + 0.10 * bbox_loss
        + 0.05 * area_loss
        + roll_prior
        + focal_prior
    )
    return Score(
        total=float(total),
        iou=float(iou),
        chamfer=float(chamfer),
        bbox=float(bbox_loss),
        area=float(area_loss),
    )


def coarse_search(
    args: argparse.Namespace,
    camera: bpy.types.Object,
    target: Vector,
    geometry: Geometry,
    context: ScoreContext,
    distance_bounds: tuple[float, float],
    initial_distance: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    azimuths = np.arange(-180.0, 180.0, args.azimuth_step)
    total = len(azimuths) * len(args.elevations) * len(args.focals)
    started = time.perf_counter()
    index = 0
    for focal in args.focals:
        for elevation in args.elevations:
            for azimuth in azimuths:
                index += 1
                base = Pose(
                    azimuth_deg=float(azimuth),
                    elevation_deg=float(elevation),
                    distance=initial_distance,
                    focal_mm=float(focal),
                    shift_x=0.0,
                    shift_y=0.0,
                    roll_deg=0.0,
                )
                pose = fit_distance_and_shift(
                    base,
                    camera,
                    target,
                    geometry,
                    context,
                    distance_bounds,
                )
                mask = render_proxy_mask(pose, camera, target, geometry, context)
                score = context.evaluate(mask, pose)
                candidates.append(Candidate(pose=pose, score=score, stage="coarse"))
                if index % 60 == 0 or index == total:
                    elapsed = time.perf_counter() - started
                    print(
                        "OMINI_CAMERA_PROGRESS",
                        f"coarse={index}/{total}",
                        f"elapsed={elapsed:.1f}s",
                        f"best={min(item.score.total for item in candidates):.6f}",
                        flush=True,
                    )
    candidates.sort(key=lambda item: item.score.total)
    return diverse_candidates(candidates, args.coarse_keep)


def diverse_candidates(candidates: Sequence[Candidate], keep: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in candidates:
        duplicate = False
        for existing in selected:
            azimuth_delta = abs(
                ((candidate.pose.azimuth_deg - existing.pose.azimuth_deg + 180.0) % 360.0)
                - 180.0
            )
            if (
                azimuth_delta < 10.0
                and abs(candidate.pose.elevation_deg - existing.pose.elevation_deg) < 8.0
                and abs(candidate.pose.focal_mm - existing.pose.focal_mm) < 12.0
            ):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= keep:
            break
    return selected


def refine_candidates(
    args: argparse.Namespace,
    seeds: Sequence[Candidate],
    camera: bpy.types.Object,
    target: Vector,
    geometry: Geometry,
    context: ScoreContext,
    distance_bounds: tuple[float, float],
) -> list[Candidate]:
    refined: list[Candidate] = []
    for seed_index, seed in enumerate(seeds[: args.refine_seeds]):
        start = seed.pose.as_array()
        bounds = [
            (start[0] - 22.0, start[0] + 22.0),
            (
                max(args.elevation_min, start[1] - 20.0),
                min(args.elevation_max, start[1] + 20.0),
            ),
            (
                max(distance_bounds[0], start[2] * 0.65),
                min(distance_bounds[1], start[2] * 1.35),
            ),
            (max(24.0, start[3] - 25.0), min(120.0, start[3] + 25.0)),
            (max(-1.5, start[4] - 0.25), min(1.5, start[4] + 0.25)),
            (max(-1.5, start[5] - 0.25), min(1.5, start[5] + 0.25)),
            (-args.roll_bound, args.roll_bound),
        ]

        evaluations = 0

        def objective(values: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            pose = Pose.from_array(values)
            mask = render_proxy_mask(pose, camera, target, geometry, context)
            return context.evaluate(mask, pose).total

        result = minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={
                "maxfev": args.refine_maxfev,
                "xtol": 1.0e-3,
                "ftol": 1.0e-5,
                "disp": False,
            },
        )
        pose = Pose.from_array(result.x)
        mask = render_proxy_mask(pose, camera, target, geometry, context)
        score = context.evaluate(mask, pose)
        candidate = Candidate(
            pose=pose, score=score, stage="refined",
            optimizer={"method": "Powell", "success": bool(result.success),
                       "status": int(result.status), "message": str(result.message),
                       "nfev": int(result.nfev), "nit": int(result.nit),
                       "seed_score": asdict(seed.score)},
        )
        refined.append(candidate)
        print(
            "OMINI_CAMERA_PROGRESS",
            f"refine_seed={seed_index + 1}",
            f"evals={evaluations}",
            f"success={result.success}",
            f"loss={score.total:.6f}",
            f"iou={score.iou:.6f}",
            flush=True,
        )
    refined.sort(key=lambda item: item.score.total)
    return refined


def rasterize_full_image(
    pose: Pose,
    camera: bpy.types.Object,
    target: Vector,
    geometry: Geometry,
    width: int,
    height: int,
) -> np.ndarray:
    pixels, valid = project_geometry(camera, target, pose, geometry, width, height)
    triangles = np.rint(np.clip(pixels[geometry.faces], -1.0e7, 1.0e7)).astype(np.int32)
    valid_faces = valid[geometry.faces].all(axis=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    fill_triangle_union(mask, triangles[valid_faces])
    return mask


def write_mask_overlay(
    original: np.ndarray,
    target_mask: np.ndarray,
    rendered_mask: np.ndarray,
    path: Path,
) -> None:
    overlay = original.astype(np.float32).copy()
    target_only = target_mask & ~(rendered_mask >= 128)
    rendered_only = (rendered_mask >= 128) & ~target_mask
    overlap = target_mask & (rendered_mask >= 128)
    colors = np.zeros_like(original, dtype=np.float32)
    colors[target_only] = (0, 220, 0)
    colors[rendered_only] = (0, 0, 255)
    colors[overlap] = (255, 255, 255)
    union = target_mask | (rendered_mask >= 128)
    overlay[union] = 0.45 * overlay[union] + 0.55 * colors[union]
    if not cv2.imwrite(str(path), np.clip(overlay, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"Could not save mask overlay: {path}")


def candidate_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "stage": candidate.stage,
        "pose": asdict(candidate.pose),
        "score": asdict(candidate.score),
        "optimizer": candidate.optimizer,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    mesh_path = validate_file(args.mesh, "mesh")
    mask_path = validate_file(args.mask, "mask")
    image_path = validate_file(args.image, "image") if args.image else None
    target_mask = load_target_mask(mask_path)
    height, width = target_mask.shape
    if image_path:
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(f"Cannot decode source image: {image_path}")
        if original.shape[:2] != target_mask.shape:
            raise ValueError("Mask and source image must have identical pixel dimensions")
    else:
        original = np.full((height, width, 3), 48, dtype=np.uint8)
    if mesh_path.suffix.lower() not in {".obj", ".glb", ".gltf"}:
        raise ValueError("Supported mesh formats are OBJ, GLB, and GLTF")
    output_dir = prepare_output_dir(args.output)

    # Build an independent scene; never alter the user's source mesh or .blend.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    originals = import_model(mesh_path)
    minimum, maximum = model_bounds(originals)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(maximum - center))
    if not np.isfinite(radius) or radius <= 1.0e-12:
        raise ValueError("Mesh must have finite, nonzero spatial extent")
    target = Vector(center.tolist())
    distance_bounds = (radius * 1.08, radius * 20.0)
    camera = create_camera()
    camera.data.clip_start = max(radius * 1.0e-5, 1.0e-8)
    camera.data.clip_end = max(radius * 100.0, camera.data.clip_start * 100)
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    proxies = duplicate_proxy(originals, args.proxy_faces)
    geometry = extract_geometry(proxies)
    context = ScoreContext(target_mask, args.search_long_side)
    coarse = coarse_search(args, camera, target, geometry, context,
                           distance_bounds, radius * 4.0)
    refined = refine_candidates(args, coarse, camera, target, geometry,
                                context, distance_bounds)
    full_geometry = extract_geometry(originals)
    final_context = ScoreContext(target_mask, args.final_score_long_side)
    rescored = []
    # Always retain coarse candidates: a finite Powell budget is not a guarantee
    # of improvement, and proxy ranking can change on the original mesh.
    for candidate in [*coarse, *refined]:
        mask = render_proxy_mask(candidate.pose, camera, target,
                                 full_geometry, final_context)
        rescored.append(Candidate(candidate.pose, final_context.evaluate(mask, candidate.pose),
                                   "full_mesh_rescore:" + candidate.stage, candidate.optimizer))
    rescored.sort(key=lambda item: item.score.total)
    best = rescored[0]
    if not math.isfinite(best.score.total) or best.score.iou <= 0:
        raise RuntimeError(
            "No viable projected overlap was found. Check mesh geometry, "
            "mask coordinates, and the camera search ranges."
        )
    aligned = rasterize_full_image(best.pose, camera, target, full_geometry, width, height)
    exact_context = ScoreContext(target_mask, max(height, width), full_frame=True)
    exact_score = score_masks(exact_context.target, aligned >= 128, exact_context, best.pose)
    for name, image in [("target_mask.png", target_mask.astype(np.uint8) * 255),
                        ("aligned_mask.png", aligned)]:
        if not cv2.imwrite(str(output_dir / name), image):
            raise RuntimeError(f"Could not save {name}")
    write_mask_overlay(original, target_mask, aligned, output_dir / "mask_overlay.png")
    for proxy in proxies:
        bpy.data.objects.remove(proxy, do_unlink=True)
    set_camera_pose(camera, target, best.pose)
    projection = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
                                           x=width, y=height, scale_x=1.0, scale_y=1.0)
    # Image textures imported from the explicit asset are packed for portability.
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "scene.blend"), check_existing=False)
    payload = {
        "schema_version": 1,
        "tool": "omini-camera-render",
        "blender_version": bpy.app.version_string,
        "source_mesh": str(mesh_path),
        "source_mask": str(mask_path),
        "source_image": str(image_path) if image_path else None,
        "resolution": [width, height],
        "pixel_aspect": [1.0, 1.0],
        "model_mesh_names": [obj.name for obj in originals],
        "camera_object_name": camera.name,
        "pose": asdict(best.pose),
        "score_exact": asdict(exact_score),
        "score_rescore": asdict(best.score),
        "camera_matrix_world": [list(row) for row in camera.matrix_world],
        "projection_matrix": [list(row) for row in projection],
        "sensor_width_mm": float(camera.data.sensor_width),
        "sensor_height_mm": float(camera.data.sensor_height),
        "model_bounds_min": minimum.tolist(),
        "model_bounds_max": maximum.tolist(),
        "model_radius": radius,
        "proxy_faces": int(len(geometry.faces)),
        "full_faces": int(len(full_geometry.faces)),
        "elapsed_seconds": time.perf_counter() - started,
        "search_parameters": {name: getattr(args, name) for name in
            ["proxy_faces", "search_long_side", "final_score_long_side", "azimuth_step",
             "elevations", "focals", "coarse_keep", "refine_seeds", "refine_maxfev",
             "elevation_min", "elevation_max", "roll_bound"]},
        "coarse_candidates": [candidate_payload(item) for item in coarse],
        "refined_candidates": [candidate_payload(item) for item in refined],
        "full_mesh_candidates": [candidate_payload(item) for item in rescored],
        "interpretation": "Projected silhouette alignment, not ground-truth camera-pose recovery.",
        "selection_policy": "Full-mesh reranking of retained coarse and refined candidates.",
    }
    write_json(output_dir / "camera.json", payload)
    print(f"OMINI_CAMERA_RESULT iou={exact_score.iou:.6f} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
