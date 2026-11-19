## OMINI: Towards Consistent 3D Object Manipulation via Generative Texturing and Adaptive Lighting｜IEEE TIP

<p align="center">
  <strong>Camera Calibration &amp; Blender Rendering</strong><br><br>
  <a href="docs/usage.md"><img src="https://img.shields.io/badge/Documentation-Usage_Guide-blue?style=flat-square" alt="Usage guide"></a>
  <a href="docs/VALIDATION.md"><img src="https://img.shields.io/badge/Blender-3.6_Tested-orange?style=flat-square" alt="Tested with Blender 3.6"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square" alt="Python 3.10 or newer"></a>
</p>

> [!NOTE]
> OMINI was originally developed in September 2024. With the rapid progress of
> generative models, some model-dependent components of the original pipeline
> are no longer up to date. This release therefore focuses on reusable Blender
> utilities that we believe remain useful for future research, including
> automatic camera calibration, rendering, shadow handling, and background
> compositing.

### 🌟 Overview

This repository provides the **automatic camera calibration** and **Blender rendering**
components of OMINI:

- **Camera calibration:** align a textured 3D asset with a target mask through
  coarse camera search and Powell-based refinement.
- **Blender rendering:** render the aligned object with a fixed camera, optional
  HDR lighting and contact shadows, then composite it with a prepared background.

The current release focuses on these two components. Users provide the 3D asset
and mask; reconstruction, texture optimization, and illumination estimation or
VLM-based correction are outside this repository's scope.
Only source code, documentation, and tests are included; local test assets,
comparison helpers, and rendered results are not distributed.

### 💻 Installation

Requirements: Blender with Cycles, Python 3.10 or newer, NumPy, SciPy, and OpenCV.
A GPU is optional. The end-to-end example has been tested with **Blender 3.6.23
on CPU**; see the [validation notes](docs/VALIDATION.md) for tested configurations.

```bash
git clone https://github.com/Whalesong-zrs/omini-camera-render.git
cd omini-camera-render
```

Set the paths to your Blender installation and its bundled Python interpreter:

```bash
export BLENDER="/path/to/blender"
export BLENDER_PYTHON="/path/to/blender/python/bin/python3.10"

# Install into a checkout-local directory, without changing Blender's environment.
PY_TAG="$("$BLENDER_PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
"$BLENDER_PYTHON" -m pip install --target ".blender-deps/$PY_TAG" -r requirements.txt
```

The launchers load dependencies matching Blender's Python version automatically.
See [environment setup](docs/usage.md#isolated-python-dependencies) for details.
Run all commands below from the repository root.

### 🔧 Inputs and Configuration

- **3D asset:** a static or baked OBJ, GLB, or GLTF file with its materials/textures.
- **Target mask:** a grayscale mask or RGBA alpha in the **original image coordinates**.
  Do not use a recentered or resized object-upload cutout.
- **Background:** an optional image with the target already removed and inpainted.
- **Lighting:** an optional equirectangular HDR/EXR environment map.

Rendering defaults to the calibrated image dimensions. Use `--device cpu` for the
portable example, or `--device auto` to select an available Cycles backend.
No model download or external API configuration is needed.

### ▶️ Running

#### Automatic camera calibration

```bash
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/fit_camera.py -- \
  --mesh /path/to/asset.glb --mask /path/to/mask.png \
  --image /path/to/original.png --output outputs/my_fit
```

The output includes `camera.json`, `scene.blend`, and mask-alignment previews.
The original image is optional and is used only for the overlay.
Calibration optimizes silhouette alignment, not image appearance, so front/back
ambiguity can remain. Inspect the exported previews before rendering.

#### Blender rendering

```bash
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/render.py -- \
  --scene outputs/my_fit/scene.blend --output outputs/my_render \
  --background /path/to/clean_background.png \
  --environment /path/to/lighting.exr \
  --device cpu --samples 32 --yaw 0 30 --shadow-catcher
```

This renders two object rotations with the same camera. Outputs include RGBA,
foreground masks, depth, normals, albedo, and background composites.
Omit `--background` for transparent outputs. Existing nonempty output directories
are preserved; choose a new directory for each run.

For a runnable example with locally generated assets, follow the
[minimal end-to-end demo](docs/usage.md#minimal-end-to-end-example).

### 📚 Documentation

- [Detailed usage guide](docs/usage.md): environment setup, synthetic demo,
  parameters, output files, and tests.
- [Validation notes](docs/VALIDATION.md): tested environment and completed checks.
- [Source and scope](docs/PROVENANCE.md): implementation lineage and release boundaries.

### 📄 License

A code license will be confirmed before the public release. This repository
currently remains private for review.
