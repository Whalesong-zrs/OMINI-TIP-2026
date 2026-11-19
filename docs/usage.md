# Detailed usage

All commands below run from the repository root, not from this documentation directory.
For an overview and installation instructions, see the [project README](../README.md).
This guide covers the standalone camera-calibration and Blender-rendering tools.

## Layout

```text
omini_camera_render/
  camera.py           # coarse camera search, Powell refinement, scene export
  render.py           # object rendering, passes, shadow catcher, compositing
  device.py           # CPU / supported Cycles GPU selection
  common.py           # input validation and non-destructive output handling
scripts/
  fit_camera.py       # run inside Blender
  render.py           # run inside Blender
  make_demo.py        # generate a tiny synthetic example locally
tests/                # lightweight tests and integration-output checks
docs/PROVENANCE.md    # source lineage and intentionally excluded components
```

## Requirements

- Blender with Cycles, Python 3.10 or newer.
- NumPy, SciPy, and OpenCV installed for **Blender's Python**, not just the system Python.
- No GPU is required. Use `--device cpu` for the portable smoke test.
- OBJ, GLB and GLTF inputs are supported. Keep OBJ/MTL/texture sidecars together;
  GLB with embedded materials is the easiest portable input.

The code includes Blender 3.6/4.x import branches; see
[VALIDATION.md](VALIDATION.md) for versions actually tested. A compatibility
branch is not a claim that every Blender/platform combination has been tested.

### Isolated Python dependencies

Set these to your own installation. The paths below are examples, not hardcoded
requirements in the implementation:

```bash
export BLENDER="/path/to/blender"
export BLENDER_PYTHON="/path/to/blender/python/bin/python3.10"

# Run from the repository root. This does NOT change Blender's bundled environment.
PY_TAG="$("$BLENDER_PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
"$BLENDER_PYTHON" -m pip install --target ".blender-deps/$PY_TAG" -r requirements.txt
```

For example, a macOS Blender 3.6 application normally contains its interpreter
under `Blender.app/Contents/Resources/3.6/python/bin/`. The runtime bootstrap
loads only `.blender-deps/python<matching-major.minor>` from this checkout.
Use Blender's interpreter so compiled wheels have the correct Python ABI.
Alternatively, install the requirements into an environment already visible to
Blender. Do not copy arbitrary system `site-packages` into Blender.

## Minimal end-to-end example

All assets in this example are generated locally. No paper datasets or network
services are used. Run from the repository root; output folders must be new or
empty. Choose a different folder for each subsequent run.

```bash
# 1. Create an asymmetric synthetic object, an original-coordinate mask,
#    a reference image, a clean background, and a tiny environment map.
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/make_demo.py -- --output demo_assets

# 2. Fit a camera (reduced search for this smoke test; normal CLI defaults
#    use a denser grid and three Powell starting points).
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/fit_camera.py -- \
  --mesh demo_assets/mesh.glb --mask demo_assets/mask.png \
  --image demo_assets/reference.png --output outputs/demo_fit \
  --azimuth-step 45 --elevations 0 15 30 --focals 50 \
  --refine-seeds 1 --refine-maxfev 180

# 3. Render the same fixed camera at two object rotations; include shadows.
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/render.py -- \
  --scene outputs/demo_fit/scene.blend \
  --background demo_assets/background.png \
  --environment demo_assets/environment.exr \
  --output outputs/demo_render --device cpu --samples 16 \
  --yaw 0 30 --shadow-catcher

# 4. Check that the expected fitting and rendering artifacts were produced.
python3 tests/check_smoke.py --fit outputs/demo_fit --render outputs/demo_render
```

If your Blender distribution launches via a wrapper, set `BLENDER` to that
wrapper. Paths containing spaces are supported; keep the shell quotes.

## Use your own inputs

### Camera fitting

```bash
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/fit_camera.py -- \
  --mesh /path/to/asset.glb --mask /path/to/mask.png \
  --image /path/to/original.png --output outputs/my_fit
```

- The mask must retain the **original image coordinates and dimensions**. Do not
  use a cropped/recentered/resized asset-upload cutout as a full-image mask.
- Use an 8/16-bit grayscale mask (white foreground), or an RGBA image whose alpha
  contains the foreground. A supplied original image must have the same dimensions.
- `--image` is optional and is used for overlay visualization, not appearance fitting.
- The optimizer searches azimuth, elevation, distance, focal length, lens shifts,
  and roll. The objective combines IoU, boundary, bounding-box and area alignment.
- Search uses a decimated proxy and CPU silhouette rasterization; it does not run
  Cycles for each candidate. Retained coarse and refined candidates are reranked
  on the full mesh. Optimizer budget/convergence is recorded explicitly.
- Camera fitting is silhouette alignment, not verified physical camera recovery;
  symmetric geometry, occlusion, segmentation errors and shape errors can cause ambiguity.

Outputs include `camera.json`, `scene.blend`, `target_mask.png`, `aligned_mask.png`
and `mask_overlay.png`. Imported image textures are packed into the saved scene.
`camera.json` records image resolution, camera/projection matrices, target object
names, fitted parameters, scores, search configuration and candidate history.

### Rendering

```bash
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python scripts/render.py -- \
  --scene outputs/my_fit/scene.blend --output outputs/my_render \
  --background /path/to/clean_background.png \
  --environment /path/to/lighting.exr --strength 1.0 \
  --device auto --samples 32 --yaw 0
```

- `--camera-json` defaults to `camera.json` beside the scene.
- Omit `--background` for transparent object outputs. A background must already
  have the original target removed/inpainted; this tool does not do that step.
- Omit `--environment` to use the scene's World lighting (or a neutral fallback).
  HDR/EXR estimation and automatic illumination correction are not included.
- `--environment-rotation` rotates the environment around world Z in degrees.
- `--shadow-catcher` adds a horizontal ground plane below the target. This is a
  simple contact-plane assumption, not arbitrary scene geometry reconstruction.
- `--yaw 0 30 -30` renders a list of object rotations with a fixed camera. This is
  not a general skeletal-animation or simulation pipeline.
- `--device auto` tries available Cycles GPU backends and falls back to CPU.
  Explicit `cpu`, `cuda`, `optix`, `hip`, or `metal` requests fail clearly if unavailable.
- Output dimensions default to the calibrated image size. Use `--width` and/or
  `--height` to override; a single dimension preserves aspect ratio, and setting
  both must preserve the calibrated aspect ratio. A mismatched
  background is rejected unless `--resize-background` is explicitly supplied.
- Render target selection uses the camera fitter's `omini_target` tags/object
  names. Arbitrary untagged multi-object `.blend` scenes are not assumed to be a
  single target. The source `.blend` file is only read, never saved over.

Each `view_000`, `view_001`, ... contains object RGBA, mask, depth EXR, normal
EXR/preview, albedo, camera metadata, and (with a background) `composite.png`.
When a background is supplied, views also retain `shadow_mask.png` (all zero if
the shadow catcher is disabled). The root `manifest.json`
records file paths, dimensions and rendering settings.

Depth is the raw Cycles Z pass, not an antialiased coverage estimate. Unhit rays
can contain large sentinel values (for example `1e10`), including a few partial-
coverage edge pixels. Check depth validity as well as the mask; soft alpha alone
does not guarantee a valid depth sample. Normal EXR preserves raw signed values;
`normal.png` is a mapped display preview, not the raw normal data.

Run `--help` after the Blender argument separator `--` for the complete options.

## Safety, tests and distribution

- A nonempty output directory is always refused. There is no recursive-delete
  or overwrite option. Source meshes/images/scenes are never modified.
- Generated demo assets, `.blend`/EXR files, renders, dependencies, caches and
  credentials are ignored by Git. Only code, documentation and tests are intended
  for the private repository.
- Unit tests, using the same isolated dependencies installed above:

  ```bash
  PYTHONPATH=".blender-deps/$PY_TAG" "$BLENDER_PYTHON" -m unittest discover -s tests -v
  ```
- Synthetic checks establish basic operation only, not accuracy on arbitrary user assets.
- No open-source license is selected yet. Keep this review repository private
  until the authors choose a license and approve the release scope.

See [provenance and scope](PROVENANCE.md) for the extraction history.
