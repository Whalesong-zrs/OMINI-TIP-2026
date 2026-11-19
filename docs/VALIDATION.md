# Local validation record

Date: 2026-11-19. This is a synthetic software smoke test, not a paper benchmark.

## Tested environment

- macOS, Apple Silicon
- Blender 3.6.23 / embedded Python 3.10.13
- NumPy 1.26.4, SciPy 1.13.1, OpenCV-headless 4.10.0.84
- Cycles CPU rendering; no external API, model weight or paper asset was used

Dependencies were installed into the checkout's ignored `.blender-deps/python3.10`
directory, not into the original project or Blender's bundled environment.

## Completed checks

1. All **19 unit tests passed**: output preservation, symlink rejection, strict
   JSON, launcher bootstrapping, compositing/partial alpha, shadow isolation and
   duplicate/overlapping-triangle coverage.
2. `make_demo.py` generated the asymmetric mesh, original-coordinate mask,
   reference image, clean background and synthetic HDR environment at 192×192.
3. Camera fitting used `--azimuth-step 45 --elevations 0 15 30 --focals 50
   --refine-seeds 1 --refine-maxfev 180`. Full-image silhouette IoU was
   **0.9711138819617623**. The local Powell run used its 180-evaluation budget;
   the metadata correctly reports this as budget exhaustion, not convergence.
   Both retained coarse and refined candidates participated in final selection.
4. HDR + shadow-catcher rendering completed at 192×192, 16 samples, CPU, yaw
   angles 0° and 30°. This used four render calls (object-only and object+shadow
   for each view). Both views' camera/projection matrices were identical, while
   target transforms differed.
5. PNG/EXR pass dimensions, alpha/mask agreement, signed normal values, and
   nonzero shadow masks were independently checked. Fully opaque foreground
   colors were unchanged by compositing; transparent unshadowed pixels matched
   the supplied background. Composites were visually inspected.
6. A separate no-background/no-HDR/no-shadow render completed at 128×128,
   2 samples, CPU. It produced transparent outputs and used the neutral World
   fallback. No composite image was fabricated without a background.
7. `tests/check_smoke.py` passed for the fitted scene and both rendered modes.
8. A credential-pattern scan of the intended text sources found no common
   API-token/private-key patterns. Generated media, local dependencies and
   caches are excluded from Git. This is not a guarantee of a comprehensive
   security audit.

The successful local outputs are `outputs/demo_fit_v2`, `outputs/demo_render`
and `outputs/demo_render_transparent` (ignored by Git). The earlier failed
diagnostic was preserved rather than overwritten. A fresh clone should use
the new/empty output paths shown in the README.

## Compatibility not yet exercised

Blender 4.x, Linux/Windows, GPU backends and third-party user models were not
executed in this validation run. OBJ import has version-specific code paths,
but this end-to-end fixture uses GLB. Arbitrary skeletal animation and full
multi-object scene reconstruction are outside the first release's scope.

No open-source license or public release is implied by this private review.
