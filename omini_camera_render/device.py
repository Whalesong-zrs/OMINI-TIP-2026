"""Cycles backend selection without machine-specific paths or device IDs."""

from __future__ import annotations


def configure_device(scene, requested: str = "auto") -> dict:
    import bpy

    requested = requested.lower()
    backends = {"cuda": "CUDA", "optix": "OPTIX", "hip": "HIP", "metal": "METAL"}
    if requested not in {"auto", "cpu", *backends}:
        raise ValueError(f"Unknown rendering device: {requested}")
    if requested == "cpu":
        scene.cycles.device = "CPU"
        return {"requested": requested, "backend": "CPU", "devices": ["CPU"]}

    addon = bpy.context.preferences.addons.get("cycles")
    preferences = addon.preferences if addon else None
    attempts = []
    candidates = list(backends.values()) if requested == "auto" else [backends[requested]]
    # Prefer OptiX over CUDA where both are available.
    if requested == "auto":
        candidates = ["OPTIX", "CUDA", "HIP", "METAL"]
    if preferences is not None:
        for backend in candidates:
            try:
                preferences.compute_device_type = backend
                preferences.get_devices()
                selected = [device for device in preferences.devices if device.type == backend]
                if not selected:
                    attempts.append(f"{backend}: no compatible device")
                    continue
                for device in preferences.devices:
                    device.use = device.type == backend
                scene.cycles.device = "GPU"
                return {"requested": requested, "backend": backend,
                        "devices": [device.name for device in selected]}
            except (TypeError, ValueError, RuntimeError) as error:
                attempts.append(f"{backend}: {error}")
    if requested != "auto":
        raise RuntimeError(
            f"Requested Cycles backend {requested!r} is unavailable. "
            "Use --device cpu or --device auto. " + "; ".join(attempts)
        )
    scene.cycles.device = "CPU"
    return {"requested": requested, "backend": "CPU", "devices": ["CPU"],
            "fallback": "No compatible GPU backend was found"}
