"""Small, Blender-independent file and metadata helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def validate_file(path: str | Path, label: str = "Input") -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is not a file: {candidate}")
    return candidate


def prepare_output_dir(path: str | Path) -> Path:
    """Create a new/empty directory; never remove or overwrite user assets."""
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"Output must not be a symbolic link: {candidate}")
    candidate = candidate.resolve()
    if candidate == Path(candidate.anchor) or candidate == Path.home().resolve():
        raise ValueError("Choose a dedicated output directory, not a root/home directory")
    if candidate.exists():
        if not candidate.is_dir():
            raise NotADirectoryError(candidate)
        if any(candidate.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {candidate}. Use a new directory; "
                "existing results are never deleted automatically."
            )
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write strict JSON (reject NaN/Infinity)."""
    target = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
