"""Small dependency-free safety checks: python -m unittest discover -s tests."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from omini_camera_render.common import prepare_output_dir, validate_file, write_json


class CommonTests(unittest.TestCase):
    def test_create_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new-output"
            result = prepare_output_dir(target)
            self.assertEqual(result, target.resolve())
            self.assertTrue(result.is_dir())

    def test_empty_output_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(prepare_output_dir(Path(temporary)), Path(temporary).resolve())

    def test_nonempty_directory_is_not_destroyed(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            sentinel = target / "important.txt"
            sentinel.write_text("keep me", encoding="utf-8")
            with self.assertRaises((FileExistsError, ValueError)):
                prepare_output_dir(target)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")

    def test_output_cannot_be_an_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "original.txt"
            path.write_text("keep me", encoding="utf-8")
            with self.assertRaises((NotADirectoryError, FileExistsError, ValueError)):
                prepare_output_dir(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep me")

    def test_output_cannot_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                prepare_output_dir(link)
            self.assertEqual(list(target.iterdir()), [])

    def test_validate_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.dat"
            path.write_bytes(b"fixture")
            self.assertEqual(validate_file(path, "asset"), path.resolve())
            with self.assertRaises((FileNotFoundError, ValueError)):
                validate_file(path.parent / "missing.dat", "asset")
            with self.assertRaises((IsADirectoryError, FileNotFoundError, ValueError)):
                validate_file(path.parent, "asset")

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            payload = {"resolution": [192, 192], "score": 0.95, "label": "synthetic"}
            write_json(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_json_rejects_nonfinite_values_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            write_json(path, {"safe": True})
            with self.assertRaises(ValueError):
                write_json(path, {"score": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"safe": True})

    def test_source_syntax_and_no_legacy_absolute_paths(self):
        root = Path(__file__).resolve().parents[1]
        candidates = list((root / "omini_camera_render").glob("*.py"))
        candidates += list((root / "scripts").glob("*.py"))
        self.assertTrue(candidates)
        for path in candidates:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertFalse(node.value.startswith(("/mnt/", "/Users/", "/home/")))

    def test_blender_launchers_bootstrap_their_script_directory(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("fit_camera.py", "render.py", "make_demo.py"):
            with self.subTest(script=name):
                text = (root / "scripts" / name).read_text(encoding="utf-8")
                insertion = text.find("sys.path.insert")
                bootstrap_import = text.find("from _bootstrap import bootstrap")
                self.assertGreaterEqual(insertion, 0)
                self.assertGreater(bootstrap_import, insertion)


if __name__ == "__main__":
    unittest.main()
