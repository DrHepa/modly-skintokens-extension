from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skintokens_ext.paths import EXTENSION_ID, PathSafetyError, resolve_modly_layout, validate_logical_path, validate_mesh_input
from tests.helpers import write_minimal_glb


class PathTests(unittest.TestCase):
    def test_rejects_unsafe_logical_paths(self) -> None:
        for value in ["../model.ckpt", "/abs/model.ckpt", "C:/model.ckpt", "models/AUX/file", ".backup/file", "models/CON.txt"]:
            with self.subTest(value=value):
                with self.assertRaises(PathSafetyError):
                    validate_logical_path(value)

    def test_accepts_safe_logical_path(self) -> None:
        self.assertEqual(validate_logical_path("models/skintokens/file.ckpt"), "models/skintokens/file.ckpt")

    def test_validate_mesh_input(self) -> None:
        with TemporaryDirectory() as tmp:
            mesh = write_minimal_glb(Path(tmp) / "mesh.glb")
            self.assertEqual(validate_mesh_input(mesh), mesh.resolve())

    def test_resolve_modly_layout_from_installed_extension(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / "Modly"
            ext_dir = home / "extensions" / EXTENSION_ID
            ext_dir.mkdir(parents=True)
            layout = resolve_modly_layout(ext_dir, ext_dir=ext_dir)
            self.assertEqual(layout.modly_home, home.resolve())
            self.assertEqual(layout.models_root, (home / "models").resolve())


if __name__ == "__main__":
    unittest.main()
