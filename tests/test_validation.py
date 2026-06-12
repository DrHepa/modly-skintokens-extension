from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skintokens_ext.validation import validate_output
from tests.helpers import write_minimal_glb


class ValidationTests(unittest.TestCase):
    def test_minimal_glb_validation_with_skin(self) -> None:
        with TemporaryDirectory() as tmp:
            output = write_minimal_glb(Path(tmp) / "out.glb", with_skin=True)
            result = validate_output(output)
            self.assertTrue(result.ok)
            self.assertEqual(result.details["skin_count"], 1)
            self.assertEqual(result.details["joint_count"], 1)

    def test_minimal_glb_validation_warns_without_skin(self) -> None:
        with TemporaryDirectory() as tmp:
            output = write_minimal_glb(Path(tmp) / "out.glb", with_skin=False)
            result = validate_output(output)
            self.assertTrue(result.ok)
            self.assertTrue(any("no skins" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
