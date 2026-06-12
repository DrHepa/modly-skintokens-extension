from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skintokens_ext.runtime import FakeBackend, normalize_params, run_pipeline
from tests.helpers import write_minimal_glb


class RuntimeTests(unittest.TestCase):
    def test_normalize_select_booleans(self) -> None:
        params = normalize_params({"use_skeleton": "true", "use_transfer": "false", "use_postprocess": "1"})
        self.assertTrue(params.use_skeleton)
        self.assertFalse(params.use_transfer)
        self.assertTrue(params.use_postprocess)

    def test_fake_backend_stage_order(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh = write_minimal_glb(tmp_path / "mesh.glb")
            progress_events = []
            logs = []
            result = run_pipeline(
                mesh_path=mesh,
                params={"use_transfer": "true"},
                progress=lambda percent, label, stage=None: progress_events.append((percent, stage, label)),
                log=lambda message, stage=None: logs.append((stage, message)),
                workspace_dir=tmp_path,
                backend=FakeBackend(),
            )
            self.assertTrue(result.file_path.exists())
            self.assertEqual(result.file_path.parent, tmp_path / "Workflows")
            stages = [stage for _, stage, _ in progress_events]
            self.assertEqual(stages[0], "validate-input")
            self.assertIn("generate-tokens", stages)
            self.assertEqual(stages[-1], "validate-output")

    def test_workspace_workflows_dir_is_not_nested(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workflows_dir = tmp_path / "Workflows"
            workflows_dir.mkdir()
            mesh = write_minimal_glb(tmp_path / "mesh.glb")
            result = run_pipeline(
                mesh_path=mesh,
                params={},
                progress=lambda percent, label, stage=None: None,
                log=lambda message, stage=None: None,
                workspace_dir=workflows_dir,
                backend=FakeBackend(),
            )
            self.assertEqual(result.file_path.parent, workflows_dir)


if __name__ == "__main__":
    unittest.main()
